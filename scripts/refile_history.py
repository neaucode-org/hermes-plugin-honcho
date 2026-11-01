#!/usr/bin/env python3
"""Re-file Hermes chat history into Honcho under the author tags ``james`` / ``jarvis``, with a context line.

One-time migration for memory-plan steps 0 + 4. Every live-chat turn in the Hermes transcripts
(``state.db`` of the root profile and of the ``z_k-*`` worker profiles) is filed again as two Honcho
messages, ``james`` (the user message that opened the turn) and ``jarvis`` (the final reply), each
``<[context] line>\\n\\n<original>`` chunked exactly as ``sync_turn`` chunks, with the pointer metadata
``hermes_session_id`` / ``hermes_message_id`` (+ ``hermes_profile`` for worker DBs) and the original
timestamp. One Honcho session per Hermes session, ``hermes-<profile>-<hermes_session_id>``, with
observation, reasoning and summaries off, so the local deriver re-derives nothing: the user's facts
move by copying conclusions instead.

The dry run (default) writes nothing to Honcho: it inventories the turns, reads the Honcho workspace
(sessions per peer, conclusions, message counts of the sessions that would go), writes context lines
for a sample of turns through the real ``memory_context_line`` auxiliary task and extrapolates the
token cost. Lines are cached in a jsonl, so no turn is paid for twice.

Run with a Python that imports Hermes, e.g. the managed runtime venv:

    PY=$(ls -t ~/.hermes/installs/*/environments/*/venv/bin/python | head -1)
    $PY scripts/refile_history.py --dry-run --sample 30
    # phase B, in this order, after a pg dump to ~/.hermes/backups/honcho-<ts>.dump:
    $PY scripts/refile_history.py --apply rename --yes        # snapshot old sessions, rename peers in place
    $PY scripts/refile_history.py --apply devlog --yes
    $PY scripts/refile_history.py --apply refile --yes
    $PY scripts/refile_history.py --apply refile-honcho --yes # the messages only Honcho holds
    $PY scripts/refile_history.py --apply delete-old --yes    # also the catch-up after the restart
    # --apply conclusions is the fallback when a rename is impossible (copies lose created_at)
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import importlib.util
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
HERMES_HOME = Path(os.environ.get("HERMES_ROOT", Path.home() / ".hermes"))
BASE_URL = "http://localhost:8000/v3/workspaces/hermes"

USER_PEER, AGENT_PEER = "james", "jarvis"
OLD_USER_PEER, OLD_AGENT_PEER, DEVLOG_PEER, MAIL_PEER = "jarvis", "hermes", "devlog", "correspondent"
SESSION_PREFIX = "refiled-"  # "hermes-" is taken: per-directory sessions are named after the cwd (hermes-agent)
DEVLOG_SESSION = "devlog"
# Sources Honcho files live; subagent, cron and oneshot chats are transcript-only by design.
SOURCES = ("webui", "telegram", "api_server", "tui", "cli", "kanban", "cursor-specstory")
SAMPLE_STRATA = ("webui", "telegram", "api_server", "cursor-specstory", "worker")
THREAD_ROWS = 8  # dialogue rows kept before a turn; the writer itself keeps the last three exchanges
# Haiku list price per million tokens (input, output), for the extrapolation only.
PRICE_PER_MTOK = (1.0, 5.0)

_SESSION_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


# ----- Turns (pure) -----

@dataclass
class Turn:
    profile: str
    source: str
    session_id: str
    user_id: int
    user_ts: float
    user: str
    reply_id: int
    reply_ts: float
    reply: str
    thread: List[Dict[str, Any]] = field(default_factory=list)  # dialogue before the turn, then the turn
    texts: List[str] = field(default_factory=list)  # every user row and the reply, cleaned: what live filing stored

    @property
    def key(self) -> str:
        return f"{self.profile}/{self.session_id}/{self.user_id}"


def _is_final_reply(row: Dict[str, Any]) -> bool:
    return row["role"] == "assistant" and bool((row.get("content") or "").strip()) and not row.get("tool_calls")


def parse_tool_calls(raw: Any) -> Any:
    """``state.db`` ``tool_calls`` as the harness holds it: a non-empty list, or None."""
    if not raw:
        return None
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return raw
    return value or None


def walk_turns(rows: Iterable[Dict[str, Any]], *, profile: str, source: str, session_id: str,
               is_internal: Callable[[str], bool] = lambda _t: False,
               clean: Callable[[str], str] = lambda t: t,
               skipped: Optional[collections.Counter] = None) -> List[Turn]:
    """The turns of one session, read as ``sync_turn`` files them live.

    ``rows``: the session's user and assistant rows in id order (``role``, ``content``, ``tool_calls``,
    ``id``, ``timestamp``). A turn opens at the first user row after the previous final reply (an
    assistant row with content and no tool calls) and closes at the next final reply; every user row
    in between is joined into the user side, so an interjection is kept, and the id is the opener's,
    as live. A turn whose opener is a machine-generated gateway notification is skipped, as live;
    a user message never answered is not a turn. ``thread`` holds the dialogue rows before the turn
    (bounded) plus the turn's own two rows, the shape the context-line writer reads."""
    turns: List[Turn] = []
    dialogue: List[Dict[str, Any]] = []
    users: List[Dict[str, Any]] = []
    for row in rows:
        if row["role"] == "user":
            if (row.get("content") or "").strip():
                users.append(row)
            continue
        if not _is_final_reply(row):
            continue
        reply = row
        if not users:
            dialogue.append({"role": "assistant", "content": reply["content"]})
            continue
        opener, opener_rows = users[0], list(users)
        user_text = clean("\n\n".join(u["content"] for u in users)).strip()
        reply_text = clean(reply["content"]).strip()
        prior = dialogue[-THREAD_ROWS:]
        dialogue.extend({"role": "user", "content": u["content"]} for u in users)
        dialogue.append({"role": "assistant", "content": reply["content"]})
        if len(users) > 1 and skipped is not None:
            skipped["joined_user_rows (not skipped)"] += 1
        users = []
        if is_internal(opener["content"]):
            if skipped is not None:
                skipped["internal"] += 1
            continue
        if not user_text and not reply_text:
            if skipped is not None:
                skipped["empty"] += 1
            continue
        turns.append(Turn(
            profile=profile, source=source, session_id=session_id,
            user_id=opener["id"], user_ts=opener["timestamp"], user=user_text,
            reply_id=reply["id"], reply_ts=reply["timestamp"], reply=reply_text,
            thread=prior + [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply_text}],
            texts=[clean(u["content"]).strip() for u in opener_rows] + [reply_text],
        ))
    if users and skipped is not None:
        skipped["unanswered"] += 1
    return turns


_CONTEXT_PREFIX_RE = re.compile(r"^\[context\][^\n]*\n\n")
COVER_CHARS = 80


def cover_key(text: str) -> str:
    """A message's identity for coverage: its first characters, whitespace-normalized, with the
    ``[context]`` line and the ``[continued]`` chunk prefix stripped."""
    text = _CONTEXT_PREFIX_RE.sub("", (text or "").lstrip())
    if text.startswith("[continued] "):
        text = text[len("[continued] "):]
    return re.sub(r"\s+", " ", text).strip()[:COVER_CHARS]


def coverage_index(turns: Iterable[Turn]) -> tuple[set, set]:
    """``(text keys, (hermes_session_id, row id) pairs)`` of everything the re-file writes."""
    keys, ids = set(), set()
    for t in turns:
        keys.update(cover_key(x) for x in t.texts if x)
        ids.update({(t.session_id, t.user_id), (t.session_id, t.reply_id)})
    return keys, ids


def classify_message(msg: Dict[str, Any], keys: set, ids: set) -> str:
    """``covered`` when the re-file writes this message again (by pointer or by text); ``continuation``
    for a later chunk (it goes with its first chunk); ``migration`` for a ``<prior_memory_file>`` or
    ``<backfilled_conversation>`` wrapper; else ``uncovered``: the only copy is in Honcho."""
    meta = msg.get("metadata") or {}
    content = msg.get("content") or ""
    if (meta.get("hermes_session_id"), meta.get("hermes_message_id")) in ids:
        return "covered"
    stripped = _CONTEXT_PREFIX_RE.sub("", content.lstrip())
    if stripped.startswith("[continued] "):
        return "continuation"
    if stripped.startswith(("<prior_memory_file>", "<backfilled_conversation>")):
        return "migration"
    return "covered" if cover_key(content) in keys else "uncovered"


def honcho_session_name(profile: str, session_id: str) -> str:
    return SESSION_PREFIX + _SESSION_NAME_RE.sub("-", f"{profile}-{session_id}")


# ----- Hermes (lazy: the dry run's import of the plugin needs the Hermes venv) -----

_plugin_mod = None


def plugin():
    """The fork's package, loaded by path, for ``_is_internal_gateway_turn`` and ``_chunk_message``."""
    global _plugin_mod
    if _plugin_mod is None:
        spec = importlib.util.spec_from_file_location(
            "honcho_plugin_refile", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _plugin_mod = mod
    return _plugin_mod


def databases() -> List[tuple[str, Path]]:
    dbs = [("default", HERMES_HOME / "state.db")]
    dbs += [(p.name, p / "state.db") for p in sorted((HERMES_HOME / "profiles").glob("z_k-*")) if (p / "state.db").exists()]
    return dbs


def load_turns(skipped: collections.Counter) -> List[Turn]:
    from agent.memory_manager import sanitize_context

    pkg = plugin()
    turns: List[Turn] = []
    for profile, db in databases():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        marks = ",".join("?" * len(SOURCES))
        sources = dict(con.execute(f"select id, source from sessions where source in ({marks})", SOURCES).fetchall())
        rows_by_session: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        q = (f"select m.id, m.session_id, m.role, m.content, m.tool_calls, m.timestamp from messages m "
             f"join sessions s on s.id = m.session_id where s.source in ({marks}) "
             f"and m.role in ('user', 'assistant') order by m.session_id, m.id")
        for r in con.execute(q, SOURCES):
            rows_by_session[r["session_id"]].append({
                "id": r["id"], "role": r["role"], "content": r["content"] or "",
                "tool_calls": parse_tool_calls(r["tool_calls"]), "timestamp": r["timestamp"]})
        con.close()
        for sid, rows in rows_by_session.items():
            turns += walk_turns(rows, profile=profile, source=sources[sid], session_id=sid,
                                is_internal=pkg._is_internal_gateway_turn, clean=sanitize_context, skipped=skipped)
    return turns


def stratum(turn: Turn) -> str:
    return "worker" if turn.profile != "default" else turn.source


def sample_turns(turns: List[Turn], n: int, seed: int = 31) -> List[Turn]:
    """``n`` turns spread evenly over SAMPLE_STRATA; the worker stratum is drawn from one worker DB."""
    rng = random.Random(seed)
    groups: Dict[str, List[Turn]] = collections.defaultdict(list)
    for t in turns:
        groups[stratum(t)].append(t)
    workers = sorted({t.profile for t in groups.get("worker", [])})
    if workers:
        one = max(workers, key=lambda p: sum(t.profile == p for t in groups["worker"]))
        groups["worker"] = [t for t in groups["worker"] if t.profile == one]
    strata = [s for s in SAMPLE_STRATA if groups.get(s)]
    picked: List[Turn] = []
    for i, s in enumerate(strata):
        k = n // len(strata) + (1 if i < n % len(strata) else 0)
        picked += rng.sample(groups[s], min(k, len(groups[s])))
    return picked


# ----- Context lines (Haiku, cached) -----

class LineWriter:
    """Writes the context line per turn through the real writer, records token usage per call and
    caches every result (line or empty) in a jsonl keyed by turn, so a re-run never pays twice."""

    def __init__(self, cache: Path, workers: int = 6):
        self.cache, self.workers = cache, workers
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.done: Dict[str, Dict[str, Any]] = {}
        if cache.exists():
            for raw in cache.read_text().splitlines():
                rec = json.loads(raw)
                self.done[rec["key"]] = rec
        self._lock = threading.Lock()
        self._local = threading.local()
        self.route = ""

    def _install(self) -> None:
        import agent.auxiliary_client as aux
        sys.path.insert(0, str(ROOT / "scripts"))
        from context_line_dry_run import declare_task

        self.route = declare_task()
        if getattr(aux.call_llm, "_refile_wrapped", False):
            return
        original = aux.call_llm

        def counted(*args, **kwargs):
            response = original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            self._local.usage = (getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0,
                                 getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
            self._local.model = getattr(response, "model", "") or ""
            return response

        counted._refile_wrapped = True
        aux.call_llm = counted

    def _one(self, turn: Turn) -> Dict[str, Any]:
        from _context_line import context_line

        self._local.usage, self._local.model = (0, 0), ""
        started = time.monotonic()
        line = context_line(turn.user, turn.reply, messages=turn.thread, summary=None)
        rec = {"key": turn.key, "line": line, "input_tokens": self._local.usage[0],
               "output_tokens": self._local.usage[1], "model": self._local.model,
               "seconds": round(time.monotonic() - started, 2)}
        with self._lock, self.cache.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done[turn.key] = rec
        return rec

    def write(self, turns: List[Turn]) -> Dict[str, Dict[str, Any]]:
        todo = [t for t in turns if t.key not in self.done]
        if todo:
            self._install()
            with cf.ThreadPoolExecutor(self.workers) as pool:
                for i, _ in enumerate(pool.map(self._one, todo), 1):
                    if i % 100 == 0:
                        print(f"  lines: {i}/{len(todo)}", flush=True)
        return {t.key: self.done[t.key] for t in turns}


# ----- Honcho (reads for the dry run; writes only behind --apply ... --yes) -----

class Honcho:
    def __init__(self, base: str = BASE_URL):
        import httpx
        self.c = httpx.Client(base_url=base, timeout=120)

    def call(self, method: str, path: str, body: Any = None, **params) -> Any:
        r = self.c.request(method, path, json=body, params=params or None)
        r.raise_for_status()
        return r.json() if r.content else None

    def pages(self, path: str, body: Any = None, size: int = 100) -> Iterable[Dict[str, Any]]:
        page = 1
        while True:
            d = self.call("POST", path, body if body is not None else {}, page=page, size=size)
            yield from d["items"]
            if page >= d.get("pages", 1):
                return
            page += 1

    def total(self, path: str, body: Any = None) -> int:
        return self.call("POST", path, body if body is not None else {}, page=1, size=1)["total"]

    def conclusions(self, observer: Optional[str] = None, observed: Optional[str] = None) -> List[Dict[str, Any]]:
        filters = {k: v for k, v in (("observer_id", observer), ("observed_id", observed)) if v}
        return list(self.pages("/conclusions/list", {"filters": filters}))


def inventory_honcho(h: Honcho, turns: List[Turn]) -> Dict[str, Any]:
    """Which existing sessions hold old-tag turns and whether every message in each is written again by
    the re-file (``would_delete``: fully covered) or has content found nowhere else (``kept_uncovered``).
    Mail sessions, the re-filed ones and any session with a ``james`` member always stay."""
    peers = [p["id"] for p in h.pages("/peers/list")]
    members: Dict[str, set] = collections.defaultdict(set)
    for p in peers:
        for s in h.pages(f"/peers/{p}/sessions"):
            members[s["id"]].add(p)
    all_sessions = {s["id"]: s for s in h.pages("/sessions/list")}
    keep, go, empty = {}, {}, []
    for sid in all_sessions:
        m = members.get(sid, set())
        if MAIL_PEER in m:
            keep[sid] = "mail"
        elif sid.startswith(SESSION_PREFIX) or sid == DEVLOG_SESSION or USER_PEER in m:
            keep[sid] = "new-tags"
        elif m:
            go[sid] = sorted(m)
        else:
            empty.append(sid)
    keys, ids = coverage_index(turns)
    with cf.ThreadPoolExecutor(8) as pool:
        msgs = dict(zip(go, pool.map(lambda s: list(h.pages(f"/sessions/{s}/messages/list")), go)))
    per_session = {s: collections.Counter(classify_message(m, keys, ids) for m in ms) for s, ms in msgs.items()}
    covered = sorted(s for s, c in per_session.items() if not c["uncovered"] and not c["migration"])
    uncovered = sorted(s for s in go if s not in set(covered))
    examples = {s: next((m["content"][:100] for m in msgs[s] if classify_message(m, keys, ids) == "uncovered"), "")
                for s in uncovered}
    total = lambda ss, k=None: sum((per_session[s][k] if k else sum(per_session[s].values())) for s in ss)
    return {"peers": peers, "sessions_total": len(all_sessions), "keep": collections.Counter(keep.values()),
            "old_sessions": len(go), "old_messages": total(go),
            "old_messages_by_class": sum(per_session.values(), collections.Counter()),
            "old_by_peers": collections.Counter(",".join(v) for v in go.values()),
            "would_delete": covered, "would_delete_messages": total(covered),
            "kept_uncovered": uncovered, "kept_uncovered_messages": total(uncovered),
            "kept_uncovered_by_class": sum((per_session[s] for s in uncovered), collections.Counter()),
            "kept_uncovered_by_prefix": collections.Counter(_group(s) for s in uncovered),
            "kept_uncovered_examples": examples, "no_members": empty}


def _group(session_name: str) -> str:
    for prefix in ("cursor-", "agent-main-telegram-", "agent-housekeeper-", "agent-", "api-"):
        if session_name.startswith(prefix):
            return prefix
    return "hex12" if re.fullmatch(r"[0-9a-f]{12}", session_name) else "other"


# ----- Phase B steps -----

def apply_conclusions(h: Honcho) -> None:
    """Copy the user's facts (``jarvis`` about ``jarvis``) to ``james`` about ``james``, session-less so
    deleting old sessions never cascades into the copies; then copy the card."""
    h.call("POST", "/peers", {"id": USER_PEER})
    source = h.conclusions(OLD_USER_PEER, OLD_USER_PEER)
    have = {c["content"] for c in h.conclusions(USER_PEER, USER_PEER)}
    todo = [c["content"] for c in source if c["content"] not in have]
    for i in range(0, len(todo), 100):
        h.call("POST", "/conclusions", {"conclusions": [
            {"content": c, "observer_id": USER_PEER, "observed_id": USER_PEER} for c in todo[i:i + 100]]})
    card = (h.call("GET", f"/peers/{OLD_USER_PEER}/card") or {}).get("peer_card") or []
    if card:
        h.call("PUT", f"/peers/{USER_PEER}/card", {"peer_card": card})
    print(f"conclusions: {len(source)} under {OLD_USER_PEER}, {len(todo)} copied, "
          f"{h.total('/conclusions/list', {'filters': {'observer_id': USER_PEER, 'observed_id': USER_PEER}})} under {USER_PEER}; "
          f"card {len(card)} facts copied")


def _ensure_session(h: Honcho, name: str, metadata: Dict[str, Any]) -> None:
    off = {"observe_me": False, "observe_others": False}
    h.call("POST", "/sessions", {
        "id": name, "metadata": metadata, "peers": {USER_PEER: off, AGENT_PEER: off},
        "configuration": {"reasoning": {"enabled": False}, "summary": {"enabled": False}}})


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


def apply_devlog(h: Honcho, ledger: Path) -> None:
    """Each journal conclusion becomes an entry by ``jarvis`` with ``kind: decision`` in session ``devlog``."""
    done = set(ledger.read_text().splitlines()) if ledger.exists() else set()
    rows = [c for c in h.conclusions(observed=DEVLOG_PEER) if f"devlog/{c['id']}" not in done]
    _ensure_session(h, DEVLOG_SESSION, {"kind": "decision", "source": "devlog"})
    for i in range(0, len(rows), 100):
        batch = rows[i:i + 100]
        h.call("POST", f"/sessions/{DEVLOG_SESSION}/messages", {"messages": [
            {"peer_id": AGENT_PEER, "content": c["content"], "created_at": c["created_at"],
             "metadata": {"kind": "decision", "source": "devlog", "devlog_conclusion_id": c["id"]}} for c in batch]})
        with ledger.open("a") as f:
            f.writelines(f"devlog/{c['id']}\n" for c in batch)
    print(f"devlog: {len(rows)} entries filed in session '{DEVLOG_SESSION}'")


def turn_messages(turn: Turn, line: str, limit: int) -> List[Dict[str, Any]]:
    """The Honcho messages for one turn, as ``sync_turn`` builds them: the line prepended before chunking."""
    chunk = plugin().HonchoMemoryProvider._chunk_message
    meta = {"hermes_session_id": turn.session_id}
    if turn.profile != "default":
        meta["hermes_profile"] = turn.profile
    out = []
    for peer, text, row_id, ts in ((USER_PEER, turn.user, turn.user_id, turn.user_ts),
                                   (AGENT_PEER, turn.reply, turn.reply_id, turn.reply_ts)):
        if not text:
            continue
        full = f"{line}\n\n{text}" if line else text
        out += [{"peer_id": peer, "content": c, "created_at": _iso(ts),
                 "metadata": {**meta, "hermes_message_id": row_id}} for c in chunk(full, limit)]
    return out


def apply_refile(h: Honcho, turns: List[Turn], lines: Dict[str, Dict[str, Any]], ledger: Path, limit: int) -> None:
    done = set(ledger.read_text().splitlines()) if ledger.exists() else set()
    by_session: Dict[tuple, List[Turn]] = collections.defaultdict(list)
    for t in turns:
        if t.key not in done:
            by_session[(t.profile, t.session_id, t.source)].append(t)
    filed = msgs = 0
    for (profile, sid, source), group in by_session.items():
        name = honcho_session_name(profile, sid)
        _ensure_session(h, name, {"hermes_session_id": sid, "hermes_profile": profile, "source": source})
        batch: List[Dict[str, Any]] = []
        keys: List[str] = []
        for t in group:
            batch += turn_messages(t, lines.get(t.key, {}).get("line", ""), limit)
            keys.append(t.key)
        for i in range(0, len(batch), 100):
            h.call("POST", f"/sessions/{name}/messages", {"messages": batch[i:i + 100]})
        with ledger.open("a") as f:
            f.writelines(k + "\n" for k in keys)
        filed, msgs = filed + len(group), msgs + len(batch)
        if filed and len(by_session) > 50 and filed % 500 < len(group):
            print(f"  refile: {filed} turns, {msgs} messages", flush=True)
    print(f"refile: {filed} turns in {len(by_session)} sessions, {msgs} messages filed")


def session_members(h: Honcho) -> Dict[str, set]:
    members: Dict[str, set] = collections.defaultdict(set)
    for p in [p["id"] for p in h.pages("/peers/list")]:
        for s in h.pages(f"/peers/{p}/sessions"):
            members[s["id"]].add(p)
    return members


def is_protected(name: str, members: set) -> bool:
    """Mail threads and the sessions this migration writes are never deleted or re-filed."""
    return MAIL_PEER in members or name.startswith(SESSION_PREFIX) or name == DEVLOG_SESSION


def snapshot_old_sessions(h: Honcho, path: Path) -> Dict[str, Any]:
    """The old-tag sessions, taken once BEFORE the rename (afterwards every one of them has a ``james``
    member, so membership no longer tells them apart): name -> members and metadata."""
    if path.exists():
        return json.loads(path.read_text())
    members = session_members(h)
    meta = {s["id"]: s.get("metadata") or {} for s in h.pages("/sessions/list")}
    old = {sid: {"members": sorted(m), "metadata": meta.get(sid, {})} for sid, m in members.items()
           if m and not is_protected(sid, m) and USER_PEER not in m}
    snap = {"taken_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sessions": old}
    path.write_text(json.dumps(snap, indent=1, ensure_ascii=False))
    print(f"snapshot: {len(old)} old-tag sessions written to {path}")
    return snap


# The rename: every column that holds a peer name in workspace ``hermes`` (Postgres keys peers by name,
# the 8 foreign keys are NO ACTION and not deferrable, no triggers), old name -> new name, in this order.
RENAME = ((OLD_USER_PEER, USER_PEER), (OLD_AGENT_PEER, AGENT_PEER))
PEER_COLUMNS = (("peers", "name"), ("messages", "peer_name"), ("message_embeddings", "peer_name"),
                ("session_peers", "peer_name"), ("collections", "observer"), ("collections", "observed"),
                ("documents", "observer"), ("documents", "observed"))
FK_TABLES = ("messages", "message_embeddings", "session_peers", "collections", "documents")


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def rename_sql(old_sessions: Iterable[str], workspace: str = "hermes") -> str:
    """One transaction: count -> drop the FKs that reference peers/collections -> rename in place
    (ids, dates and embeddings untouched) -> rename card keys -> detach the user's and the journal's
    conclusions from the old sessions (deleting a session deletes the conclusions bound to it) ->
    re-add the FKs (validating every row) -> recount under the mapping; any mismatch raises, so
    psql's ON_ERROR_STOP rolls the whole transaction back."""
    ws = _q(workspace)
    case = "CASE {c} " + " ".join(f"WHEN {_q(o)} THEN {_q(n)}" for o, n in RENAME) + " ELSE {c} END"
    def counts(mapped: bool) -> str:
        m = (lambda c: case.format(c=c)) if mapped else (lambda c: c)
        parts = [f"SELECT '{t}.{c}' AS col, {m(c)} AS k, count(*) AS n FROM {t} WHERE workspace_name = {ws} "
                 f"GROUP BY 1, 2" if t != "peers" else
                 f"SELECT 'peers.name' AS col, {m('name')} AS k, count(*) AS n FROM peers WHERE workspace_name = {ws} GROUP BY 1, 2"
                 for t, c in PEER_COLUMNS]
        parts.append(f"SELECT 'documents.pair', {m('observer')} || '>' || {m('observed')}, count(*) FROM documents "
                     f"WHERE workspace_name = {ws} GROUP BY 1, 2")
        parts.append(f"SELECT 'documents.created_at_sum', {m('observer')} || '>' || {m('observed')}, "
                     f"sum(extract(epoch FROM created_at))::bigint FROM documents WHERE workspace_name = {ws} GROUP BY 1, 2")
        parts.append(f"SELECT 'messages.created_at_sum', {m('peer_name')}, sum(extract(epoch FROM created_at))::bigint "
                     f"FROM messages WHERE workspace_name = {ws} GROUP BY 1, 2")
        card_peer = m("name")
        parts.append(f"SELECT 'card.self', {card_peer}, coalesce(jsonb_array_length(internal_metadata->'peer_card'), 0) "
                     f"FROM peers WHERE workspace_name = {ws}")
        return "\nUNION ALL ".join(parts)
    sessions = ",".join(f"({_q(s)})" for s in old_sessions) or "(NULL)"
    detach_where = (f"workspace_name = {ws} AND session_name IN (SELECT name FROM old_sessions) AND "
                    f"((observer = {_q(USER_PEER)} AND observed = {_q(USER_PEER)}) OR "
                    f"(observer = {_q(DEVLOG_PEER)} AND observed = {_q(DEVLOG_PEER)}))")
    renames = []
    for old, new in RENAME:
        for t, c in PEER_COLUMNS:
            renames.append(f"UPDATE {t} SET {c} = {_q(new)} WHERE workspace_name = {ws} AND {c} = {_q(old)};")
        renames.append(
            f"UPDATE peers SET internal_metadata = (SELECT jsonb_object_agg(CASE k WHEN {_q(old + '_peer_card')} "
            f"THEN {_q(new + '_peer_card')} ELSE k END, v) FROM jsonb_each(internal_metadata) e(k, v)) "
            f"WHERE workspace_name = {ws} AND internal_metadata ? {_q(old + '_peer_card')};")
    fk_tables = ", ".join(f"{_q(t)}::regclass" for t in FK_TABLES)
    return f"""\\set ON_ERROR_STOP on
BEGIN;
SET LOCAL lock_timeout = '60s';
DO $$ BEGIN IF EXISTS (SELECT 1 FROM peers WHERE workspace_name = {ws} AND name = {_q(USER_PEER)}) THEN
  RAISE EXCEPTION 'peer {USER_PEER} already exists: rename aborted'; END IF; END $$;
CREATE TEMP TABLE old_sessions(name text PRIMARY KEY) ON COMMIT DROP;
INSERT INTO old_sessions VALUES {sessions};
CREATE TEMP TABLE fks ON COMMIT DROP AS SELECT conrelid::regclass::text AS tbl, conname, pg_get_constraintdef(oid) AS def
  FROM pg_constraint WHERE contype = 'f' AND confrelid IN ('peers'::regclass, 'collections'::regclass)
  AND conrelid IN ({fk_tables});
CREATE TEMP TABLE before ON COMMIT DROP AS {counts(True)};
CREATE TEMP TABLE detach_expected ON COMMIT DROP AS SELECT count(*) AS n FROM documents
  WHERE {detach_where.replace(_q(USER_PEER) + ' AND observed = ' + _q(USER_PEER), _q(OLD_USER_PEER) + ' AND observed = ' + _q(OLD_USER_PEER))};
DO $$ DECLARE r record; BEGIN FOR r IN SELECT * FROM fks LOOP
  EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', r.tbl, r.conname); END LOOP; END $$;
{chr(10).join(renames)}
UPDATE documents SET session_name = NULL WHERE {detach_where};
UPDATE workspaces SET configuration = jsonb_set(configuration, '{{reasoning,custom_instructions}}',
  to_jsonb(replace(configuration->'reasoning'->>'custom_instructions', '(peer {OLD_USER_PEER})', '(peer {USER_PEER})')))
  WHERE name = {ws} AND configuration->'reasoning'->>'custom_instructions' LIKE '%(peer {OLD_USER_PEER})%';
DO $$ DECLARE r record; BEGIN FOR r IN SELECT * FROM fks ORDER BY tbl = 'documents', conname LOOP
  EXECUTE format('ALTER TABLE %s ADD CONSTRAINT %I %s', r.tbl, r.conname, r.def); END LOOP; END $$;
CREATE TEMP TABLE after ON COMMIT DROP AS {counts(False)};
SELECT coalesce(b.col, a.col) AS col, coalesce(b.k, a.k) AS peer, b.n AS before, a.n AS after
  FROM before b FULL JOIN after a ON a.col = b.col AND a.k IS NOT DISTINCT FROM b.k
  WHERE coalesce(b.col, a.col) NOT LIKE '%created_at_sum' ORDER BY 1, 2;
SELECT (SELECT n FROM detach_expected) AS detach_expected,
  (SELECT count(*) FROM documents WHERE workspace_name = {ws} AND observer = {_q(USER_PEER)} AND observed = {_q(USER_PEER)} AND session_name IS NULL) AS user_facts_sessionless,
  (SELECT count(*) FROM fks) AS fks_readded;
DO $$ BEGIN
  IF EXISTS (SELECT * FROM before EXCEPT SELECT * FROM after) OR EXISTS (SELECT * FROM after EXCEPT SELECT * FROM before) THEN
    RAISE EXCEPTION 'count mismatch after rename: rolled back'; END IF;
  IF (SELECT count(*) FROM pg_constraint c JOIN fks f ON c.conname = f.conname AND pg_get_constraintdef(c.oid) = f.def) <> (SELECT count(*) FROM fks) THEN
    RAISE EXCEPTION 'foreign keys not restored identically: rolled back'; END IF;
  IF EXISTS (SELECT 1 FROM peers WHERE workspace_name = {ws} AND name = {_q(OLD_AGENT_PEER)}) THEN
    RAISE EXCEPTION 'old agent tag still present: rolled back'; END IF;
END $$;
COMMIT;
"""


def apply_rename(h: Honcho, snapshot_path: Path, backup_dir: Path) -> None:
    dumps = sorted(backup_dir.glob("honcho-2*.dump"), key=lambda p: p.stat().st_mtime)
    if not dumps or time.time() - dumps[-1].stat().st_mtime > 6 * 3600:
        sys.exit("rename: no pg dump in the last 6 h under ~/.hermes/backups (step 0 first)")
    snap = snapshot_old_sessions(h, snapshot_path)
    print(f"rename: {OLD_USER_PEER}->{USER_PEER}, {OLD_AGENT_PEER}->{AGENT_PEER}; backup {dumps[-1]}; "
          f"{len(snap['sessions'])} old sessions (snapshot {snapshot_path})", flush=True)
    import subprocess
    sql = rename_sql(sorted(snap["sessions"]))
    (snapshot_path.parent / "rename.sql").write_text(sql)
    r = subprocess.run(["docker", "exec", "-i", "honcho-database-1", "psql", "-U", "postgres", "-d", "postgres"],
                       input=sql, text=True, capture_output=True)
    print(r.stdout)
    if r.returncode or "ROLLBACK" in r.stdout or "ERROR" in r.stderr:
        sys.exit(f"rename: FAILED, rolled back\n{r.stderr}")
    print("rename: committed")


# ----- Re-filing the Honcho-only messages (option B) -----

def remap_peer(peer: str) -> str:
    """After the rename the agent is ``jarvis``; ``hermes`` can only reappear from a process still on the
    old config. Every other author (james and the aliases 8618229140, uid-501, basic-james, api) is James."""
    return AGENT_PEER if peer in (AGENT_PEER, OLD_AGENT_PEER) else USER_PEER


def honcho_turns(session: str, msgs: List[Dict[str, Any]], classes: List[str]) -> List[tuple]:
    """``(Turn, messages to file)`` per turn of an old Honcho session, in its own message order: a run of
    user-side messages, then the run of agent messages that answers it. A continuation chunk takes the
    class of the message before it. Only turns holding an ``uncovered`` message are returned, with only
    those messages (covered ones are filed from the transcript; migration pastes are dropped)."""
    out: List[tuple] = []
    resolved: List[str] = []
    for c in classes:
        resolved.append(resolved[-1] if c == "continuation" and resolved else c)
    dialogue: List[Dict[str, Any]] = []
    i, n = 0, len(msgs)
    while i < n:
        start = i
        while i < n and remap_peer(msgs[i]["peer_id"]) == USER_PEER:
            i += 1
        while i < n and remap_peer(msgs[i]["peer_id"]) == AGENT_PEER:
            i += 1
        idx = list(range(start, i))
        strip = lambda m: _CONTEXT_PREFIX_RE.sub("", (m.get("content") or "").lstrip())
        user = "\n\n".join(strip(msgs[j]) for j in idx if remap_peer(msgs[j]["peer_id"]) == USER_PEER).strip()
        reply = "\n\n".join(strip(msgs[j]) for j in idx if remap_peer(msgs[j]["peer_id"]) == AGENT_PEER).strip()
        keep = [msgs[j] for j in idx if resolved[j] == "uncovered"]
        if keep:
            first = msgs[idx[0]]
            turn = Turn(profile="honcho", source="honcho", session_id=session, user_id=first["id"],
                        user_ts=0.0, user=user, reply_id=msgs[idx[-1]]["id"], reply_ts=0.0, reply=reply,
                        thread=dialogue[-THREAD_ROWS:] + [{"role": "user", "content": user},
                                                          {"role": "assistant", "content": reply}])
            out.append((turn, keep))
        dialogue += [{"role": "user" if remap_peer(msgs[j]["peer_id"]) == USER_PEER else "assistant",
                      "content": strip(msgs[j])} for j in idx]
    return out


def honcho_messages(session: str, keep: List[Dict[str, Any]], line: str, limit: int) -> List[Dict[str, Any]]:
    chunk = plugin().HonchoMemoryProvider._chunk_message
    out = []
    for m in keep:
        text = _CONTEXT_PREFIX_RE.sub("", (m.get("content") or "").lstrip())
        full = f"{line}\n\n{text}" if line else text
        meta = {**(m.get("metadata") or {}), "refiled_from_session": session, "refiled_from_message": m["id"]}
        out += [{"peer_id": remap_peer(m["peer_id"]), "content": c, "created_at": m["created_at"], "metadata": meta}
                for c in chunk(full, limit)]
    return out


def plan_refile_honcho(h: Honcho, turns: List[Turn], snap: Dict[str, Any], done: set) -> Dict[str, List[tuple]]:
    """Per old session, the ``(Turn, messages)`` still to file; prints the class counts. Reads only."""
    keys, ids = coverage_index(turns)
    with cf.ThreadPoolExecutor(8) as pool:
        all_msgs = dict(zip(snap["sessions"], pool.map(
            lambda s: list(h.pages(f"/sessions/{s}/messages/list")), snap["sessions"])))
    cls_count: collections.Counter = collections.Counter()
    plan: Dict[str, List[tuple]] = {}
    for sid, msgs in all_msgs.items():
        classes = [classify_message(m, keys, ids) for m in msgs]
        cls_count.update(classes)
        plan[sid] = [(t, [m for m in keep if m["id"] not in done]) for t, keep in honcho_turns(sid, msgs, classes)]
        plan[sid] = [(t, keep) for t, keep in plan[sid] if keep]
    todo = [t for ts in plan.values() for t, _ in ts]
    print(f"refile-honcho: {sum(len(v) for v in all_msgs.values())} messages in {len(all_msgs)} old sessions "
          f"{dict(cls_count)}; to file: {sum(len(k) for ts in plan.values() for _, k in ts)} messages in "
          f"{len(todo)} turns of {sum(1 for v in plan.values() if v)} sessions", flush=True)
    return plan


def apply_refile_honcho(h: Honcho, turns: List[Turn], snapshot_path: Path, writer: "LineWriter",
                        ledger: Path, limit: int) -> None:
    snap = json.loads(snapshot_path.read_text())
    done = set(ledger.read_text().splitlines()) if ledger.exists() else set()
    plan = plan_refile_honcho(h, turns, snap, done)
    lines = writer.write([t for ts in plan.values() for t, _ in ts])
    filed = msgs_n = sessions = 0
    for sid, ts in plan.items():
        if not ts:
            continue
        name = honcho_session_name("honcho", sid)
        _ensure_session(h, name, {**snap["sessions"][sid]["metadata"], "refiled_from_session": sid, "source": "honcho"})
        batch, ids_done = [], []
        for t, keep in ts:
            batch += honcho_messages(sid, keep, lines.get(t.key, {}).get("line", ""), limit)
            ids_done += [m["id"] for m in keep]
        for i in range(0, len(batch), 100):
            h.call("POST", f"/sessions/{name}/messages", {"messages": batch[i:i + 100]})
        with ledger.open("a") as f:
            f.writelines(x + "\n" for x in ids_done)
        filed, msgs_n, sessions = filed + len(ids_done), msgs_n + len(batch), sessions + 1
    print(f"refile-honcho: {filed} old messages filed as {msgs_n} messages in {sessions} sessions")


def delete_targets(members: Dict[str, set], created: Dict[str, str], snap: Dict[str, Any]) -> List[str]:
    """The old-tag sessions: a snapshot session that still is the one snapshotted (created before
    ``taken_at``; a live per-directory or per-session key recreated under the new tags reuses its name),
    plus any session a process still on the old config wrote (a ``hermes`` member and no ``james``)."""
    taken = snap.get("taken_at", "")
    old = snap.get("sessions", {})

    def stale(sid: str) -> bool:
        return sid in old and bool(taken) and _utc(created.get(sid, "")) < _utc(taken)

    return sorted(sid for sid, m in members.items() if not is_protected(sid, m) and (
        stale(sid) or (OLD_AGENT_PEER in m and USER_PEER not in m)))


def _utc(ts: str) -> dt.datetime:
    """An ISO timestamp as aware UTC; empty or unparsable sorts last, so it never counts as stale."""
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    except ValueError:
        return dt.datetime.max.replace(tzinfo=dt.timezone.utc)


def apply_delete_old(h: Honcho, snapshot_path: Path, backup_dir: Path) -> None:
    """Delete the old-tag sessions (``delete_targets``). Mail threads, ``refiled-*`` and ``devlog`` are
    never touched. Every message goes to a backup jsonl first."""
    members = session_members(h)
    snap_all = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    snap = snap_all.get("sessions", {})
    created = {s["id"]: s.get("created_at") or "" for s in h.pages("/sessions/list")}
    targets = delete_targets(members, created, snap_all)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"honcho-refile-deleted-sessions-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    n_msgs = 0
    with backup.open("w") as f:
        for sid in targets:
            f.write(json.dumps({"session": sid, "members": sorted(members[sid]),
                                "metadata": snap.get(sid, {}).get("metadata")}, ensure_ascii=False) + "\n")
            for m in h.pages(f"/sessions/{sid}/messages/list"):
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                n_msgs += 1
    print(f"delete-old: backup {backup} ({len(targets)} sessions, {n_msgs} messages)", flush=True)
    for sid in targets:
        assert not is_protected(sid, members[sid])
        h.call("DELETE", f"/sessions/{sid}")
    print(f"delete-old: {len(targets)} sessions queued for deletion; relies on backup {backup}")


# ----- Reports -----

def report_turns(turns: List[Turn], skipped: collections.Counter, limit: int) -> None:
    by_source = collections.Counter(f"{t.profile}:{t.source}" for t in turns)
    sessions = collections.Counter(f"{t.profile}:{t.source}" for t in {(t.profile, t.source, t.session_id): t for t in turns}.values())
    print("== turns per source (profile:source  turns  sessions)")
    for k in sorted(by_source):
        print(f"  {k:28} {by_source[k]:6} {sessions[k]:6}")
    chunked = sum(len(turn_messages(t, "", limit)) for t in turns)
    print(f"  total: {len(turns)} turns in {sum(sessions.values())} sessions -> {chunked} messages "
          f"(chunked at {limit}); skipped: {dict(skipped)}")


def report_lines(sample: List[Turn], recs: Dict[str, Dict[str, Any]], total_turns: int, route: str) -> None:
    print(f"== context lines, sample of {len(sample)} (route {route or 'cached'})")
    for t in sample:
        r = recs[t.key]
        print(f"  [{stratum(t)}] ({t.profile}/{t.session_id}, {t.user_id}) "
              f"{r['input_tokens']}+{r['output_tokens']} tok: {r['line'] or '<none: files bare>'}")
    n = len(sample)
    tin, tout = sum(recs[t.key]["input_tokens"] for t in sample), sum(recs[t.key]["output_tokens"] for t in sample)
    empty = sum(1 for t in sample if not recs[t.key]["line"])
    cost = lambda i, o: i / 1e6 * PRICE_PER_MTOK[0] + o / 1e6 * PRICE_PER_MTOK[1]
    print(f"  sample tokens: {tin} in + {tout} out = ${cost(tin, tout):.4f} at ${PRICE_PER_MTOK[0]}/${PRICE_PER_MTOK[1]} per MTok; "
          f"{empty}/{n} without a line; models: {sorted({recs[t.key]['model'] for t in sample})}")
    if n:
        ein, eout = tin / n * total_turns, tout / n * total_turns
        print(f"  extrapolated to {total_turns} turns: {ein:,.0f} in + {eout:,.0f} out = ${cost(ein, eout):.2f}")


def report_honcho(h: Honcho, inv: Dict[str, Any], out: Path) -> None:
    print("== Honcho workspace (read-only)")
    print(f"  peers: {inv['peers']}")
    print(f"  sessions: {inv['sessions_total']} total; untouched: {dict(inv['keep'])}; no members: {inv['no_members']}")
    print(f"  old-tag sessions: {inv['old_sessions']} holding {inv['old_messages']} messages "
          f"{dict(inv['old_messages_by_class'])}")
    for peers, n in inv["old_by_peers"].most_common():
        print(f"    members {peers:40} {n:5} sessions")
    print(f"  would delete (every message re-filed): {len(inv['would_delete'])} sessions, {inv['would_delete_messages']} messages")
    print(f"  kept (content found only in Honcho): {len(inv['kept_uncovered'])} sessions, "
          f"{inv['kept_uncovered_messages']} messages {dict(inv['kept_uncovered_by_class'])}; "
          f"by name: {dict(inv['kept_uncovered_by_prefix'])}")
    for observer, observed in ((OLD_USER_PEER, OLD_USER_PEER), (None, DEVLOG_PEER), (USER_PEER, USER_PEER)):
        rows = h.conclusions(observer, observed)
        bound = sum(1 for c in rows if c.get("session_id"))
        print(f"  conclusions {observer or '*'}->{observed}: {len(rows)} ({bound} bound to a session, "
              f"deleted with it unless copied first)")
    card = (h.call("GET", f"/peers/{OLD_USER_PEER}/card") or {}).get("peer_card") or []
    print(f"  card {OLD_USER_PEER}: {len(card)} facts")
    out.write_text(json.dumps({k: (dict(v) if isinstance(v, collections.Counter) else v)
                               for k, v in inv.items()}, indent=1))
    print(f"  session list written: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="default: inventory + sample lines; writes nothing to Honcho")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--apply", choices=("rename", "conclusions", "devlog", "refile", "refile-honcho", "delete-old"))
    ap.add_argument("--yes", action="store_true", help="required with --apply")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HERMES_HOME", HERMES_HOME))
                    / "cache" / "scratch" / "refile_history", help="line cache, ledgers and the inventory")
    args = ap.parse_args()
    if args.apply and (args.dry_run or not args.yes):
        sys.exit("--apply needs --yes and excludes --dry-run")

    cfg = json.loads((HERMES_HOME / "honcho.json").read_text())
    limit = int(cfg.get("hosts", {}).get("hermes", {}).get("messageMaxChars") or cfg.get("messageMaxChars") or 25000)
    cache = args.cache_dir / "lines.jsonl"
    h = Honcho()

    snapshot = args.cache_dir / "old_sessions.json"
    if args.apply == "rename":
        return apply_rename(h, snapshot, HERMES_HOME / "backups")
    if args.apply == "conclusions":
        return apply_conclusions(h)
    if args.apply == "devlog":
        return apply_devlog(h, args.cache_dir / "filed_devlog.txt")
    skipped: collections.Counter = collections.Counter()
    turns = load_turns(skipped)
    if args.apply == "delete-old":
        return apply_delete_old(h, snapshot, HERMES_HOME / "backups")
    writer = LineWriter(cache, args.workers)
    if args.apply == "refile-honcho":
        return apply_refile_honcho(h, turns, snapshot, writer, args.cache_dir / "filed_honcho.txt", limit)
    if args.apply == "refile":
        print(f"refile: writing lines for {len(turns)} turns (cache {cache})", flush=True)
        lines = writer.write(turns)
        return apply_refile(h, turns, lines, args.cache_dir / "filed_turns.txt", limit)

    report_turns(turns, skipped, limit)
    sample = sample_turns(turns, args.sample)
    report_lines(sample, writer.write(sample), len(turns), writer.route)
    report_honcho(h, inventory_honcho(h, turns), args.cache_dir / "honcho_inventory.json")


if __name__ == "__main__":
    main()

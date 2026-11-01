#!/usr/bin/env python3
"""Dry run of the context line on a real past turn: no Honcho reads or writes.

Rebuilds the turn that a user message opened (that message plus the final reply after it) and the
turns before it from ``state.db``, calls the real ``context_line`` writer (a real auxiliary-model
call through Hermes' ``memory_context_line`` task), and prints the line.

Run it with a Python that imports Hermes (``agent.auxiliary_client``), e.g. the managed runtime venv:

    PY=$(ls -t ~/.hermes/installs/*/environments/*/venv/bin/python | head -1)
    $PY scripts/context_line_dry_run.py <session_id> <user_message_id>

No session summary is passed (that lives in Honcho); the thread comes from the transcript alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _context_line import MAX_LINE_CHARS, TASK_KEY, context_line, register_context_line_task  # noqa: E402

_CONTEXT_ROWS = 200  # rows of the session read before the turn; the writer keeps the last few exchanges


def declare_task() -> str:
    """Declare the aux task as the plugin's ``register`` does in Hermes, and name the route it resolves to."""
    from agent.auxiliary_client import _resolve_task_provider_model
    from hermes_cli.plugins import get_plugin_auxiliary_tasks
    from plugins.memory import _ProviderCollector

    get_plugin_auxiliary_tasks()  # discovery first, so it cannot reset the registration below
    register_context_line_task(_ProviderCollector("honcho", register_skills=False))
    provider, model, *_ = _resolve_task_provider_model(TASK_KEY)
    return f"{provider}/{model}"


def _row(r: sqlite3.Row) -> dict:
    tool_calls = r["tool_calls"]
    try:
        tool_calls = json.loads(tool_calls) if tool_calls else None
    except (TypeError, ValueError):
        pass
    return {"role": r["role"], "content": r["content"] or "", "tool_calls": tool_calls, "_row_id": r["id"]}


def load_turn(db: Path, session_id: str, message_id: int) -> tuple[str, str, list[dict]]:
    """``(user_content, reply, transcript up to and including the reply)`` for the turn ``message_id`` opened."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = "select id, role, content, tool_calls from messages where session_id = ? and id {} ? order by id {} limit ?"
    before = [_row(r) for r in con.execute(q.format("<", "desc"), (session_id, message_id, _CONTEXT_ROWS))][::-1]
    after = [_row(r) for r in con.execute(q.format(">=", "asc"), (session_id, message_id, _CONTEXT_ROWS))]
    if not after or after[0]["_row_id"] != message_id or after[0]["role"] != "user":
        sys.exit(f"message {message_id} is not a user message of session {session_id}")
    turn = [after[0]]
    for m in after[1:]:
        if m["role"] == "user":
            break
        turn.append(m)
        if m["role"] == "assistant" and m["content"] and not m["tool_calls"]:
            break
    replies = [m for m in turn if m["role"] == "assistant" and m["content"] and not m["tool_calls"]]
    return after[0]["content"], (replies[-1]["content"] if replies else ""), before + turn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_id")
    ap.add_argument("message_id", type=int, help="state.db messages.id of the user message that opened the turn")
    ap.add_argument("--db", type=Path, default=Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "state.db")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("_context_line").setLevel(logging.DEBUG)  # show why a turn would file without a line

    route = declare_task()
    user, reply, transcript = load_turn(args.db, args.session_id, args.message_id)
    line = context_line(user, reply, messages=transcript, summary=None)
    print(f"session_id: {args.session_id}  message_id: {args.message_id}  route: {route}")
    print(f"user: {user[:200]!r}")
    print(f"reply: {reply[:200]!r}")
    print(f"line ({len(line)}/{MAX_LINE_CHARS} chars): {line or '<none: the turn would file without a line>'}")


if __name__ == "__main__":
    main()

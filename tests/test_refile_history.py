"""scripts/refile_history.py: the turn walker reads a transcript the way sync_turn files it live,
and coverage decides which old Honcho messages the re-file writes again."""

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refile_history_under_test", ROOT / "scripts" / "refile_history.py")
rh = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rh
spec.loader.exec_module(rh)


def _rows(*items):
    """(role, content[, tool_calls]) tuples as state.db rows with ids 1.. and timestamps 100.."""
    out = []
    for i, item in enumerate(items, 1):
        role, content, *rest = item
        out.append({"id": i, "role": role, "content": content, "tool_calls": rest[0] if rest else None,
                    "timestamp": 100.0 + i})
    return out


def _walk(rows, **kw):
    return rh.walk_turns(rows, profile="default", source="webui", session_id="s1", **kw)


def test_turn_is_opener_plus_final_reply_skipping_tool_rounds():
    rows = _rows(("user", "fix the bug"), ("assistant", "", [{"id": "c1"}]),
                 ("assistant", "looking", [{"id": "c2"}]), ("assistant", "fixed it"),
                 ("user", "thanks"), ("assistant", "welcome"))
    turns = _walk(rows)
    assert [(t.user_id, t.user, t.reply_id, t.reply) for t in turns] == [
        (1, "fix the bug", 4, "fixed it"), (5, "thanks", 6, "welcome")]
    assert turns[0].user_ts == 101.0 and turns[0].reply_ts == 104.0
    assert turns[1].key == "default/s1/5"


def test_thread_ends_with_the_turn_and_holds_prior_dialogue_only():
    rows = _rows(("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "tool", [{"id": 1}]),
                 ("assistant", "d"))
    second = _walk(rows)[1]
    assert second.thread == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                             {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]


def test_interjection_is_joined_and_id_stays_the_openers():
    stats = collections.Counter()
    rows = _rows(("user", "first"), ("assistant", "x", [{"id": 1}]), ("user", "also this"), ("assistant", "done"))
    (turn,) = _walk(rows, skipped=stats)
    assert turn.user_id == 1 and turn.user == "first\n\nalso this"
    assert turn.texts == ["first", "also this", "done"]
    assert stats["joined_user_rows (not skipped)"] == 1


def test_internal_notifications_and_unanswered_messages_are_not_turns():
    stats = collections.Counter()
    rows = _rows(("user", "[ASYNC DELEGATION COMPLETE] child done"), ("assistant", "noted"),
                 ("user", "real question"), ("assistant", "real answer"), ("user", "left hanging"))
    turns = _walk(rows, is_internal=lambda t: t.startswith("[ASYNC"), skipped=stats)
    assert [t.user for t in turns] == ["real question"]
    assert stats["internal"] == 1 and stats["unanswered"] == 1


def test_clean_is_applied_to_both_sides():
    rows = _rows(("user", " <memory-context>x</memory-context>hi "), ("assistant", " yo "))
    (turn,) = _walk(rows, clean=lambda t: t.replace("<memory-context>x</memory-context>", ""))
    assert (turn.user, turn.reply) == ("hi", "yo")


def test_parse_tool_calls_treats_empty_json_as_none():
    assert rh.parse_tool_calls("[]") is None and rh.parse_tool_calls(None) is None
    assert rh.parse_tool_calls('[{"id": 1}]') == [{"id": 1}]


def test_session_name_is_valid_and_shows_the_hermes_id():
    assert rh.honcho_session_name("default", "2026-08-06T18:21:06-07:00") == "refiled-default-2026-08-06T18-21-06-07-00"
    assert rh.honcho_session_name("z_k-opus", "20261022_232243_862c0b") == "refiled-z_k-opus-20261022_232243_862c0b"


def test_coverage_matches_by_pointer_or_text_and_flags_honcho_only_content():
    (turn,) = _walk(_rows(("user", "what is   the plan"), ("assistant", "the plan is X")))
    keys, ids = rh.coverage_index([turn])
    by_id = {"content": "anything", "metadata": {"hermes_session_id": "s1", "hermes_message_id": 2}}
    by_text = {"content": "[context] The user asked about the plan.\n\nwhat is the plan", "metadata": {}}
    assert rh.classify_message(by_id, keys, ids) == "covered"
    assert rh.classify_message(by_text, keys, ids) == "covered"
    assert rh.classify_message({"content": "[continued] tail"}, keys, ids) == "continuation"
    assert rh.classify_message({"content": "<prior_memory_file>\n..."}, keys, ids) == "migration"
    assert rh.classify_message({"content": "a chat state.db no longer has"}, keys, ids) == "uncovered"


# ----- phase B: rename SQL, Honcho-only turns -----

def test_remap_peer_agent_tags_to_jarvis_everyone_else_to_james():
    assert rh.remap_peer("jarvis") == "jarvis" and rh.remap_peer("hermes") == "jarvis"
    assert {rh.remap_peer(p) for p in ("james", "8618229140", "uid-501", "basic-james", "api")} == {"james"}


def test_rename_sql_is_one_transaction_in_order_with_checks():
    sql = rh.rename_sql(["s1", "o'brien"])
    assert sql.startswith("\\set ON_ERROR_STOP on\nBEGIN;") and sql.rstrip().endswith("COMMIT;")
    assert sql.count("BEGIN;") == 1 and sql.count("COMMIT;") == 1
    # jarvis -> james runs before hermes -> jarvis, on every peer column
    first = sql.index("UPDATE peers SET name = 'james' WHERE workspace_name = 'hermes' AND name = 'jarvis';")
    second = sql.index("UPDATE peers SET name = 'jarvis' WHERE workspace_name = 'hermes' AND name = 'hermes';")
    assert first < second
    for t, c in rh.PEER_COLUMNS:
        assert f"UPDATE {t} SET {c} = 'james' WHERE workspace_name = 'hermes' AND {c} = 'jarvis';" in sql
    assert "('o''brien')" in sql
    assert "count mismatch after rename: rolled back" in sql
    assert "UPDATE documents SET session_name = NULL" in sql  # facts survive the session delete


def _m(i, peer, content, cls_hint=None):
    return {"id": f"m{i}", "peer_id": peer, "content": content, "created_at": f"2026-01-0{i}T00:00:00Z",
            "metadata": {"k": i}}


def test_honcho_turns_pairs_runs_and_keeps_only_uncovered():
    msgs = [_m(1, "james", "q1"), _m(2, "jarvis", "a1"),
            _m(3, "8618229140", "q2"), _m(4, "jarvis", "a2"), _m(5, "jarvis", "[continued] a2b"),
            _m(6, "james", "<prior_memory_file>x"), _m(7, "jarvis", "a3")]
    classes = ["covered", "covered", "uncovered", "uncovered", "continuation", "migration", "uncovered"]
    out = rh.honcho_turns("old", msgs, classes)
    assert [[m["id"] for m in keep] for _, keep in out] == [["m3", "m4", "m5"], ["m7"]]
    t, _ = out[0]
    assert (t.user, t.reply, t.key) == ("q2", "a2\n\n[continued] a2b", "honcho/old/m3")
    assert t.thread[:2] == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


def test_honcho_messages_remap_line_and_pointers(monkeypatch):
    class P:
        class HonchoMemoryProvider:
            @staticmethod
            def _chunk_message(text, limit):
                return [text]
    monkeypatch.setattr(rh, "plugin", lambda: P)
    out = rh.honcho_messages("old", [_m(1, "uid-501", "[context] stale\n\nhello")], "[context] new.", 25000)
    assert out == [{"peer_id": "james", "content": "[context] new.\n\nhello", "created_at": "2026-01-01T00:00:00Z",
                    "metadata": {"k": 1, "refiled_from_session": "old", "refiled_from_message": "m1"}}]


def test_delete_targets_skip_a_snapshot_name_recreated_under_the_new_tags():
    snap = {"taken_at": "2026-11-01T01:39:00+00:00",
            "sessions": {"old": {}, "a834c5f8014c": {}, "jarvis": {}}}
    members = {"old": {"james", "jarvis"},                   # snapshotted, renamed in place
               "a834c5f8014c": {"james", "jarvis", "hermes"},  # same name, recreated after the rename
               "jarvis": {"james", "jarvis"},                # per-directory key, recreated
               "agent-x": {"jarvis", "hermes"},              # written on the old config after the rename
               "live": {"james", "jarvis"},
               "mail-1": {"james", "correspondent"},
               "refiled-default-s1": {"james", "jarvis", "hermes"},
               "devlog": {"jarvis", "hermes"}}
    created = {"old": "2026-08-01T00:00:00Z", "a834c5f8014c": "2026-11-01T02:03:40.921548Z",
               "jarvis": "2026-11-01T02:11:03Z", "agent-x": "2026-11-01T02:02:03Z"}
    assert rh.delete_targets(members, created, snap) == ["agent-x", "old"]
    assert rh.delete_targets(members, {}, snap) == ["agent-x"]  # unknown age never counts as stale

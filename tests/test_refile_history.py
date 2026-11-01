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

"""Passive turn recall: a short "Related past turns" section appended to the injected memory block."""
from __future__ import annotations

import logging
import threading

import pytest


def hit(session="s-old", mid=90000, content="Done. Trial step removed.", author="assistant",
        created="2026-10-30", profile=None):
    return {"content": content, "author": author, "created_at": created,
            "hermes_session_id": session, "hermes_message_id": mid, "hermes_profile": profile}


@pytest.fixture
def tr(load_submodule):
    return load_submodule("turn_recall")


# ---- config ----------------------------------------------------------------------------------

def test_config_defaults_are_disabled(tr):
    cfg = tr.TurnRecallConfig.from_block(None)
    assert (cfg.enabled, cfg.limit, cfg.excerpt_chars, cfg.max_tokens) == (False, 3, 120, 300)


def test_config_reads_object_keys(tr):
    cfg = tr.TurnRecallConfig.from_block({"enabled": True, "limit": 5, "excerptChars": 80, "maxTokens": 100})
    assert (cfg.enabled, cfg.limit, cfg.excerpt_chars, cfg.max_tokens) == (True, 5, 80, 100)


def test_config_ignores_unusable_values(tr):
    cfg = tr.TurnRecallConfig.from_block({"enabled": True, "limit": "many", "excerptChars": -4})
    assert (cfg.enabled, cfg.limit, cfg.excerpt_chars) == (True, 3, 120)


# ---- rendering -------------------------------------------------------------------------------

def render(tr, hits, current="s-now", **cfg):
    return tr.render_section(hits, tr.TurnRecallConfig(enabled=True, **cfg), current_session_id=current)


def test_line_shape_matches_spec(tr):
    out = render(tr, [hit(session="124a0ce85ac2", content="Done. Trial step removed, Procedure is now Survey")])
    assert out == ("## Related past turns\n"
                   "- 2026-10-30 [assistant] Done. Trial step removed, Procedure is now Survey "
                   "(@session:default/124a0ce85ac2 #90000)")


def test_profile_from_metadata(tr):
    assert "(@session:z_k-haiku/s-old #90000)" in render(tr, [hit(profile="z_k-haiku")])


def test_current_session_hits_are_dropped(tr):
    out = render(tr, [hit(session="s-now", mid=1), hit(session="s-old", mid=2)])
    assert "#1)" not in out and "#2)" in out


def test_hits_without_message_id_are_dropped(tr):
    out = render(tr, [hit(mid=None), hit(mid="", content="blank id"), hit(mid=7)])
    assert out.count("\n- ") == 1 and "#7)" in out


def test_duplicate_session_message_pairs_collapse(tr):
    out = render(tr, [hit(mid=5, content="first"), hit(mid=5, content="again"), hit(session="s-b", mid=5)])
    assert out.count("\n- ") == 2 and "first" in out and "again" not in out


def test_limit_keeps_top_hits_in_rank_order(tr):
    out = render(tr, [hit(mid=i, content=f"m{i}") for i in range(1, 8)], limit=3)
    assert [ln.split("#")[1].rstrip(")") for ln in out.splitlines()[1:]] == ["1", "2", "3"]


def test_excerpt_is_one_line_and_cut_with_ellipsis(tr):
    out = render(tr, [hit(content="alpha\n\nbeta   gamma " + "x" * 50)], excerpt_chars=17)
    (line,) = out.splitlines()[1:]
    assert "\n" not in line and "[assistant] alpha beta gamma... (@session" in line


def test_short_excerpt_is_not_marked_cut(tr):
    assert "..." not in render(tr, [hit(content="short")], excerpt_chars=20)


def test_token_cap_truncates_whole_section(tr):
    hits = [hit(mid=i, content="word " * 40) for i in range(1, 4)]
    out = render(tr, hits, max_tokens=60)
    assert len(out) <= 60 * 4 and out.startswith("## Related past turns")


def test_no_surviving_hits_yields_no_section(tr):
    assert render(tr, []) == ""
    assert render(tr, [hit(session="s-now")]) == ""


def test_example_with_three_crafted_hits(tr, capsys):
    out = render(tr, [
        hit(session="124a0ce85ac2", mid=90000, author="assistant", created="2026-10-30",
            content="Done. Trial step removed, Procedure is now Survey, then Review and Export " * 3),
        hit(session="s-now", mid=1, content="current session: dropped"),
        hit(session="9f3c1d77be10", mid=88412, author="james", created="2026-10-28", profile="z_k-haiku",
            content="Can we move the plugin tests under tests/ ?"),
        hit(session="124a0ce85ac2", mid=90000, content="duplicate: dropped"),
        hit(session="0b77aa31c5d2", mid=71, author="assistant", created="2026-09-02",
            content="Pushed to origin/main."),
    ])
    with capsys.disabled():
        print("\n--- rendered section ---\n" + out + "\n------------------------")
    assert out.count("\n- ") == 3


# ---- provider wiring -------------------------------------------------------------------------

class FakeManager:
    def __init__(self, hits=None, exc=None):
        self.hits, self.exc, self.calls = hits or [], exc, []

    def recall_hits(self, session_key, query, *, limit):
        self.calls.append((session_key, query, limit))
        if self.exc:
            raise self.exc
        return self.hits

    def pop_context_result(self, key):
        return {}


@pytest.fixture
def make_provider(plugin, load_submodule):
    client = load_submodule("client")
    tr = load_submodule("turn_recall")

    def _make(manager, *, enabled=True, rewriter=None, rewrite=False, limit=3, mode="hybrid", cadence=1):
        p = plugin.HonchoMemoryProvider(query_rewriter=rewriter)
        p._manager, p._session_key, p._session_initialized = manager, "k", True
        p._config = client.HonchoClientConfig(context_tokens=None)
        p._recall_mode, p._query_rewrite_enabled, p._context_cadence = mode, rewrite, cadence
        p._turn_recall = tr.TurnRecallConfig(enabled=enabled, limit=limit)
        p._base_context_cache = ""   # skip the first-turn base fetch
        p._turn_count = 2
        return p

    return _make


def test_build_overfetches_by_three_and_filters(make_provider):
    m = FakeManager([hit(session="cur", mid=1), hit(mid=2), hit(mid=3), hit(mid=4), hit(mid=5)])
    p = make_provider(m, limit=2)
    out = p._build_turn_recall("what about trial?", "cur")
    assert m.calls == [("k", "what about trial?", 5)]
    assert out.count("\n- ") == 2 and "#1)" not in out


def test_build_uses_rewritten_query(make_provider):
    m = FakeManager([hit()])
    p = make_provider(m, rewriter=lambda q: "  rewritten terms ", rewrite=True)
    p._build_turn_recall("raw question", "cur")
    assert m.calls[0][1] == "rewritten terms"


def test_build_falls_back_to_raw_query_when_rewriter_fails(make_provider):
    def boom(q):
        raise RuntimeError("aux llm down")
    m = FakeManager([hit()])
    p = make_provider(m, rewriter=boom, rewrite=True)
    assert "## Related past turns" in p._build_turn_recall("raw question", "cur")
    assert m.calls[0][1] == "raw question"


def test_build_disabled_makes_no_search(make_provider):
    m = FakeManager([hit()])
    assert make_provider(m, enabled=False)._build_turn_recall("q", "cur") == ""
    assert m.calls == []


def test_build_search_failure_logs_debug_and_yields_nothing(make_provider, caplog):
    p = make_provider(FakeManager(exc=RuntimeError("search down")))
    with caplog.at_level(logging.DEBUG):
        assert p._build_turn_recall("q", "cur") == ""
    assert any(r.levelno == logging.DEBUG and "search down" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_prefetch_appends_section_after_existing_parts(make_provider):
    p = make_provider(FakeManager())
    p._base_context_cache = "## Base\nfacts"
    p._turn_recall_ready = ("## Related past turns\n- x", 1)   # (section, turn it was fired at)
    assert p.prefetch("tell me more about the plugin", session_id="cur") == "## Base\nfacts\n\n## Related past turns\n- x"


def test_prefetch_without_section_is_unchanged(make_provider):
    p = make_provider(FakeManager())
    p._base_context_cache = "## Base\nfacts"
    assert p.prefetch("tell me more about the plugin", session_id="cur") == "## Base\nfacts"


def test_queue_prefetch_runs_search_in_background_and_prefetch_consumes_it(make_provider):
    m = FakeManager([hit(mid=9)])
    m.prefetch_context = lambda key, q: None
    p = make_provider(m)
    p._turn_count = 3
    p.queue_prefetch("what did we decide about trial?", session_id="cur")
    p._turn_recall_thread.join(timeout=5)
    p._turn_count = 4
    p._base_context_cache = "## Base\nfacts"
    out = p.prefetch("and the survey step?", session_id="cur")
    assert out.endswith("(@session:default/s-old #9)") and "## Related past turns" in out
    # consumed once: the next turn does not repeat stale pointers
    assert "Related past turns" not in p.prefetch("and the survey step?", session_id="cur")


def test_queue_prefetch_respects_context_cadence(make_provider):
    m = FakeManager([hit()])
    m.prefetch_context = lambda key, q: None
    p = make_provider(m, cadence=3)
    p._last_context_turn, p._turn_count = 2, 3   # only 1 turn since the last refresh
    p.queue_prefetch("what did we decide about trial?", session_id="cur")
    assert m.calls == []


def test_tools_mode_never_searches(make_provider):
    m = FakeManager([hit()])
    p = make_provider(m, mode="tools")
    p.queue_prefetch("what did we decide about trial?", session_id="cur")
    assert m.calls == [] and p.prefetch("anything at all here", session_id="cur") == ""


def test_stalled_search_never_holds_up_the_turn(make_provider):
    gate = threading.Event()

    class Slow(FakeManager):
        def recall_hits(self, *a, **k):
            gate.wait(10)
            return []
    m = Slow()
    m.prefetch_context = lambda key, q: None
    p = make_provider(m)
    p._base_context_cache = "## Base\nfacts"
    p.queue_prefetch("what did we decide about trial?", session_id="cur")
    p._turn_count = 3
    try:
        assert p.prefetch("next question about trial", session_id="cur") == "## Base\nfacts"
    finally:
        gate.set()


# ---- manager: search hits -> dicts -----------------------------------------------------------

def _manager(plugin, load_submodule, honcho):
    session_mod = load_submodule("session")
    from types import SimpleNamespace
    class Manager(session_mod.HonchoSessionManager):
        pass

    Manager.honcho = property(lambda self: honcho)   # the real property builds a network client
    m = object.__new__(Manager)
    sess = SimpleNamespace(assistant_peer_id="ai", user_peer_id="u")
    m._cached_session = lambda key: sess
    m._resolve_peer_id = lambda s, peer: "u"
    m._authed_call = lambda label, fn: fn()
    return m


def test_recall_hits_normalizes_search_results(plugin, load_submodule):
    from datetime import datetime
    from types import SimpleNamespace as NS
    seen = {}

    class FakeHoncho:
        def search(self, q, filters=None, limit=None):
            seen.update(q=q, filters=filters, limit=limit)
            return [NS(content="a", peer_id="ai", created_at=datetime(2026, 10, 30, 9, 0),
                       metadata={"hermes_session_id": "S1", "hermes_message_id": 12}),
                    NS(content="b", peer_id="u", created_at=None, metadata=None)]

    m = _manager(plugin, load_submodule, FakeHoncho())
    hits = m.recall_hits("k", " query ", limit=6)
    assert seen == {"q": "query", "filters": {"peer_perspective": "u"}, "limit": 6}
    assert hits[0]["author"] == "assistant" and hits[0]["hermes_session_id"] == "S1" and hits[0]["hermes_message_id"] == 12
    assert hits[1]["author"] == "u" and hits[1]["hermes_message_id"] is None


def test_recall_hits_search_failure_yields_empty(plugin, load_submodule, caplog):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("index down")

    m = _manager(plugin, load_submodule, Broken())
    m._get_or_create_peer = lambda pid: Broken()
    with caplog.at_level(logging.DEBUG):
        assert m.recall_hits("k", "query", limit=6) == []
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_search_context_label_still_carries_hermes_ids(plugin, load_submodule):
    from types import SimpleNamespace as NS

    class FakeHoncho:
        def search(self, q, filters=None, limit=None):
            return [NS(content="hello", peer_id="ai", session_id="hs",
                       metadata={"hermes_session_id": "S1", "hermes_message_id": 12})]

    out = _manager(plugin, load_submodule, FakeHoncho()).search_context("k", "query")
    assert out == "[assistant · hs · hermes_session_id=S1 hermes_message_id=12] hello"


# ---- initialize reads the host block ---------------------------------------------------------

@pytest.mark.parametrize("block,enabled,limit", [
    ({}, False, 3),
    ({"turnRecall": {"enabled": True, "limit": 4}}, True, 4),
])
def test_initialize_reads_turn_recall_from_host_block(plugin, load_submodule, monkeypatch, tmp_path, block, enabled, limit):
    import json
    client = load_submodule("client")
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps({"apiKey": "k-test", "hosts": {"hermes": {"recallMode": "tools", **block}}}))
    real = client.HonchoClientConfig.from_global_config
    monkeypatch.setattr(client.HonchoClientConfig, "from_global_config",
                        classmethod(lambda cls, host=None, config_path=None: real(host="hermes", config_path=path)))
    p = plugin.HonchoMemoryProvider()
    p.initialize("sess-1", platform="cli")
    assert (p._turn_recall.enabled, p._turn_recall.limit) == (enabled, limit)

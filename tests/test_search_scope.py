"""honcho_search scope: no peer searches the whole workspace; a peer keeps the perspective filter."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_session_context():
    """Import session_context as a package submodule without running the plugin's __init__ (Hermes deps)."""
    if "agent.redact" not in sys.modules:  # Hermes' redactor, imported by session_auth
        agent = sys.modules.setdefault("agent", types.ModuleType("agent"))
        redact = types.ModuleType("agent.redact")
        redact.redact_sensitive_text = lambda text, *a, **k: text
        agent.redact = redact
        sys.modules["agent.redact"] = redact
    pkg = types.ModuleType("honcho_plugin_under_test")
    pkg.__path__ = [str(ROOT)]
    sys.modules.setdefault("honcho_plugin_under_test", pkg)
    return importlib.import_module("honcho_plugin_under_test.session_context")


session_context = _load_session_context()

USER, AI = "jarvis", "hermes-ai"


class FakeHoncho:
    def __init__(self, messages):
        self.calls: list[dict] = []
        self._messages = messages

    def search(self, query, filters=None, limit=10):
        self.calls.append({"query": query, "filters": filters, "limit": limit})
        return self._messages


class FakeManager(session_context.SessionContextMixin):
    def __init__(self, messages):
        self.honcho = FakeHoncho(messages)
        self.session = SimpleNamespace(user_peer_id=USER, assistant_peer_id=AI)

    def _cached_session(self, key):
        return self.session

    def _resolve_peer_id(self, session, peer):
        return {"user": session.user_peer_id, "ai": session.assistant_peer_id}.get(peer or "user", peer)

    def _authed_call(self, op_name, operation):
        return operation()


def _msg(peer_id, content, **meta):
    return SimpleNamespace(peer_id=peer_id, content=content, session_id="s1", metadata=meta)


HITS = [
    _msg(USER, "did we fix the search?", hermes_session_id="hs1", hermes_message_id=7),
    _msg(AI, "search spans the whole workspace", hermes_session_id="hs1", hermes_message_id=8),
]


@pytest.mark.parametrize("raise_errors", [False, True])
def test_no_peer_searches_workspace_without_perspective_filter(raise_errors):
    mgr = FakeManager(HITS)
    out = mgr.search_context("k", "search", peer=None, raise_errors=raise_errors)
    assert len(mgr.honcho.calls) == 1
    assert not mgr.honcho.calls[0]["filters"]  # no peer_perspective, no filter at all
    # Hits by both authors, labels and hermes ids unchanged.
    assert "[jarvis · s1 · hermes_session_id=hs1 hermes_message_id=7] did we fix the search?" in out
    assert "[assistant · s1 · hermes_session_id=hs1 hermes_message_id=8] search spans the whole workspace" in out


@pytest.mark.parametrize("peer, peer_id", [("user", USER), ("ai", AI)])
def test_peer_keeps_perspective_filter(peer, peer_id):
    mgr = FakeManager(HITS)
    mgr.search_context("k", "search", peer=peer)
    assert mgr.honcho.calls[0]["filters"] == {"peer_perspective": peer_id}


def test_old_callers_default_to_user_perspective():
    mgr = FakeManager(HITS)
    mgr.search_context("k", "search")
    assert mgr.honcho.calls[0]["filters"] == {"peer_perspective": USER}


def test_workspace_search_failure_raises_only_when_asked():
    mgr = FakeManager(HITS)

    def boom(*a, **k):
        raise RuntimeError("backend down")

    mgr.honcho.search = boom
    assert mgr.search_context("k", "search", peer=None) == ""
    with pytest.raises(RuntimeError, match="backend down"):
        mgr.search_context("k", "search", peer=None, raise_errors=True)

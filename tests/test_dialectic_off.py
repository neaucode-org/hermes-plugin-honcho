"""dialecticCadence 0 turns automatic dialectic off: no spawn path and no sync recall reaches .chat()."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub(name: str, **attrs) -> None:
    """Register ``name`` (and its parents) in sys.modules, adding ``attrs`` that are not there yet."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        mod = sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
        if i > 1:
            setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    for key, value in attrs.items():
        if not hasattr(sys.modules[name], key):
            setattr(sys.modules[name], key, value)


def _spawn(target, *, name, daemon=True, args=(), owner=None):
    return threading.Thread(target=target, name=name, daemon=daemon, args=args)


class _SingletonSlot:
    def __init__(self, *a, **k):
        self.value = None


def _load(name: str):
    """Import a plugin submodule as a package member without running the plugin's __init__;
    the Hermes modules client.py reaches are stubbed, as in test_context_line."""
    _stub("agent.memory_provider", spawn_context_thread=_spawn)
    _stub("agent.secret_scope", get_secret=lambda *a, **k: None)
    _stub("agent.redact", redact_sensitive_text=lambda text, *a, **k: text)
    _stub("hermes_cli.profiles", _get_default_hermes_home=lambda: ROOT, get_active_profile_name=lambda: "default")
    _stub("hermes_constants", get_hermes_home=lambda: ROOT)
    _stub("hermes_state_common", TITLE_SOURCE_DERIVED="derived", TITLE_SOURCE_LLM="llm")
    _stub("plugins.plugin_utils", SingletonSlot=_SingletonSlot)
    pkg = types.ModuleType("honcho_plugin_under_test")
    pkg.__path__ = [str(ROOT)]
    sys.modules.setdefault("honcho_plugin_under_test", pkg)
    return importlib.import_module(f"honcho_plugin_under_test.{name}")


dialectic = _load("dialectic")
recall_sync = _load("recall_sync")


class Provider(dialectic.DialecticMixin):
    """The provider state the dialectic mixin and prefetch_sync read; counts dialectic runs."""

    def __init__(self, cadence: int):
        self._dialectic_cadence = cadence
        self._dialectic_empty_streak = 0
        self._last_dialectic_turn = -999
        self._last_context_turn = -999
        self._context_cadence = 1
        self._injection_frequency = "every-turn"
        self._dialectic_depth = 1
        self._query_rewriter = None
        self._query_rewrite_enabled = False
        self._prefetch_lock = threading.Lock()
        self._prefetch_result, self._prefetch_result_fired_at = "", -999
        self._prefetch_thread = None
        self._prefetch_thread_started_at = 0.0
        self._recall_sync_lock = threading.Lock()
        self._recall_sync_thread = None
        self._recall_generation = None
        self._session_key, self._turn_count = "s1", 1
        self._config = SimpleNamespace(timeout=5.0)
        self.runs: list[str] = []
        self.dialectic_queries: list[str] = []
        self._manager = SimpleNamespace(
            get_prefetch_context=lambda *a, **k: {"card": "base"},
            dialectic_query=lambda session, prompt, **k: self.dialectic_queries.append(prompt) or "answer",
        )

    def _run_dialectic_depth(self, query, use_query_rewrite=True):
        self.runs.append(query)
        return "answer"

    # prefetch_sync collaborators
    def _is_trivial_prompt(self, query):
        return False

    def _session_ready(self):
        return True

    def _resolve_pass_level(self, i, query=""):
        return "minimal"

    def _format_first_turn_context(self, ctx):
        return ctx.get("card", "")

    def _pop_auth_notice(self):
        return ""

    def _truncate_to_budget(self, text):
        return text


@pytest.mark.parametrize("label", ["dialectic prewarm", "first-turn dialectic", "prefetch"])
def test_cadence_zero_spawns_nothing(label):
    p = Provider(cadence=0)
    assert p._spawn_dialectic("q", thread_name="t", fired_at=1, log_label=label) is None
    assert p._prefetch_thread is None
    assert p.runs == []


def test_cadence_positive_still_spawns():
    p = Provider(cadence=1)
    thread = p._spawn_dialectic("q", thread_name="t", fired_at=1, log_label="prefetch")
    thread.join(timeout=2)
    assert p.runs == ["q"]
    assert p._last_dialectic_turn == 1


def test_sync_recall_cadence_zero_injects_base_and_skips_dialectic():
    p = Provider(cadence=0)
    for turn in range(1, 9):
        p._turn_count = turn
        assert recall_sync.prefetch_sync(p, "what have I been working on?") == "base"
    assert p.dialectic_queries == []


def test_sync_recall_cadence_positive_still_queries_dialectic():
    p = Provider(cadence=1)
    out = recall_sync.prefetch_sync(p, "what have I been working on?")
    assert len(p.dialectic_queries) == 1
    assert "answer" in out

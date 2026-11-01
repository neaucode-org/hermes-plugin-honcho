"""contextLine: sync_turn prepends a one-line thread context to both messages of a filed turn.

The plugin's __init__ needs a running Hermes to import, so the Hermes modules it reaches are
stubbed here and the package is loaded under its own name; call_llm is faked per test."""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "honcho_plugin_full_under_test"

LINE = "[context] James approves adding the context line to Honcho turn filing behind the contextLine flag."


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


def _spawn(target, *, name, daemon=True, args=()):
    return threading.Thread(target=target, name=name, daemon=daemon, args=args)


class _SingletonSlot:
    def __init__(self, *a, **k):
        self.value = None


def _load_plugin():
    _stub("agent.memory_manager", sanitize_context=lambda text: text)
    _stub("agent.memory_provider", MemoryProvider=type("MemoryProvider", (), {}),
          is_trivial_prompt=lambda *a, **k: False, spawn_context_thread=_spawn)
    _stub("agent.coding_context", INTERACTIVE_CODING_PLATFORMS=frozenset())
    _stub("agent.turn_author", a2a_key=lambda author: "")
    _stub("agent.secret_scope", get_secret=lambda *a, **k: None)
    _stub("agent.redact", redact_sensitive_text=lambda text, *a, **k: text)
    _stub("agent.auxiliary_client", call_llm=lambda **kw: None)
    _stub("hermes_cli.profiles", _get_default_hermes_home=lambda: ROOT, get_active_profile_name=lambda: "default")
    _stub("hermes_constants", get_hermes_home=lambda: ROOT)
    _stub("hermes_state_common", TITLE_SOURCE_DERIVED="derived", TITLE_SOURCE_LLM="llm")
    _stub("tools.registry", tool_error=lambda msg, **k: msg)
    _stub("plugins.plugin_utils", SingletonSlot=_SingletonSlot)
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class FakeSession:
    honcho_session_id = "hs1"

    def __init__(self):
        self.added: list[tuple[str, str]] = []

    def add_message(self, role, content, **kwargs):
        self.added.append((role, content))


class FakeManager:
    def __init__(self, summary="The session is about the Honcho memory plugin fork."):
        self.session = FakeSession()
        self.saved = 0
        self.summary = summary

    def get_or_create(self, key, **kwargs):
        return self.session

    def save(self, session):
        self.saved += 1

    def resolve_author_peer_id(self, *a, **k):
        return "james"

    def _sdk_session(self, session_id):
        def summaries():
            if isinstance(self.summary, Exception):
                raise self.summary
            return SimpleNamespace(short_summary=SimpleNamespace(content=self.summary), long_summary=None)
        return SimpleNamespace(summaries=summaries)


def _provider(context_line: bool, *, max_chars: int = 25000, manager: FakeManager | None = None):
    p = plugin.HonchoMemoryProvider()
    p._config = SimpleNamespace(context_line=context_line, message_max_chars=max_chars, a2a_sessions=True)
    p._manager = manager or FakeManager()
    p._session_key = "k"
    p._writes_enabled = lambda: True
    p._ready_or_kick_init = lambda: True
    return p


TRANSCRIPT = [
    {"role": "user", "content": "Plan: prepend a context line to filed Honcho turns behind a contextLine flag."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
    {"role": "tool", "content": "TOOL OUTPUT SECRET"},
    {"role": "assistant", "content": "Shall I add the context line behind contextLine, off by default?"},
    {"role": "user", "content": "ok do it"},
    {"role": "assistant", "content": "Done: the context line ships behind contextLine."},
]


def _file_turn(p, user="ok do it", reply="Done: the context line ships behind contextLine."):
    p.sync_turn(user, reply, session_id="s1", messages=TRANSCRIPT)
    p._sync_thread.join(timeout=5)
    return p._manager


def _fake_llm(monkeypatch, result):
    calls: list[dict] = []

    def call_llm(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])

    monkeypatch.setattr(sys.modules["agent.auxiliary_client"], "call_llm", call_llm)
    return calls


def test_flag_off_files_content_unchanged(monkeypatch):
    calls = _fake_llm(monkeypatch, LINE)
    mgr = _file_turn(_provider(False))
    assert mgr.session.added == [("user", "ok do it"), ("assistant", "Done: the context line ships behind contextLine.")]
    assert calls == []


def test_flag_on_prepends_line_to_both_messages(monkeypatch):
    calls = _fake_llm(monkeypatch, LINE)
    mgr = _file_turn(_provider(True))
    assert mgr.session.added == [
        ("user", f"{LINE}\n\nok do it"),
        ("assistant", f"{LINE}\n\nDone: the context line ships behind contextLine."),
    ]
    assert mgr.saved == 1
    assert len(calls) == 1 and calls[0]["task"] == "memory_context_line"
    prompt = calls[0]["messages"][-1]["content"]
    # Thread: earlier user messages and final replies plus the summary; tool output stays out.
    assert "Plan: prepend a context line" in prompt and "Shall I add the context line" in prompt
    assert "Honcho memory plugin fork" in prompt
    assert "TOOL OUTPUT SECRET" not in prompt


def test_model_reply_without_prefix_gets_it(monkeypatch):
    _fake_llm(monkeypatch, LINE.removeprefix("[context] "))
    mgr = _file_turn(_provider(True))
    assert mgr.session.added[0][1].startswith(f"{LINE}\n\n")


@pytest.mark.parametrize("failure", ["call_llm", "writer", "summary"])
def test_writer_failure_files_turn(monkeypatch, failure):
    _fake_llm(monkeypatch, RuntimeError("aux down") if failure == "call_llm" else LINE)
    manager = FakeManager(summary=RuntimeError("honcho down") if failure == "summary" else "s")
    if failure == "writer":
        def boom(*a, **k):
            raise RuntimeError("writer bug")
        monkeypatch.setattr(plugin, "context_line", boom)
    mgr = _file_turn(_provider(True, manager=manager))
    if failure == "summary":  # no summary: the line is still written from the transcript
        assert mgr.session.added[0][1].startswith(LINE)
    else:
        assert mgr.session.added == [("user", "ok do it"),
                                     ("assistant", "Done: the context line ships behind contextLine.")]
    assert mgr.saved == 1


@pytest.mark.parametrize("reply", [
    "[context] " + "x" * 200,                                   # over 200 chars
    "",                                                         # empty
    "   ",                                                      # blank
    "[context] The thread is about X. James approved it.",      # two sentences
])
def test_invalid_line_is_dropped(monkeypatch, reply):
    _fake_llm(monkeypatch, reply)
    mgr = _file_turn(_provider(True))
    assert mgr.session.added == [("user", "ok do it"), ("assistant", "Done: the context line ships behind contextLine.")]


def test_register_declares_the_aux_task():
    """One task key, inheriting the cheap title_generation route; an undeclared key would run on the chat model."""
    calls: list = []
    ctx = SimpleNamespace(register_memory_provider=lambda p: calls.append(("provider", p)),
                          register_auxiliary_task=lambda key, **kw: calls.append((key, kw["inherit_from"])))
    plugin.register(ctx)
    assert ("memory_context_line", "title_generation") in calls


def test_line_on_first_chunk_only(monkeypatch):
    _fake_llm(monkeypatch, LINE)
    reply = " ".join(["word"] * 200)  # ~1000 chars, chunked at 400
    mgr = _file_turn(_provider(True, max_chars=400), reply=reply)
    assistant_chunks = [c for role, c in mgr.session.added if role == "assistant"]
    assert len(assistant_chunks) > 1
    assert assistant_chunks[0].startswith(f"{LINE}\n\n")
    assert all(LINE not in c for c in assistant_chunks[1:])

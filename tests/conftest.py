"""Shared loader: imports the plugin directory as a package so relative imports resolve.

Run with the hermes venv (it carries honcho-ai and the agent deps) plus pytest on PYTHONPATH:
  PYTHONPATH=<dir with pytest> <venv>/bin/python -m pytest tests -q
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.environ.get("HERMES_AGENT_DIR", os.path.expanduser("~/code/hermes/hermes-agent")))
PLUGIN_DIR = Path(os.environ.get("PLUGIN_DIR", Path(__file__).resolve().parent.parent))
PKG = "honcho_plugin_under_test"


def _load():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(PKG, PLUGIN_DIR / "__init__.py",
                                                  submodule_search_locations=[str(PLUGIN_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def plugin():
    return _load()


@pytest.fixture(scope="session")
def load_submodule(plugin):
    return lambda name: importlib.import_module(f"{PKG}.{name}")

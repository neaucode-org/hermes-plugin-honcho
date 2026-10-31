"""Passive turn recall: pointers to the best-matching past messages, appended to the injected memory block.

Pure helpers (config, selection, rendering); the search itself lives on the session manager
(``recall_hits``) and the scheduling on the provider (queue_prefetch / prefetch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

HEADING = "## Related past turns"
DEFAULT_PROFILE = "default"
OVERFETCH = 3  # extra hits requested so filtering still leaves ``limit``
_ELLIPSIS = "..."


def _positive_int(value: Any, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


@dataclass(frozen=True)
class TurnRecallConfig:
    enabled: bool = False
    limit: int = 3
    excerpt_chars: int = 120
    max_tokens: int = 300  # approximated as chars / 4

    @classmethod
    def from_block(cls, block: Any) -> "TurnRecallConfig":
        """Build from the ``turnRecall`` object of honcho.json; anything unusable keeps the default."""
        if not isinstance(block, dict):
            return cls()
        d = cls()
        return cls(
            enabled=block.get("enabled") is True or str(block.get("enabled")).strip().lower() in {"true", "1", "yes", "on"},
            limit=_positive_int(block.get("limit"), d.limit),
            excerpt_chars=_positive_int(block.get("excerptChars"), d.excerpt_chars),
            max_tokens=_positive_int(block.get("maxTokens"), d.max_tokens),
        )


def _date(created_at: Any) -> str:
    if hasattr(created_at, "strftime"):
        return created_at.strftime("%Y-%m-%d")
    return str(created_at or "")[:10] or "undated"


def _excerpt(content: Any, limit: int) -> str:
    text = " ".join(str(content or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + _ELLIPSIS


def select_hits(hits: Iterable[dict], *, current_session_id: str, limit: int) -> list[dict]:
    """Rank order kept; drops the current Hermes session, hits with no message id, and repeated pairs."""
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for h in hits:
        session, mid = str(h.get("hermes_session_id") or ""), h.get("hermes_message_id")
        if not session or mid in (None, "") or session == current_session_id:
            continue
        key = (session, str(mid))
        if key in seen:
            continue
        seen.add(key)
        kept.append(h)
        if len(kept) >= limit:
            break
    return kept


def _line(h: dict, excerpt_chars: int) -> str:
    profile = h.get("hermes_profile") or DEFAULT_PROFILE
    return (f"- {_date(h.get('created_at'))} [{h.get('author') or 'unknown'}] {_excerpt(h.get('content'), excerpt_chars)} "
            f"(@session:{profile}/{h['hermes_session_id']} #{h['hermes_message_id']})")


def _cap(lines: list[str], max_chars: int) -> str:
    """Whole lines while they fit; a section too small for even its first line is cut hard."""
    out = HEADING
    for line in lines:
        if len(out) + 1 + len(line) > max_chars:
            break
        out += "\n" + line
    return out if out != HEADING else (HEADING + "\n" + lines[0])[:max_chars].rstrip()


def render_section(hits: Iterable[dict], cfg: TurnRecallConfig, *, current_session_id: str = "") -> str:
    """The ``## Related past turns`` section, or "" when no hit survives filtering."""
    kept = select_hits(hits, current_session_id=current_session_id, limit=cfg.limit)
    if not kept:
        return ""
    return _cap([_line(h, cfg.excerpt_chars) for h in kept], cfg.max_tokens * 4)

"""Local preference memory for Nerve HITL v2 (Phase 5).

Learns which exact action shapes the operator has explicitly approved, so the
uncertain band can shrink over time without weakening security classes.

Hard constraints:

- Memory is keyed by CONTENT-SCOPED shape hashes (tool + argv shape + workdir),
  never by generic conditions like "low confidence" or tool name alone.
- Only ``uncertain``-class shapes are ever learned. Consequential, prohibited,
  safe and project classes are never recorded and never consulted.
- A learned preference auto-allows ONLY the identical shape: same tool, same
  arguments (modulo inert numeric/pytest-selector normalization), same
  workdir. Any variation re-classifies from scratch.
- Consent is never inferred from Hermes's blanket "approve for session" of a
  generic key: shapes are recorded only when THIS module observed the operator
  approve a prompt that carried this exact scoped shape key.

Storage: one JSON document at ``<hermes_home>/nerve/preferences.json`` written
atomically; safe to corrupt (falls back to empty).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .paths import hermes_home

_SCHEMA = "hermes-nerve-preferences/v2"
_LEARNABLE_CLASSES = frozenset({"uncertain"})


def preferences_path() -> Any:
    explicit = str(os.getenv("HERMES_NERVE_PREFERENCES") or "").strip()
    return Path(explicit).expanduser() if explicit else hermes_home() / "nerve" / "preferences.json"


def _valid_terminal_segment(segment: Any, tool_name: str = "terminal") -> bool:
    if not isinstance(segment, dict) or segment.get("tool") != tool_name:
        return False
    argv = segment.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and bool(a.strip()) for a in argv):
        return False
    workdir = segment.get("workdir")
    if not isinstance(workdir, str) or not workdir.strip():
        return False
    if workdir.strip() == "/" or Path(workdir).name == "nonexistent-nerve-scope":
        return False
    category = segment.get("category")
    return isinstance(category, str) and bool(category.strip())


def has_valid_scope(tool_name: str, shape: Any) -> bool:
    """Return True only when *shape* is a complete, content-scoped classifier key."""
    if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(shape, dict) or not shape:
        return False
    if tool_name in {"terminal", "shell", "bash"}:
        if shape.get("kind") == "pipeline":
            segments = shape.get("segments")
            return (isinstance(segments, list) and len(segments) >= 2
                    and all(_valid_terminal_segment(seg, "terminal") for seg in segments))
        return _valid_terminal_segment(shape, tool_name)
    declared_tool = shape.get("tool")
    if declared_tool != tool_name and not (
        shape.get("dispatched_via") == tool_name
        and isinstance(declared_tool, str) and bool(declared_tool.strip())
    ):
        return False
    if shape.get("scope") == "unknown":
        return False
    if isinstance(shape.get("args_fp"), str) and bool(shape["args_fp"].strip()):
        return True
    paths = shape.get("paths")
    paths_sha = shape.get("paths_sha256")
    if (isinstance(paths, list) and bool(paths)
            and all(isinstance(p, str) and bool(p.strip()) for p in paths)
            and isinstance(paths_sha, str) and bool(paths_sha.strip())):
        return True
    path = shape.get("path")
    scope = shape.get("scope")
    if (isinstance(path, str) and bool(path.strip()) and path != "None"
            and isinstance(scope, str) and bool(scope.strip()) and scope != "unknown"
            and "content_sha256" in shape and isinstance(shape.get("content_sha256"), str)):
        return True
    return False


def shape_fingerprint(shape: dict[str, Any]) -> str:
    """Stable 16-hex fingerprint of the content-scoped shape material."""
    payload = json.dumps(shape, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def scoped_rule_key(assessment_category: str, tool_name: str, shape: dict[str, Any]) -> str:
    """Rule key that authorizes exactly this action shape — nothing broader.

    Unlike the legacy ``nerve:low-confidence`` / ``nerve:{tool_name}`` keys, an
    approval stored under this key can never authorize a different command,
    different arguments, a different working directory, or a different tool.
    When the classifier fails or yields an incomplete scope, a per-invocation
    nonce prevents session/permanent reuse across unclassified calls.
    """
    if (not isinstance(assessment_category, str) or not assessment_category.strip()
            or not has_valid_scope(tool_name, shape)):
        safe_cat = str(assessment_category or "uncertain").strip() or "uncertain"
        safe_tool = str(tool_name or "call").strip() or "call"
        return f"nerve:hitl:{safe_cat}:{safe_tool}:unscoped-{os.urandom(8).hex()}"
    return f"nerve:hitl:{assessment_category}:{tool_name}:{shape_fingerprint(shape)}"


def _load() -> dict[str, Any]:
    try:
        raw = json.loads(preferences_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": _SCHEMA, "shapes": {}}
    if (not isinstance(raw, dict) or raw.get("schema") != _SCHEMA
            or not isinstance(raw.get("shapes"), dict)):
        return {"schema": _SCHEMA, "shapes": {}}
    return raw


def _save(data: dict[str, Any]) -> None:
    path = preferences_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, sort_keys=True, indent=1)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pref-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except OSError:
        # Preference memory is best-effort; never break the gate over it.
        pass


def record_approval(category: str, tool_name: str, shape: dict[str, Any]) -> None:
    """Record that the operator approved this exact shape (uncertain class only)."""
    if category not in _LEARNABLE_CLASSES or not has_valid_scope(tool_name, shape):
        return
    data = _load()
    fp = shape_fingerprint(shape)
    entry = data["shapes"].setdefault(fp, {
        "category": category,
        "tool_name": str(tool_name),
        "count": 0,
        "first_ts": time.time(),
    })
    entry["count"] = int(entry.get("count") or 0) + 1
    entry["last_ts"] = time.time()
    _save(data)


def approval_count(category: str, tool_name: str, shape: dict[str, Any]) -> int:
    if category not in _LEARNABLE_CLASSES or not has_valid_scope(tool_name, shape):
        return 0
    data = _load()
    entry = data["shapes"].get(shape_fingerprint(shape))
    if (not isinstance(entry, dict)
            or entry.get("category") != category
            or entry.get("tool_name") != str(tool_name)):
        return 0
    try:
        return int(entry.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def learned_allow(category: str, tool_name: str, shape: dict[str, Any], min_samples: int) -> bool:
    """True when the operator has approved this exact shape often enough."""
    if category not in _LEARNABLE_CLASSES or not has_valid_scope(tool_name, shape):
        return False
    return approval_count(category, tool_name, shape) >= max(1, int(min_samples))


# --- Pending-approval tracking (pre_tool_call -> post_tool_call handoff) ---------

_pending: dict[tuple[str, str, str], dict[str, Any]] = {}
_PENDING_CAP = 512


def remember_pending(
    call: tuple[str, str, str], category: str, shape: dict[str, Any], args_digest: str,
) -> None:
    """Stash an exact foreground-terminal approval candidate until its post hook."""
    if (not isinstance(call, tuple) or len(call) != 3
            or not all(isinstance(part, str) and part for part in call)
            or category not in _LEARNABLE_CLASSES
            or not isinstance(args_digest, str) or not args_digest.strip()
            or not has_valid_scope("terminal", shape)):
        return
    if len(_pending) >= _PENDING_CAP:
        # Drop the oldest half; learning is best-effort.
        for key in list(_pending)[: _PENDING_CAP // 2]:
            _pending.pop(key, None)
    _pending[call] = {"category": category, "shape": dict(shape), "args_digest": args_digest, "approved": False}


def confirm_pending(call: tuple[str, str, str], pattern_key: str) -> None:
    """Attach a human response only to the exact plugin rule that was presented."""
    entry = _pending.get(call)
    if (entry is not None
            and has_valid_scope("terminal", entry.get("shape"))
            and pattern_key == "plugin_rule:" + scoped_rule_key(
                entry["category"], "terminal", entry["shape"],
            )):
        entry["approved"] = True


def consume_pending(call: tuple[str, str, str]) -> dict[str, Any] | None:
    return _pending.pop(call, None)

"""Deterministic 3-pass <= 8,000-char completion packet bounder and criterion sub-batcher."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

MAX_COMPLETION_PACKET_CHARS = 8000
MAX_COMPLETION_BATCHES = 4
MAX_CRITERIA_PER_BATCH = 8
TRUNCATION_MARKER = "...[truncated]"
_MAX_SCALAR_INT = (1 << 63) - 1

COMPLETION_BATCH_INSTRUCTIONS = (
    "Verify the remaining locked Definition-of-Done criteria as one batch. PASS only if the independently "
    "observed evidence plus completion proposal proves every listed criterion. "
    "If `evidence_truncated` is true or `omitted_evidence_count > 0` and direct proof for any listed criterion "
    "is absent from the visible packet, you MUST NOT infer or assume PASS — return NEEDS_EVIDENCE or RETRY. "
    "RETRY means the current approach is valid but more execution/evidence is required; REPLAN means the approach "
    "must change; ESCALATE means human/external judgment is required."
)


def _bound_int(value: Any, *, min_val: int = -_MAX_SCALAR_INT, max_val: int = _MAX_SCALAR_INT) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value
    try:
        parsed = int(value or 0)
    except Exception:
        return max(min_val, min(max_val, 0))
    if parsed < min_val:
        return min_val
    if parsed > max_val:
        return max_val
    return parsed


def _truncate_str_tracked(value: Any, limit: int) -> tuple[str, bool]:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < -_MAX_SCALAR_INT or value > _MAX_SCALAR_INT:
            sign = "-" if value < 0 else ""
            text = f"{sign}<int:{value.bit_length()}b>"
        else:
            text = str(value)
    else:
        try:
            text = str(value if value is not None else "")
        except Exception:
            text = f"<{type(value).__name__}:oversized>"
    if len(text) <= limit:
        return text, False
    if limit <= len(TRUNCATION_MARKER):
        return text[: max(0, limit)], True
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER, True


def _truncate_str(value: Any, limit: int) -> str:
    out, _ = _truncate_str_tracked(value, limit)
    return out


def _bound_scalar(value: Any, max_str: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _bound_int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) <= 1e15:
            return round(value, 6)
        return _truncate_str(f"{value:.6e}", min(max_str, 32))
    return _truncate_str(value, max_str)


def _truncate_nested(value: Any, *, max_str: int = 400, max_items: int = 8, depth: int = 2) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return _bound_scalar(value, max_str)
    if isinstance(value, str):
        return _truncate_str(value, max_str)
    if depth <= 0:
        return _truncate_str(value, max_str)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated_keys"] = _bound_int(len(value) - max_items, min_val=0)
                break
            key = _truncate_str(k, 80)
            out[key] = _truncate_nested(v, max_str=max_str, max_items=max_items, depth=depth - 1)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        sliced = items[-max_items:] if len(items) > max_items else items
        return [_truncate_nested(x, max_str=max_str, max_items=max_items, depth=depth - 1) for x in sliced]
    return _truncate_str(value, max_str)


def _has_unbounded_scalar(value: Any, *, depth: int = 6) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value < -_MAX_SCALAR_INT or value > _MAX_SCALAR_INT
    if isinstance(value, float):
        return not math.isfinite(value) or abs(value) > 1e15
    if isinstance(value, str):
        return False
    if depth <= 0:
        return True
    if isinstance(value, Mapping):
        if len(value) > 64:
            return True
        return any(
            _has_unbounded_scalar(k, depth=depth - 1) or _has_unbounded_scalar(v, depth=depth - 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return True
        return any(_has_unbounded_scalar(x, depth=depth - 1) for x in value)
    return False


def safe_json_len(payload: Any) -> int:
    """Dual-encoding JSON character length (max of ASCII-escaped and raw UTF-8)."""
    if _has_unbounded_scalar(payload):
        return MAX_COMPLETION_PACKET_CHARS * 16
    try:
        raw_ascii = json.dumps(
            payload,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda o: _truncate_str(o, 120),
        )
        raw_utf8 = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=lambda o: _truncate_str(o, 120),
        )
        return max(len(raw_ascii), len(raw_utf8))
    except Exception:
        return MAX_COMPLETION_PACKET_CHARS * 16


def _prioritize_evidence(
    evidence: list[Mapping[str, Any]],
    remaining_ids: set[str],
) -> list[Mapping[str, Any]]:
    scored: list[tuple[int, int, Mapping[str, Any]]] = []
    verdict_kinds = {
        "crew_verdict",
        "deterministic_check",
        "deterministic_behavior_probe",
        "deterministic_token_budget",
        "deterministic_completion_summary",
    }
    verdict_tools = {"crew_verdict", "crew_review_handoff", "crew_review"}
    signal_tokens = ("pytest", "unittest", "verdict:", "passed", "failed", "error")
    for idx, item in enumerate(evidence):
        cid = str(item.get("criterion_id") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        tool = str(item.get("tool_name") or "").strip().lower()
        is_err = bool(item.get("is_error"))
        preview_head = str(item.get("preview") or "")[:500].lower()
        if cid and cid in remaining_ids:
            tier = 0
        elif kind in verdict_kinds or tool in verdict_tools:
            tier = 1
        elif is_err or (kind == "tool_result" and any(tok in preview_head or tok in tool for tok in signal_tokens)):
            tier = 2
        else:
            tier = 3
        scored.append((tier, -idx, item))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in scored]


def _format_evidence_slice(
    prioritized: list[Mapping[str, Any]],
    *,
    max_items: int,
    preview_cap: int,
) -> tuple[list[dict[str, Any]], bool]:
    kept: list[dict[str, Any]] = []
    any_preview_truncated = False
    if max_items <= 0:
        return kept, bool(prioritized)
    for item in prioritized[:max_items]:
        raw_prev = str(item.get("preview") or "")
        bounded_prev, truncated = _truncate_str_tracked(raw_prev, preview_cap)
        if truncated or raw_prev.endswith("...") or raw_prev.endswith(TRUNCATION_MARKER):
            any_preview_truncated = True
        cid = item.get("criterion_id")
        kept.append(
            {
                "evidence_id": _truncate_str(item.get("evidence_id", ""), 80),
                "criterion_id": _truncate_str(cid, 80) if cid else None,
                "kind": _truncate_str(item.get("kind", ""), 64),
                "tool_name": _truncate_str(item.get("tool_name") or "", 64),
                "is_error": bool(item.get("is_error")),
                "preview": bounded_prev,
            }
        )
    return kept, any_preview_truncated


def _compact_proposal_summary(proposal: Mapping[str, Any], limit: int) -> str:
    if limit <= 0 or not proposal:
        return ""
    for key in ("summary", "final_response", "response", "result"):
        if key in proposal and proposal[key] not in (None, ""):
            return _truncate_str(proposal[key], limit)
    bounded = _truncate_nested(dict(proposal), max_str=160, max_items=4, depth=2)
    raw = json.dumps(bounded, separators=(",", ":"), ensure_ascii=False, default=lambda o: _truncate_str(o, 80))
    return _truncate_str(raw, limit)


def bound_completion_packet(
    goal: str,
    remaining_criteria: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    proposal: dict[str, Any] | None = None,
    *,
    max_chars: int = MAX_COMPLETION_PACKET_CHARS,
) -> dict[str, Any]:
    """Serialize completion state through a deterministic 3-pass <= max_chars ladder."""
    max_chars = max(256, int(max_chars))
    criteria_list = [c for c in (remaining_criteria or []) if isinstance(c, Mapping)]
    evidence_list = [e for e in (evidence or []) if isinstance(e, Mapping)]
    raw_proposal: Mapping[str, Any]
    if isinstance(proposal, Mapping):
        raw_proposal = proposal
    elif proposal is None:
        raw_proposal = {}
    else:
        raw_proposal = {"value": proposal}

    total_evidence = len(evidence_list)
    remaining_ids = {
        str(c.get("id") or "").strip()
        for c in criteria_list
        if str(c.get("id") or "").strip()
    }
    prioritized = _prioritize_evidence(evidence_list, remaining_ids)

    # Pass 1: Compact full context (goal<=400, criteria desc<=400, <=12 priority evidence x 350, bounded proposal)
    pass1_evidence, pass1_prev_trunc = _format_evidence_slice(prioritized, max_items=12, preview_cap=350)
    pass1_omitted = max(0, total_evidence - len(pass1_evidence))
    pass1 = {
        "goal": _truncate_str(goal, 400),
        "remaining_criteria": [
            {
                "id": _truncate_str(c.get("id", ""), 80),
                "description": _truncate_str(c.get("description", ""), 400),
                "required": bool(c.get("required", True)),
            }
            for c in criteria_list
        ],
        "evidence": pass1_evidence,
        "total_evidence_count": total_evidence,
        "omitted_evidence_count": pass1_omitted,
        "evidence_truncated": bool(pass1_omitted > 0 or pass1_prev_trunc),
        "proposal": _truncate_nested(dict(raw_proposal), max_str=400, max_items=8, depth=2),
    }
    if safe_json_len(pass1) <= max_chars:
        return pass1

    # Pass 2: Criterion-linked priority compression (criteria desc<=220, <=6 evidence x 180, compact proposal<=400)
    pass2_evidence, pass2_prev_trunc = _format_evidence_slice(prioritized, max_items=6, preview_cap=180)
    pass2_omitted = max(0, total_evidence - len(pass2_evidence))
    pass2 = {
        "goal": _truncate_str(goal, 260),
        "remaining_criteria": [
            {
                "id": _truncate_str(c.get("id", ""), 80),
                "description": _truncate_str(c.get("description", ""), 220),
                "required": bool(c.get("required", True)),
            }
            for c in criteria_list
        ],
        "evidence": pass2_evidence,
        "total_evidence_count": total_evidence,
        "omitted_evidence_count": pass2_omitted,
        "evidence_truncated": bool(total_evidence > 0 and (pass2_omitted > 0 or pass2_prev_trunc or True)),
        "proposal": _compact_proposal_summary(raw_proposal, 400),
    }
    if safe_json_len(pass2) <= max_chars:
        return pass2

    # Pass 3: Hard structural envelope guaranteed <= max_chars while preserving goal + unmet criterion IDs.
    ladder = (
        (160, 120, 3, 80, 160),
        (100, 72, 2, 48, 80),
        (64, 40, 1, 24, 0),
        (48, 18, 0, 0, 0),
        (24, 0, 0, 0, 0),
    )
    last_out: dict[str, Any] = {}
    for goal_cap, desc_cap, ev_cap, prev_cap, prop_cap in ladder:
        ev_slice, _ = _format_evidence_slice(prioritized, max_items=ev_cap, preview_cap=prev_cap)
        omitted = max(0, total_evidence - len(ev_slice))
        if desc_cap > 0:
            crit_payload = [
                {
                    "id": _truncate_str(c.get("id", ""), 64),
                    "description": _truncate_str(c.get("description", ""), desc_cap),
                }
                for c in criteria_list
            ]
        else:
            crit_payload = [{"id": _truncate_str(c.get("id", ""), 48)} for c in criteria_list]
        candidate: dict[str, Any] = {
            "goal": _truncate_str(goal, goal_cap) or _truncate_str(goal or "goal", max(1, goal_cap)),
            "remaining_criteria": crit_payload,
            "evidence": ev_slice,
            "total_evidence_count": total_evidence,
            "omitted_evidence_count": omitted,
            "evidence_truncated": True,
            "proposal_summary": _compact_proposal_summary(raw_proposal, prop_cap),
        }
        last_out = candidate
        if safe_json_len(candidate) <= max_chars:
            return candidate

    # Final emergency shrink if criterion count itself exceeds an artificially tiny max_chars.
    kept_crit = list(last_out.get("remaining_criteria") or [])
    while kept_crit and safe_json_len(last_out) > max_chars:
        kept_crit.pop()
        last_out["remaining_criteria"] = kept_crit
    return last_out


def _criteria_group_fits_cleanly(
    *,
    goal: str,
    group: list[dict[str, Any]],
    prioritized_evidence: list[Mapping[str, Any]],
    raw_proposal: Mapping[str, Any],
    max_chars: int,
) -> bool:
    if len(group) > MAX_CRITERIA_PER_BATCH:
        return False
    min_ev, _ = _format_evidence_slice(prioritized_evidence, max_items=min(3, len(prioritized_evidence)), preview_cap=180)
    probe = {
        "goal": _truncate_str(goal, 400),
        "remaining_criteria": [
            {
                "id": _truncate_str(c.get("id", ""), 80),
                "description": _truncate_str(c.get("description", ""), 400),
                "required": bool(c.get("required", True)),
            }
            for c in group
        ],
        "evidence": min_ev,
        "total_evidence_count": len(prioritized_evidence),
        "omitted_evidence_count": max(0, len(prioritized_evidence) - len(min_ev)),
        "evidence_truncated": len(prioritized_evidence) > len(min_ev),
        "proposal": _compact_proposal_summary(raw_proposal, 200),
    }
    return safe_json_len(probe) <= max_chars


def partition_completion_batches(
    goal: str,
    remaining_criteria: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    proposal: dict[str, Any] | None = None,
    *,
    max_chars: int = MAX_COMPLETION_PACKET_CHARS,
    max_batches: int = MAX_COMPLETION_BATCHES,
) -> list[dict[str, Any]] | None:
    """Partition unmet criteria into <= max_batches bounded completion packets, or None if overflowed."""
    max_chars = max(256, int(max_chars))
    max_batches = max(1, int(max_batches))
    criteria_list = [dict(c) for c in (remaining_criteria or []) if isinstance(c, Mapping)]
    evidence_list = [e for e in (evidence or []) if isinstance(e, Mapping)]
    raw_proposal: Mapping[str, Any] = proposal if isinstance(proposal, Mapping) else ({"value": proposal} if proposal else {})

    if not criteria_list:
        return [
            bound_completion_packet(
                goal=goal,
                remaining_criteria=[],
                evidence=evidence_list,
                proposal=dict(raw_proposal),
                max_chars=max_chars,
            )
        ]

    total = len(criteria_list)
    for batch_count in range(1, min(max_batches, total) + 1):
        base, rem = divmod(total, batch_count)
        groups: list[list[dict[str, Any]]] = []
        cursor = 0
        for idx in range(batch_count):
            size = base + (1 if idx < rem else 0)
            groups.append(criteria_list[cursor : cursor + size])
            cursor += size

        all_fit = True
        for group in groups:
            group_ids = {str(c.get("id") or "").strip() for c in group if str(c.get("id") or "").strip()}
            prioritized = _prioritize_evidence(evidence_list, group_ids)
            if not _criteria_group_fits_cleanly(
                goal=goal,
                group=group,
                prioritized_evidence=prioritized,
                raw_proposal=raw_proposal,
                max_chars=max_chars,
            ):
                all_fit = False
                break
        if all_fit:
            return [
                bound_completion_packet(
                    goal=goal,
                    remaining_criteria=group,
                    evidence=evidence_list,
                    proposal=dict(raw_proposal),
                    max_chars=max_chars,
                )
                for group in groups
            ]

    return None

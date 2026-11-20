"""Bounded nonsecret accounting receipts for Nerve progress and prerequisite supervision.

Separates lifetime observation counters, host prerequisite capacity/byte budgets,
active control source, guard disposition, and deliverable verification without
persisting source text, raw tool results, or credentials.
"""

from __future__ import annotations

from typing import Any
import uuid

from .privacy import canonical_hash, redact
from .progress_integrity import mutation_is_relevant, tool_kind

ACCOUNTING_RECEIPT_SCHEMA = "hermes-nerve-accounting-receipt/v1"
MAX_RETAINED_RECEIPTS = 256

_VALID_PHASES = frozenset({"preparing", "output", "exhausted", "closed", "unavailable"})
_VALID_OUTCOMES = frozenset({
    "ok",
    "success",
    "passed",
    "error",
    "failed",
    "blocked",
    "failure",
    "cancelled",
    "timeout",
    "reserved",
    "unavailable",
    "verified",
    "no_relevant_mutation",
    "no_relevant_persisted",
    "missing_verification",
    "shadow_observed",
    "challenged",
    "allowed",
})


def sanitize_identifier(value: Any, *, max_len: int = 128) -> str:
    """Bound and scrub identifiers so embedded credentials never enter receipts."""
    if value is None:
        return ""
    raw = "".join(ch for ch in str(value) if ord(ch) >= 32)
    clipped = raw[:max_len]
    scrubbed = redact(clipped)
    if scrubbed != clipped:
        return f"redacted:{canonical_hash(raw)[:16]}"
    return clipped


def normalize_outcome_token(value: Any, *, fallback: str = "other") -> str:
    token = str(value or "").strip().lower()
    if token in _VALID_OUTCOMES:
        return token
    return fallback


def _nonneg_int(value: Any, *, cap: int = 999_999_999) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, min(cap, value))
    try:
        return max(0, min(cap, int(value or 0)))
    except (TypeError, ValueError):
        return 0


def normalize_prerequisite_snapshot(raw: dict[str, Any] | None) -> dict[str, int | str]:
    if not isinstance(raw, dict):
        return {
            "phase": "unavailable",
            "operations_reserved": 0,
            "operations_remaining": 0,
            "operations_limit": 0,
            "source_bytes_reserved": 0,
            "source_bytes_remaining": 0,
            "source_bytes_limit": 0,
        }
    raw_phase = str(raw.get("phase") or "unavailable").strip().lower()
    phase = raw_phase if raw_phase in _VALID_PHASES else "unavailable"
    return {
        "phase": phase,
        "operations_reserved": _nonneg_int(raw.get("operations_reserved")),
        "operations_remaining": _nonneg_int(raw.get("operations_remaining")),
        "operations_limit": _nonneg_int(raw.get("operations_limit")),
        "source_bytes_reserved": _nonneg_int(raw.get("source_bytes_reserved")),
        "source_bytes_remaining": _nonneg_int(raw.get("source_bytes_remaining")),
        "source_bytes_limit": _nonneg_int(raw.get("source_bytes_limit")),
    }


def classify_operation_category(
    *,
    stage: str,
    tool_name: str,
    args: dict[str, Any],
    is_prerequisite_read: bool,
    change_required: bool,
    trusted_intent: bool,
    requested_paths: tuple[str, ...],
) -> str:
    if stage == "completion_gate":
        return "completion_check"
    if is_prerequisite_read:
        return "prerequisite_read"
    kind = tool_kind(tool_name, args)
    if kind == "mutation":
        relevant = (not trusted_intent or change_required) and mutation_is_relevant(tool_name, args, requested_paths)
        return "relevant_mutation" if relevant else "unrelated_mutation"
    if kind == "test":
        return "verification_test"
    return "read_or_inspect"


def build_accounting_receipt(
    *,
    observation_seq: int,
    assignment_observation_seq: int,
    actor_session_id: str,
    parent_session_id: str,
    assignment_id: str,
    task_intent_disposition: str,
    turn_id: str,
    tool_call_id: Any,
    tool_name: str,
    action_fingerprint: str,
    stage: str,
    operation_category: str,
    guard_disposition: str,
    active_control_source: str,
    decision: str,
    outcome: str,
    counters: dict[str, int],
    prerequisite: dict[str, Any] | None,
    output_verification: dict[str, bool],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema": ACCOUNTING_RECEIPT_SCHEMA,
        "receipt_id": uuid.uuid4().hex[:16],
        "observation_seq": _nonneg_int(observation_seq),
        "assignment_observation_seq": _nonneg_int(assignment_observation_seq),
        "actor_session_id": sanitize_identifier(actor_session_id),
        "parent_session_id": sanitize_identifier(parent_session_id),
        "assignment_id": sanitize_identifier(assignment_id),
        "task_intent_disposition": sanitize_identifier(task_intent_disposition, max_len=32),
        "turn_id": sanitize_identifier(turn_id),
        "tool_call_id": sanitize_identifier(tool_call_id),
        "tool_name": sanitize_identifier(tool_name, max_len=64),
        "action_fingerprint": sanitize_identifier(action_fingerprint, max_len=64),
        "stage": sanitize_identifier(stage, max_len=32),
        "operation_category": sanitize_identifier(operation_category, max_len=32),
        "guard_disposition": sanitize_identifier(guard_disposition, max_len=64),
        "active_control_source": sanitize_identifier(active_control_source, max_len=64),
        "decision": sanitize_identifier(decision, max_len=32),
        "outcome": normalize_outcome_token(outcome),
        "counters": {
            "tool_actions": _nonneg_int(counters.get("tool_actions")),
            "mutation_attempts": _nonneg_int(counters.get("mutation_attempts")),
            "relevant_mutation_attempts": _nonneg_int(counters.get("relevant_mutation_attempts")),
            "successful_relevant_mutations": _nonneg_int(counters.get("successful_relevant_mutations")),
            "verified_test_passes": _nonneg_int(counters.get("verified_test_passes")),
            "produced_outputs_count": _nonneg_int(counters.get("produced_outputs_count")),
            "required_outputs_count": _nonneg_int(counters.get("required_outputs_count")),
        },
        "prerequisite": normalize_prerequisite_snapshot(prerequisite),
        "output_verification": {
            "change_required": bool(output_verification.get("change_required")),
            "verification_required": bool(output_verification.get("verification_required")),
            "outputs_satisfied": bool(output_verification.get("outputs_satisfied")),
            "verification_satisfied": bool(output_verification.get("verification_satisfied")),
        },
    }
    digest = canonical_hash(core)
    return {**core, "receipt_digest": digest}


def verify_accounting_receipt(
    receipt: Any,
    *,
    actor_session_id: str,
    assignment_id: str,
) -> bool:
    """Validate receipt integrity and bind it strictly to (actor_session_id, assignment_id)."""
    if not isinstance(receipt, dict):
        return False
    if receipt.get("schema") != ACCOUNTING_RECEIPT_SCHEMA:
        return False
    expected_actor = sanitize_identifier(actor_session_id)
    expected_assignment = sanitize_identifier(assignment_id)
    if not expected_actor or not expected_assignment:
        return False
    if receipt.get("actor_session_id") != expected_actor:
        return False
    if receipt.get("assignment_id") != expected_assignment:
        return False
    claimed_digest = receipt.get("receipt_digest")
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    return canonical_hash(core) == claimed_digest

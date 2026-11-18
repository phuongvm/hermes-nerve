# Jev Nervous System (v0.2.1.2)

Nerve v0.2.1.2 keeps Jev off Hermes' normal critical path. Hermes remains the reasoning and execution engine; Jev is an asynchronous decision supervisor.

## Turn lifecycle

1. `pre_llm_call` creates/resolves a `turn_id` and starts Jev admission in a background worker.
2. Admission returns `OFF`, `WATCH`, or `ON` using contract `hermes/jev-turn-admission/v1`.
3. Hermes continues immediately; admission never waits in the user-turn critical path.
4. Runtime/tool events are normalized into compact JSON events and processed locally.
5. The adaptive router forwards only state changes where a remote Jev opinion could matter.
6. Jev agreement and low-confidence disagreement are logged silently.
7. High-confidence disagreement becomes a `Challenge`.
8. `transform_tool_result` attaches a still-current challenge to the next model-bound tool result. Stale challenges are discarded from delivery and retained as telemetry.
9. `post_llm_call`/`on_session_end` closes turn-scoped supervision.

## Admission

- `OFF`: ordinary conversation/creative output with no accountable decision plane expected.
- `WATCH`: a decision plane may emerge; events stay local until material work appears.
- `ON`: active asynchronous supervision for the turn.

Pure roleplay is normally OFF. Roleplay with an accountable director/training objective can be WATCH/ON at the director plane; actor dialogue itself is not a decision event.

## Local adaptive router

The router does not use "every N events" as its primary trigger. It scores semantic change from structured fields including:

- materiality and consequence
- uncertainty
- novelty
- contradiction
- strategy change
- repeated failure/recovery
- completion pressure
- risk/reversibility
- freshness of the last remote assessment

Equivalent state is suppressed by semantic hysteresis. Significant events arriving while a Jev call is already in flight are retained in a local pending batch and summarized into a later assessment.

A hard per-turn provider-call budget exists only as a safety valve.

## Decision events

For the strongest supervision, emit `nerve_nervous_event` with bounded alternatives and Hermes' proposed answer in the original event:

```json
{
  "type": "DECISION",
  "goal": "repair failing service",
  "choices": ["PATCH", "ROLLBACK", "GATHER_MORE"],
  "hermes_decision": "PATCH",
  "materiality": 0.9,
  "uncertainty": 0.6,
  "state_version": "git:abc123",
  "decision_version": "repair-4"
}
```

This avoids a second round trip asking what Hermes decided.

## Confidence-gated challenges

`nervous_challenge_confidence` controls when disagreement becomes actionable. Agreement and lower-confidence disagreement remain telemetry. `correct_next` is the recommended default: Jev can challenge the next still-actionable decision without attempting to rewind already-committed actions.

Modes:

- `shadow`: observe only; never inject a challenge.
- `correct_next`: inject current high-confidence disagreement into the next tool-result turn context.
- `precommit`: retains asynchronous operation but lets `pre_verify` consume an already-arrived completion challenge. It never synchronously waits for Jev.

## Staleness and reversibility

Events may carry `state_version` and `decision_version`. A challenge is deliverable only while the relevant decision state is current. Late opinions remain receipts, not commands. Completed irreversible actions are never automatically undone.

## Long-running workers

Supervision is scoped to the turn, not a wall-clock timeout. A two-hour turn with hundreds of tool calls can remain ON. Raw events are accumulated locally; remote calls are driven by semantic relevance and batching rather than volume.

## Local learning

`OutcomeStore` can record labeled decision outcomes. After `nervous_local_learning_min_samples`, a tiny local historical model can calibrate router sensitivity from observed usefulness. It does not add another remote LLM.

## Prerequisite policy limits

The operator settings `prerequisite_max_operations` and `prerequisite_max_seconds` default to 64 attempts and 600 seconds. They accept integers in the ranges 1–128 and 1–900, respectively. The effective limits are the lower of the operator and trusted host limits. An invalid operator value disables prerequisite eligibility and records `prerequisite_config_invalid` with the setting name, not the value. This does not turn off the ordinary progress guard.

These settings define ceilings only. Host integration must reserve attempts before dispatch and treat exhausted or expired capacity as unavailable, without approving the call or resetting progress. Setting these values alone does not enable a prerequisite read.

## Privacy

The nervous system should receive compact structured state rather than full transcripts. Tool-result previews are bounded by `nervous_event_preview_chars`, and existing recursive secret-redaction remains available at provider boundaries. Raw roleplay/chat transcripts are not required for event supervision.

## 0.2.1 repeated-failure control lease

0.2.1 adds an execution-side recovery seam for a failure mode observed in Muna:
Jev could repeatedly recommend `REPLAN` or `GATHER_EVIDENCE` while Hermes still
issued the exact same failing action.

Each failed tool action now receives two stable local identities:

- **action fingerprint**: tool name + canonicalized arguments
- **failure fingerprint**: action fingerprint + status/available exit code + normalized error signature

The first failure can be assessed remotely. Exact repeats are deduplicated locally
until materially new state/evidence appears. At the configured repeat threshold
(default `3`), a provider-independent local `REPLAN` control is created.

Under `correct_next` and `precommit`:

- `REPLAN`, `GATHER_EVIDENCE`, and `ESCALATE` block the exact controlled action;
- a materially different next action counts as following the control and consumes it;
- `RETRY` explicitly allows one next action and consumes the control;
- `shadow` records the same lifecycle but does not constrain execution.

The plugin uses one composed `pre_tool_call` callback: local control is checked
first, and the optional legacy synchronous gate is only consulted if local control
allows the action. This prevents a known-bad repeated action from also paying an
unnecessary gate provider round trip.

Control receipts use stable `decision_id` values and lifecycle stages so telemetry
can connect decision -> delivery -> next action -> followed/enforced/expired -> outcome.
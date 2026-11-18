# Architecture — Nerve v0.2.2.dev4

## Public seam

The v0.2 decision architecture is deliberately deep: `NervousSystem` is the main orchestration seam. It owns turn admission, structured event intake, local relevance routing, remote assessment dispatch, challenge lifecycle, batching, staleness checks, and telemetry. `router.py` is local-only significance logic; `outcomes.py` owns decision/outcome learning data; `client.py` owns provider transport.

```text
user prompt
   |\
   | +--> async Jev turn admission --> OFF / WATCH / ON
   |
   +----> Hermes work loop -------------------------------+
                         |                                |
                         +--> structured events           |
                                  |                       |
                                  v                       |
                         local adaptive router             |
                           | no        | yes               |
                           v           v                   |
                       accumulate   async Jev              |
                                      |                    |
                         agree/low confidence              |
                              |       high-conflict        |
                              v            |               |
                             log           v               |
                                      challenge queue      |
                                           |               |
                              transform_tool_result <------+
```

Hermes remains the reasoning and execution engine. Jev is not placed in front of ordinary tool calls. The old pre-tool gate remains an opt-in compatibility/safety surface and defaults off.

## Hook mapping

- `pre_llm_call`: start/reuse turn; enqueue admission; returns immediately.
- `post_tool_call`: observe tool outcome and update the local event stream.
- `transform_tool_result`: attach a still-current high-confidence challenge to the next model-bound result.
- `pre_verify`: completion boundary; may consume an already-arrived challenge in precommit mode, never waits for one.
- `post_llm_call` / `on_session_end`: close turn-scoped supervision.
- `pre_tool_call`: legacy selective gate only.

## Native assignment intent (Task 2.2)

`pre_llm_call` forwards optional `task_intent` and `parent_session_id` to
`NervousSystem.start_turn`. On compatible Hermes hosts, the consumer accepts the
record only when it is the same object returned by
`agent.delegation_context.current_task_intent(session_id, parent_session_id)`
inside the active native assignment scope. The host accessor validates version,
field types, identity binding and goal digest. Dictionaries, copied records,
expired scopes and mismatched actor/parent identities do not establish provenance.
This is cooperative in-process provenance, not a sandbox against arbitrary Python
execution in the host and not a permission grant.

For a trusted record, only `goal_text` supplies `change_required`,
`verification_required` and `requested_paths`. Supporting context cannot add an
output target or verification obligation. The original combined prompt remains
available to admission and contributes its original telemetry hash; it is not
replaced in the model-visible brief. Mutation relevance uses the goal-derived
paths, and trusted non-output assignments receive no relevant-mutation credit.

The deterministic change classifier recognizes create/write/generate requests,
including document output after an audit. Quoted examples and fenced code are
excluded from change-intent classification; quoted filenames remain usable as
objects of an affirmative instruction. A line-initial `Change slug` label is
metadata, not an imperative. Runtime background-receipt handling retains its
legacy policy. These heuristics do not parse `TASK`/`CONTEXT` markers into trusted
metadata.

`status()` reports `task_intent_disposition`: `trusted`, `missing`, `invalid`, or
`unsupported-host`. Missing/invalid provenance uses the complete legacy message;
an unavailable host accessor reports `unsupported-host` and also uses legacy
handling. An old host that omits metadata therefore requires no new API. A new
host with an old plugin receives the old plugin's behavior, not these guarantees.

A relevant write attempt is not successful output. With an explicit empty
`changed_paths` report, completion remains blocked even after a relevant failed
write; unrelated/context-only and scratch writes do not satisfy a named output
target. This slice introduces no prerequisite allowance, PREPARING state,
continuation retention, revocation protocol or exemption from independent guards.
Those require subsequent implementation and integration proof.

### Scoped verification

Run from the Nerve worktree with a separate developer interpreter and
`HERMES_TASK_INTENT_HOST` set to the compatible Hermes worktree:

```text
python -E -B -m pytest -o pythonpath=tests -q -p no:cacheprovider tests/test_task_intent_contract.py
```

The host fixture loads the actual `agent/delegation_context.py` from that explicit
checkout and asserts its path, avoiding an already-imported live host module.
Host-dependent cases skip when the variable is absent; such a run is not complete
cross-repository proof. These tests cover native record production and the Nerve
hook/observation/completion seams, not the full registered dispatch/Gateway E2E.
Full-suite baseline failures must be reported separately from scoped proof.

## Provider seam

All three transports implement the same System One contract:

- OpenRouter Decisions -> `openrouter-decisions`
- TypeSafe System One direct -> `typesafe-system-one`
- OpenCode Zen System One -> `opencode-zen-system-one`

Provider selection does not change Hermes-visible tools or nervous-system schemas.

## State and challenge safety

Every event can carry state and decision versions. The challenge queue validates freshness before delivery. Late Jev answers are telemetry, not retroactive commands. Already-committed irreversible actions are never blindly rewound.

## Context governor

The v0.1.x context-value governor remains a separate, optional deep seam. The nervous system can coexist with shadow/apply context curation; neither requires the other.
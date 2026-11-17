# Designer prerequisite incident fixture

## Purpose and provenance

Task 1.3 of `nerve-task-context-and-prerequisite-progress` preserves a sanitized historical derivative and pins the reproduction sources independently of future implementation changes. It is not a Gateway integration test or an archive of the incident process binary.

`designer_prerequisite_incident.json` retains the original task wording with the change slug replaced by `example-topic`. Context is reduced to four relevant noun fragments; operational context, machine paths, endpoints and the full transcript are omitted. Its provenance records the original prompt canonical hash, source hashes, session/message IDs and historical counter/control evidence. The sanitized fixture is NOT claimed to have the original prompt hash.

Original prompt canonical SHA-256: `e1defd679466ca4ccef3f793b439abca8f68ac0ceea142bee6a0bc92478fc9c4`. During authoring, the parent independently recomputed this from the original SQLite message using a read-only connection. CI never reads that database.

Fixture SHA-256 (UTF-8 JSON serialization with LF newlines): `9f574f801684c392b0ccfe6b73fa33c88c52eb49ed1359298624a6618799a13b`.

Historical evidence: progress_stall_replan at `2026-11-17T14:00:31.812690+00:00`, tool_actions=12, mutation_attempts=0, relevant_mutation_attempts=0; decision `local-9f69d03aa017499593653bfbb3e5becb`, source local-progress-stall, enforced at `2026-11-17T14:01:09.175912+00:00`. These are the archived investigation's historical observations, not newly generated runtime events.

Approved canonical exploration and same-project snapshot were compared byte-for-byte during packaging: SHA-256 `4d9e0c1da0a8b3fbfd588128f2e18a8945075aa9912eebd3f8b97c3b3e1b9f8d`. Embedded archived probe SHA-256: `f6ec6170dcc3e45c839a08b3d58fd77afc88eca560cd9eee374e5564f145ddae`. Neither approved artifact was edited.

## Reproduction contract

Run from the owning Nerve worktree with an existing pytest interpreter:

`python -E -B -m pytest -q -p no:cacheprovider tests/test_historical_prerequisite_fixture.py`

Author run: `4 passed in 0.73s`, exit 0.

The test requires local Git objects for commit `323431b824c225eefd3d013f1d718d60e7b5b08e`. A shallow clone lacking that object fails explicitly; the test never fetches or falls back to current/live source. The repository path is resolved from the test file, not a machine-specific checkout.

All three Git blob digests are verified before loading any source module. The pinned privacy and progress modules have only standard-library imports. The nervous-system AST extracts the same isolated seams used in the approved exploration, replacing constructor, worker, logging, outcome sinks and risk labels. It never constructs a provider or executes the denied read command. Verified bytes are compiled directly; no path is reopened after verification.

The initial packaging probe failed because Git LF blobs were compared to Windows checkout CRLF digests (3 failed, 1 passed). Byte comparison proved CRLF/LF was the only difference for all three sources. Both sets of digests remain in the fixture. Tests validate the LF blobs and reconstruct their CRLF representation to verify the original pins; this is not silent hash normalization of unknown source changes.

Positive cases reproduce full sanitized input at 11/12/16 observations, task-only input at 12, exact denial text and context-noun ablation. Task-only False records OLD behavior and is not the repaired classifier's target. Negative tests independently corrupt each expected digest, including privacy.py, and assert rejection before any module-loader invocation.

## Limits

- Full historical prompt replay was performed in the archived exploration, not by this sanitized fixture.
- Hashes pin reproduction sources, not independently archived incident binaries.
- The seam harness is intentionally implementation-coupled historical evidence, not a future public-API acceptance test.
- Concurrent scheduling, native hooks, dispatch, permissions and Gateway E2E remain outside this test.
- Task 1.2 RED cases remain deliberately failing until implementation task 2.2; these four passing historical tests do not mean the product bug is fixed.

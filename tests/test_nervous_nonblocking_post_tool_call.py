"""Regression tests for non-blocking nervous post_tool_call / observe_tool_call."""
from __future__ import annotations

import json
import threading
import time

from hermes_nerve import experience, jsonl, nervous


def _isolated_nervous_env(tmp_path, monkeypatch):
    events_path = tmp_path / "nervous-events.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"
    policy_path = tmp_path / "policy-memory.jsonl"
    exp_path = tmp_path / "experience.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events_path))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(outcomes_path))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(policy_path))
    monkeypatch.setenv("HERMES_NERVE_EXPERIENCE", str(exp_path))
    return events_path, outcomes_path, policy_path, exp_path


def test_post_tool_call_does_not_block_when_jsonl_append_stalls(tmp_path, monkeypatch):
    events_path, _outcomes_path, _policy_path, _exp_path = _isolated_nervous_env(tmp_path, monkeypatch)
    system = nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No remote calls expected"))
    )
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0)

    real_append = jsonl.append_jsonl
    write_entered = threading.Event()
    release_write = threading.Event()

    system.start_turn(user_message="Inspect repository status.", session_id="s-stall", turn_id="t-stall")

    def stalled_append(path, record):
        write_entered.set()
        release_write.wait(timeout=1.5)
        return real_append(path, record)

    monkeypatch.setattr(jsonl, "append_jsonl", stalled_append)
    monkeypatch.setattr(nervous, "append_jsonl", stalled_append)

    try:
        started = time.monotonic()
        nervous.post_tool_call(
            turn_id="t-stall",
            session_id="s-stall",
            tool_name="read_file",
            args={"path": "README.md"},
            tool_call_id="call-stall-1",
            status="ok",
            result="ok",
        )
        elapsed = time.monotonic() - started

        assert elapsed < 0.15, f"post_tool_call blocked hook thread for {elapsed:.3f}s on append_jsonl"
        assert system.status(turn_id="t-stall")["tool_actions"] == 1
        receipts = system.accounting_receipts(session_id="s-stall", turn_id="t-stall")
        assert any(r["tool_call_id"] == "call-stall-1" and r["stage"] == "post_tool" for r in receipts)

        assert write_entered.wait(timeout=1.0)
        release_write.set()
        assert system.drain(timeout=2.0)

        rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert any(r.get("kind") == "accounting_receipt" and r.get("tool_call_id") == "call-stall-1" for r in rows)
        assert any(r.get("kind") == "router" for r in rows)
    finally:
        release_write.set()
        system.shutdown()


def test_post_tool_call_does_not_block_when_experience_store_stalls(tmp_path, monkeypatch):
    _events_path, _outcomes_path, _policy_path, exp_path = _isolated_nervous_env(tmp_path, monkeypatch)
    experience.configure(enabled=True)
    system = nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No remote calls expected"))
    )
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0)
    system.start_turn(user_message="Read missing config.", session_id="s-exp", turn_id="t-exp")

    real_store_factory = experience.ExperienceStore
    claim_entered = threading.Event()
    release_claim = threading.Event()

    class StalledExperienceStore(experience.ExperienceStore):
        def create_claim(self, **kwargs):
            claim_entered.set()
            release_claim.wait(timeout=1.5)
            return super().create_claim(**kwargs)

    monkeypatch.setattr(experience, "default_store", lambda: StalledExperienceStore(exp_path))

    try:
        started = time.monotonic()
        nervous.post_tool_call(
            turn_id="t-exp",
            session_id="s-exp",
            tool_name="read_file",
            args={"path": "missing.yaml"},
            tool_call_id="call-exp-1",
            status="error",
            result="file not found: missing.yaml",
            error_message="file not found: missing.yaml",
        )
        elapsed = time.monotonic() - started

        assert elapsed < 0.15, f"post_tool_call blocked hook thread for {elapsed:.3f}s on experience.default_store()"
        active = system.status(turn_id="t-exp")["active_control"]
        assert active is not None and active["control"] == "REPLAN"

        assert claim_entered.wait(timeout=1.0)
        release_claim.set()
        assert system.drain(timeout=2.0)

        persisted = real_store_factory(exp_path).claims()
        assert any(c.get("claim_type") == "disproven" for c in persisted.values())
    finally:
        release_claim.set()
        experience.configure(enabled=False)
        system.shutdown()


def test_io_queue_backpressure_drops_excess_writes_without_blocking_or_breaking_state(tmp_path, monkeypatch):
    _isolated_nervous_env(tmp_path, monkeypatch)
    experience.configure(enabled=True)
    default_sys = nervous.NervousSystem()
    assert default_sys._config.io_queue_maxsize == 64
    assert default_sys._io_queue.maxsize == 64

    system = nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No remote calls expected"))
    )
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(
        enabled=True,
        admission_enabled=False,
        mode="correct_next",
        call_threshold=1.0,
        local_learning=True,
        policy_memory_enabled=True,
        repeated_failure_local_replan_at=3,
        io_queue_maxsize=4,
    )
    system.start_turn(user_message="Repair service.", session_id="s-bp", turn_id="t-bp")
    assert system.flush_io(timeout=1.0)

    real_append = jsonl.append_jsonl
    first_entered = threading.Event()
    release_write = threading.Event()

    def stalled_append(path, record):
        if threading.current_thread().name == "hermes-nerve-nervous-io":
            first_entered.set()
            release_write.wait(timeout=1.5)
        return real_append(path, record)

    monkeypatch.setattr(jsonl, "append_jsonl", stalled_append)
    monkeypatch.setattr(nervous, "append_jsonl", stalled_append)

    try:
        started = time.monotonic()
        for idx in range(20):
            nervous.post_tool_call(
                turn_id="t-bp",
                session_id="s-bp",
                tool_name="terminal",
                args={"command": "python broken_check.py"},
                tool_call_id=f"call-bp-{idx}",
                status="error",
                result="connection failed after 501 ms",
                error_message="connection failed after 501 ms",
            )
        elapsed = time.monotonic() - started

        assert elapsed < 0.30, f"20 post_tool_call invocations under backpressure took {elapsed:.3f}s"
        assert first_entered.wait(timeout=1.0)
        assert system.flush_io(timeout=0.03) is False

        st = system.status(turn_id="t-bp")
        assert st["tool_actions"] == 20
        assert st["active_control"] is not None and st["active_control"]["source"] == "local-loop-breaker"
        verdict = system.before_tool_call(
            turn_id="t-bp",
            session_id="s-bp",
            tool_name="terminal",
            args={"command": "python broken_check.py"},
            tool_call_id="call-bp-next",
        )
        assert verdict is not None and verdict["action"] == "block"

        receipts = [
            r for r in system.accounting_receipts(session_id="s-bp", turn_id="t-bp")
            if r["stage"] == "post_tool"
        ]
        assert len(receipts) == 20
        assert [r["assignment_observation_seq"] for r in receipts] == list(range(1, 21))
        assert len(receipts[-1]["receipt_digest"]) == 64

        metrics = system.report()["metrics"]
        assert metrics.get("io_queue_dropped", 0) > 0

        release_write.set()
        assert nervous.flush_io(timeout=2.0) is True
    finally:
        release_write.set()
        experience.configure(enabled=False)
        system.shutdown()


def test_shutdown_stays_bounded_when_io_worker_is_stalled(tmp_path, monkeypatch):
    _isolated_nervous_env(tmp_path, monkeypatch)
    system = nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No remote calls expected"))
    )
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0)
    system.start_turn(user_message="Check bounded shutdown.", session_id="s-sd", turn_id="t-sd")

    real_append = jsonl.append_jsonl
    entered = threading.Event()
    release = threading.Event()

    def hung_append(path, record):
        entered.set()
        release.wait(timeout=3.0)
        return real_append(path, record)

    monkeypatch.setattr(jsonl, "append_jsonl", hung_append)
    monkeypatch.setattr(nervous, "append_jsonl", hung_append)

    try:
        for idx in range(5):
            nervous.post_tool_call(
                turn_id="t-sd",
                session_id="s-sd",
                tool_name="read_file",
                args={"path": f"f{idx}.txt"},
                tool_call_id=f"sd-{idx}",
                status="ok",
                result="ok",
            )
        assert entered.wait(timeout=1.0)

        started = time.monotonic()
        system.shutdown()
        shutdown_elapsed = time.monotonic() - started

        assert shutdown_elapsed < 0.60, f"shutdown() hung for {shutdown_elapsed:.3f}s on stalled I/O worker"
        assert system.report()["enabled"] is False
        assert system.report()["active_turns"] == 0
    finally:
        release.set()


def test_delayed_io_worker_writes_to_caller_captured_paths_not_later_env(tmp_path, monkeypatch):
    dir_a = tmp_path / "scope_a"
    dir_b = tmp_path / "scope_b"
    dir_a.mkdir()
    dir_b.mkdir()
    events_a = dir_a / "nervous-events.jsonl"
    exp_a = dir_a / "experience.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(dir_a))
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events_a))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(dir_a / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(dir_a / "policy.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_EXPERIENCE", str(exp_a))

    experience.configure(enabled=True)
    system = nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No remote calls expected"))
    )
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0)
    system.start_turn(user_message="Verify path capture.", session_id="s-cap", turn_id="t-cap")
    assert system.flush_io(timeout=1.0)

    real_append = jsonl.append_jsonl
    entered = threading.Event()
    release = threading.Event()

    def gated_append(path, record):
        entered.set()
        release.wait(timeout=1.5)
        return real_append(path, record)

    monkeypatch.setattr(jsonl, "append_jsonl", gated_append)
    monkeypatch.setattr(nervous, "append_jsonl", gated_append)

    try:
        nervous.post_tool_call(
            turn_id="t-cap",
            session_id="s-cap",
            tool_name="read_file",
            args={"path": "missing.txt"},
            tool_call_id="cap-1",
            status="error",
            result="file not found: missing.txt",
            error_message="file not found: missing.txt",
        )
        assert entered.wait(timeout=1.0)

        # Mutate environment paths BEFORE the background worker finishes flushing.
        monkeypatch.setenv("HERMES_HOME", str(dir_b))
        monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(dir_b / "nervous-events.jsonl"))
        monkeypatch.setenv("HERMES_NERVE_EXPERIENCE", str(dir_b / "experience.jsonl"))

        release.set()
        assert system.drain(timeout=2.0)

        assert events_a.exists()
        assert exp_a.exists()
        assert not (dir_b / "nervous-events.jsonl").exists()
        assert not (dir_b / "experience.jsonl").exists()
    finally:
        release.set()
        experience.configure(enabled=False)
        system.shutdown()

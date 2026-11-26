"""Prerequisite limits and reservation accounting, without content dispatch."""
import json
import os
import runpy
from pathlib import Path

import pytest

from hermes_nerve.nervous import NervousSystem


@pytest.mark.parametrize("operator, host, expected", [
    ({}, (128, 900), (64, 600)),
    ({}, (1, 1), (1, 1)),
    ({"prerequisite_max_operations": 128, "prerequisite_max_seconds": 900}, (128, 900), (128, 900)),
    ({"prerequisite_max_operations": 1, "prerequisite_max_seconds": 900}, (128, 1), (1, 1)),
])
def test_prerequisite_limits_take_lower_bound(operator, host, expected):
    with NervousSystem() as system:
        system.configure(**operator)
        assert system.prerequisite_limits(*host) == expected


@pytest.mark.parametrize("key, value", [
    ("prerequisite_max_operations", 0), ("prerequisite_max_operations", 129),
    ("prerequisite_max_operations", True), ("prerequisite_max_operations", 64.0),
    ("prerequisite_max_seconds", 0), ("prerequisite_max_seconds", 901),
    ("prerequisite_max_seconds", None), ("prerequisite_max_seconds", "must-not-echo"),
])
def test_invalid_prerequisite_config_disables_only_allowance(key, value, tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider_attempts = []

    def forbidden_provider():
        provider_attempts.append(True)
        raise AssertionError("No inference in prerequisite limits tests")

    with NervousSystem(engine_factory=forbidden_provider) as system:
        system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0,
                         local_learning=False, policy_memory_enabled=False, **{key: value})
        assert system.prerequisite_limits(64, 600) is None
        assert system.report()["enabled"] is True
        system.start_turn(user_message="Implement the fix in src/parser.py.", session_id="actor", turn_id="turn")
        args = {"path": "docs/prerequisite.md"}
        for index in range(12):
            system.observe_tool_call(tool_name="read_file", args=args, status="ok", result="fixture",
                                     tool_call_id=f"read-{index}", turn_id="turn")
        decision = system.before_tool_call(tool_name="read_file", args=args, turn_id="turn")
        assert decision is not None and decision["action"] == "block"
        assert "task-relevant implementation" in decision["message"]
    assert not provider_attempts
    data = events.read_text(encoding="utf-8")
    assert "must-not-echo" not in data
    assert any(row.get("kind") == "prerequisite_config_invalid" and row.get("setting") == key
               for row in map(json.loads, data.splitlines()))


@pytest.mark.parametrize("host", [(None, 1), (0, 1), (129, 900), (1, 901), (1, True)])
def test_invalid_host_ceiling_is_not_given_defaults(host):
    with NervousSystem() as system:
        assert system.prerequisite_limits(*host) is None


@pytest.mark.parametrize("profile, sidecar_limits, expected", [
    pytest.param("legacy", None, (21, 210), id="legacy-plugin-settings"),
    pytest.param("legacy", (7, 70), (7, 70), id="legacy-sidecar-valid"),
    pytest.param("legacy", (7, "must-not-echo"), None, id="legacy-sidecar-invalid"),
    pytest.param("operator", (7, 70), (7, 70), id="operator-sidecar-valid"),
    pytest.param("operator", (7, "must-not-echo"), None, id="operator-sidecar-invalid"),
])
def test_registration_resolves_prerequisite_ceilings(tmp_path, monkeypatch, profile, sidecar_limits, expected):
    from hermes_nerve.profiles import save_profile
    from tests.test_module_absence import Ctx, load_plugin

    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK_ID", raising=False)
    values = {"prerequisite_max_operations": 21, "prerequisite_max_seconds": 210,
              "nervous_mode": "shadow"}
    if sidecar_limits is not None:
        operations, seconds = sidecar_limits
        save_profile({"version": 1, "nerve_profile": profile, "nerve_modules": {}, "advanced": {
            "prerequisite_max_operations": operations, "prerequisite_max_seconds": seconds,
            "nervous_mode": "correct_next",
        }}, tmp_path)
    plugin = load_plugin()
    with NervousSystem() as system:
        monkeypatch.setattr(plugin.nervous, "_default", system)
        plugin.register(Ctx(tmp_path, values))

        assert system.prerequisite_limits(128, 900) == expected
        assert system.report()["mode"] == ("shadow" if profile == "legacy" else "correct_next")
        if expected is None:
            assert system.drain(timeout=2.0)
            data = events.read_text(encoding="utf-8")
            assert "must-not-echo" not in data
            assert any(row.get("kind") == "prerequisite_config_invalid"
                       and row.get("setting") == "prerequisite_max_seconds"
                       for row in map(json.loads, data.splitlines()))


@pytest.mark.parametrize("case", ["valid", "invalid", "no-catalog", "read-only", "output"])
def test_native_hook_starts_preparation_once(tmp_path, monkeypatch, case):
    from hermes_nerve import nervous

    source = os.environ.get("HERMES_TASK_INTENT_HOST")
    if not source:
        pytest.skip("Set HERMES_TASK_INTENT_HOST to the matching Hermes worktree")
    host = Path(source).resolve()
    monkeypatch.syspath_prepend(str(host))
    from agent import delegation_context as dc
    from agent.prerequisite import catalog, reservations

    assert Path(dc.__file__).resolve().is_relative_to(host)
    assert Path(reservations.__file__).resolve().is_relative_to(host)
    fixture = runpy.run_path(str(host / "tests/tools/proposal_catalog_fixture.py"))
    planning = tmp_path / "planning"
    config = fixture["proposal_entry"](planning)
    selection = None if case == "no-catalog" else catalog.freeze_proposal_catalog(fixture["CHANGE"], config)
    monkeypatch.setattr(dc, "child_proposal_selection", lambda *a, **kw: selection)
    now = [5.0]
    monkeypatch.setattr(reservations, "_monotonic", lambda: now[0])
    for key, name in (("HERMES_NERVE_NERVOUS_EVENTS", "events.jsonl"),
                      ("HERMES_NERVE_OUTCOMES", "outcomes.jsonl"),
                      ("HERMES_NERVE_POLICY_MEMORY", "policy.jsonl")):
        monkeypatch.setenv(key, str(tmp_path / name))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider_attempts = []

    def forbidden_provider():
        provider_attempts.append(True)
        raise AssertionError("No inference in prerequisite accounting tests")

    with NervousSystem(engine_factory=forbidden_provider) as system:
        monkeypatch.setattr(nervous, "_default", system)
        system.configure(enabled=True, admission_enabled=False, call_threshold=1.0,
                         local_learning=False, policy_memory_enabled=False,
                         prerequisite_max_operations=1, prerequisite_max_seconds=0 if case == "invalid" else 10)
        goal = "Summarize the report." if case == "read-only" else "Write proposal.md."
        scope = dict(actor_session_id="actor", parent_session_id="parent", goal=goal, context=None)
        args = {"path": str(planning / "rules/AGENTS.md")}
        with dc.native_task_intent(**scope) as intent:
            hook = dict(user_message=goal, session_id="actor", parent_session_id="parent", task_intent=intent)
            nervous.pre_llm_call(**hook, turn_id="initial")
            decision = reservations.reserve_read(intent, "actor", "page-1", "read_file", args)
            assert decision == ("reserved" if case in {"valid", "output"} else "unavailable")
            if case == "output":
                reservations.note_output_dispatch("actor", "parent", "write_file", {"path": "scratch.md"})
                assert reservations.reserve_read(intent, "actor", "page-2", "read_file", args) == "exhausted"
                reservations.note_output_dispatch("actor", "parent", "write_file", {"path": "proposal.md"})
                assert reservations.begin_read(intent, "actor", "page-1", "read_file", args) == "output"
            assert system.status(turn_id="initial")["tool_actions"] == 0
            system.observe_tool_call(turn_id="initial", tool_name="read_file", args=args,
                                     tool_call_id="ordinary", status="ok", result="fixture")
            system.finish_turn(turn_id="initial")
            handoff = dc._ResumeIntent(intent)
        system.configure(prerequisite_max_operations=128, prerequisite_max_seconds=900)
        with dc.native_task_intent(**scope, resumed=True, handoff=handoff) as resumed:
            assert resumed is intent
            nervous.pre_llm_call(**hook, turn_id="continued")
            assert system.status(turn_id="continued")["tool_actions"] == 1
            decision = reservations.reserve_read(intent, "actor", "page-2", "read_file", args)
            assert decision == {"valid": "exhausted", "output": "output"}.get(case, "unavailable")
    assert not provider_attempts


def _host_fixture_setup(tmp_path, monkeypatch, *, max_ops=8, max_seconds=120):
    from hermes_nerve import nervous

    source = os.environ.get("HERMES_TASK_INTENT_HOST")
    if not source:
        pytest.skip("Set HERMES_TASK_INTENT_HOST to the matching Hermes worktree")
    host = Path(source).resolve()
    monkeypatch.syspath_prepend(str(host))
    from agent import delegation_context as dc
    from agent.prerequisite import catalog, reservations

    fixture = runpy.run_path(str(host / "tests/tools/proposal_catalog_fixture.py"))
    planning = tmp_path / "planning"
    entry = fixture["proposal_entry"](planning)
    selection = catalog.freeze_proposal_catalog(fixture["CHANGE"], entry)
    monkeypatch.setattr(dc, "child_proposal_selection", lambda *_a, **_kw: selection)
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def forbidden_provider():
        raise AssertionError("No inference in accounting receipt tests")

    system = NervousSystem(engine_factory=forbidden_provider)
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(
        enabled=True,
        admission_enabled=False,
        mode="correct_next",
        call_threshold=1.0,
        local_learning=False,
        policy_memory_enabled=False,
        prerequisite_max_operations=max_ops,
        prerequisite_max_seconds=max_seconds,
    )
    return dc, reservations, system, planning, events


def test_accounting_receipts_capture_concurrent_completion_order_independent_of_transcript_order(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    dc, reservations, system, planning, _events = _host_fixture_setup(tmp_path, monkeypatch)
    doc_a = str(planning / "rules/AGENTS.md")
    doc_b = str(planning / "rules/shared/safety.md")
    scope = dict(actor_session_id="actor", parent_session_id="parent", goal="Write proposal.md.", context=None)
    second_observed = Event()
    both_entered = Event()

    try:
        with dc.native_task_intent(**scope) as intent:
            system.start_turn(
                user_message="Write proposal.md.",
                session_id="actor",
                parent_session_id="parent",
                turn_id="turn-race",
                task_intent=intent,
            )

            def run_first_in_transcript():
                both_entered.wait(5)
                assert second_observed.wait(5)
                system.observe_tool_call(
                    turn_id="turn-race",
                    session_id="actor",
                    tool_name="read_file",
                    args={"path": doc_a, "offset": 1, "limit": 10},
                    tool_call_id="call-transcript-0",
                    status="ok",
                    result="line 1",
                )
                return {"tool_call_id": "call-transcript-0", "transcript_row": 0}

            def run_second_in_transcript():
                both_entered.set()
                system.observe_tool_call(
                    turn_id="turn-race",
                    session_id="actor",
                    tool_name="read_file",
                    args={"path": doc_b, "offset": 1, "limit": 10},
                    tool_call_id="call-transcript-1",
                    status="ok",
                    result="line 1",
                )
                second_observed.set()
                return {"tool_call_id": "call-transcript-1", "transcript_row": 1}

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut0 = pool.submit(run_first_in_transcript)
                fut1 = pool.submit(run_second_in_transcript)
                # Transcript persists rows in dispatch/request array order [0, 1], opposite completion order!
                transcript_rows = [fut0.result(timeout=10), fut1.result(timeout=10)]

            assert [r["tool_call_id"] for r in transcript_rows] == ["call-transcript-0", "call-transcript-1"]

            receipts = [
                r for r in system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)
                if r["stage"] == "post_tool"
            ]
            assert [r["tool_call_id"] for r in receipts] == ["call-transcript-1", "call-transcript-0"]
            assert receipts[0]["observation_seq"] < receipts[1]["observation_seq"]
            assert [r["assignment_observation_seq"] for r in receipts] == [1, 2]
            assert [r["counters"]["tool_actions"] for r in receipts] == [1, 2]
            for r in receipts:
                assert r["schema"] == "hermes-nerve-accounting-receipt/v1"
                assert r["actor_session_id"] == "actor"
                assert r["parent_session_id"] == "parent"
                assert r["assignment_id"] == intent.assignment_id
                assert r["prerequisite"]["phase"] == "preparing"
                assert r["prerequisite"]["operations_limit"] == 8
                assert system.verify_accounting_receipt(
                    r, actor_session_id="actor", assignment_id=intent.assignment_id
                ) is True
    finally:
        system.drain(timeout=2.0)
        system.shutdown(flush_timeout=2.0, join_timeout=2.0, drop_pending=True)


def test_accounting_receipts_reject_secret_and_source_content_leakage(tmp_path, monkeypatch):
    dc, reservations, system, planning, events = _host_fixture_setup(tmp_path, monkeypatch)
    secret_sk = "sk-liveSecretKey1234567890abcdef"
    secret_gh = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    secret_kv = "ULTRA_SECRET_CREDENTIAL_VALUE_991"
    source_sentinel = "PREREQ_SOURCE_BODY_SENTINEL_88421"
    doc = str(planning / "rules/AGENTS.md")
    scope = dict(actor_session_id="actor", parent_session_id="parent", goal="Write proposal.md.", context=None)

    try:
        with dc.native_task_intent(**scope) as intent:
            system.start_turn(
                user_message=f"Write proposal.md. API_KEY={secret_kv}",
                session_id="actor",
                parent_session_id="parent",
                turn_id="turn-sec",
                task_intent=intent,
            )
            noisy_args = {
                "path": doc,
                "api_key": secret_kv,
                "note": f"{source_sentinel} {secret_sk}",
            }
            system.before_tool_call(
                turn_id="turn-sec",
                session_id="actor",
                tool_name="read_file",
                args=noisy_args,
                tool_call_id=f"call-{secret_sk}",
            )
            system.observe_tool_call(
                turn_id="turn-sec",
                session_id="actor",
                tool_name="read_file",
                args=noisy_args,
                tool_call_id=f"call-{secret_sk}",
                status="error",
                result=f"1: {source_sentinel} Bearer {secret_gh}",
                error_message=f"boom {source_sentinel} {secret_gh} API_KEY={secret_kv}",
            )
            system.completion_gate(
                turn_id="turn-sec",
                session_id="actor",
                changed_paths=[f"scratch-{secret_sk}.md"],
                final_response=f"{source_sentinel} {secret_gh}",
            )

            receipts = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)
            assert len(receipts) >= 3
            serialized_receipts = json.dumps(receipts)
            assert system.drain(timeout=2.0)
            disk_jsonl = events.read_text(encoding="utf-8")
            for forbidden in (secret_sk, secret_gh, secret_kv, source_sentinel):
                assert forbidden not in serialized_receipts
                assert forbidden not in disk_jsonl
    finally:
        system.drain(timeout=2.0)
        system.shutdown(flush_timeout=2.0, join_timeout=2.0, drop_pending=True)


def test_accounting_receipt_verification_rejects_replay_across_actor_or_assignment_and_tampering(
    tmp_path, monkeypatch
):
    dc, _reservations, system, planning, _events = _host_fixture_setup(tmp_path, monkeypatch)
    doc = str(planning / "rules/AGENTS.md")
    try:
        with dc.native_task_intent(
            actor_session_id="actor-a", parent_session_id="parent", goal="Write proposal.md.", context=None
        ) as intent_a:
            system.start_turn(
                user_message="Write proposal.md.",
                session_id="actor-a",
                parent_session_id="parent",
                turn_id="turn-a",
                task_intent=intent_a,
            )
            system.observe_tool_call(
                turn_id="turn-a",
                session_id="actor-a",
                tool_name="read_file",
                args={"path": doc},
                tool_call_id="call-a1",
                status="ok",
                result="ok",
            )
            receipt_a = system.accounting_receipts(session_id="actor-a", assignment_id=intent_a.assignment_id)[-1]
            assert system.verify_accounting_receipt(
                receipt_a, actor_session_id="actor-a", assignment_id=intent_a.assignment_id
            ) is True
            # Reject cross-actor replay
            assert system.verify_accounting_receipt(
                receipt_a, actor_session_id="actor-b", assignment_id=intent_a.assignment_id
            ) is False

        with dc.native_task_intent(
            actor_session_id="actor-a", parent_session_id="parent", goal="Write proposal.md.", context=None
        ) as intent_b:
            assert intent_b.assignment_id != intent_a.assignment_id
            # Reject prior-assignment receipt replay for a new assignment on the same actor
            assert system.verify_accounting_receipt(
                receipt_a, actor_session_id="actor-a", assignment_id=intent_b.assignment_id
            ) is False

            # Reject tampered counter/disposition
            tampered = json.loads(json.dumps(receipt_a))
            tampered["counters"]["successful_relevant_mutations"] = 1
            assert system.verify_accounting_receipt(
                tampered, actor_session_id="actor-a", assignment_id=intent_a.assignment_id
            ) is False
    finally:
        system.drain(timeout=2.0)
        system.shutdown(flush_timeout=2.0, join_timeout=2.0, drop_pending=True)


def test_completion_gate_and_receipts_reject_scratch_unrelated_and_failed_writes_without_fake_progress(
    tmp_path, monkeypatch
):
    dc, reservations, system, _planning, _events = _host_fixture_setup(tmp_path, monkeypatch)
    goal = "Write proposal.md and verify it."
    context = "Reference context mentions src/context.py and scratch.md."
    try:
        with dc.native_task_intent(
            actor_session_id="actor", parent_session_id="parent", goal=goal, context=context
        ) as intent:
            system.start_turn(
                user_message=f"CONTEXT:\n{context}\nTASK:\n{goal}",
                session_id="actor",
                parent_session_id="parent",
                turn_id="turn-prog",
                task_intent=intent,
            )

            # 1. Scratch write: neither ends preparation nor grants relevant progress or completion.
            reservations.note_output_dispatch("actor", "parent", "write_file", {"path": "scratch.md"})
            system.observe_tool_call(
                turn_id="turn-prog",
                session_id="actor",
                tool_name="write_file",
                args={"path": "scratch.md", "content": "temp"},
                tool_call_id="w-scratch",
                status="ok",
                result="written",
            )
            r_scratch = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)[-1]
            assert r_scratch["operation_category"] == "unrelated_mutation"
            assert r_scratch["counters"]["mutation_attempts"] == 1
            assert r_scratch["counters"]["relevant_mutation_attempts"] == 0
            assert r_scratch["counters"]["successful_relevant_mutations"] == 0
            assert r_scratch["prerequisite"]["phase"] == "preparing"
            assert r_scratch["output_verification"]["outputs_satisfied"] is False
            gate_scratch = system.completion_gate(
                turn_id="turn-prog", session_id="actor", changed_paths=["scratch.md"]
            )
            assert gate_scratch is not None and gate_scratch["action"] == "continue"

            # 2. Unrelated context-only path write: also rejected as fake progress.
            reservations.note_output_dispatch("actor", "parent", "write_file", {"path": "src/context.py"})
            system.observe_tool_call(
                turn_id="turn-prog",
                session_id="actor",
                tool_name="write_file",
                args={"path": "src/context.py", "content": "unrelated"},
                tool_call_id="w-unrelated",
                status="ok",
                result="written",
            )
            r_unrel = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)[-1]
            assert r_unrel["operation_category"] == "unrelated_mutation"
            assert r_unrel["counters"]["relevant_mutation_attempts"] == 0
            assert r_unrel["prerequisite"]["phase"] == "preparing"
            gate_unrel = system.completion_gate(
                turn_id="turn-prog", session_id="actor", changed_paths=["src/context.py"]
            )
            assert gate_unrel is not None and gate_unrel["action"] == "continue"

            # 3. Failed write to requested target proposal.md: retires preparation to OUTPUT,
            #    counts attempt, but NEVER grants successful_relevant_mutations or completion!
            reservations.note_output_dispatch("actor", "parent", "write_file", {"path": "proposal.md"})
            system.observe_tool_call(
                turn_id="turn-prog",
                session_id="actor",
                tool_name="write_file",
                args={"path": "proposal.md", "content": "draft"},
                tool_call_id="w-failed",
                status="error",
                result="",
                error_message="ENOSPC: disk full",
            )
            r_fail = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)[-1]
            assert r_fail["operation_category"] == "relevant_mutation"
            assert r_fail["outcome"] == "error"
            assert r_fail["counters"]["relevant_mutation_attempts"] == 1
            assert r_fail["counters"]["successful_relevant_mutations"] == 0
            assert r_fail["prerequisite"]["phase"] == "output"
            assert r_fail["output_verification"]["outputs_satisfied"] is False
            gate_fail = system.completion_gate(
                turn_id="turn-prog", session_id="actor", changed_paths=["proposal.md"]
            )
            assert gate_fail is not None and gate_fail["action"] == "continue"
            r_gate_fail = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)[-1]
            assert r_gate_fail["stage"] == "completion_gate"
            assert r_gate_fail["guard_disposition"] == "completion_blocked"

            # 4. Successful write to proposal.md + passing verification test -> completion_gate allows!
            system.observe_tool_call(
                turn_id="turn-prog",
                session_id="actor",
                tool_name="write_file",
                args={"path": "proposal.md", "content": "# Proposal"},
                tool_call_id="w-ok",
                status="ok",
                result="written",
            )
            gate_no_test = system.completion_gate(
                turn_id="turn-prog", session_id="actor", changed_paths=["proposal.md"]
            )
            assert gate_no_test is not None and "verification/test" in gate_no_test["message"]

            system.observe_tool_call(
                turn_id="turn-prog",
                session_id="actor",
                tool_name="terminal",
                args={"command": "pytest -q"},
                tool_call_id="t-verify",
                status="ok",
                result='{"exit_code": 0, "output": "1 passed"}',
                exit_code=0,
            )
            gate_ok = system.completion_gate(
                turn_id="turn-prog", session_id="actor", changed_paths=["proposal.md"]
            )
            assert gate_ok is None
            r_done = system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)[-1]
            assert r_done["stage"] == "completion_gate"
            assert r_done["guard_disposition"] == "completion_allowed"
            assert r_done["output_verification"]["outputs_satisfied"] is True
            assert r_done["output_verification"]["verification_satisfied"] is True
    finally:
        system.drain(timeout=2.0)
        system.shutdown(flush_timeout=2.0, join_timeout=2.0, drop_pending=True)


def test_receipt_bookkeeping_failure_fails_closed_without_erasing_blocks(tmp_path, monkeypatch):
    from hermes_nerve import nervous

    dc, _reservations, system, planning, _events = _host_fixture_setup(tmp_path, monkeypatch)
    doc = str(planning / "rules/AGENTS.md")
    try:
        with dc.native_task_intent(
            actor_session_id="actor", parent_session_id="parent", goal="Write proposal.md.", context=None
        ) as intent:
            system.start_turn(
                user_message="Write proposal.md.",
                session_id="actor",
                parent_session_id="parent",
                turn_id="turn-err",
                task_intent=intent,
            )
            for i in range(12):
                system.observe_tool_call(
                    turn_id="turn-err",
                    session_id="actor",
                    tool_name="read_file",
                    args={"path": "notes.md"},
                    tool_call_id=f"obs-{i}",
                    status="ok",
                    result="ok",
                )
            assert system.status(turn_id="turn-err")["active_control"]["source"] == "local-progress-stall"

            def boom_receipt(**_kw):
                raise RuntimeError("injected receipt bookkeeping fault")

            monkeypatch.setattr(nervous, "build_accounting_receipt", boom_receipt)

            # 1. Prerequisite admission with receipt fault must fail CLOSED, never advisory-allow.
            admitted_fault = system.before_tool_call(
                turn_id="turn-err",
                session_id="actor",
                tool_name="read_file",
                args={"path": doc},
                tool_call_id="adm-fault",
                reserve_prerequisite_read=lambda **_k: "reserved",
            )
            assert admitted_fault is not None and admitted_fault["action"] == "block"
            assert "bookkeeping unavailable" in admitted_fault["message"]

            # 2. Ordinary progress-stall block & completion_gate block remain enforced despite receipt fault.
            stall_block = system.before_tool_call(
                turn_id="turn-err",
                session_id="actor",
                tool_name="read_file",
                args={"path": "unpinned.md"},
                tool_call_id="stall-fault",
                reserve_prerequisite_read=lambda **_k: "unavailable",
            )
            assert stall_block is not None and stall_block["action"] == "block"
            assert "task-relevant implementation" in stall_block["message"]

            comp_block = system.completion_gate(
                turn_id="turn-err", session_id="actor", changed_paths=["scratch.md"]
            )
            assert comp_block is not None and comp_block["action"] == "continue"
    finally:
        system.drain(timeout=2.0)
        system.shutdown(flush_timeout=2.0, join_timeout=2.0, drop_pending=True)

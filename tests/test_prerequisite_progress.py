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
    system = NervousSystem()
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

    system = NervousSystem(engine_factory=forbidden_provider)
    system.configure(enabled=True, admission_enabled=False, mode="correct_next", call_threshold=1.0,
                     local_learning=False, policy_memory_enabled=False, **{key: value})
    try:
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
    finally:
        system.configure(enabled=False)
    assert not provider_attempts
    data = events.read_text(encoding="utf-8")
    assert "must-not-echo" not in data
    assert any(row.get("kind") == "prerequisite_config_invalid" and row.get("setting") == key
               for row in map(json.loads, data.splitlines()))


@pytest.mark.parametrize("host", [(None, 1), (0, 1), (129, 900), (1, 901), (1, True)])
def test_invalid_host_ceiling_is_not_given_defaults(host):
    assert NervousSystem().prerequisite_limits(*host) is None


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
    system = NervousSystem()
    monkeypatch.setattr(plugin.nervous, "_default", system)
    plugin.register(Ctx(tmp_path, values))

    assert system.prerequisite_limits(128, 900) == expected
    assert system.report()["mode"] == ("shadow" if profile == "legacy" else "correct_next")
    if expected is None:
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

    system = NervousSystem(engine_factory=forbidden_provider)
    monkeypatch.setattr(nervous, "_default", system)
    system.configure(enabled=True, admission_enabled=False, call_threshold=1.0,
                     local_learning=False, policy_memory_enabled=False,
                     prerequisite_max_operations=1, prerequisite_max_seconds=0 if case == "invalid" else 10)
    goal = "Summarize the report." if case == "read-only" else "Write proposal.md."
    scope = dict(actor_session_id="actor", parent_session_id="parent", goal=goal, context=None)
    args = {"path": str(planning / "rules/AGENTS.md")}
    try:
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
    finally:
        system.configure(enabled=False)
    assert not provider_attempts

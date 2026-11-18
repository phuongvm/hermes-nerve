"""Task 3.1: policy bounds only; no reservations or guard changes."""
import json

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

"""PR #64 must coexist with catalog telemetry and sensitive-path approval."""
import pytest
from hermes_nerve import gate


@pytest.mark.parametrize("tool,args,expected", [
    ("skills_list", {}, "read-only-tool"),
    ("skill_view", {"name": "digital-assistant"}, "read-only-tool"),
    ("kanban_show", {"task_id": "t_x"}, "read-only-tool"),
    ("nerve_stats", {}, "catalog-read-only"),
])
def test_cataloged_introspection_retains_catalog_reason(monkeypatch, tool, args, expected):
    monkeypatch.setenv("HERMES_NERVE_GATE_SCOPE", "selective")
    assert gate.bypass_reason(tool, args) == expected


@pytest.mark.parametrize("target", ["/etc/shadow", "/etc/sudoers", "~/.ssh/id_rsa"])
def test_sensitive_skill_reference_does_not_fall_through_to_read_only(monkeypatch, target):
    monkeypatch.setenv("HERMES_NERVE_GATE_SCOPE", "selective")
    assert gate.bypass_reason("skill_view", {"name": "digital-assistant", "file_path": target}) is None

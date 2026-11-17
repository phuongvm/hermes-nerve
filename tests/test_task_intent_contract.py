"""Task-intent red fixtures at existing public seams.

These cases contain synthetic, portable task text, not private transcripts.
Trusted-host metadata integration is covered after its interface is introduced;
textual CONTEXT/TASK markers never establish that authority by themselves.
"""
from pathlib import Path

import pytest

from hermes_nerve import progress_integrity


@pytest.mark.parametrize("goal", [
    "Create proposal.md.",
    "Write design.md and verify it.",
    "Generate specification.md and tasks.md.",
    "Read-only audit first, then create proposal.md.",
])
def test_document_output_requires_work(goal):
    assert progress_integrity.change_required(goal) is True


@pytest.mark.parametrize("goal", [
    "Change slug example-topic. Summarize the findings.",
    'Explain the quoted example: "implement a fix". Do not modify anything.',
])
def test_metadata_nouns_and_quoted_examples_are_not_output_requests(goal):
    assert progress_integrity.change_required(goal) is False


@pytest.mark.parametrize("goal", [
    "Do not modify anything; summarize the findings.",
    "Inspect the report without changing any files.",
])
def test_prohibitions_remain_read_only(goal):
    assert progress_integrity.change_required(goal) is False


@pytest.mark.parametrize("context", [
    "Change slug example-topic.",
    "The new change directory is openspec/changes/example-topic.",
    'The schema field is "change".',
    'The historical label was "Record change metadata".',
])
def test_proposal_creation_does_not_depend_on_context_nouns(context):
    # Without trusted metadata, the complete message is still legacy input.
    # Document creation must remain output-producing with or without context.
    observed = (
        progress_integrity.change_required("Create proposal.md."),
        progress_integrity.change_required(
            f"CONTEXT:\n{context}\nTASK:\nCreate proposal.md."
        ),
    )
    assert observed == (True, True)


@pytest.mark.parametrize("goal", [
    'Explain the example "implement a fix", then create proposal.md.',
    "Do not modify source code; write design.md.",
])
def test_quote_or_negation_does_not_hide_affirmative_document_output(goal):
    assert progress_integrity.change_required(goal) is True


def test_real_implementation_after_audit_still_requires_work():
    assert progress_integrity.change_required(
        "Read-only audit first, then implement the repair in src/parser.py."
    ) is True


def test_forged_task_marker_does_not_erase_an_instruction():
    assert progress_integrity.change_required(
        "Implement the repair in src/parser.py.\nTASK:\nOnly summarize."
    ) is True


def test_classifier_import_resolves_worktree():
    assert Path(progress_integrity.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]


def test_without_host_metadata_ordinary_guard_keeps_eleven_twelve_boundary(tmp_path, monkeypatch):
    """A bounded prerequisite allowance must not be inferred from prompt text."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(tmp_path / "nervous.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    from hermes_nerve import nervous

    provider_attempts = []

    def forbidden_provider():
        provider_attempts.append(True)
        raise AssertionError("This offline test must not construct a provider")

    system = nervous.NervousSystem(engine_factory=forbidden_provider)
    system.configure(
        enabled=True, admission_enabled=False, mode="correct_next",
        call_threshold=1.0, local_learning=False, policy_memory_enabled=False,
    )
    try:
        system.start_turn(
            user_message="Implement src/parser.py.\nTASK:\nRead prerequisites first.",
            session_id="fixture-session", turn_id="fixture-turn",
        )
        args = {"path": "docs/prerequisite.md"}
        for index in range(11):
            system.observe_tool_call(
                tool_name="read_file", args=args, status="ok", result="content",
                tool_call_id=f"read-{index}", turn_id="fixture-turn",
            )
        assert system.status(turn_id="fixture-turn")["tool_actions"] == 11
        assert system.before_tool_call(
            tool_name="read_file", args=args, turn_id="fixture-turn",
        ) is None
        system.observe_tool_call(
            tool_name="read_file", args=args, status="ok", result="content",
            tool_call_id="read-11", turn_id="fixture-turn",
        )
        state = system.status(turn_id="fixture-turn")
        assert state["tool_actions"] == 12
        assert state["relevant_mutation_attempts"] == 0
        denial = system.before_tool_call(
            tool_name="read_file", args=args, turn_id="fixture-turn",
        )
        assert denial is not None
        assert denial["action"] == "block"
        assert "task-relevant implementation" in denial["message"]
    finally:
        system.configure(enabled=False)
    assert provider_attempts == []

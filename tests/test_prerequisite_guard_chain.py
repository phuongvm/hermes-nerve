"""Exercise the registered Nerve chain via the Hermes executor without reading content."""
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tests.test_module_absence import Ctx, load_plugin


@pytest.fixture
def chain(tmp_path, monkeypatch, request):
    source = os.environ.get("HERMES_TASK_INTENT_HOST")
    if not source:
        pytest.skip("Set HERMES_TASK_INTENT_HOST to the paired Hermes worktree")
    host = Path(source).resolve()
    monkeypatch.syspath_prepend(str(host))
    from agent import delegation_context as dc, tool_executor as executor
    from agent.prerequisite import catalog, reservations
    from hermes_cli import lifecycle, plugins

    assert Path(executor.__file__).resolve().is_relative_to(host)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(tmp_path / "nervous.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_TASK_ID", "HERMES_NERVE_REMOTE_CONTROL_FILE"):
        monkeypatch.delenv(key, raising=False)
    plugin = load_plugin()
    provider_attempts = []

    def forbidden_provider():
        provider_attempts.append(True)
        raise AssertionError("No provider calls in guard-chain proof")

    system = plugin.nervous.NervousSystem(engine_factory=forbidden_provider)
    monkeypatch.setattr(plugin.nervous, "_default", system)
    gate_events = []
    control = {"gate": None, "work": None}

    def gate(**kwargs):
        gate_events.append(kwargs["tool_call_id"])
        verdict = control["gate"]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    real_gate = getattr(request, "param", None) == "real-gate"
    if real_gate:
        monkeypatch.setenv("HERMES_NERVE_PREFERENCES", str(tmp_path / "preferences.json"))
        monkeypatch.setenv("HERMES_NERVE_GATE_EVENTS", str(tmp_path / "gate-events.jsonl"))
    else:
        monkeypatch.setattr(plugin.gate, "pre_tool_call", gate)
    monkeypatch.setattr(plugin.work_hooks, "pre_tool_call", lambda **kw: control["work"])
    ctx = Ctx(tmp_path, {"nervous_turn_admission": False, "nervous_local_learning": False,
                         "nervous_policy_memory_enabled": False, "nervous_call_threshold": 1.0,
                         "prerequisite_max_operations": 1})
    if getattr(request, "param", "current-host") != "old-host":
        # Use the real, optional public method; the fake ctx supplies only settings/registration.
        host_ctx = plugins.PluginContext(plugins.PluginManifest(name="nerve"), plugins.PluginManager(str(tmp_path)))
        capability = getattr(host_ctx, "reserve_prerequisite_read", None)
        if capability is not None:
            monkeypatch.setattr(ctx, "reserve_prerequisite_read", capability, raising=False)
    plugin.register(ctx)
    if real_gate:
        plugin.gate.configure(mode="enforce", scope="all", risk_preflight=True, learned_preferences=True)
    callbacks = [cb for event, cb in ctx.hooks if event == "pre_tool_call"]
    assert len(callbacks) == 1
    manager = plugins.PluginManager(str(tmp_path))
    manager._hooks["pre_tool_call"] = callbacks
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    monkeypatch.setattr(lifecycle, "_observe", lambda *_a, **_kw: None)
    manager._hooks["post_tool_call"] = [cb for event, cb in ctx.hooks if event == "post_tool_call"]
    assert plugin.nervous.post_tool_call in manager._hooks["post_tool_call"]
    manager._hooks["post_approval_response"] = [cb for event, cb in ctx.hooks if event == "post_approval_response"]
    fixture = runpy.run_path(str(host / "tests/tools/proposal_catalog_fixture.py"))
    planning = tmp_path / "planning"
    selection = catalog.freeze_proposal_catalog(fixture["CHANGE"], fixture["proposal_entry"](planning))
    monkeypatch.setattr(dc, "child_proposal_selection", lambda *_a, **_kw: selection)
    tool = "openspec_status"
    args = {"workdir": str(planning), "change": fixture["CHANGE"]}
    scope = dict(actor_session_id="actor", parent_session_id="parent", goal="Write proposal.md.", context=None)
    with dc.native_task_intent(**scope) as intent:
        system.start_turn(user_message="Write proposal.md.", session_id="actor", parent_session_id="parent",
                          turn_id="turn", task_intent=intent)
        for i in range(12):
            system.observe_tool_call(tool_name="read_file", args={"path": "notes.md"}, status="ok",
                                     result="fixture", turn_id="turn", tool_call_id=f"observation-{i}")
        agent = SimpleNamespace(session_id="actor", _parent_session_id="parent", _current_turn_id="turn",
                                _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)))
        try:
            yield SimpleNamespace(executor=executor, ctx=ctx, system=system, control=control, agent=agent,
                                  reservations=reservations, tool=tool, args=args, intent=intent,
                                  gate_events=gate_events, manager=manager, gate=plugin.gate)
        finally:
            system.configure(enabled=False)
    assert not provider_attempts


def dispatch(chain, call_id="candidate"):
    executor = chain.executor
    state = executor._ManagedToolResult(None, chain.args, [], False, False)
    ref = executor._ToolCallRef(chain.tool, chain.args, "fixture-task", call_id, [])

    def unexpected_execution(_args):
        pytest.fail("A prerequisite must not fall back to the ordinary handler")

    result = executor._dispatch_authorized_once(chain.agent, state, ref, execute=unexpected_execution,
                                               scope_block=None, display_index=None, begin_execution=None,
                                               authorization_gate=None)
    assert state.blocked
    return json.loads(result)["error"]


@pytest.mark.parametrize("case", ["no-veto", "deny", "approval-denied", "approval-accepted", "critical-error"])
def test_registered_chain_preserves_downstream_authority(chain, monkeypatch, case):
    from tools import approval

    approvals = []

    def approve(tool, message, *, rule_key):
        approvals.append((tool, message, rule_key))
        return {"approved": case == "approval-accepted", "message": "native approval refused"}

    monkeypatch.setattr(approval, "request_tool_approval", approve)
    chain.control["gate"] = {
        "deny": {"action": "block", "message": "independent veto"},
        "approval-denied": {"action": "approve", "message": "fixture approval", "rule_key": "fixture:rule"},
        "approval-accepted": {"action": "approve", "message": "fixture approval", "rule_key": "fixture:rule"},
        "critical-error": RuntimeError("fixture critical error"),
    }.get(case)
    expected = {
        "no-veto": "verified reader unavailable", "deny": "independent veto",
        "approval-denied": "native approval refused", "approval-accepted": "verified reader unavailable",
        "critical-error": "native approval refused",
    }
    assert expected[case] in dispatch(chain).lower()
    assert chain.gate_events == ["candidate"]
    assert len(approvals) == (0 if case in {"no-veto", "deny"} else 1)
    if case == "critical-error":
        assert "action gate unavailable" in approvals[0][1]
    assert chain.reservations.reserve_read(chain.intent, "actor", "candidate", chain.tool, chain.args) == "replay"
    assert chain.reservations.begin_read(chain.intent, "actor", "candidate", chain.tool, chain.args) != "started"
    observed = chain.system.status(turn_id="turn")
    assert observed["tool_actions"] == 13
    if case in {"no-veto", "approval-accepted"}:
        assert observed["active_control"]["source"] == "local-progress-stall"
        assert "progress guard" in dispatch(chain, "next").lower()
        assert chain.gate_events == ["candidate"]
    else:
        # A real independent refusal retains its existing post-tool semantics.
        assert observed["active_control"]["source"] == "local-deterministic-failure"


def test_new_bookkeeping_failure_blocks_without_exposing_exception(chain, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("fixture confidential details")

    monkeypatch.setattr(chain.reservations, "reserve_read", unavailable)
    message = dispatch(chain)
    assert "prerequisite bookkeeping unavailable" in message.lower()
    assert "fixture confidential details" not in message
    assert chain.gate_events == []


@pytest.mark.parametrize("case", ["remote", "work", "deterministic"])
def test_other_controls_are_not_waived(chain, tmp_path, monkeypatch, case):
    if case == "remote":
        path = tmp_path / "control.json"
        path.write_text('{"control":"STOP_REQUESTED"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_NERVE_REMOTE_CONTROL_FILE", str(path))
    elif case == "work":
        chain.control["work"] = {"action": "block", "message": "work supervision veto"}
    else:
        chain.system.observe_tool_call(tool_name=chain.tool, args=chain.args, status="blocked",
                                       result="permission denied", turn_id="turn", tool_call_id="prior")
        assert chain.system.status(turn_id="turn")["active_control"]["source"] == "local-deterministic-failure"
    assert {"remote": "STOP_REQUESTED", "work": "work supervision veto", "deterministic": "prevents repeating"}[case] in dispatch(chain)
    assert chain.gate_events == []
    assert chain.reservations.reserve_read(chain.intent, "actor", "probe", chain.tool, chain.args) == "reserved"


def test_replay_does_not_discard_the_original_pending_ticket(chain):
    request = (chain.intent, "actor", "candidate", chain.tool, chain.args)
    assert chain.reservations.reserve_read(*request) == "reserved"
    assert "progress guard" in dispatch(chain).lower()
    assert chain.gate_events == []
    assert chain.reservations.begin_read(*request) == "started"
    assert chain.reservations.begin_read(*request) == "replay"


def test_timed_out_callback_discards_its_late_ticket_without_refund(chain, monkeypatch):
    from threading import Event
    from hermes_cli import plugins

    reserve = chain.reservations.reserve_read
    discard = chain.reservations.discard_read
    reserved, release, settled = Event(), Event(), Event()

    def pause_after_reservation(*args, **kwargs):
        result = reserve(*args, **kwargs)
        if result == "reserved":
            reserved.set()
            if not release.wait(10):
                raise AssertionError("Timeout fixture did not release its worker")
        return result

    def signal_discard(*args):
        discard(*args)
        settled.set()

    monkeypatch.setattr(chain.reservations, "reserve_read", pause_after_reservation)
    monkeypatch.setattr(chain.reservations, "discard_read", signal_discard)
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 1.0)
    try:
        assert "callback timed out" in dispatch(chain).lower()
        assert reserved.is_set()
        assert not settled.is_set()
    finally:
        release.set()
        if reserved.is_set():
            assert settled.wait(10)
    request = (chain.intent, "actor", "candidate", chain.tool, chain.args)
    assert chain.reservations.begin_read(*request) == "replay"
    assert reserve(chain.intent, "actor", "fresh", chain.tool, chain.args) == "exhausted"


@pytest.mark.parametrize("chain", ["old-host"], indirect=True)
def test_missing_public_host_capability_keeps_the_progress_veto(chain):
    assert "progress guard" in dispatch(chain).lower()
    assert chain.gate_events == []
    assert chain.reservations.reserve_read(chain.intent, "actor", "probe", chain.tool, chain.args) == "reserved"


def test_public_capability_matches_only_the_current_host_call(chain):
    reserve = chain.ctx.reserve_prerequisite_read
    identity = dict(session_id="actor", assignment_id=chain.intent.assignment_id,
                    tool_call_id="candidate", tool_name=chain.tool)
    assert reserve(**identity) == "unavailable"
    decisions = []

    def probe(**_kwargs):
        for field in identity:
            decisions.append(reserve(**{**identity, field: "unrelated"}))

    chain.manager._hooks["pre_tool_call"].insert(0, probe)
    assert "verified reader unavailable" in dispatch(chain).lower()
    assert decisions == ["unavailable"] * len(identity)
    assert chain.gate_events == ["candidate"]
    assert reserve(**identity) == "unavailable"


def test_cleanup_error_invalidates_the_ticket_and_preserves_veto(chain, monkeypatch, caplog):
    def broken_cleanup(*args, **kwargs):
        raise RuntimeError("fixture private cleanup information")

    chain.control["gate"] = {"action": "block", "message": "independent veto"}
    monkeypatch.setattr(chain.reservations, "discard_read", broken_cleanup)
    assert dispatch(chain) == "independent veto"
    assert "prerequisite_ticket_cleanup_failed" in caplog.text
    assert "fixture private cleanup information" not in caplog.text
    ledger = chain.reservations
    assert ledger.begin_read(chain.intent, "actor", "candidate", chain.tool, chain.args) == "replay"
    assert ledger.reserve_read(chain.intent, "actor", "fresh", chain.tool, chain.args) == "exhausted"


@pytest.mark.parametrize("chain", ["real-gate"], indirect=True)
def test_accepted_approval_does_not_learn_from_host_refusal(chain, monkeypatch):
    from tools import approval

    gate = chain.gate
    assessment = gate.risk.assess_tool_call(chain.tool, chain.args)
    assert assessment.category == gate.risk.UNCERTAIN
    approvals = []
    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs["tool_name"])
        return SimpleNamespace(value="APPROVAL", confidence=0.9, latency_ms=1.0,
                               probabilities={"ALLOW": 0.05, "APPROVAL": 0.9, "BLOCK": 0.05})

    def approve(tool, message, *, rule_key):
        approvals.append((tool, rule_key))
        return {"approved": True}

    monkeypatch.setattr(gate, "evaluate_tool_call", evaluate)
    monkeypatch.setattr(approval, "request_tool_approval", approve)
    shape = (assessment.category, chain.tool, assessment.shape)
    assert gate.prefs.approval_count(*shape) == 0
    assert "verified reader unavailable" in dispatch(chain).lower()
    assert evaluations == [chain.tool]
    assert approvals == [(chain.tool, gate.prefs.scoped_rule_key(*shape))]
    assert gate.prefs.consume_pending(("actor", "turn", "candidate")) is None
    assert chain.system.status(turn_id="turn")["active_control"]["source"] == "local-progress-stall"
    assert gate.prefs.approval_count(*shape) == 0


@pytest.mark.parametrize("case", ["started", "yielded", "exception", "other-rule", "completed"])
@pytest.mark.parametrize("chain", ["real-gate"], indirect=True)
def test_approval_learning_requires_completed_execution(chain, monkeypatch, tmp_path, case):
    from model_tools import _approval_observability, _CallIds
    from tools import approval, approval_context, approval_prompt, terminal_tool_result

    gate = chain.gate
    executor = chain.executor
    args = {"command": "./scripts/fixture.sh", "workdir": str(tmp_path)}
    ref = executor._ToolCallRef("terminal", args, "fixture-task", "terminal-fixture", [])
    assessment = gate.risk.assess_tool_call(ref.name, args)
    assert assessment.category == gate.risk.UNCERTAIN
    approvals = []
    monkeypatch.setattr(gate, "evaluate_tool_call", lambda **_: SimpleNamespace(
        value="APPROVAL", confidence=0.9, latency_ms=1.0, probabilities={"APPROVAL": 0.9},
    ))

    def answer(_command, reason, **_kwargs):
        approvals.append(reason)
        return "once"

    monkeypatch.setattr(approval, "_presence", lambda *_: (None, True, False, False))
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "prompt_dangerous_approval", answer)
    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", answer)
    monkeypatch.setattr(approval, "is_approved", lambda *_: False)
    if case == "other-rule":
        chain.manager._hooks["pre_tool_call"].insert(0, lambda **_: {
            "action": "approve", "message": "fixture separate rule", "rule_key": "other:rule",
        })
    # Exercise the normal gate without the fixture's unrelated progress stall.
    chain.system.configure(enabled=False)
    chain.agent.tool_progress_callback = None
    monkeypatch.setattr(executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(executor, "_run_with_activity_heartbeat", lambda _agent, _tool, run: run())
    executions = []

    def handler(final_args):
        executions.append(final_args)
        if case == "exception":
            raise RuntimeError("fixture exception")
        if case == "started":
            return '{"output":"started","exit_code":0,"error":null,"session_id":"fixture"}'
        if case == "yielded":
            return '{"output":"pending","exit_code":null,"error":null,"status":"yielded_to_background"}'
        with _approval_observability(_CallIds(ref.task_id, "actor", ref.call_id, "turn")):
            return terminal_tool_result.finalize_foreground_result(
                command=args["command"], result={"returncode": 0, "output": "fixture"},
                env=SimpleNamespace(), env_type="local", effective_task_id=ref.task_id,
                task_id=ref.task_id, session_id="actor", session_key="actor", workdir=args["workdir"],
                command_cwd=args["workdir"], approval_note=None,
            )

    def run(_agent, **kwargs):
        state = executor._ManagedToolResult(None, args, [], False, True)
        state.result = executor._dispatch_authorized_once(
            chain.agent, state, ref, execute=kwargs["execute"], scope_block=None,
            display_index=None, begin_execution=None, authorization_gate=None,
        )
        return state

    monkeypatch.setattr(executor, "_run_sequential_tool_execution_middleware", run)
    monkeypatch.setattr(executor, "_commit_tool_result", lambda *_a, **_kw: None)
    dispatch_plan = executor._SequentialDispatch(
        execute=handler, error_result=lambda exc: f"Error executing tool 'terminal': {exc}",
        error_log="Fixture tool %s raised: %s", finish_spinner=False,
    )
    messages = []
    state, duration = executor._run_sequential_call(
        chain.agent, dispatch_plan, ref, scope_block=None, messages=messages, remaining_calls=[],
        display_index=1, tool_start_time=executor.time.time(),
    )
    assert not state.blocked
    executor._publish_sequential_result(
        chain.agent, messages, ref, state, tool_duration=duration, index=0, budget=None, transform_applied=True,
    )
    shape = (assessment.category, ref.name, assessment.shape)
    assert executions == [args]
    assert len(approvals) == 1
    assert ("fixture separate rule" in approvals[0]) == (case == "other-rule")
    assert gate.prefs.consume_pending(("actor", "turn", ref.call_id)) is None
    assert gate.prefs.approval_count(*shape) == (1 if case == "completed" else 0)


@pytest.mark.parametrize("chain", ["real-gate"], indirect=True)
def test_classifier_scope_error_neither_records_nor_reuses_preference(chain, monkeypatch, tmp_path):
    from model_tools import _approval_observability, _CallIds
    from tools import approval, approval_context, approval_prompt, terminal_tool_result

    gate = chain.gate
    executor = chain.executor
    gate.configure(mode="enforce", scope="all", risk_preflight=True,
                   learned_preferences=True, preference_min_samples=1)
    monkeypatch.setattr(gate, "evaluate_tool_call", lambda **_: SimpleNamespace(
        value="APPROVAL", confidence=0.9, latency_ms=1.0, probabilities={"APPROVAL": 0.9},
    ))

    prompts = []

    def answer(_command, reason, **_kwargs):
        prompts.append(reason)
        return "session"

    monkeypatch.setattr(approval, "_presence", lambda *_: (None, True, False, False))
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "prompt_dangerous_approval", answer)
    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", answer)
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})

    chain.system.configure(enabled=False)
    chain.agent.tool_progress_callback = None
    monkeypatch.setattr(executor, "_begin_tool_execution", lambda *_: None)
    monkeypatch.setattr(executor, "_run_with_activity_heartbeat", lambda _agent, _tool, run: run())

    # Pre-seed an empty-scope v2 entry to verify lookup-side rejection in the live chain.
    pref_path = gate.prefs.preferences_path()
    empty_fp = gate.prefs.shape_fingerprint({})
    pref_path.write_text(json.dumps({
        "schema": "hermes-nerve-preferences/v2",
        "shapes": {empty_fp: {"category": "uncertain", "tool_name": "terminal", "count": 5}},
    }), encoding="utf-8")

    def dispatch_terminal(command: str, call_id: str) -> None:
        call_args = {"command": command, "workdir": str(tmp_path)}
        ref = executor._ToolCallRef("terminal", call_args, "fixture-task", call_id, [])

        def handler(_final_args):
            with _approval_observability(_CallIds(ref.task_id, "actor", ref.call_id, "turn")):
                return terminal_tool_result.finalize_foreground_result(
                    command=call_args["command"], result={"returncode": 0, "output": "fixture"},
                    env=SimpleNamespace(), env_type="local", effective_task_id=ref.task_id,
                    task_id=ref.task_id, session_id="actor", session_key="actor", workdir=call_args["workdir"],
                    command_cwd=call_args["workdir"], approval_note=None,
                )

        def run(_agent, **kwargs):
            state = executor._ManagedToolResult(None, call_args, [], False, True)
            state.result = executor._dispatch_authorized_once(
                chain.agent, state, ref, execute=kwargs["execute"], scope_block=None,
                display_index=None, begin_execution=None, authorization_gate=None,
            )
            return state

        monkeypatch.setattr(executor, "_run_sequential_tool_execution_middleware", run)
        monkeypatch.setattr(executor, "_commit_tool_result", lambda *_a, **_kw: None)
        plan = executor._SequentialDispatch(
            execute=handler, error_result=lambda exc: f"Error executing tool 'terminal': {exc}",
            error_log="Fixture tool %s raised: %s", finish_spinner=False,
        )
        messages = []
        state, duration = executor._run_sequential_call(
            chain.agent, plan, ref, scope_block=None, messages=messages, remaining_calls=[],
            display_index=1, tool_start_time=executor.time.time(),
        )
        assert not state.blocked
        executor._publish_sequential_result(
            chain.agent, messages, ref, state, tool_duration=duration, index=0, budget=None, transform_applied=True,
        )

    real_assess = gate.risk.assess_tool_call

    def failing_assess(*_a, **_kw):
        raise RuntimeError("classifier failure")

    monkeypatch.setattr(gate.risk, "assess_tool_call", failing_assess)
    dispatch_terminal("./scripts/unclassified-one.sh", "err-1")
    assert len(prompts) == 1
    assert gate.prefs.approval_count("uncertain", "terminal", {}) == 0
    assert json.loads(pref_path.read_text(encoding="utf-8"))["shapes"][empty_fp]["count"] == 5

    # Second call with classifier error must NOT be auto-allowed by Nerve preferences or session rule cache.
    dispatch_terminal("./scripts/unclassified-two.sh", "err-2")
    assert len(prompts) == 2
    assert gate.prefs.approval_count("uncertain", "terminal", {}) == 0

    # Positive control: valid nonempty classifier scope learns after completion and auto-allows next call.
    monkeypatch.setattr(gate.risk, "assess_tool_call", real_assess)
    monkeypatch.setattr(approval, "_session_approved", {})
    valid_args = {"command": "./scripts/valid-chain-control.sh", "workdir": str(tmp_path)}
    valid_shape = real_assess("terminal", valid_args).shape
    dispatch_terminal(valid_args["command"], "valid-1")
    assert len(prompts) == 3
    assert gate.prefs.approval_count("uncertain", "terminal", valid_shape) == 1
    dispatch_terminal(valid_args["command"], "valid-2")
    assert len(prompts) == 3
    assert gate.prefs.learned_allow("uncertain", "terminal", valid_shape, 1) is True


def test_registered_chain_dispatches_verified_read_file_beyond_stall_and_records_nonsecret_receipts(
    tmp_path, monkeypatch
):
    source = os.environ.get("HERMES_TASK_INTENT_HOST")
    if not source:
        pytest.skip("Set HERMES_TASK_INTENT_HOST to the paired Hermes worktree")
    host = Path(source).resolve()
    monkeypatch.syspath_prepend(str(host))
    from agent import delegation_context as dc, tool_executor as executor
    from agent.prerequisite import catalog, reader
    from hermes_cli import plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    events_path = tmp_path / "nervous.jsonl"
    monkeypatch.setenv("HERMES_NERVE_NERVOUS_EVENTS", str(events_path))
    monkeypatch.setenv("HERMES_NERVE_OUTCOMES", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setenv("HERMES_NERVE_POLICY_MEMORY", str(tmp_path / "policy.jsonl"))
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_TASK_ID", "HERMES_NERVE_REMOTE_CONTROL_FILE"):
        monkeypatch.delenv(key, raising=False)

    plugin = load_plugin()
    system = plugin.nervous.NervousSystem(
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("No provider calls"))
    )
    monkeypatch.setattr(plugin.nervous, "_default", system)
    monkeypatch.setattr(plugin.gate, "pre_tool_call", lambda **_kw: None)
    monkeypatch.setattr(plugin.work_hooks, "pre_tool_call", lambda **_kw: None)

    ctx = Ctx(
        tmp_path,
        {
            "nervous_turn_admission": False,
            "nervous_local_learning": False,
            "nervous_policy_memory_enabled": False,
            "nervous_call_threshold": 1.0,
            "prerequisite_max_operations": 4,
        },
    )
    host_ctx = plugins.PluginContext(plugins.PluginManifest(name="nerve"), plugins.PluginManager(str(tmp_path)))
    monkeypatch.setattr(ctx, "reserve_prerequisite_read", host_ctx.reserve_prerequisite_read, raising=False)
    plugin.register(ctx)

    manager = plugins.PluginManager(str(tmp_path))
    manager._hooks["pre_tool_call"] = [cb for event, cb in ctx.hooks if event == "pre_tool_call"]
    manager._hooks["post_tool_call"] = [cb for event, cb in ctx.hooks if event == "post_tool_call"]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)

    fixture = runpy.run_path(str(host / "tests/tools/proposal_catalog_fixture.py"))
    planning = tmp_path / "planning"
    entry = fixture["proposal_entry"](planning)
    secret_tok = "sk-chainReceiptSecret1234567890abcdef"
    body_sentinel = "CHAIN_PREREQ_BODY_SENTINEL_5519"
    agents_doc = planning / "rules" / "AGENTS.md"
    raw_bytes = f"# Rules\n{body_sentinel}\nTOKEN={secret_tok}\n".encode("utf-8")
    agents_doc.write_bytes(raw_bytes)
    st = agents_doc.lstat()
    from hashlib import sha256

    entry["resources"]["governance/AGENTS.md"] = {
        "path": str(agents_doc),
        "device": st.st_dev,
        "inode": st.st_ino,
        "size": st.st_size,
        "max_file_bytes": 1024,
        "sha256": sha256(raw_bytes).hexdigest(),
    }
    selection = catalog.freeze_proposal_catalog(fixture["CHANGE"], entry)
    monkeypatch.setattr(dc, "child_proposal_selection", lambda *_a, **_kw: selection)

    scope = dict(actor_session_id="actor", parent_session_id="parent", goal="Write proposal.md.", context=None)
    with dc.native_task_intent(**scope) as intent:
        system.start_turn(
            user_message="Write proposal.md.",
            session_id="actor",
            parent_session_id="parent",
            turn_id="turn-13",
            task_intent=intent,
        )
        for i in range(12):
            system.observe_tool_call(
                tool_name="read_file",
                args={"path": "notes.md"},
                status="ok",
                result="fixture",
                turn_id="turn-13",
                session_id="actor",
                tool_call_id=f"obs-{i}",
            )
        assert system.status(turn_id="turn-13")["active_control"]["source"] == "local-progress-stall"

        agent = SimpleNamespace(
            session_id="actor",
            _parent_session_id="parent",
            _current_turn_id="turn-13",
            tool_progress_callback=None,
            _tool_guardrails=SimpleNamespace(before_call=lambda *_: SimpleNamespace(allows_execution=True)),
        )
        monkeypatch.setattr(executor, "_begin_tool_execution", lambda *_: None)
        monkeypatch.setattr(executor, "_commit_tool_result", lambda *_a, **_kw: None)

        read_args = {"path": str(agents_doc), "offset": 1, "limit": 20}
        ref = executor._ToolCallRef("read_file", dict(read_args), "task-chain", "prereq-13", [])

        def run_mw(_agent, **kwargs):
            st_res = executor._ManagedToolResult(None, dict(read_args), [], False, True)
            st_res.result = executor._dispatch_authorized_once(
                agent,
                st_res,
                ref,
                execute=kwargs["execute"],
                scope_block=None,
                display_index=None,
                begin_execution=None,
                authorization_gate=None,
            )
            return st_res

        monkeypatch.setattr(executor, "_run_sequential_tool_execution_middleware", run_mw)
        plan = executor._SequentialDispatch(
            execute=lambda _a: pytest.fail("Must not fall back to legacy read_file handler"),
            error_result=lambda exc: f"Error: {exc}",
            error_log="err %s: %s",
            finish_spinner=False,
        )
        messages = []
        with reader.measure_physical_reads() as metrics:
            state, duration = executor._run_sequential_call(
                agent, plan, ref, scope_block=None, messages=messages, remaining_calls=[],
                display_index=1, tool_start_time=executor.time.time(),
            )
            assert not state.blocked
            executor._publish_sequential_result(
                agent, messages, ref, state, tool_duration=duration, index=0, budget=None, transform_applied=True,
            )

        assert metrics.bytes_read == len(raw_bytes)
        payload = json.loads(state.result)
        assert body_sentinel in payload["content"]
        assert secret_tok not in payload["content"]

        # Lifetime tool_actions increased monotonically to 13 rather than resetting.
        assert system.status(turn_id="turn-13")["tool_actions"] == 13
        call_receipts = [
            r for r in system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id)
            if r["tool_call_id"] == "prereq-13"
        ]
        assert [r["stage"] for r in call_receipts] == ["pre_tool", "post_tool"]
        assert call_receipts[0]["guard_disposition"] == "prerequisite_admitted"
        assert call_receipts[0]["active_control_source"] == "local-progress-stall"
        assert call_receipts[1]["operation_category"] == "prerequisite_read"
        assert call_receipts[1]["counters"]["tool_actions"] == 13
        assert call_receipts[1]["prerequisite"]["operations_reserved"] == 1
        assert call_receipts[1]["prerequisite"]["source_bytes_reserved"] == 1024

        raw_receipts_json = json.dumps(system.accounting_receipts(session_id="actor", assignment_id=intent.assignment_id))
        disk_text = events_path.read_text(encoding="utf-8")
        assert secret_tok not in raw_receipts_json and secret_tok not in disk_text
        assert body_sentinel not in raw_receipts_json and body_sentinel not in disk_text
    system.configure(enabled=False)

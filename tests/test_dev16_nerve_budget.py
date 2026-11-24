from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from hermes_nerve.work import hooks, runtime
from hermes_nerve.work.models import RunIdentity
from hermes_nerve.work.deterministic import _evaluate_criterion
from hermes_nerve.work.nerve import budget_criterion, estimate_task_budget, evaluate
from hermes_nerve.work.supervisor import CardSupervisor


class Accept:
    value = "ACCEPT"
    confidence = 1.0
    receipt_id = "local"


def _bound(tmp: str, *, target: int = 1_000_000, reviewer: str = "reviewer"):
    runtime.configure(
        enabled=True,
        mode="advisory",
        store_path=str(Path(tmp) / "work.db"),
        reviewer=reviewer,
        provider_decisions_enabled=False,
        nerve_observer_enabled=True,
        nerve_auto_kill=False,
        nerve_watch_fraction=0.65,
        nerve_forecast_fraction=0.80,
        nerve_extension_fraction=0.25,
        nerve_extension_min_confidence=0.60,
        nerve_no_handoff_fraction=0.90,
        nerve_replan_fraction=0.90,
        nerve_hard_budget_multiplier=1.75,
        nerve_min_calls_before_kill=12,
    )
    sup = runtime.supervisor()
    # One substantive criterion carries the target. DOD-BUDGET is zero-weight
    # economically so the allocation remains exactly target.
    bound = sup.bind_contract(
        task_id="t",
        goal="ship",
        criteria=[
            {"id": "DOD-01", "description": "manual criterion", "estimated_tokens": target},
            budget_criterion(target),
        ],
        reserve_tokens=0,
        preflight_result=Accept(),
    )
    ident = RunIdentity("t", 1, bound["contract"]["contract_hash"], "claim", "worker")
    sup.bind_run(ident)
    workspace = Path(tmp) / "repo"
    workspace.mkdir()
    sup.store.save_run_context(ident, workspace_path=str(workspace), base_revision="", session_id="s", bound_at="now")
    return sup, ident


def _env(ident: RunIdentity):
    return {
        "HERMES_KANBAN_TASK": ident.task_id,
        "HERMES_KANBAN_TASK_ID": ident.task_id,
        "HERMES_KANBAN_RUN_ID": str(ident.run_id),
        "HERMES_KANBAN_CLAIM_LOCK": ident.claim_identity,
        "HERMES_KANBAN_WORKER_ID": ident.worker_id or "worker",
        "HERMES_NERVE_DOD_HASH": ident.contract_hash,
    }


def test_estimator_is_conservative_for_eight_criterion_task():
    body = "x" * 4000
    estimate = estimate_task_budget(body, 8, floor_tokens=70000)
    assert 900_000 <= estimate <= 1_300_000
    assert estimate % 10_000 == 0


def test_budget_criterion_is_locked_without_double_counting():
    c = budget_criterion(1_050_000)
    assert c["id"] == "DOD-BUDGET"
    assert c["required"] is False
    assert c["estimated_tokens"] == 0
    assert "1050000" in c["description"]
    assert "1155000" in c["description"]
    assert "execution policy" in c["description"]
    assert "must not make otherwise correct work impossible" in c["description"]


def test_nerve_does_not_kill_legitimate_near_budget_run():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        # 9 calls / 900k is expensive but still within target. Observer may REPLAN,
        # but must not trip the kill switch merely for being hard.
        for i in range(9):
            sup.record_api_usage(ident, api_request_id=f"a{i}", usage={"input_tokens": 90_000, "output_tokens": 10_000})
        d = evaluate(sup, ident, cfg=runtime.settings())
        assert d.level in {"WATCH", "FORECAST", "REPLAN"}
        assert d.should_kill is False


def test_nerve_hard_ceiling_returns_authority_to_orchestrator_not_kill():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        for i in range(11):
            sup.record_api_usage(ident, api_request_id=f"a{i}", usage={"input_tokens": 170_000, "output_tokens": 0})
        d = evaluate(sup, ident, cfg=runtime.settings())
        assert d.level == "ORCH_REVIEW"
        assert d.should_kill is False
        assert d.requires_orchestrator_review is True
        assert d.confidence >= 0.99




def test_frozen_delivery_benchmark_estimates_to_960k():
    body = (Path(__file__).resolve().parents[1] / "benchmarks/dev6_event_delivery/task-body.md").read_text()
    from hermes_nerve.work.autobind import extract_dod_lines
    assert estimate_task_budget(body, len(extract_dod_lines(body)), floor_tokens=70000) == 960_000


def test_budget_telemetry_records_overrun_but_is_not_required_dod():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        criterion = sup.active_contract("t").criteria[-1]
        # 1.05x target remains inside the default 1.10x completion tolerance.
        sup.record_api_usage(ident, api_request_id="near", usage={"input_tokens": 1_050_000, "output_tokens": 0})
        passed, reason = _evaluate_criterion(sup, ident, criterion, workspace=str(Path(td) / "repo"), base="")
        assert passed is True
        assert "ceiling=1100000" in reason
        # A genuine overrun remains visible telemetry, but DOD-BUDGET is optional and cannot block completion.
        sup.record_api_usage(ident, api_request_id="over", usage={"input_tokens": 100_000, "output_tokens": 0})
        passed, reason = _evaluate_criterion(sup, ident, criterion, workspace=str(Path(td) / "repo"), base="")
        assert passed is False
        assert "exceeded" in reason
        assert criterion.required is False


class ForecastEngine:
    def __init__(self, value="YES", confidence=0.95):
        self.value=value; self.confidence=confidence
    def assess(self, *, state, questions, contract):
        return {
            "id": "forecast-test",
            "answers": {
                "finish_with_extension": {
                    "type": "choice",
                    "choice": self.value,
                    "probabilities": {"YES": self.confidence if self.value=="YES" else 0.02, "NO": self.confidence if self.value=="NO" else 0.02, "MAYBE": self.confidence if self.value=="MAYBE" else 0.03},
                    "confidence": self.confidence,
                }
            },
            "usage": {"input_tokens": 50, "output_tokens": 0},
        }


def test_post_api_request_yes_grants_one_bounded_extension_then_reviews_on_exhaustion():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        sup.engine_factory = lambda: ForecastEngine("YES", 0.95)
        calls = []
        runtime.set_tool_dispatcher(lambda name, args: calls.append((name, args)) or '{"ok": true, "status": "review"}')
        try:
            with patch.dict(os.environ, _env(ident), clear=False):
                # Crossing 80% asks the forecast. YES grants +25%; it must not hand off yet.
                hooks.post_api_request(
                    task_id="t", session_id="s", api_request_id="api-forecast",
                    usage={"input_tokens": 810_000, "output_tokens": 0},
                )
                assert not calls
                ext = sup.store.latest_diagnostic(task_id="t", run_id=1, kind="nerve_budget_extension")
                assert (ext["payload"] or {})["extension_tokens"] == 250_000
                assert (ext["payload"] or {})["effective_token_target"] == 1_250_000
                # Exhaust the one automatic extension. Nerve must request review, never block/kill.
                hooks.post_api_request(
                    task_id="t", session_id="s", api_request_id="api-exhaust",
                    usage={"input_tokens": 450_000, "output_tokens": 0},
                )
            assert calls
            assert calls[-1][0] == "kanban_request_review"
            assert all(name != "kanban_block" for name, _ in calls)
            control = sup.control_for_run(ident)
            assert control["control"] == "REPLAN"
            assert (control["payload"] or {}).get("orchestrator_review_required") is True
            assert (control["payload"] or {}).get("kill_switch") is False
        finally:
            runtime.set_tool_dispatcher(None)


def test_maybe_forecast_goes_straight_to_orchestrator_review():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        sup.engine_factory = lambda: ForecastEngine("MAYBE", 0.72)
        calls=[]
        runtime.set_tool_dispatcher(lambda name,args: calls.append((name,args)) or '{"ok": true, "status": "review"}')
        try:
            with patch.dict(os.environ, _env(ident), clear=False):
                hooks.post_api_request(task_id="t",session_id="s",api_request_id="m",usage={"input_tokens":810_000,"output_tokens":0})
            assert calls and calls[-1][0] == "kanban_request_review"
            assert all(name != "kanban_block" for name,_ in calls)
        finally:
            runtime.set_tool_dispatcher(None)


def test_no_forecast_gets_checkpoint_window_then_review_by_90_percent():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        sup.engine_factory = lambda: ForecastEngine("NO", 0.91)
        calls=[]
        runtime.set_tool_dispatcher(lambda name,args: calls.append((name,args)) or '{"ok": true, "status": "review"}')
        try:
            with patch.dict(os.environ, _env(ident), clear=False):
                hooks.post_api_request(task_id="t",session_id="s",api_request_id="n1",usage={"input_tokens":810_000,"output_tokens":0})
                assert not calls
                hooks.post_api_request(task_id="t",session_id="s",api_request_id="n2",usage={"input_tokens":100_000,"output_tokens":0})
            assert calls and calls[-1][0] == "kanban_request_review"
        finally:
            runtime.set_tool_dispatcher(None)


def test_post_verified_provider_call_prefers_controller_completion_over_kill():
    with tempfile.TemporaryDirectory() as td:
        sup, ident = _bound(td)
        sup.store.set_control(
            ident,
            control="COMPLETE_READY",
            decision_id="done",
            payload={"lifecycle_state": "VERIFIED", "reason": "verified", "confidence": 1.0},
            created_at="now",
        )
        calls = []
        runtime.set_tool_dispatcher(lambda name, args: calls.append((name, args)) or '{"ok": true, "status": "done"}')
        try:
            with patch.dict(os.environ, _env(ident), clear=False):
                hooks.post_api_request(
                    task_id="t", session_id="s", api_request_id="late",
                    usage={"input_tokens": 100, "output_tokens": 10},
                )
            assert calls and calls[-1][0] == "kanban_complete"
            control = sup.control_for_run(ident)
            assert control["control"] == "COMPLETE_READY"
            assert (control["payload"] or {}).get("lifecycle_state") == "COMPLETED"
        finally:
            runtime.set_tool_dispatcher(None)


def test_review_handoff_without_distinct_reviewer_falls_back_to_kanban_block():
    for configured_reviewer in ("", "coder", "@Coder"):
        with tempfile.TemporaryDirectory() as td:
            sup, ident = _bound(td, reviewer=configured_reviewer)
            sup.engine_factory = lambda: ForecastEngine("MAYBE", 0.72)
            calls = []
            runtime.set_tool_dispatcher(lambda name, args: calls.append((name, args)) or '{"ok": true, "status": "blocked"}')
            try:
                env = {**_env(ident), "HERMES_PROFILE": "coder"}
                with patch.dict(os.environ, env, clear=False):
                    hooks.post_api_request(
                        task_id="t", session_id="s", api_request_id="m",
                        usage={"input_tokens": 810_000, "output_tokens": 0},
                    )
                assert [name for name, _ in calls] == ["kanban_block"]
                _, block_args = calls[0]
                assert block_args["task_id"] == "t"
                assert block_args["kind"] == "needs_input"
                assert set(block_args.keys()) <= {"task_id", "kind", "reason"}
            finally:
                runtime.set_tool_dispatcher(None)

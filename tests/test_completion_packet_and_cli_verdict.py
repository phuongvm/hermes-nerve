"""TDD regression suite for bounded completion packets, criterion sub-batching, CLI crew_card.py verdict observer, and supervision mode precedence."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hermes_nerve.config_resolver import resolve_config
from hermes_nerve.work import autobind, deterministic, hooks, runtime
from hermes_nerve.work.models import RunIdentity
from hermes_nerve.work.packet_bound import (
    MAX_COMPLETION_BATCHES,
    MAX_COMPLETION_PACKET_CHARS,
    bound_completion_packet,
    partition_completion_batches,
    safe_json_len,
)
from hermes_nerve.work.supervisor import CardSupervisor


@dataclass
class _FakeVerifyResult:
    value: str
    confidence: float = 0.94
    probabilities: dict[str, float] | None = None
    receipt_id: str = "receipt-batch"
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 120, "output_tokens": 15})

    def __post_init__(self) -> None:
        if self.probabilities is None:
            self.probabilities = {self.value: self.confidence}


class _RecordingEngine:
    def __init__(self, verifies: list[_FakeVerifyResult] | None = None) -> None:
        self.verifies = list(verifies or [_FakeVerifyResult("PASS")])
        self.verify_calls: list[dict[str, Any]] = []
        self.decide_calls: list[dict[str, Any]] = []

    def decide(self, **kwargs: Any) -> _FakeVerifyResult:
        self.decide_calls.append(kwargs)
        return _FakeVerifyResult("ACCEPT", receipt_id="preflight-accept")

    def verify(self, **kwargs: Any) -> _FakeVerifyResult:
        self.verify_calls.append(kwargs)
        if self.verifies:
            return self.verifies.pop(0)
        return _FakeVerifyResult("PASS")


class CompletionPacketBoundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.db_path = self.root / "work-supervision.sqlite3"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_supervisor(self, engine: _RecordingEngine | None = None) -> tuple[CardSupervisor, _RecordingEngine]:
        eng = engine or _RecordingEngine()
        sup = CardSupervisor(store_path=self.db_path, engine_factory=lambda: eng, mode="shadow", preview_chars=4000)
        return sup, eng

    def test_45k_packet_bounds_under_8000_chars_ascii_and_utf8_with_wrapper_headroom(self) -> None:
        raw_evidence = []
        # Criterion-linked item early in history must survive ahead of 23 later generic tool_result items.
        raw_evidence.append(
            {
                "evidence_id": "jevev-priority-c2",
                "criterion_id": "DOD-02",
                "kind": "tool_result",
                "tool_name": "terminal",
                "is_error": False,
                "preview": "PRIORITY_PROOF_TOKEN_C2 " + ("αβγ✓" * 380),
            }
        )
        for idx in range(1, 24):
            raw_evidence.append(
                {
                    "evidence_id": f"jevev-noise-{idx:02d}",
                    "criterion_id": None,
                    "kind": "tool_result",
                    "tool_name": "read_file",
                    "is_error": False,
                    "preview": f"noise-{idx:02d}-" + ("x" * 1500),
                }
            )
        raw_proposal = {
            "summary": "Completed implementation " + ("P" * 6000),
            "details": "Extended walkthrough " + ("Δ" * 6000),
            "extra_notes": [f"item-{i}-" + ("z" * 400) for i in range(15)],
        }
        criteria = [
            {"id": "DOD-01", "description": "Architecture invariants hold under load", "required": True},
            {"id": "DOD-02", "description": "Priority evidence proves edge-case handling", "required": True},
        ]

        raw_total = len(json.dumps({"evidence": raw_evidence, "proposal": raw_proposal}))
        self.assertGreater(raw_total, 45000)

        bounded = bound_completion_packet(
            goal="Verify completion resilience " + ("G" * 600),
            remaining_criteria=criteria,
            evidence=raw_evidence,
            proposal=raw_proposal,
        )

        ascii_json = json.dumps(bounded, separators=(",", ":"), allow_nan=False)
        utf8_json = json.dumps(bounded, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self.assertLessEqual(len(ascii_json), MAX_COMPLETION_PACKET_CHARS)
        self.assertLessEqual(len(utf8_json), MAX_COMPLETION_PACKET_CHARS)
        self.assertLessEqual(safe_json_len(bounded), MAX_COMPLETION_PACKET_CHARS)

        self.assertTrue(bounded.get("goal"))
        retained_ids = [c["id"] for c in bounded["remaining_criteria"]]
        self.assertEqual(retained_ids, ["DOD-01", "DOD-02"])
        self.assertTrue(bounded["evidence_truncated"])
        self.assertGreater(bounded["omitted_evidence_count"], 0)

        kept_evidence_ids = [e["evidence_id"] for e in bounded["evidence"]]
        self.assertIn("jevev-priority-c2", kept_evidence_ids)
        self.assertEqual(kept_evidence_ids[0], "jevev-priority-c2")

        # Also exercise end-to-end through CardSupervisor.verify_completion() and verify <= 9,500 wrapper headroom.
        sup, eng = self._make_supervisor(_RecordingEngine(verifies=[_FakeVerifyResult("PASS")]))
        bound = sup.bind_contract(
            task_id="t_pkt001",
            goal="Verify completion resilience " + ("G" * 600),
            criteria=[
                {"id": "DOD-01", "description": "Architecture invariants hold under load", "estimated_tokens": 500},
                {"id": "DOD-02", "description": "Priority evidence proves edge-case handling", "estimated_tokens": 500},
            ],
        )
        ident = RunIdentity("t_pkt001", 1, bound["contract"]["contract_hash"], "claim-1", "coder")
        sup.bind_run(ident)
        sup.observe_evidence(
            ident,
            value="PRIORITY_PROOF_TOKEN_C2 " + ("αβγ✓" * 900),
            criterion_id="DOD-02",
            tool_name="terminal",
        )
        for idx in range(23):
            sup.observe_evidence(
                ident,
                value=f"noise-{idx:02d}-" + ("w" * 3800),
                tool_name="read_file",
            )

        verdict = sup.verify_completion(ident, proposal=raw_proposal)
        self.assertTrue(verdict.allow)
        self.assertEqual(verdict.value, "PASS")
        self.assertEqual(len(eng.verify_calls), 1)

        sent_state = eng.verify_calls[0]["state"]
        sent_instructions = eng.verify_calls[0]["instructions"]
        self.assertLessEqual(safe_json_len(sent_state), MAX_COMPLETION_PACKET_CHARS)
        self.assertLessEqual(safe_json_len(sent_state) + len(sent_instructions), 9500)

    def test_pass2_and_pass3_ladder_enforces_hard_cap_under_tight_budgets(self) -> None:
        criteria = [
            {"id": f"DOD-{i:02d}", "description": f"Prose criterion {i}: " + ("C" * 350), "required": True}
            for i in range(1, 7)
        ]
        evidence = [
            {
                "evidence_id": f"jevev-{i:02d}",
                "criterion_id": "DOD-01" if i == 0 else None,
                "kind": "tool_result",
                "tool_name": "terminal",
                "is_error": False,
                "preview": "E" * 2000,
            }
            for i in range(16)
        ]
        proposal = {f"key_{i}": "V" * 1000 for i in range(10)}

        # Tight cap forces Pass 2 / Pass 3 compression while keeping goal + all criterion IDs intact.
        for tight_cap in (3500, 1400):
            pkt = bound_completion_packet(
                goal="Goal header " + ("H" * 300),
                remaining_criteria=criteria,
                evidence=evidence,
                proposal=proposal,
                max_chars=tight_cap,
            )
            self.assertLessEqual(safe_json_len(pkt), tight_cap)
            self.assertIn("goal", pkt)
            self.assertEqual([c["id"] for c in pkt["remaining_criteria"]], [f"DOD-{i:02d}" for i in range(1, 7)])
            self.assertTrue(pkt["evidence_truncated"])
            self.assertGreater(pkt["omitted_evidence_count"], 0)

    def test_truncation_fidelity_instructions_and_empty_evidence_zero_provider_calls(self) -> None:
        sup, eng = self._make_supervisor(_RecordingEngine(verifies=[_FakeVerifyResult("RETRY")]))
        bound = sup.bind_contract(
            task_id="t_empty001",
            goal="Verify fidelity rules",
            criteria=[{"id": "DOD-01", "description": "Prose requirement verified", "estimated_tokens": 500}],
        )
        ident = RunIdentity("t_empty001", 1, bound["contract"]["contract_hash"], "claim-1", "coder")
        sup.bind_run(ident)

        # Empty evidence MUST return NEEDS_EVIDENCE with 0 verify calls.
        empty_verdict = sup.verify_completion(ident, proposal={"summary": "trust me"})
        self.assertFalse(empty_verdict.allow)
        self.assertEqual(empty_verdict.value, "NEEDS_EVIDENCE")
        self.assertEqual(len(eng.verify_calls), 0)

        # Truncated evidence includes metadata and anti-false-PASS instruction; RETRY propagates allow=False.
        for i in range(15):
            sup.observe_evidence(ident, value=f"log-{i}-" + ("L" * 2000), tool_name="terminal")
        retry_verdict = sup.verify_completion(ident, proposal={"summary": "still missing direct proof"})
        self.assertFalse(retry_verdict.allow)
        self.assertEqual(retry_verdict.value, "RETRY")
        self.assertEqual(len(eng.verify_calls), 1)

        sent_state = eng.verify_calls[0]["state"]
        sent_instructions = eng.verify_calls[0]["instructions"]
        self.assertTrue(sent_state["evidence_truncated"])
        self.assertEqual(sent_state["omitted_evidence_count"], 3)
        self.assertIn("evidence_truncated", sent_instructions)
        self.assertIn("omitted_evidence_count", sent_instructions)
        self.assertIn("MUST NOT", sent_instructions)
        self.assertIn("NEEDS_EVIDENCE", sent_instructions)

    def test_criterion_sub_batching_all_pass_short_circuit_and_batch_cap_fail_closed(self) -> None:
        many_criteria = [
            {
                "id": f"DOD-{i:02d}",
                "description": f"Independent prose requirement #{i}: " + ("R" * 340),
                "required": True,
                "estimated_tokens": 200,
            }
            for i in range(1, 19)
        ]
        evidence = [
            {
                "evidence_id": f"jevev-{i:02d}",
                "criterion_id": f"DOD-{i:02d}",
                "kind": "tool_result",
                "tool_name": "terminal",
                "is_error": False,
                "preview": f"Proof output for DOD-{i:02d}",
            }
            for i in range(1, 19)
        ]

        batches = partition_completion_batches(
            goal="Verify multi-batch completion",
            remaining_criteria=many_criteria,
            evidence=evidence,
            proposal={"summary": "All 18 prose criteria addressed"},
        )
        self.assertIsNotNone(batches)
        assert batches is not None
        self.assertGreaterEqual(len(batches), 2)
        self.assertLessEqual(len(batches), MAX_COMPLETION_BATCHES)
        covered: list[str] = []
        for batch in batches:
            self.assertLessEqual(safe_json_len(batch), MAX_COMPLETION_PACKET_CHARS)
            covered.extend(c["id"] for c in batch["remaining_criteria"])
        self.assertEqual(covered, [f"DOD-{i:02d}" for i in range(1, 19)])

        # 1. Partial PASS + RETRY must NOT pass overall and must short-circuit remaining batches.
        eng_fail = _RecordingEngine(verifies=[_FakeVerifyResult("PASS"), _FakeVerifyResult("RETRY")])
        sup_fail, _ = self._make_supervisor(eng_fail)
        b_fail = sup_fail.bind_contract(task_id="t_batch_fail", goal="Multi-batch fail", criteria=many_criteria)
        id_fail = RunIdentity("t_batch_fail", 1, b_fail["contract"]["contract_hash"], "claim-f", "coder")
        sup_fail.bind_run(id_fail)
        sup_fail.observe_evidence(id_fail, value="some evidence", tool_name="terminal")
        v_fail = sup_fail.verify_completion(id_fail, proposal={"summary": "attempt"})
        self.assertFalse(v_fail.allow)
        self.assertEqual(v_fail.value, "RETRY")
        self.assertEqual(len(eng_fail.verify_calls), 2)

        # 2. All batches returning PASS marks every criterion VERIFIED_PASS and records usage per batch.
        eng_pass = _RecordingEngine(verifies=[_FakeVerifyResult("PASS") for _ in range(len(batches))])
        sup_pass, _ = self._make_supervisor(eng_pass)
        b_pass = sup_pass.bind_contract(task_id="t_batch_pass", goal="Multi-batch pass", criteria=many_criteria)
        id_pass = RunIdentity("t_batch_pass", 1, b_pass["contract"]["contract_hash"], "claim-p", "coder")
        sup_pass.bind_run(id_pass)
        sup_pass.observe_evidence(id_pass, value="complete evidence", tool_name="terminal")
        v_pass = sup_pass.verify_completion(id_pass, proposal={"summary": "complete"})
        self.assertTrue(v_pass.allow)
        self.assertEqual(v_pass.value, "PASS")
        self.assertEqual(len(eng_pass.verify_calls), len(batches))
        self.assertEqual(sup_pass.store.supervisor_usage_total(id_pass), len(batches) * 135)

        # 3. Pathological 50-criterion contract exceeds MAX_COMPLETION_BATCHES and fails closed with ESCALATE.
        overflow_criteria = [
            {
                "id": f"DOD-{i:02d}",
                "description": f"Pathological prose criterion #{i}: " + ("X" * 300),
                "required": True,
                "estimated_tokens": 100,
            }
            for i in range(1, 51)
        ]
        self.assertIsNone(
            partition_completion_batches(
                goal="Overflow goal",
                remaining_criteria=overflow_criteria,
                evidence=evidence,
                proposal={"summary": "overflow"},
            )
        )
        eng_over = _RecordingEngine(verifies=[_FakeVerifyResult("PASS") for _ in range(10)])
        sup_over, _ = self._make_supervisor(eng_over)
        b_over = sup_over.bind_contract(task_id="t_batch_over", goal="Overflow contract", criteria=overflow_criteria)
        id_over = RunIdentity("t_batch_over", 1, b_over["contract"]["contract_hash"], "claim-o", "coder")
        sup_over.bind_run(id_over)
        sup_over.observe_evidence(id_over, value="some evidence", tool_name="terminal")
        v_over = sup_over.verify_completion(id_over, proposal={"summary": "overflow"})
        self.assertFalse(v_over.allow)
        self.assertEqual(v_over.value, "ESCALATE")
        self.assertEqual(len(eng_over.verify_calls), 0)


class CliCrewVerdictAndModePrecedenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.db_path = self.root / "work-supervision.sqlite3"
        runtime.set_supervisor_for_tests(None, enabled_value=False)
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "HERMES_KANBAN_TASK_ID",
                "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_RUN_ID",
                "HERMES_NERVE_DOD_HASH",
                "HERMES_KANBAN_CLAIM_IDENTITY",
                "HERMES_KANBAN_CLAIM_LOCK",
                "HERMES_KANBAN_CLAIM_TOKEN",
                "HERMES_PROFILE",
                "HERMES_KANBAN_WORKSPACE",
            }
        }
        p = patch.dict(os.environ, clean_env, clear=True)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: runtime.set_supervisor_for_tests(None, enabled_value=False))
        self.addCleanup(self._tmp.cleanup)

    def _bootstrap_card(
        self,
        *,
        task_id: str = "t_abc123",
        body: str | None = None,
        mode: str = "shadow",
        profile: str = "coder",
    ) -> tuple[CardSupervisor, RunIdentity]:
        runtime.configure(
            enabled=True,
            mode=mode,
            store_path=str(self.db_path),
            provider_decisions_enabled=False,
        )
        card_body = body or (
            "Role: verifier\n"
            "Verify: agent\n"
            "GOAL: Verify CLI Crew verdict recognition\n"
            "## Definition of Done\n"
            "- Crew verdict PASS recorded for card\n"
        )
        task_obj = SimpleNamespace(
            id=task_id,
            title="CLI verdict card",
            body=card_body,
            current_run_id=4,
            claim_lock="claim-4",
        )
        env = {
            "HERMES_KANBAN_TASK_ID": task_id,
            "HERMES_KANBAN_RUN_ID": "4",
            "HERMES_KANBAN_CLAIM_LOCK": "claim-4",
            "HERMES_KANBAN_WORKSPACE": str(self.ws),
            "HERMES_PROFILE": profile,
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            autobind, "_load_canonical_task", side_effect=lambda tid: task_obj if tid == task_id else None
        ):
            ident = hooks.bootstrap_kanban_worker(task_id=task_id, session_id="sess-cli")
        assert ident is not None
        return runtime.supervisor(), ident

    def test_cli_crew_card_verdict_pass_marks_verified_pass_with_zero_provider_calls(self) -> None:
        sup, ident = self._bootstrap_card(task_id="t_abc123")
        provider_calls = 0

        def _forbid_provider(*_a: Any, **_kw: Any) -> Any:
            nonlocal provider_calls
            provider_calls += 1
            raise AssertionError("Semantic provider must not be called when CLI crew_card.py verdict PASS succeeded")

        sup.engine_factory = _forbid_provider

        cli_output = (
            "verdict: PASS\n"
            "proof command: git diff --check\n"
            "rc=0\n"
            "raw output:\n"
            "(no output)\n"
            "fails on this card: 0  file: /tmp/verdict.json\n"
        )
        obs = deterministic.observe_test_result(
            sup,
            ident,
            tool_name="terminal",
            args={
                "command": (
                    r'O:\workspaces\oss\hermes-agent\.venv\Scripts\python.exe -B -E '
                    r'"O:\workspaces\oss\crew\scripts\crew_card.py" verdict --card t_abc123'
                )
            },
            result=json.dumps({"output": cli_output, "exit_code": 0}),
            kwargs={},
        )
        self.assertTrue(obs["observed"])
        self.assertTrue(obs["passed"])
        self.assertFalse(obs["failed"])
        self.assertEqual(obs["provider_calls"], 0)
        self.assertGreaterEqual(obs["marked"], 1)

        # Completion verification must succeed deterministically with 0 semantic provider calls.
        verdict = sup.verify_completion(ident, proposal={"summary": "CLI verdict passed"})
        self.assertTrue(verdict.allow)
        self.assertEqual(verdict.value, "PASS")
        self.assertEqual(verdict.receipt_id, "deterministic")
        self.assertEqual(provider_calls, 0)

    def test_cli_crew_card_verdict_fail_rc1_and_rc3_with_pytest_flag_records_authoritative_fail(self) -> None:
        sup, ident = self._bootstrap_card(task_id="t_abc123")
        eng = _RecordingEngine(verifies=[_FakeVerifyResult("PASS")])
        sup.engine_factory = lambda: eng

        # First promote to PASS so we prove a subsequent CLI FAIL overwrites it with deterministic FAIL.
        deterministic.observe_crew_verdict(
            sup,
            ident,
            tool_name="terminal",
            args={"command": "python scripts/crew_card.py verdict --card t_abc123"},
            result=json.dumps({"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
            kwargs={},
        )

        # Case A: rc=1 with `--command "pytest -q"` MUST route via observe_test_result -> observe_crew_verdict
        # (never swallowed by L177 pytest branch or dropped at L108).
        fail_rc1_text = (
            "verdict: FAIL (fails on this card: 1)\n"
            "proof command: pytest -q\n"
            "rc=1\n"
            "raw output:\n"
            "1 failed in 0.15s\n"
            "fails on this card: 1  file: /tmp/verdict.json\n"
        )
        obs_rc1 = deterministic.observe_test_result(
            sup,
            ident,
            tool_name="terminal",
            args={"command": 'python scripts/crew_card.py verdict --card t_abc123 --command "pytest -q"'},
            result=json.dumps({"output": fail_rc1_text, "exit_code": 1}),
            kwargs={},
        )
        self.assertTrue(obs_rc1["observed"])
        self.assertTrue(obs_rc1["failed"])
        self.assertFalse(obs_rc1.get("passed", True))
        self.assertEqual(obs_rc1["provider_calls"], 0)
        self.assertGreaterEqual(obs_rc1["marked"], 1)

        # Case B: rc=3 handed-back output via post_tool_call also records authoritative deterministic FAIL.
        fail_rc3_text = (
            "verdict: FAIL, handed back - returned to its writer (request-changes): ok\n"
            "proof command: pytest -q\n"
            "rc=1\n"
            "second FAIL: the rework loop is closed\n"
            "fails on this card: 2  file: /tmp/verdict.json\n"
        )
        obs_rc3 = deterministic.observe_test_result(
            sup,
            ident,
            tool_name="bash",
            args={"command": 'cd /repo && python3 scripts/crew_card.py verdict --card=t_abc123 --command "pytest -q"'},
            result=json.dumps({"output": fail_rc3_text, "exit_code": 3}),
            kwargs={},
        )
        self.assertTrue(obs_rc3["observed"])
        self.assertTrue(obs_rc3["failed"])
        self.assertEqual(obs_rc3["provider_calls"], 0)

        # Semantic engine returning PASS MUST NOT override deterministic FAIL.
        sup.observe_evidence(ident, value="worker claims everything works", tool_name="terminal")
        comp = sup.verify_completion(ident, proposal={"summary": "please pass anyway"})
        self.assertFalse(comp.allow)
        self.assertEqual(comp.value, "RETRY")
        self.assertEqual(comp.receipt_id, "deterministic")
        self.assertEqual(len(eng.verify_calls), 0)

    def test_cli_verdict_ignores_mismatched_card_id_and_spoofed_commands(self) -> None:
        sup, ident = self._bootstrap_card(task_id="t_abc123")

        spoof_cases = [
            ("python scripts/crew_card.py verdict --card t_other999", {"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
            ('echo "verdict: PASS"', {"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
            ('python -c "print(\'verdict: PASS\')" crew_card.py verdict --card t_abc123', {"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
            ("python scripts/crew_card.py show-contract --card t_abc123", {"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
            ('python scripts/crew_card.py verdict --card t_abc123 || echo "verdict: PASS"', {"output": "verdict: PASS\nrc=0\n", "exit_code": 0}),
        ]
        for cmd, payload in spoof_cases:
            with self.subTest(cmd=cmd):
                obs = deterministic.observe_crew_verdict(
                    sup,
                    ident,
                    tool_name="terminal",
                    args={"command": cmd},
                    result=json.dumps(payload),
                    kwargs={},
                )
                self.assertFalse(obs["observed"])
                self.assertEqual(obs.get("marked", 0), 0)

        verdicts = sup.store.latest_contract_verdicts("t_abc123", ident.contract_hash)
        self.assertEqual(verdicts, {})

    def test_sidecar_work_supervision_mode_override_populates_advanced_override_sources_only(self) -> None:
        sidecar = {
            "version": 1,
            "nerve_profile": "operator",
            "nerve_modules": {"receipts": True},
            "advanced": {"work_supervision_mode": "enforce"},
        }
        host_cfg = {
            "nerve_profile": "operator",
            "work_supervision_mode": "shadow",
        }
        resolved = resolve_config(get_config=lambda k, d=None: host_cfg.get(k, d), profile=sidecar)
        self.assertEqual(resolved.get_config("work_supervision_mode", "shadow"), "enforce")
        self.assertEqual(
            resolved.advanced_override_sources,
            {"work_supervision_mode": "profile.json:advanced"},
        )
        self.assertNotIn("work_supervision_mode", resolved.override_sources)
        self.assertEqual(resolved.override_sources, {"receipts": "sidecar"})

        # Matching config.yaml value produces no override conflict entry.
        matching_cfg = {"nerve_profile": "operator", "work_supervision_mode": "enforce"}
        resolved_same = resolve_config(get_config=lambda k, d=None: matching_cfg.get(k, d), profile=sidecar)
        self.assertEqual(resolved_same.advanced_override_sources, {})

    def test_shadow_mode_preserves_pre_tool_call_dod_gating_while_disabling_trajectory_control(self) -> None:
        body = (
            "Role: worker\n"
            "Verify: dual\n"
            "Budget: 200000 tokens\n"
            "GOAL: Preserve NERVE-INV-5 shadow completion gating\n"
            "Artifact: must_exist.txt\n"
            "proof command: `pytest -q`\n"
        )
        sup, ident = self._bootstrap_card(task_id="t_shadow01", body=body, mode="shadow", profile="coder")
        task_obj = SimpleNamespace(
            id="t_shadow01",
            title="CLI verdict card",
            body=body,
            current_run_id=4,
            claim_lock="claim-4",
        )
        env = {
            "HERMES_KANBAN_TASK_ID": "t_shadow01",
            "HERMES_KANBAN_RUN_ID": "4",
            "HERMES_KANBAN_CLAIM_LOCK": "claim-4",
            "HERMES_KANBAN_WORKSPACE": str(self.ws),
            "HERMES_PROFILE": "coder",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            autobind, "_load_canonical_task", side_effect=lambda tid: task_obj if tid == "t_shadow01" else None
        ), patch.object(hooks, "_owns_kanban_terminal_authority", return_value=True):
            blocked_handoff = hooks.pre_tool_call(
                "crew_review_handoff",
                {"card_id": "t_shadow01"},
                task_id="t_shadow01",
                session_id="sess-cli",
            )
            assert isinstance(blocked_handoff, dict)
            self.assertEqual(blocked_handoff.get("action"), "block")

            blocked_complete = hooks.pre_tool_call(
                "kanban_complete",
                {"summary": "premature close"},
                task_id="t_shadow01",
                session_id="sess-cli",
            )
            assert isinstance(blocked_complete, dict)
            self.assertEqual(blocked_complete.get("action"), "block")

        # Mid-run trajectory controls remain suppressed in shadow mode.
        eng = _RecordingEngine()
        eng.decide = lambda **_kw: _FakeVerifyResult(  # type: ignore[method-assign]
            "REPLAN",
            confidence=0.95,
            probabilities={"REPLAN": 0.95, "CONTINUE": 0.05},
        )
        sup.engine_factory = lambda: eng
        assessment = sup.assess_trajectory(ident, trigger="test_failure", failures=["1 failed"])
        self.assertEqual(assessment.control, "REPLAN")
        self.assertIsNone(sup.control_for_run(ident))


if __name__ == "__main__":
    unittest.main()

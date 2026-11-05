"""Integration tests for Crew flat DoD autobind, zero-token crew_verdict, HANDOFF_VERIFIED, and SQLite compaction."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_nerve.work import autobind, hooks, runtime
from hermes_nerve.work.autobind import (
    extract_dod_lines,
    infer_run_terminal_mode,
    parse_crew_contract_fields,
    prune_and_vacuum_supervision_db,
)


class CrewDoDAndHandoffTests(unittest.TestCase):
    def setUp(self):
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

    def _configure_store(self, root: Path) -> Path:
        db_path = root / "work-supervision.sqlite3"
        runtime.configure(
            enabled=True,
            store_path=str(db_path),
            provider_decisions_enabled=False,
        )
        return db_path

    def test_autobind_crew_flat_contract_binds_and_blocks_premature_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            db_path = self._configure_store(root)

            body = (
                "Role: worker\n"
                "Coordinator: default\n"
                "Verify: proof\n"
                "Budget: 2000 tokens\n\n"
                "GOAL: Implement deterministic check\n\n"
                "Done when: (1) Artifact report.txt exists; (2) Proof suite passes\n"
                "Units: hermes_nerve/work/autobind.py | hermes_nerve/work/hooks.py\n"
                "Artifact: report.txt (verification receipt)\n"
                "proof command: `pytest -q`\n"
            )
            fields = parse_crew_contract_fields(body)
            self.assertEqual(fields["role"], "worker")
            self.assertEqual(fields["verify"], "proof")
            dod_lines = extract_dod_lines(body)
            self.assertEqual(len(dod_lines), 5)

            task_obj = SimpleNamespace(
                id="t_abc12345",
                title="Crew worker card",
                body=body,
                current_run_id=7,
                claim_lock="claim-7",
            )
            env = {
                "HERMES_KANBAN_TASK_ID": "t_abc12345",
                "HERMES_KANBAN_RUN_ID": "7",
                "HERMES_KANBAN_CLAIM_LOCK": "claim-7",
                "HERMES_KANBAN_WORKSPACE": str(ws),
                "HERMES_PROFILE": "coder",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                autobind, "_load_canonical_task", side_effect=lambda tid: task_obj if tid == "t_abc12345" else None
            ), patch.object(hooks, "_owns_kanban_terminal_authority", return_value=True):
                ident = hooks.bootstrap_kanban_worker(task_id="t_abc12345", session_id="sess-1")
                self.assertIsNotNone(ident)
                contract = runtime.supervisor().active_contract("t_abc12345")
                assert contract is not None
                self.assertGreaterEqual(len(contract.criteria), 2)

                # Premature kanban_complete must be blocked because Artifact: report.txt and proof command are unmet
                directive = hooks.pre_tool_call(
                    "kanban_complete",
                    {"summary": "Trying to close early"},
                    task_id="t_abc12345",
                    session_id="sess-1",
                )
                assert isinstance(directive, dict)
                self.assertEqual(directive.get("action"), "block")
                self.assertIn("Missing:", directive.get("message", ""))

            con = sqlite3.connect(str(db_path))
            try:
                cnt = con.execute("SELECT COUNT(*) FROM contracts WHERE task_id = 't_abc12345'").fetchone()[0]
                self.assertEqual(cnt, 1)
            finally:
                con.close()

    def test_autobind_non_kanban_chat_and_cron_sessions_write_zero_missing_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = self._configure_store(root)

            for non_card_id in ("20261105_chat_session", "cron_job_tick_01", "subagent_99", ""):
                self.assertIsNone(hooks.pre_llm_call(task_id=non_card_id, session_id="chat-1"))
                self.assertIsNone(
                    hooks.pre_tool_call("read_file", {"path": "README.md"}, task_id=non_card_id, session_id="chat-1")
                )
                hooks.post_tool_call(
                    tool_name="read_file",
                    args={"path": "README.md"},
                    result="ok",
                    task_id=non_card_id,
                    session_id="chat-1",
                )

            con = sqlite3.connect(str(db_path))
            try:
                missing_rows = con.execute(
                    "SELECT COUNT(*) FROM supervision_diagnostics WHERE kind = 'autobind_task_missing'"
                ).fetchone()[0]
                self.assertEqual(missing_rows, 0)
            finally:
                con.close()

    def test_deterministic_crew_verdict_pass_and_fail_zero_provider_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "findings.md").write_text("Verdict: APPROVED\n", encoding="utf-8")
            db_path = self._configure_store(root)

            body = (
                "Role: verifier\n"
                "Verify: agent\n"
                "GOAL: Verify implementation\n"
                "Artifact: findings.md\n"
                "proof command: `pytest -q`\n"
            )
            task_obj = SimpleNamespace(
                id="t_beef0001",
                title="Verifier card",
                body=body,
                current_run_id=3,
                claim_lock="claim-3",
            )
            provider_calls = 0

            def _forbid_provider(*_a, **_kw):
                nonlocal provider_calls
                provider_calls += 1
                raise AssertionError("Provider must not be called for deterministic crew_verdict observation")

            env = {
                "HERMES_KANBAN_TASK_ID": "t_beef0001",
                "HERMES_KANBAN_RUN_ID": "3",
                "HERMES_KANBAN_CLAIM_LOCK": "claim-3",
                "HERMES_KANBAN_WORKSPACE": str(ws),
                "HERMES_PROFILE": "reviewer",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                autobind, "_load_canonical_task", side_effect=lambda tid: task_obj if tid == "t_beef0001" else None
            ):
                ident = hooks.bootstrap_kanban_worker(task_id="t_beef0001", session_id="ver-1")
                assert ident is not None
                sup = runtime.supervisor()
                sup.engine_factory = _forbid_provider

                # Direct observe_test_result call on crew_verdict asserts provider_calls == 0
                obs_direct = hooks.observe_test_result(
                    sup,
                    ident,
                    tool_name="crew_verdict",
                    args={"card": "t_beef0001"},
                    result=json.dumps({"verdict": "PASS", "rc": 0, "file": "findings.md"}),
                    kwargs={},
                )
                self.assertTrue(obs_direct["observed"])
                self.assertTrue(obs_direct["passed"])
                self.assertEqual(obs_direct["provider_calls"], 0)

                # 1. Passing crew_verdict (also wrapped via tool_call) through post_tool_call
                hooks.post_tool_call(
                    tool_name="tool_call",
                    args={"calls": [{"name": "crew_verdict", "arguments": {"card": "t_beef0001"}}]},
                    result=json.dumps({"results": [{"verdict": "PASS", "rc": 0, "file": "findings.md"}]}),
                    task_id="t_beef0001",
                    session_id="ver-1",
                )

                self.assertEqual(provider_calls, 0)
                con = sqlite3.connect(str(db_path))
                try:
                    verdicts = con.execute(
                        "SELECT criterion_id, state, receipt_id FROM criterion_verdicts WHERE task_id = 't_beef0001' ORDER BY rowid"
                    ).fetchall()
                finally:
                    con.close()
                self.assertTrue(any(state == "VERIFIED_PASS" and rec == "deterministic" for _, state, rec in verdicts))

                # 2. Failing crew_verdict overrides with authoritative deterministic FAIL
                hooks.post_tool_call(
                    tool_name="crew_verdict",
                    args={"card": "t_beef0001"},
                    result=json.dumps({"verdict": "FAIL", "rc": 1}),
                    task_id="t_beef0001",
                    session_id="ver-1",
                )

                self.assertEqual(provider_calls, 0)
                con = sqlite3.connect(str(db_path))
                try:
                    latest = con.execute(
                        "SELECT state, receipt_id FROM criterion_verdicts WHERE task_id = 't_beef0001' ORDER BY rowid DESC LIMIT 1"
                    ).fetchone()
                finally:
                    con.close()
                self.assertEqual(latest, ("FAIL", "deterministic"))

    def test_handoff_verified_writer_on_dual_and_independent_cards(self):
        self.assertEqual(infer_run_terminal_mode("Role: worker\nVerify: dual\n", "coder"), "handoff")
        self.assertEqual(infer_run_terminal_mode("Role: worker\nVerify: independent\n", "designer"), "handoff")
        self.assertEqual(infer_run_terminal_mode("Role: verifier\nVerify: dual\n", "reviewer"), "complete")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "workspace"
            ws.mkdir()
            db_path = self._configure_store(root)

            body = (
                "Role: worker\n"
                "Verify: dual\n"
                "GOAL: Implement slice\n"
                "Artifact: slice_out.txt\n"
                "proof command: `pytest -q`\n"
            )
            task_obj = SimpleNamespace(
                id="t_cafe0002",
                title="Dual verified worker card",
                body=body,
                current_run_id=5,
                claim_lock="claim-5",
            )
            dispatched: list[tuple[str, dict]] = []

            def _fake_dispatch(tname: str, targs: dict):
                dispatched.append((tname, targs))
                return {"ok": True}

            env = {
                "HERMES_KANBAN_TASK_ID": "t_cafe0002",
                "HERMES_KANBAN_RUN_ID": "5",
                "HERMES_KANBAN_CLAIM_LOCK": "claim-5",
                "HERMES_KANBAN_WORKSPACE": str(ws),
                "HERMES_PROFILE": "coder",
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                autobind, "_load_canonical_task", side_effect=lambda tid: task_obj if tid == "t_cafe0002" else None
            ), patch.object(hooks, "tool_dispatcher_available", return_value=True), patch.object(
                hooks, "dispatch_tool", side_effect=_fake_dispatch
            ), patch.object(hooks, "_owns_kanban_terminal_authority", return_value=True):
                ident = hooks.bootstrap_kanban_worker(task_id="t_cafe0002", session_id="dual-1")
                self.assertIsNotNone(ident)

                # 1. Before DoD is met, crew_review_handoff is blocked
                blocked = hooks.pre_tool_call(
                    "crew_review_handoff",
                    {"card_id": "t_cafe0002"},
                    task_id="t_cafe0002",
                    session_id="dual-1",
                )
                assert isinstance(blocked, dict)
                self.assertEqual(blocked.get("action"), "block")

                # 2. Satisfy DoD: write artifact + observe passing proof
                (ws / "slice_out.txt").write_text("done\n", encoding="utf-8")
                hooks.post_tool_call(
                    tool_name="crew_verdict",
                    args={"card": "t_cafe0002"},
                    result=json.dumps({"verdict": "PASS", "rc": 0, "file": "slice_out.txt"}),
                    task_id="t_cafe0002",
                    session_id="dual-1",
                )

                # 3. Direct kanban_complete by writer on Verify: dual card is refused (handoff required)
                refused_complete = hooks.pre_tool_call(
                    "kanban_complete",
                    {"summary": "trying to complete directly"},
                    task_id="t_cafe0002",
                    session_id="dual-1",
                )
                assert isinstance(refused_complete, dict)
                self.assertEqual(refused_complete.get("action"), "block")
                self.assertEqual(refused_complete.get("rule_key"), "jev:work-handoff-required")

                # 4. crew_review_handoff is allowed and records HANDOFF_VERIFIED
                allowed = hooks.pre_tool_call(
                    "crew_review_handoff",
                    {"card_id": "t_cafe0002"},
                    task_id="t_cafe0002",
                    session_id="dual-1",
                )
                self.assertIsNone(allowed)

                # 5. Neither pre_verify nor on_session_end dispatches kanban_complete
                pv = hooks.pre_verify(task_id="t_cafe0002", session_id="dual-1", final_response="Handed off")
                self.assertIsNone(pv)
                hooks.on_session_end(task_id="t_cafe0002", session_id="dual-1")

                complete_dispatches = [name for name, _ in dispatched if name == "kanban_complete"]
                self.assertEqual(complete_dispatches, [])

            con = sqlite3.connect(str(db_path))
            try:
                row = con.execute(
                    "SELECT payload_json FROM run_controls WHERE task_id = 't_cafe0002' ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(row)
                payload = json.loads(row[0])
                self.assertEqual(payload.get("lifecycle_state"), "HANDOFF_VERIFIED")
            finally:
                con.close()

    def test_prune_and_vacuum_supervision_db_backup_integrity_and_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            db_path = tmp_dir / "work-supervision.sqlite3"
            bdir = tmp_dir / "backups"
            con = sqlite3.connect(str(db_path))
            try:
                con.execute("CREATE TABLE contracts (id INTEGER PRIMARY KEY, task_id TEXT)")
                con.execute("CREATE TABLE run_bindings (id INTEGER PRIMARY KEY, run_id INTEGER)")
                con.execute("CREATE TABLE run_controls (id INTEGER PRIMARY KEY, control TEXT)")
                con.execute("CREATE TABLE supervision_diagnostics (id INTEGER PRIMARY KEY, kind TEXT)")
                con.execute("INSERT INTO contracts (task_id) VALUES ('t_001')")
                con.execute("INSERT INTO run_bindings (run_id) VALUES (1)")
                con.execute("INSERT INTO run_controls (control) VALUES ('COMPLETE_READY')")
                con.executemany(
                    "INSERT INTO supervision_diagnostics (kind) VALUES (?)",
                    [("autobind_task_missing",)] * 5 + [("important_event",)],
                )
                con.commit()
            finally:
                con.close()

            # 1. Preflight low-disk check fails closed without mutating DB
            with patch("shutil.disk_usage", return_value=SimpleNamespace(free=100)):
                with self.assertRaises(RuntimeError):
                    prune_and_vacuum_supervision_db(db_path, backup_dir=bdir)

            # 2. Simulated mid-compaction failure rolls back atomically from online backup
            with self.assertRaises(RuntimeError):
                prune_and_vacuum_supervision_db(db_path, backup_dir=bdir, simulate_failure=True)

            con_rb = sqlite3.connect(str(db_path))
            try:
                rolled_back_missing = con_rb.execute(
                    "SELECT COUNT(*) FROM supervision_diagnostics WHERE kind = 'autobind_task_missing'"
                ).fetchone()[0]
                self.assertEqual(rolled_back_missing, 5)
            finally:
                con_rb.close()

            # 3. Clean prune + VACUUM succeeds and preserves control tables
            res = prune_and_vacuum_supervision_db(db_path, backup_dir=bdir)
            self.assertTrue(res["ok"])
            self.assertEqual(res["pruned_rows"], 5)
            self.assertEqual(res["remaining_autobind_task_missing"], 0)
            self.assertEqual(res["integrity_check"], "ok")
            self.assertEqual(res["preserved_counts"], {"contracts": 1, "run_bindings": 1, "run_controls": 1})
            self.assertTrue(Path(res["backup_path"]).is_file())


if __name__ == "__main__":
    unittest.main()

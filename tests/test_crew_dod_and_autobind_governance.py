"""Tests for Crew flat DoD extraction, non-Kanban session fencing, and zero-token deterministic verdict proof."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hermes_nerve.work.autobind import (
    extract_dod_lines,
    infer_run_terminal_mode,
    parse_crew_contract_fields,
    prune_and_vacuum_supervision_db,
)
from hermes_nerve.work.deterministic import (
    _safe_command_from_description,
    observe_crew_verdict,
)
from hermes_nerve.work.models import CriterionSpec, DoDContract, RunIdentity


class CrewDoDAutobindTests(unittest.TestCase):
    def test_crew_flat_contract_parsing(self):
        body = """Role: worker
Coordinator: default
Verify: dual
Budget: 1000 tokens

GOAL: Implement new telemetry gate

Done when: (1) All unit tests pass; (2) Artifact docs/report.md generated
Units: hermes_nerve/work/hooks.py | hermes_nerve/work/autobind.py
Artifact: docs/report.md (detailed summary)
proof command: `python -m unittest discover`
"""
        fields = parse_crew_contract_fields(body)
        self.assertEqual(fields["role"], "worker")
        self.assertEqual(fields["verify"], "dual")
        self.assertIn("unit tests pass", fields["done when"])

        dod_lines = extract_dod_lines(body)
        self.assertEqual(len(dod_lines), 5)
        self.assertIn("Crew Done-when satisfied: All unit tests pass", dod_lines)
        self.assertIn("Crew Done-when satisfied: Artifact docs/report.md generated", dod_lines)
        self.assertIn("Crew Units satisfied: hermes_nerve/work/hooks.py, hermes_nerve/work/autobind.py", dod_lines)
        self.assertIn("Required artifact exists: `docs/report.md`", dod_lines)
        self.assertIn("Proof command passes (rc=0): `python -m unittest discover`", dod_lines)

    def test_infer_run_terminal_mode(self):
        body_dual = "Role: worker\nVerify: dual\nGOAL: x\n"
        self.assertEqual(infer_run_terminal_mode(body_dual, "coder"), "handoff")
        self.assertEqual(infer_run_terminal_mode(body_dual, "worker"), "handoff")
        self.assertEqual(infer_run_terminal_mode(body_dual, "reviewer"), "complete")

        body_proof = "Role: worker\nVerify: proof\nGOAL: x\n"
        self.assertEqual(infer_run_terminal_mode(body_proof, "coder"), "complete")

    def test_safe_command_extraction(self):
        desc = "Proof command passes (rc=0): `python -m unittest discover -s tests`"
        cmd = _safe_command_from_description(desc)
        self.assertEqual(cmd, ["python", "-m", "unittest", "discover", "-s", "tests"])

    def test_zero_token_crew_verdict_observer(self):
        marked_calls = []

        class FakeStore:
            def evidence(self, identity, criterion_id=None, include_stale=True):
                return [{"evidence_id": "ev_1", "kind": "crew_verdict", "is_error": False}]

        class FakeSupervisor:
            store = FakeStore()

            def active_contract(self, task_id):
                return DoDContract(
                    task_id=task_id,
                    version=1,
                    goal="Test",
                    criteria=(
                        CriterionSpec(id="c1", description="Crew review verdict PASS observed"),
                        CriterionSpec(id="c2", description="Required artifact exists: `out.txt`"),
                    ),
                    reserve_tokens=1000,
                    checkpoint_fractions=(0.4, 0.7),
                    contract_hash="hash_123",
                    locked_at="2026-11-03T00:00:00Z",
                    actor="coordinator",
                )

            def mark_deterministic_verdict(self, identity, criterion_id, *, passed, reason, evidence_ids):
                marked_calls.append((criterion_id, passed, reason))

        sup = FakeSupervisor()
        ident = RunIdentity(
            task_id="t_123456",
            run_id=1,
            contract_hash="hash_123",
            claim_identity="worker_1",
        )

        obs = observe_crew_verdict(
            sup,
            ident,
            tool_name="crew_verdict",
            args={"card": "t_123456"},
            result=json.dumps({"verdict": "PASS", "file": "out.txt"}),
            kwargs={},
        )
        self.assertTrue(obs["observed"])
        self.assertTrue(obs["passed"])
        self.assertEqual(len(marked_calls), 2)
        self.assertEqual(marked_calls[0][0], "c1")
        self.assertTrue(marked_calls[0][1])
        self.assertEqual(marked_calls[1][0], "c2")
        self.assertTrue(marked_calls[1][1])

    def test_supervision_db_prune_and_vacuum(self):
        tmp_dir = Path(tempfile.mkdtemp())
        db_path = tmp_dir / "work-supervision.sqlite3"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE contracts (id INTEGER PRIMARY KEY, task_id TEXT)")
        con.execute("CREATE TABLE run_bindings (id INTEGER PRIMARY KEY, run_id INTEGER)")
        con.execute("CREATE TABLE run_controls (id INTEGER PRIMARY KEY, control TEXT)")
        con.execute("CREATE TABLE supervision_diagnostics (id INTEGER PRIMARY KEY, kind TEXT)")
        con.execute("INSERT INTO contracts (task_id) VALUES ('t_abc')")
        con.execute("INSERT INTO supervision_diagnostics (kind) VALUES ('autobind_task_missing')")
        con.execute("INSERT INTO supervision_diagnostics (kind) VALUES ('autobind_task_missing')")
        con.execute("INSERT INTO supervision_diagnostics (kind) VALUES ('other_diagnostic')")
        con.commit()
        con.close()

        res = prune_and_vacuum_supervision_db(db_path, backup_dir=tmp_dir / "backups")
        self.assertTrue(res["ok"])
        self.assertEqual(res["pruned_rows"], 2)
        self.assertEqual(res["preserved_counts"]["contracts"], 1)

        con2 = sqlite3.connect(str(db_path))
        rows = con2.execute("SELECT kind FROM supervision_diagnostics").fetchall()
        con2.close()
        self.assertEqual(rows, [("other_diagnostic",)])


if __name__ == "__main__":
    unittest.main()

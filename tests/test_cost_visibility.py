"""Regression: missing provider cost must surface as None, never a bare 0.0.

The typesafe transport returns usage without a cost field, so every decision
row can lack cost. quality_metrics() used to default jev_provider_reported_cost
to 0.0, which reads as "Jev is free" in jev_stats output.
"""

import tempfile
import unittest
from pathlib import Path
import os
from unittest.mock import patch

from hermes_nerve import outcomes as outcomes_mod
from hermes_nerve import nervous
from hermes_nerve import receipts as receipts_mod


class CostVisibilityTest(unittest.TestCase):
    def test_reported_cost_is_none_when_every_decision_lacks_cost(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
            "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
        }, clear=False):
            store = outcomes_mod.OutcomeStore()
            store.append({
                "record_type": "decision",
                "decision_id": "d1",
                "decision_type": "tool_supervision",
                "latency_ms": 640.0,
                "usage": {"input_tokens": 100, "output_tokens": 10},  # no cost key value
            })
            report = store.report()
            self.assertIsNone(report["provider_cost"])
            self.assertIsNone(report["provider_reported_cost"])
            self.assertEqual(report["provider_cost_missing_decisions"], 1)
            self.assertEqual(report["provider_cost_reported_decisions"], 0)

    def test_partial_cost_reports_sum_and_keeps_missing_count(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
            "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
        }, clear=False):
            store = outcomes_mod.OutcomeStore()
            store.append({
                "record_type": "decision",
                "decision_id": "d1",
                "decision_type": "tool_supervision",
                "latency_ms": 640.0,
                "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.5},
            })
            store.append({
                "record_type": "decision",
                "decision_id": "d2",
                "decision_type": "tool_supervision",
                "latency_ms": 620.0,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            })
            report = store.report()
            self.assertIsNone(report["provider_cost"])          # any missing -> aggregate None
            self.assertEqual(report["provider_reported_cost"], 0.5)  # partial sum still surfaced
            self.assertEqual(report["provider_cost_missing_decisions"], 1)
            self.assertEqual(report["provider_cost_reported_decisions"], 1)

    def test_quality_metrics_does_not_mask_missing_cost_as_zero(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
            "HERMES_NERVE_NERVOUS_EVENTS": str(Path(td) / "nervous.jsonl"),
            "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
        }, clear=False):
            store = outcomes_mod.OutcomeStore()
            store.append({
                "record_type": "decision",
                "decision_id": "d1",
                "decision_type": "tool_supervision",
                "latency_ms": 640.0,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            })
            with nervous.NervousSystem(engine_factory=lambda: None) as system:
                system._outcomes = store
                q = system.quality_metrics()
                self.assertIsNone(q["jev_provider_cost"])
                self.assertIsNone(q["jev_provider_reported_cost"])
                self.assertEqual(q["jev_provider_cost_missing_decisions"], 1)


    def test_explicit_zero_cost_remains_zero(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {
            "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
        }, clear=False):
            store = outcomes_mod.OutcomeStore()
            store.append({
                "record_type": "decision",
                "decision_id": "d1",
                "decision_type": "tool_supervision",
                "latency_ms": 10.0,
                "usage": {"cost": 0.0},
            })
            report = store.report()
            self.assertEqual(report["provider_cost"], 0.0)
            self.assertEqual(report["provider_reported_cost"], 0.0)
            self.assertEqual(report["provider_cost_reported_decisions"], 1)
            self.assertEqual(report["provider_cost_missing_decisions"], 0)

    def test_receipts_report_unknown_zero_and_partial_costs_distinctly(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "receipts.jsonl"

            def write_rows(rows):
                path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")

            base = {
                "contract": "test",
                "model": "typesafe/jev-test",
                "latency_ms": 1.0,
                "execution": {"live_provider_call": True},
            }

            write_rows([{**base, "result": {"usage": {}}}])
            unknown = receipts_mod.report(path)
            self.assertIsNone(unknown["total_cost"])
            self.assertIsNone(unknown["provider_reported_cost"])
            self.assertEqual(unknown["provider_cost_reported_calls"], 0)
            self.assertEqual(unknown["provider_cost_missing_calls"], 1)

            write_rows([{**base, "result": {"usage": {"cost": 0.0}}}])
            zero = receipts_mod.report(path)
            self.assertEqual(zero["total_cost"], 0.0)
            self.assertEqual(zero["provider_reported_cost"], 0.0)
            self.assertEqual(zero["provider_cost_reported_calls"], 1)
            self.assertEqual(zero["provider_cost_missing_calls"], 0)

            write_rows([
                {**base, "result": {"usage": {"cost": 0.25}}},
                {**base, "result": {"usage": {}}},
            ])
            partial = receipts_mod.report(path)
            self.assertIsNone(partial["total_cost"])
            self.assertEqual(partial["provider_reported_cost"], 0.25)
            self.assertEqual(partial["provider_cost_reported_calls"], 1)
            self.assertEqual(partial["provider_cost_missing_calls"], 1)


if __name__ == "__main__":
    unittest.main()

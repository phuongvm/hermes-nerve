"""Regression tests for bounded boot/skill overhead exemption and TDD RED baseline allowance."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_nerve.nervous import (
    NervousSystem,
    _MAX_BASELINE_TESTS_UNDER_STALL,
    _MAX_EXEMPT_BOOT_SKILL_ACTIONS,
    _PROGRESS_CHALLENGE_AFTER,
)
from hermes_nerve.progress_integrity import is_boot_or_skill_overhead, tool_kind
from tests.test_nervous import ScriptedEngine


class TestBootSkillAndTddBaselineGuard(unittest.TestCase):
    def _system(self, td: str) -> NervousSystem:
        ScriptedEngine.reset()
        system = NervousSystem(engine_factory=ScriptedEngine)
        system.configure(enabled=True, admission_enabled=False, mode="correct_next")
        return system

    def test_windows_python_exe_pytest_classified_as_test(self):
        cmd = (
            "PYTHONDONTWRITEBYTECODE=1 "
            "O:/workspaces/oss/hermes-agent/.venv/Scripts/python.exe -m pytest -q -s"
        )
        self.assertEqual(tool_kind("terminal", {"command": cmd}), "test")
        self.assertEqual(
            tool_kind("terminal", {"command": "O:/venv/Scripts/pytest.exe -q"}),
            "test",
        )

    def test_is_boot_or_skill_overhead_classification(self):
        self.assertTrue(is_boot_or_skill_overhead("skill_view", {"name": "tdd"}))
        self.assertTrue(is_boot_or_skill_overhead("skills_list", {}))
        self.assertTrue(is_boot_or_skill_overhead("tool_describe", {"names": ["patch"]}))
        self.assertTrue(is_boot_or_skill_overhead("read_file", {"path": "O:/workspaces/rules/AGENTS.md"}))
        self.assertTrue(is_boot_or_skill_overhead("read_file", {"path": "rules/shared/safety.md"}))
        self.assertFalse(is_boot_or_skill_overhead("read_file", {"path": "src/calculator.py"}))
        self.assertFalse(is_boot_or_skill_overhead("read_file", {"path": "spec.md"}))
        self.assertFalse(is_boot_or_skill_overhead("patch", {"path": "rules/AGENTS.md"}))

    def test_distinct_boot_and_skill_reads_do_not_trigger_premature_stall(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {
                "HERMES_NERVE_NERVOUS_EVENTS": str(Path(td) / "nervous.jsonl"),
                "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
            },
            clear=False,
        ):
            with self._system(td) as system:
                system.start_turn(
                    turn_id="t-boot",
                    session_id="s-boot",
                    user_message="Fix the two bugs in calculator.py and run pytest.",
                )
                # 9 boot/skill actions + 6 project file reads = 15 total calls,
                # but only 6 non-exempt task actions (< _PROGRESS_CHALLENGE_AFTER=12).
                boot_calls = [
                    ("read_file", {"path": "O:/workspaces/rules/AGENTS.md"}),
                    ("read_file", {"path": "O:/workspaces/rules/shared/startup-rules.md"}),
                    ("read_file", {"path": "O:/workspaces/rules/shared/safety.md"}),
                    ("skill_view", {"name": "tdd"}),
                    ("skill_view", {"name": "verification-before-claim"}),
                    ("skill_view", {"name": "diagnosing-bugs"}),
                    ("skill_view", {"name": "llm-coding-guardrails"}),
                    ("skill_view", {"name": "windows-msys-terminal-handling"}),
                    ("tool_describe", {"names": ["patch"]}),
                ]
                for tool, args in boot_calls:
                    system.observe_tool_call(
                        tool_name=tool, args=args, status="ok", turn_id="t-boot", session_id="s-boot"
                    )
                for idx in range(6):
                    system.observe_tool_call(
                        tool_name="read_file",
                        args={"path": f"mini_calc/file_{idx}.py"},
                        status="ok",
                        turn_id="t-boot",
                        session_id="s-boot",
                    )
                st = system.status(turn_id="t-boot")
                self.assertIsNone(st.get("active_control"))
                # TDD RED baseline pytest runs completely unblocked.
                verdict = system.before_tool_call(
                    tool_name="terminal",
                    args={"command": "O:/workspaces/oss/hermes-agent/.venv/Scripts/python.exe -m pytest -q -s"},
                    turn_id="t-boot",
                    session_id="s-boot",
                )
                self.assertIsNone(verdict)

    def test_repeated_skill_view_spam_still_triggers_progress_stall(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {
                "HERMES_NERVE_NERVOUS_EVENTS": str(Path(td) / "nervous.jsonl"),
                "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
            },
            clear=False,
        ):
            with self._system(td) as system:
                system.start_turn(
                    turn_id="t-spam",
                    session_id="s-spam",
                    user_message="Fix calculator.py now.",
                )
                # Repeating the exact same skill_view only exempts the first call; subsequent ones count.
                for _ in range(_PROGRESS_CHALLENGE_AFTER + 2):
                    system.observe_tool_call(
                        tool_name="skill_view",
                        args={"name": "tdd"},
                        status="ok",
                        turn_id="t-spam",
                        session_id="s-spam",
                    )
                st = system.status(turn_id="t-spam")
                self.assertEqual((st.get("active_control") or {}).get("source"), "local-progress-stall")

    def test_exempt_boot_skill_cap_enforced(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {
                "HERMES_NERVE_NERVOUS_EVENTS": str(Path(td) / "nervous.jsonl"),
                "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
            },
            clear=False,
        ):
            with self._system(td) as system:
                system.start_turn(
                    turn_id="t-cap",
                    session_id="s-cap",
                    user_message="Fix calculator.py now.",
                )
                # Distinct skill_view calls beyond _MAX_EXEMPT_BOOT_SKILL_ACTIONS count toward stall.
                for idx in range(_MAX_EXEMPT_BOOT_SKILL_ACTIONS + _PROGRESS_CHALLENGE_AFTER):
                    system.observe_tool_call(
                        tool_name="skill_view",
                        args={"name": f"skill-{idx}"},
                        status="ok",
                        turn_id="t-cap",
                        session_id="s-cap",
                    )
                st = system.status(turn_id="t-cap")
                self.assertEqual((st.get("active_control") or {}).get("source"), "local-progress-stall")

    def test_tdd_baseline_test_allowed_bounded_under_stall_before_mutation(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {
                "HERMES_NERVE_NERVOUS_EVENTS": str(Path(td) / "nervous.jsonl"),
                "HERMES_NERVE_OUTCOMES": str(Path(td) / "outcomes.jsonl"),
            },
            clear=False,
        ):
            with self._system(td) as system:
                system.start_turn(
                    turn_id="t-tdd",
                    session_id="s-tdd",
                    user_message="Fix calculator.py and verify with pytest.",
                )
                for idx in range(_PROGRESS_CHALLENGE_AFTER):
                    system.observe_tool_call(
                        tool_name="read_file",
                        args={"path": f"src/mod_{idx}.py"},
                        status="ok",
                        turn_id="t-tdd",
                        session_id="s-tdd",
                    )
                self.assertEqual(
                    (system.status(turn_id="t-tdd").get("active_control") or {}).get("source"),
                    "local-progress-stall",
                )
                # Ordinary read_file is STILL blocked under stall.
                read_verdict = system.before_tool_call(
                    tool_name="read_file",
                    args={"path": "src/extra.py"},
                    turn_id="t-tdd",
                    session_id="s-tdd",
                )
                self.assertIsNotNone(read_verdict)
                self.assertEqual(read_verdict["action"], "block")

                # Up to _MAX_BASELINE_TESTS_UNDER_STALL baseline test runs ARE allowed before mutation.
                for i in range(_MAX_BASELINE_TESTS_UNDER_STALL):
                    test_verdict = system.before_tool_call(
                        tool_name="terminal",
                        args={"command": f"O:/venv/Scripts/python.exe -m pytest -q test_{i}.py"},
                        turn_id="t-tdd",
                        session_id="s-tdd",
                    )
                    self.assertIsNone(test_verdict)

                # A 3rd test run without any mutation is blocked so the agent cannot loop on tests.
                blocked_test = system.before_tool_call(
                    tool_name="terminal",
                    args={"command": "O:/venv/Scripts/python.exe -m pytest -q test_extra.py"},
                    turn_id="t-tdd",
                    session_id="s-tdd",
                )
                self.assertIsNotNone(blocked_test)
                self.assertEqual(blocked_test["action"], "block")


if __name__ == "__main__":
    unittest.main()

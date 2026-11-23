"""Regression tests for Issue #7: Nerve plugin unload / hot-reload teardown."""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class UnloadableHostCtx:
    """Minimal PluginContext double matching Hermes' registration + on_unload contract."""

    def __init__(self, home: Path, settings: dict | None = None):
        self.home = home
        self.plugin_id = "nerve"
        self.settings = {
            "work_supervision_db": str(home / "work.db"),
            "remote_data_dir": str(home / "remote"),
            **(settings or {}),
        }
        self.tools: dict[str, tuple] = {}
        self.hooks: list[tuple[str, object]] = []
        self.middleware: list[tuple[str, object]] = []
        self.pre_verify_no_edits: list[object] = []
        self.engine = None
        self._unload_callbacks: list = []

    def get_config(self, key: str, default=None):
        return self.settings.get(key, default)

    def register_tool(self, *, name: str, schema=None, handler=None, **_kwargs):
        self.tools[name] = (schema, handler)

    def register_hook(self, name: str, callback, **_kwargs):
        self.hooks.append((name, callback))

    def register_pre_verify_no_edits(self, callback):
        self.pre_verify_no_edits.append(callback)
        self.hooks.append(("pre_verify_no_edits", callback))

    def register_middleware(self, kind: str, callback):
        self.middleware.append((kind, callback))

    def register_context_engine(self, engine):
        self.engine = engine

    def on_unload(self, callback):
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        self._unload_callbacks.append(callback)

    def unload(self) -> None:
        callbacks, self._unload_callbacks = list(self._unload_callbacks), []
        for cb in reversed(callbacks):
            cb()


def _load_fresh_plugin(module_name: str = "hermes_nerve_plugin_unload_test"):
    prefix = f"{module_name}."
    for key in list(sys.modules):
        if key == module_name or key.startswith(prefix):
            sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class PluginUnloadTests(unittest.TestCase):
    def _isolated_env(self, td: str) -> dict[str, str]:
        home = Path(td)
        return {
            "HERMES_HOME": str(home),
            "HERMES_NERVE_NERVOUS_EVENTS": str(home / "nervous-events.jsonl"),
            "HERMES_NERVE_OUTCOMES": str(home / "outcomes.jsonl"),
            "HERMES_NERVE_POLICY_MEMORY": str(home / "policy.jsonl"),
            "HERMES_NERVE_PREFERENCES": str(home / "preferences.json"),
            "HERMES_NERVE_GATE_EVENTS": str(home / "gate-events.jsonl"),
            "HERMES_KANBAN_TASK": "",
            "HERMES_KANBAN_TASK_ID": "",
        }

    def test_modular_profile_unload_disables_nervous_selective_memory_and_hooks(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            with patch.dict(os.environ, self._isolated_env(td), clear=False):
                mod = _load_fresh_plugin("nerve_unload_modular")
                no_remote = mod.nervous.NervousSystem(
                    engine_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline test"))
                )
                mod.nervous._default = no_remote

                ctx = UnloadableHostCtx(
                    home,
                    {
                        "nerve_profile": "fat_cat",
                        "gate_mode": "advisory",
                        "nervous_enabled": True,
                        "nervous_turn_admission": True,
                        "nervous_emit_prompt_hint": True,
                        "selective_context_enabled": True,
                        "selective_context_memory_mode": "apply",
                    },
                )
                mod.register(ctx)

                # Seed an admitted verified-experience claim so memory_context emits [Nerve selected memory].
                store = mod.experience.default_store()
                claim = store.create_claim(
                    subject="plugin-unload-architecture",
                    statement="Plugin disable must detach sticky pre_llm_call memory hooks immediately.",
                    claim_type="verified",
                    evidence_ids=["ev-unload-1"],
                    confidence=0.95,
                    admit=True,
                )
                self.assertTrue(claim["admitted"])

                before = mod.nervous.report()
                self.assertTrue(before["enabled"])
                self.assertTrue(before["admission_enabled"])
                self.assertTrue(mod.selective_context.report()["enabled"])
                self.assertEqual(mod.gate.gate_mode(), "advisory")
                self.assertTrue(mod.work_runtime.enabled())

                pre_llm_cbs = [cb for name, cb in ctx.hooks if name == "pre_llm_call"]
                pre_tool_cbs = [cb for name, cb in ctx.hooks if name == "pre_tool_call"]
                mw_cbs = [cb for kind, cb in ctx.middleware if kind == "llm_request"]
                self.assertGreaterEqual(len(pre_llm_cbs), 2)
                self.assertEqual(len(pre_tool_cbs), 1)
                self.assertEqual(len(mw_cbs), 1)

                outputs = [
                    cb(
                        session_id="ae56b4",
                        turn_id="turn-before-unload",
                        user_message="check plugin-unload-architecture sticky memory hooks",
                    )
                    for cb in pre_llm_cbs
                ]
                rendered = "\n".join(
                    item["context"] if isinstance(item, dict) else str(item)
                    for item in outputs
                    if item is not None
                )
                self.assertIn("[Nerve selected memory", rendered)
                self.assertIn("Nerve nervous supervision is active", rendered)
                self.assertEqual(mod.nervous.report()["active_turns"], 1)

                # Unload plugin: must register on_unload and tear down all live Nerve state.
                self.assertTrue(ctx._unload_callbacks, "register(ctx) must register an on_unload callback")
                ctx.unload()

                after = mod.nervous.report()
                self.assertFalse(after["enabled"])
                self.assertFalse(after["admission_enabled"])
                self.assertEqual(after["active_turns"], 0)
                self.assertFalse(mod.selective_context.report()["enabled"])
                self.assertEqual(mod.gate.gate_mode(), "off")
                self.assertFalse(mod.work_runtime.enabled())
                self.assertFalse(mod.experience.enabled())
                self.assertFalse(mod.ledger.enabled())
                self.assertFalse(mod.receipts.enabled())

                # Even if an in-flight turn held references to the old closures, they must be inert.
                stale_outputs = [
                    cb(
                        session_id="ae56b4",
                        turn_id="turn-after-unload",
                        user_message="check plugin-unload-architecture sticky memory hooks",
                    )
                    for cb in pre_llm_cbs
                ]
                self.assertEqual([x for x in stale_outputs if x is not None], [])
                self.assertIsNone(
                    pre_tool_cbs[0](tool_name="terminal", args={"command": "echo hi"}, session_id="ae56b4", turn_id="turn-after-unload")
                )
                self.assertIsNone(
                    mw_cbs[0](
                        request={"messages": [{"role": "user", "content": "check plugin-unload-architecture"}]},
                        session_id="ae56b4",
                        turn_id="turn-after-unload",
                    )
                )

                # Re-registering (re-enable / hot-reload) cleanly restores Nerve supervision.
                ctx2 = UnloadableHostCtx(
                    home,
                    {
                        "nerve_profile": "fat_cat",
                        "nervous_enabled": True,
                        "nervous_turn_admission": True,
                        "nervous_emit_prompt_hint": True,
                        "selective_context_enabled": True,
                        "selective_context_memory_mode": "apply",
                    },
                )
                mod.register(ctx2)
                try:
                    reloaded = mod.nervous.report()
                    self.assertTrue(reloaded["enabled"])
                    self.assertTrue(reloaded["admission_enabled"])
                    re_outputs = [
                        cb(
                            session_id="ae56b4",
                            turn_id="turn-reloaded",
                            user_message="check plugin-unload-architecture sticky memory hooks",
                        )
                        for name, cb in ctx2.hooks
                        if name == "pre_llm_call"
                    ]
                    re_rendered = "\n".join(
                        item["context"] if isinstance(item, dict) else str(item)
                        for item in re_outputs
                        if item is not None
                    )
                    self.assertIn("[Nerve selected memory", re_rendered)
                finally:
                    ctx2.unload()

    def test_legacy_profile_unload_disables_nervous_and_clears_active_controls(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            with patch.dict(os.environ, self._isolated_env(td), clear=False):
                mod = _load_fresh_plugin("nerve_unload_legacy")
                system = mod.nervous.NervousSystem(
                    engine_factory=lambda: (_ for _ in ()).throw(RuntimeError("offline test"))
                )
                mod.nervous._default = system
                ctx = UnloadableHostCtx(
                    home,
                    {
                        "nerve_profile": "legacy",
                        "gate_mode": "enforce",
                        "nervous_enabled": True,
                        "nervous_turn_admission": True,
                        "nervous_emit_prompt_hint": True,
                    },
                )
                mod.register(ctx)
                self.assertTrue(mod.nervous.report()["enabled"])
                self.assertTrue(mod.nervous.report()["admission_enabled"])

                system.start_turn(user_message="fix app.py and run pytest", session_id="c28c44", turn_id="t-leg")
                system.observe_tool_call(
                    tool_name="read_file",
                    args={"path": "missing.py"},
                    status="error",
                    error_message="file not found: missing.py",
                    session_id="c28c44",
                    turn_id="t-leg",
                    tool_call_id="call-1",
                )
                self.assertEqual(mod.nervous.report()["active_turns"], 1)

                ctx.unload()

                after = mod.nervous.report()
                self.assertFalse(after["enabled"])
                self.assertFalse(after["admission_enabled"])
                self.assertEqual(after["active_turns"], 0)
                self.assertEqual(mod.gate.gate_mode(), "off")
                self.assertIsNone(
                    system.before_tool_call(
                        tool_name="read_file",
                        args={"path": "missing.py"},
                        session_id="c28c44",
                        turn_id="t-leg",
                        tool_call_id="call-2",
                    )
                )
                self.assertIsNone(system.inject_challenge("ok", turn_id="t-leg", session_id="c28c44"))
                self.assertIsNone(system.completion_gate(turn_id="t-leg", session_id="c28c44", coding=True, changed_paths=[]))

    def test_unload_deactivates_registered_context_engine_for_open_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            with patch.dict(os.environ, self._isolated_env(td), clear=False):
                mod = _load_fresh_plugin("nerve_unload_engine")
                ctx = UnloadableHostCtx(
                    home,
                    {
                        "nerve_profile": "operator",
                        "context_engine_register": True,
                        "context_engine_mode": "apply",
                    },
                )
                mod.register(ctx)
                engine = ctx.engine
                self.assertIsNotNone(engine)
                self.assertFalse(getattr(engine, "_unloaded", False))

                ctx.unload()
                self.assertTrue(getattr(engine, "_unloaded", False))
                with patch.object(mod.context, "curate_context", side_effect=AssertionError("must not curate after unload")):
                    msgs = [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "first"},
                        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x" * 2000},
                        {"role": "user", "content": "latest"},
                    ]
                    out = engine.compress(msgs, current_tokens=50000, force=True)
                    self.assertEqual(out, msgs)


if __name__ == "__main__":
    unittest.main()

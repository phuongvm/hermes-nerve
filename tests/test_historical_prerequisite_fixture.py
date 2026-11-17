"""Pinned, sanitized source-seam evidence; not a Gateway or binary replay."""
import ast
import collections
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import typing
import uuid
from unittest.mock import Mock

import pytest


REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "designer_prerequisite_incident.json"
FIXTURE_SHA256 = "9f574f801684c392b0ccfe6b73fa33c88c52eb49ed1359298624a6618799a13b"


def load_fixture():
    # Pin LF serialization so Git checkout newline conversion is harmless.
    raw = FIXTURE.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "fixture digest"
    return json.loads(raw)


def capture_sources(provenance):
    # Read immutable Git objects, never installed plugins or changing HEAD files.
    return {
        name: subprocess.run(
            ["git", "-C", str(REPO), "show", f"{provenance['source_commit']}:hermes_nerve/{name}"],
            check=True, capture_output=True, timeout=5,
        ).stdout
        for name in provenance["git_source_sha256"]
    }


def verified_modules(sources, expected, loader):
    assert sources.keys() == expected.keys(), "source inventory"
    for name, digest in expected.items():
        assert hashlib.sha256(sources[name]).hexdigest() == digest, name
    # Load the verified bytes themselves; never reopen a pathname after checking.
    return loader("progress_integrity.py", sources["progress_integrity.py"]), loader("privacy.py", sources["privacy.py"])


def load_module(name, source):
    module = types.ModuleType("historical_" + name.removesuffix(".py"))
    exec(compile(source, name, "exec"), module.__dict__)
    return module


@pytest.fixture
def incident():
    fixture = load_fixture()
    sources = capture_sources(fixture["provenance"])
    for name, raw in sources.items():
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == fixture["provenance"]["source_sha256"][name]
    return fixture, sources


@pytest.mark.parametrize("filename", ["progress_integrity.py", "nervous.py", "privacy.py"])
def test_bad_source_digest_rejected_before_any_module_load(incident, filename):
    fixture, sources = incident
    expected = dict(fixture["provenance"]["git_source_sha256"])
    expected[filename] = "0" * 64
    loader = Mock(side_effect=AssertionError("must not load"))
    with pytest.raises(AssertionError, match=filename.replace(".", r"\.")):
        verified_modules(sources, expected, loader)
    loader.assert_not_called()


def test_sanitized_historical_progress_seams(incident, monkeypatch):
    fixture, sources = incident
    progress, privacy = verified_modules(sources, fixture["provenance"]["git_source_sha256"], load_module)
    classes = {"NervousConfig", "Challenge", "ControlDirective", "FailureEpisode", "TurnState"}
    methods = {"start_turn", "_resolve_turn", "_activate_progress_stall_replan", "_activate_control",
               "_record_control_lifecycle", "observe_tool_call", "before_tool_call"}
    body = []
    for node in ast.parse(sources["nervous.py"]).body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "NervousSystem":
            node.body = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in methods]
            body.append(node)
    module = types.ModuleType("historical_progress_seams")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    g = module.__dict__
    g.update(Any=typing.Any, Callable=typing.Callable, dataclass=dataclasses.dataclass,
             field=dataclasses.field, deque=collections.deque, time=time, uuid=uuid,
             BLOCKING_CONTROLS={"GATHER_EVIDENCE", "REPLAN", "ESCALATE"},
             _PROGRESS_CHALLENGE_AFTER=12, _is_trivial_ack=lambda _: False,
             canonical_hash=privacy.canonical_hash, progress_change_required=progress.change_required,
             progress_verification_required=progress.verification_required,
             progress_requested_paths=progress.requested_paths,
             tool_action_fingerprint=lambda name, args: privacy.canonical_hash([name, args]),
             infer_tool_risk=lambda *args: ("low", []))
    for name in ("tool_kind", "mutation_is_relevant", "environment_failure", "environment_action",
                 "test_succeeded", "test_result_masked"):
        g[name] = getattr(progress, name)
    exec(compile(ast.Module(body=body, type_ignores=[]), "historical/nervous.py", "exec"), g)
    full = f"CONTEXT:\n{fixture['context']}\nTASK:\n{fixture['goal']}"
    for text, count, expected_block in [(full, 11, False), (full, 12, True),
                                       (fixture["goal"], 12, False), (full, 16, True)]:
        system = module.NervousSystem()
        system._config = module.NervousConfig(admission_enabled=False, policy_memory_enabled=False)
        system._lock = threading.RLock()
        system._turns = {}
        system._session_turn = {}
        system._metrics = collections.Counter()
        system._recent_controls = collections.deque()
        system._recent_challenges = collections.deque()
        system._outcomes = []
        system._policy_memory = None
        system._log = lambda *args: None
        system._now = lambda: "offline"
        system._ensure_worker = lambda: None
        system.emit_event = lambda event: event
        system.start_turn(user_message=text, session_id="offline", turn_id="offline")
        for index in range(count):
            system.observe_tool_call(tool_name="skill_view", args={"name": "offline-prerequisite"},
                                     status="ok", result="loaded", turn_id="offline",
                                     tool_call_id=f"offline-{index}")
        result = system.before_tool_call(tool_name="execute_code", args={"code": "# prerequisite read"},
                                         turn_id="offline")
        state = system._turns["offline"]
        assert state.tool_actions == count
        assert state.relevant_mutation_attempts == 0
        assert bool(result) == expected_block
        if expected_block:
            assert result["action"] == "block"
            assert result["message"] == fixture["denial"]
    assert progress.change_required("Create proposal.md. Change slug example-topic.")
    assert not progress.change_required("Create proposal.md. Project slug example-topic.")

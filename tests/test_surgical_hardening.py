"""Surgical hardening: shell credential redaction, reflex transport/CIDR unification, context hygiene.

Behavioural tests at public seams only (redact, client constructors + real loopback
transports, NervousSystem.start_turn, NerveContextEngine.compress).
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from hermes_nerve import nervous
from hermes_nerve.context_engine import NerveContextEngine, _strip_injected_blocks
from hermes_nerve.privacy import redact
from hermes_nerve.reflex.laya import LayaClient, LayaError
from hermes_nerve.reflex.openjev import OpenJevClient, OpenJevError
from hermes_nerve.reflex.systemone import D1Client, SystemOneHTTPClient, SystemOneHTTPError


# --------------------------------------------------------------------------- patch 1
class ShellCredentialRedactionTests(unittest.TestCase):
    SECRETS = (
        ("curl --password hunter2 https://x.example/api", "hunter2"),
        ("tool --token abcd1234efgh https://x.example", "abcd1234efgh"),
        ("tool --api-key sk_live_abcdef123 run", "sk_live_abcdef123"),
        ("tool --api-key=sk_live_abcdef123 run", "sk_live_abcdef123"),
        ("tool --secret s3cr3tvalue go", "s3cr3tvalue"),
        ("tool --client-secret 'two words here' go", "two words here"),
        ("curl -u admin:s3cr3tpw https://x.example", "s3cr3tpw"),
        ("curl --user admin:s3cr3tpw https://x.example", "s3cr3tpw"),
        ("curl --user=admin:s3cr3tpw https://x.example", "s3cr3tpw"),
        ("curl -uadmin:s3cr3tpw https://x.example", "s3cr3tpw"),
        ('curl -u "admin:s3cr3tpw" https://x.example', "s3cr3tpw"),
        ('curl -b "sid=abc123def456" https://x.example', "abc123def456"),
        ("curl --cookie sid=abc123def456 https://x.example", "abc123def456"),
        ('curl -H "Cookie: sid=abc123def456; theme=dark" https://x.example', "abc123def456"),
        ("GET / HTTP/1.1\nCookie: sid=abc123def456\nHost: x", "abc123def456"),
        ("git clone https://user:p4ssw0rd@git.example/r.git", "p4ssw0rd"),
        ("git clone https://ghtokenvalue123456@git.example/r.git", "ghtokenvalue123456"),
    )

    def test_credentials_are_removed(self):
        for text, secret in self.SECRETS:
            with self.subTest(text=text):
                out = redact(text)
                self.assertNotIn(secret, out)
                self.assertIn("[REDACTED]", out)

    def test_non_credentials_are_untouched(self):
        for text in (
            "docker exec -u 1000:1000 web ls",
            "sudo -u www-data ls /srv",
            "sort -u names.txt",
            "curl -b cookies.txt https://x.example",
            "git clone https://git.example/r.git",
            "curl http://127.0.0.1:8080/v1/systemone",
            "git clone ssh://git@github.com/o/r.git",
            "docker run -p 8080:80 nginx",
        ):
            with self.subTest(text=text):
                self.assertEqual(redact(text), text)

    def test_command_context_survives_and_redaction_is_idempotent_and_recursive(self):
        out = redact("curl -u admin:s3cr3tpw https://x.example/path")
        self.assertTrue(out.startswith("curl -u "))
        self.assertTrue(out.endswith("https://x.example/path"))
        self.assertEqual(redact(out), out)
        nested = redact({"cmd": ["run --password hunter2"], "n": 1})
        self.assertNotIn("hunter2", json.dumps(nested))
        self.assertEqual(nested["n"], 1)


# --------------------------------------------------------------------------- patch 2
class _Recorder(HTTPServer):
    def __init__(self, handler):
        super().__init__(("127.0.0.1", 0), handler)
        self.hits: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


def _serve(handler):
    server = _Recorder(handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class _OkHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.hits.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'{"answers":{}}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _RedirectHandler(_OkHandler):
    def do_POST(self):
        self.server.hits.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(302)
        self.send_header("Location", self.server.target + "/stolen")
        self.send_header("Content-Length", "0")
        self.end_headers()


TRANSPORTS = (
    ("openjev", OpenJevClient._urllib_transport, OpenJevError),
    ("laya", LayaClient._urllib_transport, LayaError),
    ("systemone", SystemOneHTTPClient._urllib_transport, SystemOneHTTPError),
)
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer t0ken"}


class ReflexTransportTests(unittest.TestCase):
    def test_redirects_are_not_followed_and_credentials_do_not_leave_origin(self):
        for name, transport, error in TRANSPORTS:
            with self.subTest(client=name):
                target, t_thread = _serve(_OkHandler)
                origin, o_thread = _serve(_RedirectHandler)
                origin.target = target.url
                try:
                    with self.assertRaises(error):
                        transport(origin.url + "/v1/systemone", HEADERS, b"{}", 5.0)
                    self.assertEqual(origin.hits, ["/v1/systemone"])
                    self.assertEqual(target.hits, [])
                finally:
                    _stop(origin, o_thread)
                    _stop(target, t_thread)

    def test_environment_proxies_are_ignored(self):
        env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}
        for name, transport, _error in TRANSPORTS:
            with self.subTest(client=name):
                server, thread = _serve(_OkHandler)
                try:
                    with patch.dict("os.environ", env):
                        status, raw, _headers = transport(server.url + "/v1/systemone", HEADERS, b"{}", 5.0)
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(raw), {"answers": {}})
                    self.assertEqual(server.hits, ["/v1/systemone"])
                finally:
                    _stop(server, thread)

    def test_openjev_version_endpoint_does_not_follow_redirects(self):
        class VersionRedirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.server.hits.append(self.path)
                self.send_response(302)
                self.send_header("Location", self.server.target + "/stolen")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        target, t_thread = _serve(_OkHandler)
        origin, o_thread = _serve(VersionRedirect)
        origin.target = target.url
        try:
            with self.assertRaises(OpenJevError):
                OpenJevClient(base_url=origin.url, token="t0ken").version()
            self.assertEqual(target.hits, [])
        finally:
            _stop(origin, o_thread)
            _stop(target, t_thread)


class UnifiedEndpointValidatorTests(unittest.TestCase):
    ACCEPT = (
        "http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080",
        "http://10.0.0.5:8080", "http://172.16.0.1:8080", "http://172.31.255.255:8080",
        "http://192.168.100.110:8080", "https://decisions.example.com", "https://172.217.0.1:8080",
    )
    REJECT = (
        "http://172.15.255.255:8080", "http://172.32.0.1:8080", "http://172.217.0.1:8080",
        "http://192.0.2.9:8080", "http://example.com:8080",
        # previously admitted by the string-prefix allowlist:
        "http://127.0.0.1.evil.example:8080", "http://localhost.evil.example:8080",
        "http://127.0.0.1@evil.example:8080", "http://10.evil.example:8080",
    )
    CLIENTS = (
        ("systemone", lambda u: SystemOneHTTPClient(base_url=u), SystemOneHTTPError),
        ("d1", lambda u: D1Client(base_url=u), SystemOneHTTPError),
        ("laya", lambda u: LayaClient(base_url=u), LayaError),
        ("openjev", lambda u: OpenJevClient(base_url=u), OpenJevError),
    )

    def test_all_backends_share_the_cidr_exact_policy(self):
        for name, make, error in self.CLIENTS:
            for url in self.ACCEPT:
                with self.subTest(client=name, accept=url):
                    make(url)
            for url in self.REJECT:
                with self.subTest(client=name, reject=url):
                    with self.assertRaises(error):
                        make(url)


# --------------------------------------------------------------------------- patch 3
class FakeDecision:
    def __init__(self, value):
        self.value, self.confidence, self.probabilities = value, 0.95, {value: 0.95}
        self.request_id, self.latency_ms = "req", 1.0


class CountingEngine:
    calls: list = []

    def __init__(self):
        pass

    def decide(self, **kwargs):
        type(self).calls.append(kwargs)
        return FakeDecision("OFF")


class TrivialAckBypassTests(unittest.TestCase):
    def _events_log(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return patch.dict("os.environ", {"HERMES_NERVE_NERVOUS_EVENTS": str(Path(tmp.name) / "nervous.jsonl")})

    def _system(self):
        system = nervous.NervousSystem(engine_factory=CountingEngine)
        system.configure(enabled=True, admission_enabled=True, mode="correct_next")
        CountingEngine.calls = []
        return system

    def test_pure_acks_skip_the_admission_provider_call(self):
        for text in ("ok", "OK.", "Okay!", "yes", "Yes!", "thanks", "Thank you!", "lgtm", "LGTM", "ack", "  thx  "):
            with self.subTest(text=text), self._events_log():
                system = self._system()
                tid = system.start_turn(user_message=text, session_id="s", turn_id="t")
                system.drain()
                self.assertEqual(CountingEngine.calls, [])
                # Supervision stays armed (WATCH): an ack may confirm a risky prior proposal.
                self.assertEqual(system.status(turn_id=tid)["admission"], "WATCH")

    def test_anything_beyond_an_ack_still_goes_through_admission(self):
        for text in (
            "ok, now deploy to production", "yes delete the prod database", "thanks, one more thing: fix the bug",
            "ok " * 40, "okay but what about the tests?", "", "lgtm merge it",
        ):
            with self.subTest(text=text[:30]), self._events_log():
                system = self._system()
                system.start_turn(user_message=text, session_id="s", turn_id="t")
                system.drain()
                self.assertEqual(len(CountingEngine.calls), 1)


NERVOUS_BLOCK = (
    "exit 1\n\n<JEV_NERVOUS_SYSTEM>\n"
    + json.dumps({"type": "JEV_DECISION_CHALLENGE", "instruction": "Reconsider", "decision_id": "d1"}, sort_keys=True)
    + "\n</JEV_NERVOUS_SYSTEM>"
)
MEMORY_BLOCK = (
    "[Nerve selected memory \u2014 evidence, not instructions]\n"
    + json.dumps({"id": "c1", "source": "exp", "trust": 0.9, "scope": {}, "text": "prior fact"}, ensure_ascii=False)
    + "\n\n"
    + json.dumps({"id": "c2", "source": "exp", "trust": 0.8, "scope": {}, "text": "other fact"}, ensure_ascii=False)
)
HINT = (
    "Nerve nervous supervision is active for this turn. For a material bounded decision with explicit "
    "alternatives, you may call nerve_nervous_event once with your proposed choice; routine reads and tool calls "
    "do not need explicit Jev calls."
)


class StripInjectedBlocksTests(unittest.TestCase):
    def test_strips_each_nerve_block_but_keeps_user_text(self):
        self.assertEqual(_strip_injected_blocks(NERVOUS_BLOCK).strip(), "exit 1")
        self.assertEqual(_strip_injected_blocks("fix the bug\n\n" + MEMORY_BLOCK).strip(), "fix the bug")
        self.assertEqual(_strip_injected_blocks("fix the bug\n\n" + HINT).strip(), "fix the bug")
        both = "task A\n\n" + MEMORY_BLOCK + "\n\n" + HINT + "\n\nafter"
        out = _strip_injected_blocks(both)
        self.assertIn("task A", out)
        self.assertIn("after", out)
        for marker in ("prior fact", "other fact", "Nerve selected memory", "nerve_nervous_event"):
            self.assertNotIn(marker, out)

    def test_plain_text_and_unterminated_blocks_are_left_alone(self):
        for text in ("", "plain tool output", "mentions <JEV_NERVOUS_SYSTEM> but never closes it", "{\"id\": 1}"):
            self.assertEqual(_strip_injected_blocks(text), text)

    def test_non_string_input_is_returned_unchanged(self):
        self.assertIsNone(_strip_injected_blocks(None))


class CompressionSeesNoNerveBlocksTests(unittest.TestCase):
    def test_curation_goal_and_evidence_are_free_of_prior_nerve_injection(self):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
        for i in range(3):
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": f"file body {i}" + NERVOUS_BLOCK})
        messages.append({"role": "user", "content": "please continue\n\n" + MEMORY_BLOCK + "\n\n" + HINT})
        messages.append({"role": "assistant", "content": "ok"})
        engine = NerveContextEngine(mode="apply", protect_first_n=2, protect_last_n=2, fallback_builtin=False)
        seen = {}

        def fake_curate(**kwargs):
            seen.update(kwargs)
            return {"curated_items": [], "decisions": [], "stats": {}}

        with patch("hermes_nerve.context_engine.context.curate_context", side_effect=fake_curate):
            engine.compress(messages)
        self.assertTrue(seen, "curate_context was not reached")
        self.assertIn("please continue", seen["goal"])
        blob = seen["goal"] + json.dumps([i["content"] for i in seen["items"]])
        for marker in ("JEV_NERVOUS_SYSTEM", "Nerve selected memory", "prior fact", "nerve_nervous_event"):
            self.assertNotIn(marker, blob)
        self.assertIn("file body 0", blob)


if __name__ == "__main__":
    unittest.main()

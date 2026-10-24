from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hermes_nerve.engine import DecisionEngine
from hermes_nerve.receipts import report as receipt_report
from hermes_nerve.reflex import config as reflex_config
from hermes_nerve.reflex.openjev import OpenJevClient, OpenJevError
from hermes_nerve.reflex.shadow import ShadowProvider
from hermes_nerve.reflex.telemetry import report as shadow_report


class _OpenJevHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _write(self, code, payload):
        raw=json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def _authorized(self):
        token=getattr(self.server,"token","")
        return not token or self.headers.get("Authorization") == "Bearer " + token

    def do_GET(self):
        if not self._authorized(): return self._write(401,{"error":{"code":401,"message":"unauthorized"}})
        if self.path != "/v1/version": return self._write(404,{"error":"not_found"})
        return self._write(200,{"model":"openjev-test-model T=0.85 shim=testhash","helper_sha256":"testhash"})

    def do_POST(self):
        if not self._authorized(): return self._write(401,{"error":{"code":401,"message":"unauthorized"}})
        if self.path != "/v1/systemone": return self._write(404,{"error":"not_found"})
        n=int(self.headers.get("Content-Length") or 0); payload=json.loads(self.rfile.read(n))
        answers={}
        for qid,q in payload["questions"].items():
            if q["type"] == "choice":
                labels=list(q["criteria"]); answers[qid]={"type":"choice","choice":labels[0],"probabilities":{labels[0]:0.9,labels[1]:0.1},"confidence":0.8}
            elif q["type"] == "score":
                answers[qid]={"type":"score","score":1.0,"probabilities":{"0":0.1,"1":0.9},"confidence":0.8}
            else:
                answers[qid]={"type":"noul","noul":0.8,"confidence":0.8}
        return self._write(200,{"id":"shim-test","model":"openjev-test-model T=0.85 shim=testhash","answers":answers,"usage":{"input_tokens":17,"output_tokens":0}})


class _OpenJevServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self, addr, token=""):
        super().__init__(addr,_OpenJevHandler); self.token=token


class StaticProvider:
    def __init__(self, choice="WATCH", provider="TypeSafe", transport="openrouter-decisions", live=True):
        self.choice=choice; self.provider=provider; self.transport=transport; self.live=live
    def system_one(self, *, state, questions, model=None):
        labels=list(questions["decision"]["criteria"]); choice=self.choice if self.choice in labels else labels[0]
        return type("R",(),{
            "model":model or "model",
            "answers":{"decision":{"type":"choice","choice":choice,"probabilities":{x:(0.9 if x==choice else 0.1/max(1,len(labels)-1)) for x in labels},"confidence":0.9}},
            "usage":{"input_tokens":10,"output_tokens":0},"latency_ms":1.0,"request_id":"req" if self.live else "local",
            "provider":self.provider,"transport":self.transport,"live_provider_call":self.live,
        })()


class OpenJevDev17Tests(unittest.TestCase):
    def setUp(self):
        self.old_home=os.environ.get("HERMES_HOME"); self.tmp=tempfile.TemporaryDirectory(); os.environ["HERMES_HOME"]=self.tmp.name
        reflex_config.configure(backend="jev")
    def tearDown(self):
        reflex_config.configure(backend="jev")
        if self.old_home is None: os.environ.pop("HERMES_HOME",None)
        else: os.environ["HERMES_HOME"]=self.old_home
        self.tmp.cleanup()

    def _server(self, token=""):
        srv=_OpenJevServer(("127.0.0.1",0),token=token); th=threading.Thread(target=srv.serve_forever,daemon=True); th.start(); return srv

    def test_real_http_round_trip_and_version_pin(self):
        srv=self._server(token="secret")
        try:
            base=f"http://127.0.0.1:{srv.server_address[1]}"
            c=OpenJevClient(base_url=base,token="secret",expected_identity="testhash")
            self.assertIn("testhash",json.dumps(c.version()))
            r=c.system_one(state={"x":1},questions={"decision":{"type":"choice","instructions":"choose","criteria":{"WATCH":"w","REPLAN":"r"}}})
            self.assertEqual(r.answers["decision"]["choice"],"WATCH")
            self.assertEqual(r.provider,"OpenJev"); self.assertEqual(r.transport,"openjev-local-http"); self.assertFalse(r.live_provider_call)
            self.assertIn("testhash",r.model)
        finally:
            srv.shutdown(); srv.server_close()

    def test_response_model_is_server_identity_not_request_label(self):
        def transport(url,headers,body,timeout):
            return 200,json.dumps({"id":"x","model":"served-dir T=0.85 shim=abc","answers":{"q":{"type":"noul","noul":0.7}},"usage":{}}).encode(),{}
        r=OpenJevClient(model="openjev",transport=transport).system_one(state="x",questions={"q":{"type":"noul","instructions":"yes?"}})
        self.assertEqual(r.model,"served-dir T=0.85 shim=abc")

    def test_identity_pin_rejects_mismatch(self):
        def transport(url,headers,body,timeout):
            return 200,json.dumps({"model":"wrong","answers":{"q":{"type":"noul","noul":0.7}}}).encode(),{}
        with self.assertRaises(OpenJevError):
            OpenJevClient(expected_identity="wanted",transport=transport).system_one(state="x",questions={"q":{"type":"noul","instructions":"yes?"}})

    def test_plaintext_non_loopback_rejected(self):
        with self.assertRaises(OpenJevError): OpenJevClient(base_url="http://192.0.2.9:3000")

    def test_plaintext_rfc1918_and_loopback_endpoints_accepted(self):
        # NERVE-INV-1: plaintext HTTP only for loopback or RFC 1918 (10/8, 172.16/12, 192.168/16).
        for url in (
            "http://127.0.0.1:3000",
            "http://127.0.0.1",
            "http://localhost:3000",
            "http://LOCALHOST:3000/",
            "http://[::1]:3000",
            "http://10.0.0.0:3000",
            "http://10.255.255.255:3000",
            "http://172.16.0.0:3000",
            "http://172.16.0.1:3000",
            "http://172.31.255.255:3000",
            "http://192.168.0.1:3000",
            "http://192.168.100.110:3000",
            "http://192.168.255.255:3000",
        ):
            with self.subTest(url=url):
                self.assertEqual(OpenJevClient(base_url=url).base_url, url.rstrip("/"))

    def test_plaintext_outside_rfc1918_rejected(self):
        # NERVE-INV-1 boundaries: 172.15/172.32 sit just outside 172.16.0.0/12; 172.217 is public.
        for url in (
            "http://172.15.255.255:3000",
            "http://172.32.0.1:3000",
            "http://172.217.0.1:3000",
            "http://172.0.0.1:3000",
            "http://11.0.0.1:3000",
            "http://9.255.255.255:3000",
            "http://192.167.255.255:3000",
            "http://192.169.0.1:3000",
            "http://192.0.2.9:3000",
            "http://100.64.0.1:3000",
            "http://169.254.1.1:3000",
            "http://8.8.8.8:3000",
            "http://example.com:3000",
            # Prefix-spoofing hostnames and userinfo must not pass as private/loopback.
            "http://10.evil.example:3000",
            "http://192.168.example.com:3000",
            "http://127.0.0.1.nip.io:3000",
            "http://localhost.evil.example:3000",
            "http://10.0.0.1@evil.example:3000",
            "http://10.1:3000",
            "http://[fc00::1]:3000",
            "http://[::ffff:10.0.0.1]:3000",
            "http:///nohost",
            "ftp://10.0.0.1:3000",
        ):
            with self.subTest(url=url):
                with self.assertRaises(OpenJevError):
                    OpenJevClient(base_url=url)

    def test_https_endpoints_still_accepted(self):
        for url in ("https://openjev.example.com", "https://172.217.0.1:3000", "https://8.8.8.8"):
            with self.subTest(url=url):
                self.assertEqual(OpenJevClient(base_url=url).base_url, url)

    def test_openjev_decision_is_local_only(self):
        def transport(url,headers,body,timeout):
            labels=list(json.loads(body)["questions"]["decision"]["criteria"])
            return 200,json.dumps({"id":"shim-x","model":"openjev-local","answers":{"decision":{"type":"choice","choice":labels[0],"probabilities":{labels[0]:0.9,labels[1]:0.1},"confidence":0.8}},"usage":{"input_tokens":10,"output_tokens":0}}).encode(),{}
        result=DecisionEngine(provider=OpenJevClient(transport=transport)).decide(state={"x":1},instructions="choose",choices=["WATCH","REPLAN"],contract="dev17/openjev/v1")
        self.assertEqual(result.provider,"OpenJev"); self.assertEqual(result.provenance_status,"LOCAL_ONLY"); self.assertFalse(result.live_provider_call)
        self.assertEqual(receipt_report(recent_limit=0)["provider_calls"],0)

    def test_configured_openjev_factory_requires_no_jev_credentials(self):
        old={k:os.environ.pop(k,None) for k in ("OPENROUTER_API_KEY","TYPESAFE_API_KEY","OPENCODE_API_KEY")}
        try:
            reflex_config.configure(backend="openjev",openjev_base_url="http://127.0.0.1:3000")
            self.assertIsInstance(reflex_config.get_provider(),OpenJevClient)
        finally:
            for k,v in old.items():
                if v is not None: os.environ[k]=v

    def test_shadow_can_select_openjev_and_redacts_token(self):
        old=os.environ.get("OPENROUTER_API_KEY"); os.environ["OPENROUTER_API_KEY"]="test-key"
        try:
            reflex_config.configure(backend="shadow",shadow_backend="openjev",openjev_token="secret")
            provider=reflex_config.get_provider(); self.assertIsInstance(provider,ShadowProvider); self.assertIsInstance(provider.shadow,OpenJevClient)
            report=reflex_config.report(recent_limit=0)
            self.assertTrue(report["settings"]["openjev_token_configured"]); self.assertNotIn("secret",json.dumps(report))
        finally:
            if old is None: os.environ.pop("OPENROUTER_API_KEY",None)
            else: os.environ["OPENROUTER_API_KEY"]=old

    def test_shadow_pair_telemetry_is_provider_neutral(self):
        path=Path(self.tmp.name)/"shadow.jsonl"
        p=ShadowProvider(StaticProvider("WATCH"),StaticProvider("REPLAN",provider="OpenJev",transport="openjev-local-http",live=False),path=path,asynchronous=False)
        p.system_one(state="x",questions={"decision":{"type":"choice","instructions":"x","criteria":{"WATCH":"w","REPLAN":"r"}}})
        r=shadow_report(path,recent_limit=1); self.assertEqual(r["records"],1); self.assertEqual(r["disagreements"],1); self.assertEqual(r["shadow_errors"],0)

if __name__ == "__main__": unittest.main()

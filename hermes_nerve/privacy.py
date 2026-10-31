"""Redaction and stable hashing before state leaves the machine."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "auth", "cookie", "password", "passwd", "secret",
    "token", "access_token", "refresh_token", "private_key", "client_secret", "credential",
    "credentials", "aws_access_key_id", "aws_secret_access_key", "session_token",
}
_SECRET_SUFFIXES = ("_token", "_secret", "_password", "_api_key", "_access_key", "_private_key", "_credential", "_credentials")
_SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"),
    re.compile(r"(?i)\b([A-Z0-9_.-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)[^\s,;]+"),
    # Shell-command credential shapes. Group 1 is the kept prefix, the value is masked.
    # CLI flags: --password hunter2, --api-key=KEY, --client-secret 'two words'.
    re.compile(r"""(?i)(?<![\w-])(--(?:password|token|api[-_]?key|secret|client[-_]?secret)[=\s]+)(?:"[^"]*"|'[^']*'|\S+)"""),
    # curl -u/--user user:pass (all spellings); numeric ids such as `docker exec -u 1000:1000` are not credentials.
    re.compile(r"""(?i)(?<![\w-])((?:--user|-u)[=\s]{0,2}['"]?)(?![0-9]+:)[^\s:@'"]{0,64}:[^\s@'"]{1,256}"""),
    # Cookies: `Cookie:` headers and curl -b/--cookie name=value (a bare `-b cookies.txt` is a file, not a cookie).
    re.compile(r"""(?i)(?<![\w-])(cookie\s*:\s*)[^"'\n]+"""),
    re.compile(r"""(?i)(?<![\w-])((?:-b|--cookie)[=\s]+)(?:"[^"]*=[^"]*"|'[^']*=[^']*'|\S*=\S*)"""),
    # URL userinfo: scheme://user:pass@host and scheme://TOKEN@host (host and `@` are kept).
    re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{1,15}://)[^\s:@/]{1,64}:[^\s@/]{1,256}(?=@)"),
    re.compile(r"(?i)\b((?:https?|wss?|ftp)://)[^\s:@/]{8,}(?=@)"),
]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            out[key] = "[REDACTED]" if normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES) else redact(item)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
        return text
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

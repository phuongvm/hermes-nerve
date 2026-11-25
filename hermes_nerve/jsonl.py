"""Serialized JSONL persistence helpers.

Writes are serialized within the process and use an advisory file lock on
platforms that provide fcntl. Each record is encoded before the lock is taken,
then appended with O_APPEND so concurrent Hermes/Jev writers cannot interleave
JSON fragments.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows fallback remains thread-safe.
    _fcntl = None

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.expanduser().resolve(strict=False))
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def append_jsonl(path: Path, record: dict[str, Any]) -> Path:
    selected = Path(path).expanduser()
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    lock = _lock_for(selected)
    with lock:
        fd = os.open(str(selected), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short JSONL append")
                view = view[written:]
        finally:
            if _fcntl is not None:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
    return selected


def append_jsonl_many(
    path: Path,
    records: list[dict[str, Any]],
    *,
    must_exist: bool = False,
    require_parent: bool = False,
) -> Path:
    selected = Path(path).expanduser()
    if not records:
        return selected
    if must_exist:
        flags = os.O_APPEND | os.O_WRONLY
    else:
        if require_parent and not selected.parent.is_dir():
            return selected
        selected.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    payload = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        for record in records
    ).encode("utf-8")
    lock = _lock_for(selected)
    with lock:
        try:
            fd = os.open(str(selected), flags, 0o600)
        except FileNotFoundError:
            if must_exist or require_parent:
                return selected
            raise
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short JSONL append")
                view = view[written:]
        finally:
            if _fcntl is not None:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    selected = Path(path).expanduser()
    lock = _lock_for(selected)
    with lock:
        if not selected.exists():
            return []
        text = selected.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows

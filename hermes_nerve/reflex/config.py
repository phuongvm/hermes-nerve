"""Global Reflex backend configuration used by DecisionEngine default construction."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from ..client import JevClient
from ..paths import report_home
from .laya import DEFAULT_BASE_URL as LAYA_BASE_URL, DEFAULT_MODEL as LAYA_MODEL, LayaClient
from .openjev import DEFAULT_BASE_URL as OPENJEV_BASE_URL, DEFAULT_MODEL as OPENJEV_MODEL, OpenJevClient
from .shadow import ShadowProvider
from . import telemetry

_LOCK = threading.RLock()
_SIDECAR_SIG: tuple[str, str] | None = None
_SETTINGS: dict[str, Any] = {
    "backend": "jev",
    "laya_base_url": LAYA_BASE_URL,
    "laya_model": LAYA_MODEL,
    "laya_timeout_seconds": 5.0,
    "laya_token": "",
    "openjev_base_url": OPENJEV_BASE_URL,
    "openjev_model": OPENJEV_MODEL,
    "openjev_timeout_seconds": 10.0,
    "openjev_token": "",
    "openjev_expected_identity": "",
    "shadow_backend": "laya",
    "shadow_async": True,
    "shadow_log": "",
}


def _current_sidecar_sig() -> tuple[str, str] | None:
    try:
        from ..profiles import profile_path
        path = profile_path()
        raw = path.read_bytes()
        return (str(path), hashlib.sha256(raw).hexdigest())
    except Exception:
        return None


def _sync_sidecar_if_changed() -> None:
    global _SIDECAR_SIG
    sig = _current_sidecar_sig()
    if sig is None or sig == _SIDECAR_SIG:
        return
    try:
        from ..profiles import load_profile
        doc = load_profile()
    except Exception:
        return
    with _LOCK:
        _SIDECAR_SIG = sig
        if not isinstance(doc, dict):
            return
        adv = doc.get("advanced")
        if not isinstance(adv, dict):
            return
        if "reflex_backend" in adv and str(adv["reflex_backend"] or "").strip().lower() in {"jev", "laya", "openjev", "shadow"}:
            _SETTINGS["backend"] = str(adv["reflex_backend"]).strip().lower()
        for src, dst in (
            ("reflex_laya_base_url", "laya_base_url"),
            ("reflex_laya_model", "laya_model"),
            ("reflex_laya_token", "laya_token"),
            ("reflex_openjev_base_url", "openjev_base_url"),
            ("reflex_openjev_model", "openjev_model"),
            ("reflex_openjev_token", "openjev_token"),
            ("reflex_openjev_expected_identity", "openjev_expected_identity"),
        ):
            if src in adv and adv[src] is not None:
                val = str(adv[src]).strip()
                if dst.endswith("_base_url"):
                    val = val.rstrip("/")
                if val or dst.endswith("_token") or dst.endswith("_identity"):
                    _SETTINGS[dst] = val
        for src, dst in (
            ("reflex_laya_timeout_seconds", "laya_timeout_seconds"),
            ("reflex_openjev_timeout_seconds", "openjev_timeout_seconds"),
        ):
            if src in adv and adv[src] is not None:
                try:
                    _SETTINGS[dst] = min(120.0, max(0.25, float(adv[src])))
                except (TypeError, ValueError):
                    pass


def configure(
    *,
    backend: Any = "jev",
    laya_base_url: Any = LAYA_BASE_URL,
    laya_model: Any = LAYA_MODEL,
    laya_timeout_seconds: Any = 5.0,
    laya_token: Any = "",
    openjev_base_url: Any = OPENJEV_BASE_URL,
    openjev_model: Any = OPENJEV_MODEL,
    openjev_timeout_seconds: Any = 10.0,
    openjev_token: Any = "",
    openjev_expected_identity: Any = "",
    shadow_backend: Any = "laya",
    shadow_async: Any = True,
    shadow_log: Any = "",
) -> None:
    global _SIDECAR_SIG
    selected = str(backend or "jev").strip().lower()
    if selected not in {"jev", "laya", "openjev", "shadow"}:
        raise ValueError("reflex backend must be one of: jev, laya, openjev, shadow")
    selected_shadow = str(shadow_backend or "laya").strip().lower()
    if selected_shadow not in {"laya", "openjev"}:
        raise ValueError("reflex shadow backend must be one of: laya, openjev")
    try:
        laya_timeout = min(120.0, max(0.25, float(laya_timeout_seconds)))
    except (TypeError, ValueError):
        laya_timeout = 5.0
    try:
        openjev_timeout = min(120.0, max(0.25, float(openjev_timeout_seconds)))
    except (TypeError, ValueError):
        openjev_timeout = 10.0
    with _LOCK:
        _SETTINGS.update(
            {
                "backend": selected,
                "laya_base_url": str(laya_base_url or LAYA_BASE_URL).rstrip("/"),
                "laya_model": str(laya_model or LAYA_MODEL).strip() or LAYA_MODEL,
                "laya_timeout_seconds": laya_timeout,
                "laya_token": str(laya_token or "").strip(),
                "openjev_base_url": str(openjev_base_url or OPENJEV_BASE_URL).rstrip("/"),
                "openjev_model": str(openjev_model or OPENJEV_MODEL).strip() or OPENJEV_MODEL,
                "openjev_timeout_seconds": openjev_timeout,
                "openjev_token": str(openjev_token or "").strip(),
                "openjev_expected_identity": str(openjev_expected_identity or "").strip(),
                "shadow_backend": selected_shadow,
                "shadow_async": bool(shadow_async),
                "shadow_log": str(shadow_log or "").strip(),
            }
        )
        _SIDECAR_SIG = _current_sidecar_sig()


def settings() -> dict[str, Any]:
    _sync_sidecar_if_changed()
    with _LOCK:
        out = dict(_SETTINGS)
    for name in ("laya_token", "openjev_token"):
        out[name + "_configured"] = bool(out.pop(name, ""))
    return out


def _raw_settings() -> dict[str, Any]:
    _sync_sidecar_if_changed()
    with _LOCK:
        return dict(_SETTINGS)


def _laya_client(cfg: dict[str, Any]) -> LayaClient:
    return LayaClient(
        base_url=cfg["laya_base_url"],
        model=cfg["laya_model"],
        timeout=cfg["laya_timeout_seconds"],
        token=cfg["laya_token"] or None,
    )


def _openjev_client(cfg: dict[str, Any]) -> OpenJevClient:
    return OpenJevClient(
        base_url=cfg["openjev_base_url"],
        model=cfg["openjev_model"],
        timeout=cfg["openjev_timeout_seconds"],
        token=cfg["openjev_token"] or None,
        expected_identity=cfg["openjev_expected_identity"] or None,
    )


def get_provider():
    cfg = _raw_settings()
    backend = cfg["backend"]
    if backend == "jev":
        return JevClient()
    if backend == "laya":
        return _laya_client(cfg)
    if backend == "openjev":
        return _openjev_client(cfg)
    shadow = _laya_client(cfg) if cfg["shadow_backend"] == "laya" else _openjev_client(cfg)
    shadow_path = Path(cfg["shadow_log"]).expanduser() if cfg["shadow_log"] else report_home() / "reflex" / "shadow.jsonl"
    return ShadowProvider(JevClient(), shadow, path=shadow_path, asynchronous=cfg["shadow_async"])


def report(*, recent_limit: int = 5) -> dict[str, Any]:
    cfg = settings()
    path = Path(str(cfg.get("shadow_log") or "")).expanduser() if cfg.get("shadow_log") else report_home() / "reflex" / "shadow.jsonl"
    return {"settings": cfg, "shadow": telemetry.report(path, recent_limit=recent_limit)}

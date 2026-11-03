from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from .models import RunIdentity, utc_now
from .nerve import budget_criterion, estimate_task_budget
from .runtime import settings

_DOD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(definition\s+of\s+done|acceptance\s+criteria|done\s+criteria)\s*$", re.I)
_ANY_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+?)\s*$")
_CANONICAL_TASK_ID_RE = re.compile(r"^t_[0-9a-f]{6,}$", re.I)
_CREW_FIELD_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?\**(Role|Verify|Done\s+when|Units|Artifact|proof\s+command)\**\s*:\s*(.+?)\s*$",
    re.I,
)
_NUMBERED_CLAUSE_SPLIT_RE = re.compile(r"(?:^|\s+)\(\d+\)\s+")
_WRITER_PROFILES = {"coder", "designer", "worker", "content", "researcher", "leader"}
_CLOSER_PROFILES = {"reviewer", "qa", "verifier"}


@dataclass(frozen=True)
class _LocalAccept:
    value: str = "ACCEPT"
    confidence: float = 1.0
    receipt_id: str = "local-structured-dod"


def parse_checkpoint_fractions(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        try:
            value = float(part.strip())
        except ValueError:
            continue
        if 0.0 < value < 1.0 and value not in values:
            values.append(value)
    return tuple(sorted(values)) or (0.40, 0.70)


def parse_crew_contract_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in str(body or "").splitlines():
        m = _CREW_FIELD_RE.match(raw)
        if not m:
            continue
        key = re.sub(r"\s+", " ", m.group(1).strip().lower())
        val = m.group(2).strip().strip("*_").strip()
        if val:
            fields[key] = val
    return fields


def infer_run_terminal_mode(body: str, worker_profile: str | None = None) -> str:
    """Return ``'handoff'`` when a writer run on a dual/independent Crew card must hand off to review, else ``'complete'``."""
    fields = parse_crew_contract_fields(body)
    verify_mode = str(fields.get("verify") or "").strip().lower().split()[0] if fields.get("verify") else ""
    if verify_mode not in {"dual", "independent"}:
        return "complete"
    role = str(fields.get("role") or "").strip().lower().split()[0] if fields.get("role") else ""
    prof = str(worker_profile or os.getenv("HERMES_PROFILE") or "").strip().lower().lstrip("@")
    if prof in _CLOSER_PROFILES:
        return "complete"
    if prof in _WRITER_PROFILES or role in {"worker", "content"}:
        return "handoff"
    return "complete"


def _extract_crew_flat_dod_lines(body: str) -> list[str]:
    fields = parse_crew_contract_fields(body)
    if not fields:
        return []
    items: list[str] = []
    done_when = fields.get("done when", "").strip().strip("*_").strip()
    if done_when and done_when.lower() not in {"none", "n/a", "-"}:
        parts = [p.strip().rstrip(";") for p in _NUMBERED_CLAUSE_SPLIT_RE.split(done_when) if p.strip()]
        if len(parts) <= 1 and ";" in done_when:
            parts = [p.strip() for p in done_when.split(";") if p.strip()]
        for clause in parts:
            items.append(f"Crew Done-when satisfied: {clause}")
    units = fields.get("units", "").strip().strip("*_").strip()
    if units and units.lower() not in {"none", "n/a", "-"}:
        unit_parts = [u.strip() for u in units.split("|") if u.strip()]
        if unit_parts:
            items.append(f"Crew Units satisfied: {', '.join(unit_parts)}")
    artifact = fields.get("artifact", "").strip().strip("`\"'").strip("*_").strip()
    if artifact and artifact.lower() not in {"none", "n/a", "-"}:
        # Take first path token if notes follow in parentheses
        artifact_path = artifact.split(" (")[0].strip().strip("`\"'").strip("*_").strip()
        items.append(f"Required artifact exists: `{artifact_path}`")
    proof_cmd = fields.get("proof command", "").strip().strip("`\"'").strip("*_").strip()
    if proof_cmd and proof_cmd.lower() not in {"none", "n/a", "-"}:
        if proof_cmd.startswith("`") and proof_cmd.endswith("`") and len(proof_cmd) > 2:
            proof_cmd = proof_cmd[1:-1].strip()
        items.append(f"Proof command passes (rc=0): `{proof_cmd}`")
    return items


def extract_dod_lines(body: str) -> list[str]:
    lines = str(body or "").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if _DOD_HEADING.match(line):
            start = idx + 1
            break
    if start is None:
        return _extract_crew_flat_dod_lines(body)
    items: list[str] = []
    current = ""
    for raw in lines[start:]:
        if _ANY_HEADING.match(raw):
            break
        m = _LIST_ITEM.match(raw)
        if m:
            if current:
                items.append(current.strip())
            current = m.group(1).strip()
            continue
        text = raw.strip()
        if text and current:
            current += " " + text
    if current:
        items.append(current.strip())
    extracted = [x for x in items if x]
    return extracted if extracted else _extract_crew_flat_dod_lines(body)


def build_criteria(body: str, *, total_budget: int) -> tuple[list[dict[str, Any]], int]:
    lines = extract_dod_lines(body)
    if not lines:
        return [], 0
    total = max(1000, int(total_budget))
    reserve = max(1000, int(total * 0.15))
    usable = max(len(lines), total - reserve)
    per = max(1, usable // len(lines))
    remainder = max(0, usable - per * len(lines))
    criteria = [
        {
            "id": f"DOD-{idx:02d}",
            "description": text,
            "required": True,
            "weight": 1.0,
            "estimated_tokens": per + (1 if idx <= remainder else 0),
        }
        for idx, text in enumerate(lines, 1)
    ]
    return criteria, reserve


def _git_head(workspace: str) -> str:
    if not workspace:
        return ""
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""




def _task_attr(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _load_canonical_task(task_id: str) -> Any:
    """Read a Kanban task through Hermes' canonical board connection.

    Hermes v0.21.3 split connection helpers out of ``kanban_db`` into
    ``kanban_db_connect``. Older builds still expose ``kanban_db.connect``.
    Support both layouts and prefer the dispatcher-pinned DB/board env so a
    headless worker cannot accidentally bind against whatever board happens
    to be globally selected by another controller process.
    """
    from hermes_cli import kanban_db as kb

    db_raw = str(os.getenv("HERMES_KANBAN_DB") or "").strip()
    board = str(os.getenv("HERMES_KANBAN_BOARD") or "").strip() or None
    db_path = Path(db_raw).expanduser() if db_raw else None

    try:
        from hermes_cli import kanban_db_connect as kbc
    except (ImportError, AttributeError):
        kbc = None

    if kbc is not None and hasattr(kbc, "connect_closing"):
        kwargs: dict[str, Any] = {}
        if db_path is not None:
            kwargs["db_path"] = db_path
        elif board is not None:
            kwargs["board"] = board
        with kbc.connect_closing(**kwargs) as con:
            return kb.get_task(con, task_id)

    connect = getattr(kb, "connect", None)
    if connect is None:
        raise RuntimeError(
            "Hermes Kanban connection API unavailable: expected "
            "hermes_cli.kanban_db_connect.connect_closing or hermes_cli.kanban_db.connect"
        )
    con = None
    try:
        try:
            con = connect(db_path=db_path, board=board)
        except TypeError:
            con = connect()
        return kb.get_task(con, task_id)
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _raw_worker_scope(*, session_id: str = "") -> tuple[int, str, str | None] | None:
    raw_run = str(os.getenv("HERMES_KANBAN_RUN_ID") or "").strip()
    claim = str(
        os.getenv("HERMES_KANBAN_CLAIM_LOCK")
        or os.getenv("HERMES_KANBAN_CLAIM_IDENTITY")
        or os.getenv("HERMES_KANBAN_CLAIM_TOKEN")
        or ""
    ).strip()
    if not raw_run or not claim:
        return None
    try:
        run_id = int(raw_run)
    except ValueError:
        return None
    worker = str(os.getenv("HERMES_PROFILE") or os.getenv("HERMES_KANBAN_WORKER_ID") or session_id or "").strip() or None
    return run_id, claim, worker


def _worker_identity(contract_hash: str, *, task_id: str, session_id: str = "") -> RunIdentity | None:
    scope = _raw_worker_scope(session_id=session_id)
    if scope is None:
        return None
    run_id, claim, worker = scope
    return RunIdentity(task_id, run_id, contract_hash, claim, worker)


def _validate_canonical_scope(supervisor, *, task_id: str, task: Any, session_id: str = "") -> tuple[int, str, str | None] | None:
    scope = _raw_worker_scope(session_id=session_id)
    if scope is None:
        supervisor.store.add_diagnostic(
            task_id=task_id, kind="autobind_missing_run_identity", payload={}, created_at=utc_now(),
        )
        return None
    run_id, claim, worker = scope
    canonical_run = _task_attr(task, "current_run_id", None)
    try:
        canonical_run_int = int(canonical_run) if canonical_run is not None else None
    except (TypeError, ValueError):
        canonical_run_int = None
    if canonical_run_int is not None and canonical_run_int != run_id:
        supervisor.store.add_diagnostic(
            task_id=task_id, run_id=run_id, kind="autobind_run_mismatch",
            payload={"canonical_run_id": canonical_run}, created_at=utc_now(),
        )
        return None
    canonical_claim = str(_task_attr(task, "claim_lock", "") or "").strip()
    if canonical_claim and canonical_claim != claim:
        supervisor.store.add_diagnostic(
            task_id=task_id, run_id=run_id, kind="autobind_claim_mismatch",
            payload={"canonical_claim_present": True}, created_at=utc_now(),
        )
        return None
    return scope


def ensure_kanban_binding(supervisor, *, task_id: str = "", session_id: str = "") -> RunIdentity | None:
    cfg = settings()
    if not cfg.get("auto_bind_kanban", True):
        return None
    # The dispatcher-pinned environment is canonical. Hook-local ``task_id``
    # values may be a session/subtask identifier on fallback lifecycle paths.
    env_tid = str(os.getenv("HERMES_KANBAN_TASK_ID") or os.getenv("HERMES_KANBAN_TASK") or "").strip()
    tid = env_tid or str(task_id or "").strip()
    if not tid:
        return None
    # Non-Kanban chat/cron sessions pass session_id or non-card IDs in task_id.
    # Never query Kanban DB or log autobind_task_missing unless env_tid is set or tid is a canonical t_<hex> ID.
    if not env_tid and not _CANONICAL_TASK_ID_RE.match(tid):
        return None
    # Make the canonical worker env visible to the rest of the plugin even on
    # Hermes builds that only set HERMES_KANBAN_TASK. Only propagate a
    # dispatcher-pinned value: a hook-local fallback may be a session id on
    # non-worker processes, and writing it back would make later reads treat
    # it as canonical (issue #60).
    if env_tid:
        os.environ.setdefault("HERMES_KANBAN_TASK_ID", env_tid)

    # Read and fence against canonical Kanban state before creating any Jev
    # contract. A stale/reclaimed worker must not be able to lock a DoD for a
    # run it no longer owns.
    try:
        task = _load_canonical_task(tid)
    except Exception as exc:
        supervisor.store.add_diagnostic(
            task_id=tid, kind="autobind_task_lookup_failed",
            payload={"error": f"{type(exc).__name__}: {exc}"}, created_at=utc_now(),
        )
        return None
    if task is None:
        supervisor.store.add_diagnostic(task_id=tid, kind="autobind_task_missing", payload={}, created_at=utc_now())
        return None
    scope = _validate_canonical_scope(supervisor, task_id=tid, task=task, session_id=session_id)
    if scope is None:
        return None
    run_id, claim, worker = scope

    active = supervisor.active_contract(tid)
    if active is None:
        body = str(_task_attr(task, "body", "") or "")
        dod_lines = extract_dod_lines(body)
        if bool(cfg.get("auto_estimate_task_budget", True)):
            task_budget = estimate_task_budget(
                body,
                len(dod_lines),
                floor_tokens=int(cfg.get("default_task_budget_tokens", 70000)),
                base_tokens=int(cfg.get("budget_estimator_base_tokens", 120000)),
                per_criterion_tokens=int(cfg.get("budget_estimator_per_criterion_tokens", 75000)),
                body_char_factor=float(cfg.get("budget_estimator_body_char_factor", 25.0)),
                safety_multiplier=float(cfg.get("budget_estimator_safety_multiplier", 1.25)),
                max_tokens=int(cfg.get("budget_estimator_max_tokens", 2000000)),
            )
        else:
            task_budget = int(cfg.get("default_task_budget_tokens", 70000))
        criteria, reserve = build_criteria(body, total_budget=task_budget)
        if criteria and bool(cfg.get("budget_dod_required", True)):
            criteria.append(budget_criterion(task_budget, tolerance=float(cfg.get("budget_dod_tolerance", 1.10))))
        if not criteria:
            supervisor.store.add_diagnostic(
                task_id=tid, run_id=run_id, kind="autobind_no_structured_dod",
                payload={"title": str(_task_attr(task, "title", "") or "")}, created_at=utc_now(),
            )
            return None
        out = supervisor.bind_contract(
            task_id=tid,
            goal=str(_task_attr(task, "title", "") or tid),
            criteria=criteria,
            reserve_tokens=reserve,
            checkpoint_fractions=list(parse_checkpoint_fractions(cfg.get("checkpoint_fractions", "0.40,0.70"))),
            actor="orchestrator",
            preflight_result=_LocalAccept(),
        )
        active = supervisor.active_contract(tid)
        if not out.get("accepted") or active is None:
            return None

    ident = RunIdentity(tid, run_id, active.contract_hash, claim, worker)
    current = supervisor.store.current_identity(tid)
    if current != ident:
        supervisor.bind_run(ident)
    os.environ["HERMES_NERVE_DOD_HASH"] = active.contract_hash
    workspace = str(os.getenv("HERMES_KANBAN_WORKSPACE") or os.getenv("TERMINAL_CWD") or "").strip()
    supervisor.store.save_run_context(
        ident,
        workspace_path=workspace,
        base_revision=_git_head(workspace),
        session_id=session_id,
        bound_at=utc_now(),
    )
    return ident


def prune_and_vacuum_supervision_db(
    db_path: Path | str,
    backup_dir: Path | str | None = None,
    *,
    simulate_failure: bool = False,
) -> dict[str, Any]:
    """Backup-guarded live SQLite compaction for work-supervision.sqlite3."""
    import shutil
    import sqlite3
    import time

    target = Path(db_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Supervision DB not found: {target}")

    size_before = target.stat().st_size
    free_bytes = shutil.disk_usage(str(target.parent)).free
    required_free = max(size_before * 3, 1024 * 1024)
    if free_bytes < required_free:
        raise RuntimeError(
            f"Preflight disk space check failed: free={free_bytes}B < required={required_free}B (3x DB size)"
        )

    bdir = Path(backup_dir).expanduser().resolve() if backup_dir else target.parent
    bdir.mkdir(parents=True, exist_ok=True)
    backup_path = bdir / f"{target.stem}.pre-prune-{int(time.time() * 1000)}.sqlite3"

    # Step 1 & 2: Preflight quick_check + baseline counts + online atomic backup
    src_con = sqlite3.connect(str(target), timeout=30.0)
    try:
        src_con.execute("PRAGMA busy_timeout = 30000")
        qc = src_con.execute("PRAGMA quick_check").fetchone()
        if not qc or str(qc[0]).lower() != "ok":
            raise RuntimeError(f"Preflight quick_check failed: {qc}")
        baseline_counts = {
            tbl: int(src_con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
            for tbl in ("contracts", "run_bindings", "run_controls")
        }
        missing_before = int(
            src_con.execute(
                "SELECT COUNT(*) FROM supervision_diagnostics WHERE kind = 'autobind_task_missing'"
            ).fetchone()[0]
        )
        bck_con = sqlite3.connect(str(backup_path), timeout=30.0)
        try:
            src_con.backup(bck_con)
        finally:
            bck_con.close()
    finally:
        src_con.close()

    # Step 3 & 4: Locked delete + VACUUM + post-checks, with automatic rollback on failure
    try:
        work_con = sqlite3.connect(str(target), timeout=30.0, isolation_level=None)
        try:
            work_con.execute("PRAGMA busy_timeout = 30000")
            work_con.execute("BEGIN IMMEDIATE")
            work_con.execute("DELETE FROM supervision_diagnostics WHERE kind = 'autobind_task_missing'")
            work_con.execute("COMMIT")
            if simulate_failure:
                raise RuntimeError("Simulated failure during supervision DB compaction")
            work_con.execute("VACUUM")
            ic = work_con.execute("PRAGMA integrity_check").fetchone()
            if not ic or str(ic[0]).lower() != "ok":
                raise RuntimeError(f"Post-vacuum integrity_check failed: {ic}")
            after_counts = {
                tbl: int(work_con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
                for tbl in ("contracts", "run_bindings", "run_controls")
            }
            if after_counts != baseline_counts:
                raise RuntimeError(
                    f"Control-table counts changed during compaction: before={baseline_counts}, after={after_counts}"
                )
            missing_after = int(
                work_con.execute(
                    "SELECT COUNT(*) FROM supervision_diagnostics WHERE kind = 'autobind_task_missing'"
                ).fetchone()[0]
            )
            if missing_after != 0:
                raise RuntimeError(f"Expected 0 autobind_task_missing rows after prune, found {missing_after}")
        finally:
            work_con.close()
    except Exception:
        # Step 5: Automatic atomic rollback from backup_path
        bck_con = sqlite3.connect(str(backup_path), timeout=30.0)
        dst_con = sqlite3.connect(str(target), timeout=30.0)
        try:
            dst_con.execute("PRAGMA busy_timeout = 30000")
            bck_con.backup(dst_con)
        finally:
            dst_con.close()
            bck_con.close()
        raise

    size_after = target.stat().st_size
    return {
        "ok": True,
        "db_path": str(target),
        "backup_path": str(backup_path),
        "pruned_rows": missing_before,
        "remaining_autobind_task_missing": 0,
        "preserved_counts": baseline_counts,
        "size_before": size_before,
        "size_after": size_after,
        "integrity_check": "ok",
    }

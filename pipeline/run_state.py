"""Durable, atomic state for a single resumable video generation run."""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.models import (
    PaidTakeReservation,
    RunManifest,
    RunOptions,
    PendingTaskDescriptor,
    RunStatus,
    ShotStatus,
    ShotTaskState,
    TakeRecordState,
    validate_storyboard,
)


MANIFEST_FILENAME = "run_manifest.json"
STORYBOARD_FILENAME = "storyboard.json"


class PaidTakeBudgetExhaustedError(RuntimeError):
    """No durable authorization remains for a new provider submission."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_transaction(method):
    """修改必须在同一个锁内重读、计算和提交，嵌套调用复用事务。"""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._thread_lock:
            if self._in_transaction:
                return method(self, *args, **kwargs)
            with self._file_lock(".manifest.lock"):
                self._reload_manifest()
                self._in_transaction = True
                try:
                    return method(self, *args, **kwargs)
                finally:
                    self._in_transaction = False
                    self._reload_manifest()
    return wrapped


def _merge_prompt_attempts(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace a pending checkpoint with the terminal state of the same request."""
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[object, object, object], int] = {}
    for attempt in (*previous, *current):
        key = (
            attempt.get("attempt"),
            attempt.get("fingerprint"),
            attempt.get("provider_task_id"),
        )
        if key in positions:
            merged[positions[key]] = attempt
        else:
            positions[key] = len(merged)
            merged.append(attempt)
    return merged


def _merge_take_history(
    previous: list[TakeRecordState],
    current: TakeRecordState | None,
) -> list[TakeRecordState]:
    history = list(previous)
    if current is None:
        return history
    for index, take in enumerate(history):
        if take.take_id == current.take_id:
            # 允许重新验收同一素材，但提交来源一旦已知便不可被回调覆盖。
            source = {}
            for field in (
                "prompt_fingerprint", "compiled_contract_version",
                "compiled_contract_fingerprint", "model_used", "resolution_used",
            ):
                original = getattr(take, field)
                incoming = getattr(current, field)
                if original and incoming and original != incoming:
                    raise ValueError(f"Take {take.take_id} 的提交来源 {field} 不可更改")
                source[field] = original or incoming
            history[index] = current.model_copy(update=source)
            break
    else:
        history.append(current)
    return history


def _canonical_take(state: ShotTaskState) -> TakeRecordState | None:
    if not state.canonical_take_id:
        return None
    return next(
        (
            take for take in state.take_history
            if take.take_id == state.canonical_take_id
            and take.disposition == "accepted"
        ),
        None,
    )


class RunWorkspace:
    """Own run identity, checkpoints, recovery, and task truth behind one interface."""

    def __init__(self, path: Path, manifest: RunManifest):
        self.path = path.resolve()
        self.manifest_path = self.path / MANIFEST_FILENAME
        self.manifest = manifest
        self._snapshot = manifest.model_dump(mode="json", exclude_none=True)
        self._thread_lock = threading.RLock()
        self._in_transaction = False

    @contextmanager
    def _file_lock(self, name: str, *, nonblocking: bool = False):
        fd = os.open(self.path / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            except BlockingIOError as exc:
                raise RuntimeError("该工作区正在运行；请等待当前执行完成再恢复") from exc
            yield
        finally:
            os.close(fd)

    def _reload_manifest(self) -> None:
        self.manifest = RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        self._snapshot = self.manifest.model_dump(mode="json", exclude_none=True)

    @contextmanager
    def execution(self):
        """跨 CLI、Web 和进程占有整个运行；退出或进程终止自动释放。"""
        with self._file_lock(".execution.lock", nonblocking=True):
            with self._thread_lock, self._file_lock(".manifest.lock"):
                self._reload_manifest()
            yield

    @classmethod
    def create(cls, output_root: Path, options: RunOptions) -> "RunWorkspace":
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output_root / run_id
        path.mkdir(parents=False, exist_ok=False)
        now = _now()
        workspace = cls(
            path,
            RunManifest(
                run_id=run_id,
                created_at=now,
                updated_at=now,
                options=options,
            ),
        )
        workspace._save_manifest()
        return workspace

    @classmethod
    def resume(cls, path: str | Path) -> "RunWorkspace":
        resolved = Path(path).expanduser().resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if not resolved.is_dir():
            raise ValueError(f"Resume workspace does not exist: {resolved}")
        if not manifest_path.is_file():
            raise ValueError(f"Resume workspace has no {MANIFEST_FILENAME}: {resolved}")
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = RunManifest.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid run manifest: {manifest_path}: {exc}") from exc
        return cls(resolved, manifest)

    def authorize_planned_generation(self, limit: int) -> None:
        """显式启动已完成的零预算规划；不扩充已执行任务的额度。"""
        if type(limit) is not int or limit <= 0:
            raise ValueError('授权提交次数必须是正整数')
        with self.execution(), self._thread_lock:
            if (self.manifest.stage != 'planning_ready'
                    or self.manifest.status != RunStatus.interrupted
                    or self.manifest.paid_take_budget.reservations
                    or any(s.provider_task_id or s.pending_task or s.take_history
                           for s in self.manifest.shots.values())
                    or self.options.paid_take_budget not in (0, limit)):
                raise ValueError('只能授权已完成纯规划且尚无视频提交的工作区；不能追加已有付费额度')
            self.load_storyboard(include_observations=False)
            if self.options.paid_take_budget == limit:
                return
            self.manifest.options = RunOptions.model_validate({
                **self.options.model_dump(), 'paid_take_budget': limit})
            # _save_manifest 的锁与快照校验防止并发写入覆盖最新账本。
            self._save_manifest()

    @property
    def options(self) -> RunOptions:
        return self.manifest.options

    def load_storyboard(self, *, include_observations: bool = True) -> dict[str, Any]:
        """Read the frozen plan, optionally enriching it with accepted observations.

        Planning admission must use the frozen view: execution progress is not
        an authored revision. The default preserves existing inspection callers.
        """
        path = self.path / STORYBOARD_FILENAME
        if not path.is_file():
            raise ValueError(f"Resume workspace has no validated storyboard: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid storyboard JSON: {path}: {exc}") from exc
        validated = validate_storyboard(data)
        from pipeline.storyboard import _apply_coverage_defaults

        _apply_coverage_defaults(validated["shots"])
        if not include_observations:
            return validate_storyboard(validated)
        for shot in validated["shots"]:
            state = self.manifest.shots.get(str(shot["shot_id"]))
            if not state:
                continue
            canonical = _canonical_take(state)
            if canonical is not None:
                observed_state = canonical.observed_end_state
            elif state.status == ShotStatus.success:
                observed_state = state.observed_end_state
            else:
                continue
            observed = {
                key: value
                for key, value in observed_state.items()
                if str(value).strip()
            }
            if observed:
                shot["observed_end_state"] = observed
        return validate_storyboard(validated)

    @_manifest_transaction
    def save_storyboard(self, storyboard: dict[str, Any]) -> dict[str, Any]:
        validated = validate_storyboard(storyboard)
        from pipeline.storyboard import _apply_coverage_defaults

        _apply_coverage_defaults(validated["shots"])
        validated = validate_storyboard(validated)
        self._atomic_write_json(self.path / STORYBOARD_FILENAME, validated)
        for shot in validated["shots"]:
            key = str(shot["shot_id"])
            self.manifest.shots.setdefault(key, ShotTaskState(shot_id=shot["shot_id"]))
        self.manifest.paid_take_budget.estimated_takes = len(validated["shots"])
        self.checkpoint("storyboard_ready", RunStatus.running)
        return validated

    @_manifest_transaction
    def check_paid_take(self, shot_id: int) -> None:
        """只读检查当前策略；不占用额度，真正提交前仍须原子预留。"""
        ledger = self.manifest.paid_take_budget
        active = [
            reservation
            for reservation in ledger.reservations
            if reservation.status in {"reserved", "submitted", "reconciled"}
        ]
        limit = self.options.paid_take_budget
        policy = self.options.take_policy
        if policy != 'legacy':
            if str(shot_id) not in self.manifest.shots:
                raise ValueError('提交的镜头不在冻结生产计划内')
            submitted_shots = {reservation.shot_id for reservation in active}
            if policy == 'single_pass' and shot_id in submitted_shots:
                raise PaidTakeBudgetExhaustedError('本次策略每个镜头最多提交一次；原任务与素材已保留')
            if limit is not None:
                future = sum(state.shot_id != shot_id and state.shot_id not in submitted_shots
                             and state.status != ShotStatus.success for state in self.manifest.shots.values())
                if limit - len(active) < 1 + future:
                    raise PaidTakeBudgetExhaustedError(
                        f'剩余额度不足以同时提交本次任务并保留后续 {future} 个镜头的首次机会；未提交新任务')
        if limit is not None and len(active) >= limit:
            raise PaidTakeBudgetExhaustedError(
                f"付费 take 预算已耗尽（{len(active)}/{limit}）；未提交新的 provider 任务"
            )

    @_manifest_transaction
    def reserve_paid_take(self, shot_id: int) -> str:
        """Persist authorization before any new, billable provider submission."""
        self.check_paid_take(shot_id)
        ledger = self.manifest.paid_take_budget
        prior_shot_takes = sum(
            reservation.shot_id == shot_id for reservation in ledger.reservations
        )
        reservation_id = f"{shot_id}:{prior_shot_takes + 1}"
        ledger.reservations.append(
            PaidTakeReservation(
                reservation_id=reservation_id,
                shot_id=shot_id,
                take_number=prior_shot_takes + 1,
            )
        )
        self._save_manifest()
        return reservation_id

    @_manifest_transaction
    def confirm_paid_take_submission(self, reservation_id: str, task_id: str) -> None:
        """Bind a persisted authorization to the provider's immutable task identity."""
        reservation = self._paid_take_reservation(reservation_id)
        if reservation.status != "reserved":
            raise ValueError(f"Paid take reservation is not pending: {reservation_id}")
        reservation.status = "submitted"
        reservation.provider_task_id = task_id
        self._save_manifest()

    @_manifest_transaction
    def reconcile_paid_take(
        self,
        reservation_id: str,
        provider_task_id: str | None = None,
    ) -> None:
        """Close a known provider submission without changing its charge count."""
        reservation = self._paid_take_reservation(reservation_id)
        if reservation.status == "reserved":
            if not provider_task_id:
                raise ValueError(
                    f"Paid take submission has no provider identity: {reservation_id}"
                )
            reservation.status = "submitted"
            reservation.provider_task_id = provider_task_id
        if reservation.status == "submitted":
            if (
                provider_task_id
                and reservation.provider_task_id
                and reservation.provider_task_id != provider_task_id
            ):
                raise ValueError(
                    f"Paid take provider identity changed: {reservation_id}"
                )
            reservation.status = "reconciled"
            self._save_manifest()

    @_manifest_transaction
    def release_unsubmitted_paid_take(self, reservation_id: str) -> None:
        """Release only an authorization known not to have reached the provider."""
        reservation = self._paid_take_reservation(reservation_id)
        if reservation.status != "reserved":
            raise ValueError(
                f"Submitted paid take cannot be released: {reservation_id}"
            )
        reservation.status = "released"
        self._save_manifest()

    def _paid_take_reservation(self, reservation_id: str) -> PaidTakeReservation:
        for reservation in self.manifest.paid_take_budget.reservations:
            if reservation.reservation_id == reservation_id:
                return reservation
        raise ValueError(f"Unknown paid take reservation: {reservation_id}")

    @_manifest_transaction
    def checkpoint(self, stage: str, status: RunStatus = RunStatus.running) -> None:
        self.manifest.stage = stage
        self.manifest.status = status
        self.manifest.error = None
        self._save_manifest()

    @_manifest_transaction
    def record_shot(
        self,
        *,
        shot_id: int,
        status: str,
        provider_task_id: str | None = None,
        provider_error_type: str | None = None,
        provider_error_code: str | None = None,
        provider_error_message: str | None = None,
        provider_error_locus: str | None = None,
        prompt_profile: str | None = None,
        prompt_fingerprint: str | None = None,
        compiled_contract_version: str | None = None,
        compiled_contract_fingerprint: str | None = None,
        accepted_contract_version: str | None = None,
        accepted_contract_fingerprint: str | None = None,
        semantic_evaluator_version: str | None = None,
        acceptance_policy: str | None = None,
        recovery_actions: list[str] | None = None,
        prompt_attempts: list[dict[str, Any]] | None = None,
        rejection_evidence: list[dict[str, Any]] | None = None,
        local_path: str | None = None,
        last_frame_url: str | None = None,
        quality_score: int = 0,
        technical_quality_score: int = 0,
        semantic_accepted: bool | None = None,
        observed_end_state: dict[str, str] | None = None,
        reference_chain_depth: int | None = None,
        model_used: str = "",
        resolution_used: str = "",
        attempts: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        key = str(shot_id)
        previous = self.manifest.shots.get(key)
        relative_path = self._relative_to_workspace(local_path)
        previous_errors = list(previous.errors) if previous else []
        merged_errors = previous_errors + [
            error for error in (errors or []) if error not in previous_errors
        ]
        previous_recovery = list(previous.recovery_actions) if previous else []
        merged_recovery = previous_recovery + [
            action for action in (recovery_actions or [])
            if action not in previous_recovery
        ]
        previous_attempts = (
            [
                attempt.model_dump(mode="json", exclude_none=True)
                for attempt in previous.prompt_attempts
            ]
            if previous else []
        )
        merged_attempts = _merge_prompt_attempts(
            previous_attempts, prompt_attempts or []
        )
        relative_last_frame = self._relative_to_workspace(last_frame_url)
        pending_task = None
        if status == ShotStatus.running.value and provider_task_id:
            descriptor_fields = (
                prompt_profile,
                prompt_fingerprint,
                compiled_contract_version,
                compiled_contract_fingerprint,
            )
            if all(descriptor_fields):
                pending_task = PendingTaskDescriptor(
                    task_id=provider_task_id,
                    prompt_profile=prompt_profile,
                    prompt_fingerprint=prompt_fingerprint,
                    compiled_contract_version=compiled_contract_version,
                    compiled_contract_fingerprint=compiled_contract_fingerprint,
                )
        disposition = None
        shot_status = ShotStatus(status)
        if relative_path and shot_status == ShotStatus.success:
            disposition = "accepted"
        elif relative_path and (
            semantic_accepted is False or shot_status == ShotStatus.failed
        ):
            # 已物化但技术或语义失败的付费结果仍是 Take，只是不能推进 Canonical State。
            disposition = "rejected"
        take = None
        if disposition:
            take = TakeRecordState(
                take_id=f"{shot_id}:{relative_path}",
                disposition=disposition,
                local_path=relative_path,
                last_frame_url=relative_last_frame,
                semantic_accepted=semantic_accepted,
                observed_end_state=observed_end_state or {},
                quality_score=quality_score,
                technical_quality_score=technical_quality_score,
                model_used=model_used,
                resolution_used=resolution_used,
                prompt_fingerprint=prompt_fingerprint,
                compiled_contract_version=compiled_contract_version,
                compiled_contract_fingerprint=compiled_contract_fingerprint,
                accepted_contract_version=accepted_contract_version,
                accepted_contract_fingerprint=accepted_contract_fingerprint,
                semantic_evaluator_version=semantic_evaluator_version,
                acceptance_policy=acceptance_policy,
                rejection_evidence=(
                    rejection_evidence or [] if disposition == "rejected" else []
                ),
                errors=merged_errors,
            )
        take_history = _merge_take_history(
            list(previous.take_history) if previous else [],
            take,
        )
        canonical_take_id = previous.canonical_take_id if previous else None
        if take and take.disposition == "accepted":
            canonical_take_id = take.take_id
        self.manifest.shots[key] = ShotTaskState(
            shot_id=shot_id,
            status=ShotStatus(status),
            provider_task_id=provider_task_id,
            pending_task=pending_task,
            provider_error_type=provider_error_type,
            provider_error_code=provider_error_code,
            provider_error_message=provider_error_message,
            provider_error_locus=provider_error_locus,
            prompt_profile=(
                prompt_profile or (previous.prompt_profile if previous else None)
            ),
            prompt_fingerprint=(
                prompt_fingerprint or (previous.prompt_fingerprint if previous else None)
            ),
            compiled_contract_version=(
                compiled_contract_version
                or (previous.compiled_contract_version if previous else None)
            ),
            compiled_contract_fingerprint=(
                compiled_contract_fingerprint
                or (previous.compiled_contract_fingerprint if previous else None)
            ),
            accepted_contract_version=(
                accepted_contract_version
                or (previous.accepted_contract_version if previous else None)
            ),
            accepted_contract_fingerprint=(
                accepted_contract_fingerprint
                or (previous.accepted_contract_fingerprint if previous else None)
            ),
            semantic_evaluator_version=(
                semantic_evaluator_version
                or (previous.semantic_evaluator_version if previous else None)
            ),
            acceptance_policy=(
                acceptance_policy
                or (previous.acceptance_policy if previous else None)
            ),
            recovery_actions=merged_recovery,
            prompt_attempts=merged_attempts,
            take_history=take_history,
            canonical_take_id=canonical_take_id,
            local_path=relative_path or (previous.local_path if previous else None),
            last_frame_url=relative_last_frame,
            quality_score=quality_score,
            technical_quality_score=technical_quality_score,
            semantic_accepted=semantic_accepted,
            observed_end_state=observed_end_state or {},
            reference_chain_depth=(
                reference_chain_depth
                if reference_chain_depth is not None
                else previous.reference_chain_depth if previous else 0
            ),
            model_used=model_used,
            resolution_used=resolution_used,
            attempts=max(attempts, previous.attempts if previous else 0),
            errors=merged_errors,
        )
        self.manifest.stage = "generating_shots"
        self.manifest.status = RunStatus.running
        self._save_manifest()

    def resumable_provider_tasks(self) -> dict[int, str]:
        """Legacy task-id view; callers must prefer resumable_pending_tasks()."""
        return {
            state.shot_id: state.provider_task_id
            for state in self.manifest.shots.values()
            if state.status == ShotStatus.running and state.provider_task_id
        }

    def ensure_recoverable_submissions(self) -> None:
        """没有任务身份或镜头提交记录的占用不能被当作可重新生成的失败。"""
        for reservation in self.manifest.paid_take_budget.reservations:
            state = self.manifest.shots.get(str(reservation.shot_id))
            if reservation.status == "reserved" or (
                reservation.status == "submitted" and (
                    state is None or not reservation.provider_task_id
                    or state.provider_task_id != reservation.provider_task_id
                )
            ):
                raise RuntimeError(
                    f"Shot {reservation.shot_id} 的提交结果尚未确认；"
                    "已保留预算占用，须核对原任务后恢复，不能自动重新生成。"
                )

    def resumable_pending_tasks(self) -> dict[int, dict[str, Any]]:
        """Return only fully identified submissions that may be polled safely."""
        return {
            state.shot_id: state.pending_task.model_dump(mode="json")
            for state in self.manifest.shots.values()
            if state.status == ShotStatus.running and state.pending_task is not None
        }

    def unresolved_legacy_provider_tasks(self) -> dict[int, str]:
        """Return old running submissions that lack immutable poll provenance.

        These task IDs cannot be safely resumed: polling requires proof of the
        exact submitted prompt and action contract, while submitting again would
        create a second paid provider task.  Callers must stop before generation.
        """
        return {
            state.shot_id: state.provider_task_id
            for state in self.manifest.shots.values()
            if (
                state.status == ShotStatus.running
                and state.provider_task_id
                and state.pending_task is None
            )
        }

    def terminal_materialization_tasks(self) -> dict[int, str]:
        """Return downloaded provider takes that failed local QA/materialization.

        The provider task has already consumed budget and returned footage. It
        is not an authorization to silently create another task on ``--resume``.
        """
        return {
            state.shot_id: state.provider_task_id
            for state in self.manifest.shots.values()
            if (
                state.status == ShotStatus.failed
                and state.provider_task_id
                and state.local_path
            )
        }

    def accepted_shot_artifacts(self) -> dict[int, dict[str, Any]]:
        """Return successful local takes as authoritative resume inputs."""
        accepted: dict[int, dict[str, Any]] = {}
        for state in self.manifest.shots.values():
            canonical = _canonical_take(state)
            if canonical is not None:
                local_value = canonical.local_path
                last_frame_value = canonical.last_frame_url
                semantic_accepted = canonical.semantic_accepted
                observed_end_state = canonical.observed_end_state
                quality_score = canonical.quality_score
                technical_quality_score = canonical.technical_quality_score
                model_used = canonical.model_used
                resolution_used = canonical.resolution_used
                prompt_fingerprint = canonical.prompt_fingerprint
                compiled_contract_version = canonical.compiled_contract_version
                compiled_contract_fingerprint = canonical.compiled_contract_fingerprint
                accepted_contract_version = canonical.accepted_contract_version
                accepted_contract_fingerprint = canonical.accepted_contract_fingerprint
                semantic_evaluator_version = canonical.semantic_evaluator_version
                acceptance_policy = canonical.acceptance_policy
                errors = canonical.errors
            elif state.status == ShotStatus.success and state.local_path:
                local_value = state.local_path
                last_frame_value = state.last_frame_url
                semantic_accepted = state.semantic_accepted
                observed_end_state = state.observed_end_state
                quality_score = state.quality_score
                technical_quality_score = state.technical_quality_score
                model_used = state.model_used
                resolution_used = state.resolution_used
                prompt_fingerprint = state.prompt_fingerprint
                compiled_contract_version = state.compiled_contract_version
                compiled_contract_fingerprint = state.compiled_contract_fingerprint
                accepted_contract_version = state.accepted_contract_version
                accepted_contract_fingerprint = state.accepted_contract_fingerprint
                semantic_evaluator_version = state.semantic_evaluator_version
                acceptance_policy = state.acceptance_policy
                errors = state.errors
            else:
                continue
            local_path = self._resolve_artifact(local_value)
            if not local_path or not Path(local_path).is_file():
                continue
            accepted[state.shot_id] = {
                "local_path": local_path,
                "last_frame_url": self._resolve_artifact(last_frame_value),
                "quality_score": quality_score,
                "technical_quality_score": technical_quality_score,
                "semantic_accepted": semantic_accepted,
                "observed_end_state": dict(observed_end_state),
                "reference_chain_depth": state.reference_chain_depth,
                "model_used": model_used,
                "resolution_used": resolution_used,
                "attempts": state.attempts,
                "errors": list(errors),
                "provider_error_locus": state.provider_error_locus,
                "prompt_profile": state.prompt_profile,
                "prompt_fingerprint": prompt_fingerprint,
                "compiled_contract_version": compiled_contract_version,
                "compiled_contract_fingerprint": compiled_contract_fingerprint,
                "accepted_contract_version": accepted_contract_version,
                "accepted_contract_fingerprint": accepted_contract_fingerprint,
                "semantic_evaluator_version": semantic_evaluator_version,
                "acceptance_policy": acceptance_policy,
                "recovery_actions": list(state.recovery_actions),
                "prompt_attempts": [
                    attempt.model_dump(mode="json", exclude_none=True)
                    for attempt in state.prompt_attempts
                ],
            }
        return accepted

    @_manifest_transaction
    def mark_failed(self, error: str) -> None:
        self.manifest.status = RunStatus.failed
        self.manifest.error = error
        self._save_manifest()

    @_manifest_transaction
    def mark_interrupted(self, error: str = "Run interrupted") -> None:
        self.manifest.status = RunStatus.interrupted
        self.manifest.error = error
        self._save_manifest()

    @_manifest_transaction
    def mark_succeeded(self, final_path: str) -> None:
        self.manifest.status = RunStatus.succeeded
        self.manifest.stage = "completed"
        self.manifest.final_path = self._relative_to_workspace(final_path)
        self.manifest.error = None
        self._save_manifest()

    def completed_output(self) -> str | None:
        if self.manifest.status != RunStatus.succeeded or not self.manifest.final_path:
            return None
        path = self.path / self.manifest.final_path
        return str(path) if path.is_file() and path.stat().st_size > 0 else None

    def _relative_to_workspace(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        try:
            return str(path.resolve().relative_to(self.path))
        except ValueError:
            return value

    def _resolve_artifact(self, value: str | None) -> str | None:
        if not value or value.startswith(("http://", "https://", "data:")):
            return value
        path = Path(value)
        return str(path if path.is_absolute() else self.path / path)

    def _save_manifest(self) -> None:
        with self._thread_lock:
            lock = nullcontext() if self._in_transaction else self._file_lock(".manifest.lock")
            with lock:
                # 兼容首次创建及显式保存；拒绝过期快照覆盖已提交状态。
                if not self._in_transaction and self.manifest_path.exists():
                    stored = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    if stored != self._snapshot:
                        raise RuntimeError("运行状态已变化；不能覆盖最新记录")
                self.manifest.updated_at = _now()
                snapshot = self.manifest.model_dump(mode="json", exclude_none=True)
                self._atomic_write_json(self.manifest_path, snapshot)
                self._snapshot = snapshot

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
            # 文件内容与目录中的替换记录都落盘后，才允许调用方提交远端任务。
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(name).unlink(missing_ok=True)

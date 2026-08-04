"""Stage 2: 视频生成 — 角色一致性 + 降级策略 + 即时下载"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from typing import Optional

import config
from tools.seedance_api import (
    SeedanceAPI,
    GenerationResult,
    GenerationStatus,
    SubmittedTaskCheckpointError,
)
from tools.frame_extractor import extract_frame, check_video_quality
from pipeline.storyboard import (
    _apply_coverage_defaults,
    _normalize_continuity_contract,
    _scene_id,
    _should_use_previous_tail_reference,
)
from pipeline.participants import visible_character_names
from pipeline.causality import ACTION_CONTRACT_VERSION, compile_action_contract
from pipeline.provider_prompt import (
    compile_normal_provider_prompt,
    compile_policy_safe_prompt,
    has_explicit_action_contract,
)
from pipeline.semantic_review import (
    SEMANTIC_REVIEW_VERSION,
    SemanticReview,
    SemanticReviewUnavailableError,
    SemanticTakeReviewer,
)
from pipeline.readiness import (
    GenerationReadinessError,
    ensure_shot_ready,
    ensure_storyboard_ready,
)


@dataclass
class ShotResult:
    shot_id: int
    status: str = "pending"
    local_path: Optional[str] = None
    last_frame_url: Optional[str] = None
    character_ref_path: Optional[str] = None
    quality_score: int = 0
    model_used: str = ""
    resolution_used: str = ""
    attempts: int = 0
    errors: list = field(default_factory=list)
    provider_task_id: Optional[str] = None
    provider_error_type: Optional[str] = None
    provider_error_code: Optional[str] = None
    provider_error_message: Optional[str] = None
    provider_error_locus: Optional[str] = None
    prompt_profile: Optional[str] = None
    prompt_fingerprint: Optional[str] = None
    compiled_contract_version: Optional[str] = None
    compiled_contract_fingerprint: Optional[str] = None
    accepted_contract_version: Optional[str] = None
    accepted_contract_fingerprint: Optional[str] = None
    semantic_evaluator_version: Optional[str] = None
    acceptance_policy: Optional[str] = None
    recovery_actions: list[str] = field(default_factory=list)
    prompt_attempts: list[dict] = field(default_factory=list)
    technical_quality_score: int = 0
    semantic_accepted: Optional[bool] = None
    observed_end_state: dict[str, str] = field(default_factory=dict)


class RemoteTaskPendingError(RuntimeError):
    """A paid remote task is still unresolved, so the run must pause."""

    def __init__(self, result: ShotResult, workspace: Path):
        self.result = result
        self.workspace = workspace
        task_id = result.provider_task_id or "unknown"
        super().__init__(
            f"远端任务 {task_id} 仍在处理，流水线已暂停且不会提交后续镜头；"
            f"请稍后运行 python main.py --resume {workspace}"
        )


class ProgressPersistenceError(RuntimeError):
    """Run state could not be persisted; continuing could duplicate paid work."""


class VideoGenerator:
    """视频生成引擎 — 角色一致性优先 + 画面衔接"""

    def __init__(
        self,
        output_dir: str,
        on_progress: Callable[[ShotResult], None] | None = None,
        resume_tasks: Mapping[int, Mapping[str, object]] | None = None,
        resume_task_ids: Mapping[int, str] | None = None,
        accepted_shot_artifacts: Mapping[int, Mapping[str, object]] | None = None,
    ):
        self.api = SeedanceAPI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "shots").mkdir(exist_ok=True)
        (self.output_dir / "character_refs").mkdir(exist_ok=True)
        self.character_refs: dict[str, str] = {}  # name → 本地图片路径
        self.character_ref_hashes: dict[str, str] = {}  # sha256 → character name
        self.on_progress = on_progress
        self.resume_tasks = {
            int(shot_id): dict(descriptor)
            for shot_id, descriptor in (resume_tasks or {}).items()
        }
        self._legacy_resume_task_ids: dict[int, str] = {}
        # Kept only so callers on the older constructor fail closed instead of
        # silently polling an identity that lacks prompt/contract provenance.
        self.resume_task_ids = resume_task_ids or {}
        self.accepted_shot_artifacts = {
            int(shot_id): dict(artifact)
            for shot_id, artifact in (accepted_shot_artifacts or {}).items()
        }
        self.semantic_reviewer = (
            SemanticTakeReviewer(
                self.output_dir, model=config.SEMANTIC_REVIEW_MODEL
            )
            if config.SEMANTIC_REVIEW_ENABLED
            else None
        )

    @property
    def resume_task_ids(self) -> dict[int, str]:
        """Compatibility view for callers that only retained a provider task ID."""
        return dict(self._legacy_resume_task_ids)

    @resume_task_ids.setter
    def resume_task_ids(self, task_ids: Mapping[int, str]) -> None:
        self._legacy_resume_task_ids = {
            int(shot_id): str(task_id)
            for shot_id, task_id in task_ids.items()
        }
        for shot_id, task_id in self._legacy_resume_task_ids.items():
            self.resume_tasks.setdefault(shot_id, {"task_id": task_id})

    def _notify_progress(self, result: ShotResult) -> None:
        if self.on_progress:
            try:
                self.on_progress(result)
            except Exception as exc:
                raise ProgressPersistenceError(
                    f"Shot {result.shot_id} 状态保存失败，已停止以避免重复生成: {exc}"
                ) from exc

    async def generate_all(self, storyboard: dict) -> list[ShotResult]:
        """
        Generate shots strictly in canonical order.

        角色一致性策略:
        ┌─────────────────────────────────────────────────────────────────┐
        │ 合格镜头: 从已验收中点帧裁出无污染角色身份参考                  │
        │ seamless: first_frame (仅上一尾帧，锁定真实起始状态)            │
        │ 同场景切镜: 已验收尾帧 first_frame，锁定真实起始状态         │
        │ 跨场景 intentional_cut: 仅 canonical 角色身份                  │
        └─────────────────────────────────────────────────────────────────┘
        """
        shots = storyboard["shots"]
        self._normalize_continuity(shots)
        ensure_storyboard_ready(storyboard)
        results: list[ShotResult] = [None] * len(shots)  # type: ignore
        prev_last_frame: Optional[str] = None
        prev_shot: Optional[dict] = None

        for idx, shot in enumerate(shots):
            print(f"\n  🎬 生成 Shot {shot['shot_id']}...")
            incoming_last_frame = prev_last_frame
            result = await self._generate_single_shot(
                shot=shot,
                prev_last_frame=prev_last_frame,
                prev_shot=prev_shot,
                storyboard=storyboard,
            )
            results[idx] = result

            if result.status == "success":
                shot["output_reference_depth"] = self._next_reference_depth(
                    shot, prev_shot, incoming_last_frame
                )
                prev_last_frame = result.last_frame_url
                if result.observed_end_state:
                    shot["observed_end_state"] = result.observed_end_state

                semantic_label = (
                    "通过" if result.semantic_accepted is True
                    else "未启用" if result.semantic_accepted is None
                    else "未通过"
                )
                print(
                    f"     ✓ 完成 (技术质量: {result.technical_quality_score}, "
                    f"语义验收: {semantic_label}, 模型: {result.model_used})"
                )
                # 提取角色参考帧
                if shot.get("extract_character_ref") and not self.semantic_reviewer:
                    await self._extract_character_ref(shot, result.local_path, storyboard)
                prev_shot = shot
            elif result.status == "running":
                print(
                    f"     ⏸ 远端任务 {result.provider_task_id or 'unknown'} "
                    "状态未决，暂停后续镜头"
                )
                raise RemoteTaskPendingError(result, self.output_dir)
            else:
                print(f"     ✗ 失败: {result.errors[-1] if result.errors else 'unknown'}")
                raise RuntimeError(
                    f"Shot {shot['shot_id']} 生成失败，已停止后续镜头；"
                    "失败镜头不会成为连续性参考"
                )

        return results

    @staticmethod
    def _normalize_continuity(shots: list[dict]) -> None:
        """生成入口再次收紧契约，覆盖旧工作区和外部分镜。"""
        for correction in _normalize_continuity_contract(shots):
            print(f"  [连续性校正] {correction}，改为 intentional_cut")
        _apply_coverage_defaults(shots)

    def _find_independent_shots(self, shots: list[dict]) -> set[int]:
        """识别可独立生成的镜头 (无角色 + 有意切镜 = 纯 T2V, 不依赖前帧)

        保守策略: 只在确定无依赖时才标记为独立, 避免影响画面一致性。
        """
        independent = set()
        for i in range(1, len(shots)):
            shot = shots[i]
            prev = shots[i - 1]

            # 条件 1: 无角色 (insert shot)
            if shot.get("characters"):
                continue
            # 条件 2: 同场景镜头依赖上一镜已接受尾帧，即使摄影机切镜。
            if _scene_id(shot) == _scene_id(prev):
                continue
            # 条件 3: 剪辑契约不依赖上一镜头的尾帧
            continuity = shot.get("continuity_from_previous")
            if continuity == "seamless":
                continue
            # 条件 4: 非首镜 (首镜不需要预生成, 本来就是 T2V)
            independent.add(i)

        if independent:
            ids = [shots[i]["shot_id"] for i in sorted(independent)]
            print(f"  [OPT] 识别到 {len(independent)} 个独立镜头可预生成: Shot {ids}")
        return independent

    @staticmethod
    def _shot_has_character(shot: dict) -> bool:
        """判断镜头是否含角色 (用于设定角色锚点)"""
        return bool(
            shot.get("extract_character_ref")
            or shot.get("characters")
            or shot.get("has_character")
        )

    async def _generate_single_shot(
        self, shot: dict, prev_last_frame: Optional[str],
        prev_shot: Optional[dict], storyboard: dict
    ) -> ShotResult:
        """生成单个镜头 (含缓存 + 降级链 + 429 限流退避)"""

        requested_resolution = storyboard.get("resolution", config.DEFAULT_RESOLUTION)
        self._reconcile_start_state(shot, prev_shot)
        ensure_shot_ready(
            shot,
            previous_frame=prev_last_frame,
            previous_shot=prev_shot,
            character_refs=self.character_refs,
        )
        expected_refs, expected_role = self._build_image_refs(
            shot, prev_last_frame, prev_shot
        )
        state_reference_required = self._has_state_reference(
            shot, prev_shot, prev_last_frame, expected_refs, expected_role
        )
        resume_descriptor = self.resume_tasks.pop(shot["shot_id"], None)
        resume_task_id = (
            str(resume_descriptor.get("task_id", "")).strip()
            if isinstance(resume_descriptor, dict)
            else None
        ) or None
        restored = self.accepted_shot_artifacts.pop(shot["shot_id"], None)
        if restored and not self._restored_provenance_matches(
            restored,
            str(restored["local_path"]), shot, expected_refs, expected_role
        ):
            result = ShotResult(
                shot_id=shot["shot_id"],
                status="failed",
                local_path=str(restored["local_path"]),
                errors=[
                    "已接受镜头的编译合同、有效提示词或状态参考与当前请求不匹配；"
                    "必须由当前语义验收器离线复核，且不会在 resume 中隐式提交新的付费生成任务"
                ],
            )
            if self.semantic_reviewer and Path(result.local_path).is_file():
                review = await self._review_rejected_take(
                    result,
                    shot,
                    Path(result.local_path),
                    previous_frame_path=prev_last_frame,
                    previous_shot=prev_shot,
                    storyboard=storyboard,
                )
                if review and review.accepted:
                    result.status = "success"
                    result.semantic_accepted = True
                    result.observed_end_state = review.observed_end_state
                    result.accepted_contract_version = ACTION_CONTRACT_VERSION
                    result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
                    result.semantic_evaluator_version = SEMANTIC_REVIEW_VERSION
                    result.acceptance_policy = "semantic_reviewed"
                    self._write_acceptance_context(
                        shot["shot_id"],
                        result.local_path,
                        shot,
                        expected_refs,
                        expected_role,
                    )
                    result.last_frame_url = self._extract_local_tail_frame(
                        shot["shot_id"], result.local_path
                    )
                    self._register_identity_crops(
                        shot_id=shot["shot_id"],
                        video_path=result.local_path,
                        crop_boxes=review.identity_crop_boxes,
                    )
                    result.recovery_actions.append("offline_revalidate_accepted_take")
                    result.errors = []
                    self._notify_progress(result)
                    return result
            self._notify_progress(result)
            return result
        if restored:
            result = ShotResult(
                shot_id=shot["shot_id"],
                status="success",
                local_path=str(restored["local_path"]),
                last_frame_url=(
                    str(restored["last_frame_url"])
                    if restored.get("last_frame_url")
                    else None
                ),
                quality_score=int(restored.get("quality_score", 0)),
                technical_quality_score=int(
                    restored.get("technical_quality_score", 0)
                ),
                semantic_accepted=restored.get("semantic_accepted"),
                observed_end_state=dict(restored.get("observed_end_state", {})),
                model_used=str(restored.get("model_used", "cached")),
                resolution_used=str(restored.get("resolution_used", "")),
                attempts=int(restored.get("attempts", 0)),
                errors=list(restored.get("errors", [])),
                provider_error_locus=restored.get("provider_error_locus"),
                prompt_profile=restored.get("prompt_profile"),
                prompt_fingerprint=restored.get("prompt_fingerprint"),
                compiled_contract_version=restored.get("compiled_contract_version"),
                compiled_contract_fingerprint=restored.get("compiled_contract_fingerprint"),
                accepted_contract_version=restored.get("accepted_contract_version"),
                accepted_contract_fingerprint=restored.get("accepted_contract_fingerprint"),
                semantic_evaluator_version=restored.get("semantic_evaluator_version"),
                acceptance_policy=restored.get("acceptance_policy"),
                recovery_actions=list(restored.get("recovery_actions", [])),
                prompt_attempts=list(restored.get("prompt_attempts", [])),
            )
            if not result.last_frame_url or not Path(result.last_frame_url).is_file():
                result.last_frame_url = self._extract_local_tail_frame(
                    shot["shot_id"], result.local_path
                )
            if self.semantic_reviewer and result.semantic_accepted is True:
                accepted_boxes = self.semantic_reviewer.accepted_identity_crop_boxes(
                    result.local_path
                )
                if accepted_boxes:
                    self._register_identity_crops(
                        shot_id=shot["shot_id"],
                        video_path=result.local_path,
                        crop_boxes=accepted_boxes,
                    )
            self._notify_progress(result)
            print("     ♻️ 恢复已接受镜头，不重新生成或重新判定")
            return result

        result = ShotResult(
            shot_id=shot["shot_id"],
            status="running",
            compiled_contract_version=ACTION_CONTRACT_VERSION,
            compiled_contract_fingerprint=self._compiled_contract_fingerprint(shot),
            provider_task_id=resume_task_id,
        )
        if resume_descriptor is None:
            self._notify_progress(result)
        semantic_failure = ""
        rejected_takes = self._unique_rejected_takes(shot["shot_id"])
        rejected_provider_task_ids = self._rejected_provider_task_ids(
            shot["shot_id"]
        )
        if resume_task_id in rejected_provider_task_ids:
            resume_task_id = None
            resume_descriptor = None
            result.provider_task_id = None
        semantic_retake_count = len(rejected_takes)
        if semantic_retake_count >= 2:
            if self.semantic_reviewer:
                return await self._reassess_latest_rejected_take(
                    result,
                    shot,
                    rejected_takes[-1],
                    previous_frame_path=prev_last_frame,
                    previous_shot=prev_shot,
                    storyboard=storyboard,
                    image_urls=expected_refs,
                    image_role=expected_role,
                )
            result.status = "failed"
            result.errors.append(
                "语义验收已达到上限（原始 take + 1 次定向重拍），"
                "必须修改分镜契约后创建新运行"
            )
            self._notify_progress(result)
            return result
        if semantic_retake_count == 1:
            if not self.semantic_reviewer:
                result.status = "failed"
                result.local_path = str(rejected_takes[0])
                result.errors.append(
                    "已有 rejected take，但语义验收器未启用；为避免无依据消耗唯一重拍预算，已停止"
                )
                self._notify_progress(result)
                return result
            review = await self._review_rejected_take(
                result,
                shot,
                rejected_takes[0],
                previous_frame_path=prev_last_frame,
                previous_shot=prev_shot,
                storyboard=storyboard,
            )
            if review is None:
                result.status = "failed"
                self._notify_progress(result)
                return result
            if review.accepted:
                return self._promote_rejected_take(
                    result,
                    shot,
                    rejected_takes[0],
                    review,
                    image_urls=expected_refs,
                    image_role=expected_role,
                )
            semantic_failure = (
                review.failure_reason or "镜头未满足动作与空间契约"
            )
            result.errors.append(f"历史镜头语义验收不通过: {semantic_failure}")
            result.local_path = str(rejected_takes[0])
            self._notify_progress(result)

        # ─── 缓存检查: dependent takes need matching state provenance ───
        cached_path = str(
            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
        )
        cache_is_compatible = (
            not state_reference_required
            or self._provenance_matches(
                cached_path, shot, expected_refs, expected_role
            )
        )
        if (
            Path(cached_path).exists()
            and Path(cached_path).stat().st_size > 0
            and cache_is_compatible
        ):
            qa = check_video_quality(cached_path)
            if qa["pass"]:
                result.status = "success"
                result.local_path = cached_path
                result.quality_score = qa["quality_score"]
                result.technical_quality_score = qa["quality_score"]
                result.model_used = "cached"
                cache_accepted = True
                if self.semantic_reviewer:
                    review = await self._review_take(
                        cached_path,
                        shot,
                        previous_frame_path=prev_last_frame,
                        previous_shot=prev_shot,
                        storyboard=storyboard,
                    )
                    result.semantic_accepted = review.accepted
                    result.observed_end_state = review.observed_end_state
                    if review.accepted:
                        result.accepted_contract_version = ACTION_CONTRACT_VERSION
                        result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
                        result.semantic_evaluator_version = SEMANTIC_REVIEW_VERSION
                        result.acceptance_policy = "semantic_reviewed"
                        self._write_acceptance_context(
                            shot["shot_id"], cached_path, shot, expected_refs, expected_role
                        )
                        self._register_identity_crops(
                            shot_id=shot["shot_id"],
                            video_path=cached_path,
                            crop_boxes=review.identity_crop_boxes,
                        )
                    if not review.accepted:
                        reason = review.failure_reason or "镜头未满足动作与空间契约"
                        result.errors.append(f"缓存镜头语义验收不通过: {reason}")
                        rejected_path = self._preserve_rejected_take(
                            shot["shot_id"], cached_path
                        )
                        result.local_path = rejected_path
                        result.last_frame_url = None
                        cache_accepted = False
                        if semantic_retake_count >= 1:
                            result.status = "failed"
                            self._notify_progress(result)
                            return result
                        semantic_retake_count += 1
                        semantic_failure = reason
                        result.status = "running"
                        result.provider_task_id = None
                        resume_task_id = None
                        # The local take proves this submitted task reached a
                        # terminal result and was rejected. A retake is a new
                        # contract-bound submission, never a poll of that task.
                        resume_descriptor = None
                        self._notify_progress(result)
                        print(
                            "     ⚠ 缓存镜头语义验收不通过，执行唯一一次定向重拍: "
                            f"{reason}"
                        )

                if cache_accepted:
                    if not self.semantic_reviewer:
                        result.accepted_contract_version = ACTION_CONTRACT_VERSION
                        result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
                        result.acceptance_policy = "technical_only"
                        self._write_acceptance_context(
                            shot["shot_id"],
                            cached_path,
                            shot,
                            expected_refs,
                            expected_role,
                            policy="technical_only",
                        )
                    # 从缓存视频提取尾帧, 供后续镜头衔接。
                    lastframe_path = str(
                        self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}_lastframe.jpg"
                    )
                    try:
                        from tools.ffmpeg_ops import get_video_duration
                        dur = get_video_duration(cached_path)
                        extract_frame(
                            cached_path, lastframe_path, timestamp=max(0, dur - 0.1)
                        )
                        result.last_frame_url = lastframe_path
                    except Exception:
                        result.last_frame_url = None

                    print(
                        f"     ♻️ 缓存命中: {cached_path} "
                        f"(技术质量: {qa['quality_score']})"
                    )
                    if shot.get("extract_character_ref") and not self.semantic_reviewer:
                        await self._extract_character_ref(shot, cached_path, storyboard)

                    self._notify_progress(result)
                    return result
            else:
                print(f"     ⚠ 缓存文件 QA 不通过 ({qa['issues']}), 重新生成")
        elif Path(cached_path).exists() and Path(cached_path).stat().st_size > 0:
            result.status = "failed"
            result.local_path = cached_path
            result.errors.append(
                "本地镜头的状态参考溯源与当前合同不匹配；已停止，"
                "缓存冲突本身不授权提交新的付费生成任务"
            )
            self._notify_progress(result)
            return result

        rate_limit_backoff = 0
        max_rate_limit_retries = 3
        copyright_retry_used = False
        input_text_recompile_used = False

        use_refs = True  # 是否使用参考图 (失败后会关闭)
        skip_char_refs = False  # 隐私审核失败后, 丢弃角色参考帧但保留尾帧衔接
        for level, deg_config in enumerate(
            config.GENERATION_CHAINS[requested_resolution]
        ):
            if rate_limit_backoff >= max_rate_limit_retries:
                break

            for attempt in range(3):
                result.attempts += 1
                result.model_used = deg_config["model"]
                result.resolution_used = deg_config["resolution"]

                try:
                    # 先选择参考职责，prompt 才能准确说明每张图的权限边界。
                    if not use_refs:
                        image_urls, role = [], None
                    elif skip_char_refs:
                        image_urls, role = self._build_state_only_refs(
                            shot, prev_last_frame, prev_shot
                        )
                    else:
                        image_urls, role = self._build_image_refs(
                            shot, prev_last_frame, prev_shot
                        )
                    has_state_reference = self._has_state_reference(
                        shot, prev_shot, prev_last_frame, image_urls, role
                    )
                    resuming_existing_task = resume_descriptor is not None
                    if resuming_existing_task:
                        if not self._pending_descriptor_matches(
                            resume_descriptor,
                            shot,
                            image_urls,
                            role,
                        ):
                            result.status = "failed"
                            result.errors.append(
                                "待恢复远端任务的提交描述符、编译合同、有效提示词或参考职责与当前请求不匹配；"
                                "已停止且未轮询、未提交新的付费任务"
                            )
                            self._notify_progress(result)
                            return result
                        prompt_profile = str(resume_descriptor["prompt_profile"])
                        prompt_fingerprint = str(resume_descriptor["prompt_fingerprint"])
                        # Polling has no prompt. The submitted prompt remains
                        # immutable in the persisted descriptor and lineage.
                        prompt = ""
                    else:
                        prompt_profile = (
                            "policy_safe" if input_text_recompile_used else "normal"
                        )
                        prompt = ""
                    if not resuming_existing_task and prompt_profile == "policy_safe":
                        prompt = compile_policy_safe_prompt(
                            shot,
                            storyboard=storyboard,
                            has_state_reference=has_state_reference,
                            image_role=role,
                            reference_count=len(image_urls),
                            retake_instruction=(
                                compile_action_contract(shot).safe_retake_instruction()
                                if semantic_failure
                                else None
                            ),
                        )
                    elif not resuming_existing_task:
                        if has_explicit_action_contract(shot):
                            prompt = compile_normal_provider_prompt(
                                shot,
                                storyboard,
                                has_observed_start=has_state_reference,
                            )
                        else:
                            # Legacy persisted storyboards have no phase owner.
                            prompt = self._inject_character_description(
                                shot["prompt_en"], shot, storyboard
                            )
                            prompt = self._inject_scene_continuity(
                                prompt, shot, prev_shot
                            )
                            prompt = self._inject_shot_contract(
                                prompt, shot, has_observed_start=has_state_reference
                            )
                        if semantic_failure:
                            prompt = (
                                compile_action_contract(shot).retake_instruction(
                                    semantic_failure
                                )
                                + " "
                                + prompt
                            )
                        if copyright_retry_used:
                            prompt = self._inject_copyright_boundary(prompt)
                        if not has_explicit_action_contract(shot) and shot.get("negative_prompt"):
                            prompt += f". {shot['negative_prompt']}"

                        prompt = self._inject_reference_scope(
                            prompt,
                            role,
                            reference_count=len(image_urls),
                            has_state_reference=has_state_reference,
                        )
                    if not resuming_existing_task:
                        prompt_fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
                    generation_provenance = self._generation_provenance(
                        shot,
                        image_urls,
                        role,
                        prompt_profile=prompt_profile,
                        prompt_fingerprint=prompt_fingerprint,
                    )
                    result.prompt_profile = prompt_profile
                    result.prompt_fingerprint = prompt_fingerprint
                    prompt_attempt = {
                        "attempt": result.attempts,
                        "profile": prompt_profile,
                        "fingerprint": prompt_fingerprint,
                        "outcome": "pending",
                    }
                    result.prompt_attempts.append(prompt_attempt)

                    # 调用 API
                    def remember_submission(task_id: str) -> None:
                        self._write_generation_provenance(
                            shot["shot_id"], generation_provenance, task_id
                        )
                        result.provider_task_id = task_id
                        prompt_attempt["provider_task_id"] = task_id
                        self._notify_progress(result)

                    if resuming_existing_task:
                        gen_result = await self._poll_existing_task(
                            resume_task_id or ""
                        )
                    else:
                        gen_result = await self.api.generate(
                            prompt=prompt,
                            duration=min(shot["duration"], deg_config["max_duration"]),
                            ratio=storyboard.get("aspect_ratio", "16:9"),
                            resolution=deg_config["resolution"],
                            model=deg_config["model"],
                            generate_audio=shot.get("generate_audio", True),
                            return_last_frame=self.api.supports_last_frame,
                            image_urls=image_urls if image_urls else None,
                            image_role=role,
                            timeout=config.GENERATION_TIMEOUT,
                            task_id=None,
                            on_submitted=remember_submission,
                        )
                    resume_task_id = None
                    resume_descriptor = None
                    returned_task_id = gen_result.get("provider_task_id")
                    if returned_task_id:
                        result.provider_task_id = str(returned_task_id)
                        prompt_attempt["provider_task_id"] = result.provider_task_id
                        self._notify_progress(result)
                    if self._is_privacy_failure(gen_result):
                        gen_result["error_type"] = "privacy"
                    if gen_result["status"] == "succeeded":
                        result.provider_error_type = None
                        result.provider_error_code = None
                        result.provider_error_message = None
                        result.provider_error_locus = None
                        prompt_attempt["outcome"] = "succeeded"
                    else:
                        result.provider_error_type = str(
                            gen_result.get("error_type", "unknown")
                        )
                        result.provider_error_code = (
                            str(gen_result["error_code"])
                            if gen_result.get("error_code") else None
                        )
                        result.provider_error_message = str(
                            gen_result.get("error", "unknown")
                        )
                        result.provider_error_locus = str(
                            gen_result.get("error_locus", "unknown")
                        )
                        prompt_attempt["outcome"] = (
                            "pending"
                            if gen_result.get("error_type") in {"poll_error", "poll_timeout"}
                            else "failed"
                        )
                        prompt_attempt["provider_error_locus"] = (
                            result.provider_error_locus
                        )
                        if result.provider_error_code:
                            prompt_attempt["provider_error_code"] = (
                                result.provider_error_code
                            )

                    if (
                        resuming_existing_task
                        and gen_result["status"] != "succeeded"
                        and gen_result.get("error_type") not in {"poll_error", "poll_timeout"}
                    ):
                        result.status = "failed"
                        result.errors.append(
                            "恢复的远端任务已返回终态失败；不会基于当前状态重新提交新任务"
                        )
                        self._notify_progress(result)
                        return result

                    if gen_result["status"] == "succeeded":
                        # Persist immutable submission lineage before any local
                        # materialization. A download outage must be resumable
                        # as the same remote task, never a new paid submission.
                        self._write_generation_provenance(
                            shot["shot_id"],
                            generation_provenance,
                            result.provider_task_id,
                        )
                        # ⚡ 立即下载
                        local_path = str(
                            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
                        )
                        try:
                            await self.api.download_video(
                                gen_result["video_url"], local_path
                            )
                        except Exception as exc:
                            result.status = "running"
                            result.errors.append(
                                "远端任务已成功，但本地下载暂不可用；已保留同一任务身份，"
                                f"恢复时只会继续获取该结果: {exc}"
                            )
                            self._notify_progress(result)
                            return result
                        result.local_path = local_path
                        result.last_frame_url = gen_result.get("last_frame_url")

                        # 质量检测
                        try:
                            qa = check_video_quality(local_path)
                        except Exception as exc:
                            result.status = "failed"
                            result.errors.append(
                                "远端任务已成功，但本地技术 QA 异常；已停止且不会重新提交: "
                                f"{exc}"
                            )
                            self._notify_progress(result)
                            return result
                        result.quality_score = qa["quality_score"]
                        result.technical_quality_score = qa["quality_score"]
                        self._write_generation_provenance(
                            shot["shot_id"], generation_provenance,
                            result.provider_task_id,
                        )

                        if not qa["pass"]:
                            result.errors.append(f"QA 不通过: {qa['issues']}")
                            result.status = "failed"
                            self._notify_progress(result)
                            print(f"     ⚠ QA 不通过: {qa['issues']}, 停止以避免重复提交")
                            return result

                        # Persist the downloaded take before remote semantic review.
                        # If review is unavailable, resume reuses this local video.
                        result.status = "running"
                        self._notify_progress(result)

                        if self.semantic_reviewer:
                            review = await self._review_take(
                                local_path,
                                shot,
                                previous_frame_path=prev_last_frame,
                                previous_shot=prev_shot,
                                storyboard=storyboard,
                            )
                            result.semantic_accepted = review.accepted
                            result.observed_end_state = review.observed_end_state
                            if review.accepted:
                                result.accepted_contract_version = ACTION_CONTRACT_VERSION
                                result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
                                result.semantic_evaluator_version = SEMANTIC_REVIEW_VERSION
                                result.acceptance_policy = "semantic_reviewed"
                                self._write_acceptance_context(
                                    shot["shot_id"],
                                    local_path,
                                    shot,
                                    image_urls,
                                    role,
                                )
                                self._register_identity_crops(
                                    shot_id=shot["shot_id"],
                                    video_path=local_path,
                                    crop_boxes=review.identity_crop_boxes,
                                )
                            if not review.accepted:
                                reason = review.failure_reason or "镜头未满足动作与空间契约"
                                result.errors.append(f"语义验收不通过: {reason}")
                                rejected_path = self._preserve_rejected_take(
                                    shot["shot_id"], local_path
                                )
                                if semantic_retake_count >= 1:
                                    result.local_path = rejected_path
                                    result.last_frame_url = None
                                    result.status = "failed"
                                    self._notify_progress(result)
                                    return result
                                semantic_retake_count += 1
                                semantic_failure = reason
                                result.provider_task_id = None
                                result.local_path = rejected_path
                                result.last_frame_url = None
                                self._notify_progress(result)
                                print(
                                    "     ⚠ 语义验收不通过，执行唯一一次定向重拍: "
                                    f"{reason}"
                                )
                                continue

                        try:
                            result.last_frame_url = self._extract_local_tail_frame(
                                shot["shot_id"], local_path
                            )
                        except Exception as exc:
                            # The paid take is already accepted; never regenerate it
                            # just because local observation failed. Keep provider tail
                            # when available and stop before a dependent next shot.
                            remote_tail = gen_result.get("last_frame_url")
                            if remote_tail:
                                result.last_frame_url = remote_tail
                                result.errors.append(
                                    f"本地尾帧提取失败，保留 provider 尾帧: {exc}"
                                )
                            else:
                                raise GenerationReadinessError(
                                    f"Shot {shot['shot_id']} 已生成但无法记录已接受尾帧: {exc}"
                                ) from exc
                        if not self.semantic_reviewer:
                            result.accepted_contract_version = ACTION_CONTRACT_VERSION
                            result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
                            result.acceptance_policy = "technical_only"
                            self._write_acceptance_context(
                                shot["shot_id"],
                                local_path,
                                shot,
                                image_urls,
                                role,
                                policy="technical_only",
                            )
                        result.status = "success"
                        self._notify_progress(result)
                        return result

                    elif gen_result.get("error_type") in {"poll_error", "poll_timeout"}:
                        result.errors.append(
                            f"远端任务 {result.provider_task_id} 状态暂不可确认: "
                            f"{gen_result.get('error', 'unknown')}"
                        )
                        result.status = "running"
                        self._notify_progress(result)
                        return result

                    elif gen_result.get("error_type") == "privacy":
                        # 隐私审核: 角色参考帧被判定为含真实人物
                        # 降级策略: 丢弃角色参考帧, 仅保留尾帧做画面衔接
                        if not image_urls:
                            # 纯文本请求没有可降级的参考职责；重复提交相同请求
                            # 只会增加成本，并不能改变远端审核结果。
                            result.errors.append(
                                f"L{level}: 纯文本请求仍被隐私审核拒绝, 停止重试"
                            )
                            result.status = "failed"
                            self._notify_progress(result)
                            return result
                        if (
                            not skip_char_refs
                            and self._has_removable_identity_refs(image_urls, role)
                        ):
                            skip_char_refs = True
                            result.errors.append(
                                f"L{level}: 隐私审核拒绝角色参考帧, 降级为尾帧衔接 + 文字描述"
                            )
                            print(f"     ⚠ 隐私审核: 角色参考帧含类真实人物, 改为尾帧衔接模式")
                            await asyncio.sleep(3)
                            continue
                        else:
                            # 状态镜头不能丢尾帧；独立镜头才可继续 T2V。
                            if state_reference_required:
                                result.errors.append(
                                    f"L{level}: 状态参考被隐私审核拒绝，"
                                    "没有可移除的身份参考，无法满足连续性契约"
                                )
                                result.status = "failed"
                                self._notify_progress(result)
                                return result
                            use_refs = False
                            result.errors.append(
                                f"L{level}: 隐私审核再次拒绝, 退化为纯文本 T2V"
                            )
                            print(f"     ⚠ 隐私审核: 尾帧也被拒绝, 退化为纯文本 T2V")
                            await asyncio.sleep(3)
                            continue

                    elif gen_result.get("error_type") == "copyright_policy":
                        detail = self._provider_failure_detail(gen_result)
                        if not copyright_retry_used:
                            copyright_retry_used = True
                            result.errors.append(
                                f"L{level}: 输出触发版权策略，执行唯一一次版权边界澄清重试 "
                                f"({detail})"
                            )
                            print("     ⚠ 输出触发版权策略，保持镜头合同并进行一次版权边界澄清重试")
                            continue
                        result.errors.append(
                            f"L{level}: 输出连续触发版权策略，已停止重试 ({detail})"
                        )
                        result.status = "failed"
                        self._notify_progress(result)
                        return result

                    elif gen_result.get("error_type") == "moderation":
                        detail = self._provider_failure_detail(gen_result)
                        locus = str(gen_result.get("error_locus", "unknown"))
                        if locus == "input_text" and not input_text_recompile_used:
                            input_text_recompile_used = True
                            result.recovery_actions.append(
                                "recompile_input_text_policy_safe"
                            )
                            result.errors.append(
                                f"L{level}: 输入文本审核拒绝，执行唯一一次合同等价重编译 "
                                f"({detail})"
                            )
                            print("     ⚠ 输入文本审核拒绝，使用精简制片语言重编译一次")
                            continue
                        result.errors.append(
                            f"L{level}: 内容审核拒绝，已停止重试 ({detail})"
                        )
                        result.status = "failed"
                        self._notify_progress(result)
                        return result

                    elif gen_result.get("error_type") == "transient_provider":
                        detail = self._provider_failure_detail(gen_result)
                        result.errors.append(f"L{level}: 服务暂时异常 ({detail})")
                        print(f"     ⚠ L{level} 服务暂时异常: {detail[:200]}")

                    elif gen_result.get("error_type") == "rate_limit":
                        rate_limit_backoff += 1
                        if rate_limit_backoff >= max_rate_limit_retries:
                            result.errors.append(
                                f"L{level}: 配额耗尽, 已退避 {rate_limit_backoff} 次仍失败"
                            )
                            break
                        wait_secs = 30 * (2 ** (rate_limit_backoff - 1))
                        result.errors.append(
                            f"L{level}: 429 限流, 第 {rate_limit_backoff} 次退避 {wait_secs}s..."
                        )
                        print(f"     ⏳ 429 限流, 等待 {wait_secs}s 后重试...")
                        await asyncio.sleep(wait_secs)
                        continue

                    else:
                        err_msg = gen_result.get("error", "unknown")
                        result.errors.append(f"L{level}: {err_msg}")
                        print(f"     ⚠ L{level} 错误: {err_msg[:200]}")

                        err_low = err_msg.lower()
                        is_ref_error = use_refs and any(
                            kw in err_low for kw in [
                                "download image", "image format", "image size",
                                "invalid image", "fetch image", "image url",
                                "cannot be mixed", "first_frame", "last_frame",
                                "reference_image", "首帧", "参考图",
                            ]
                        )
                        if is_ref_error:
                            if state_reference_required:
                                result.errors.append(
                                    f"L{level}: 状态参考图不可用，停止以避免生成不连续镜头"
                                )
                                result.status = "failed"
                                self._notify_progress(result)
                                return result
                            result.errors.append(f"L{level}: 参考图异常, 改为无图重试")
                            print(f"     ⚠ 参考图异常, 改为无图重试")
                            use_refs = False
                            await asyncio.sleep(3)
                            continue

                except (
                    ProgressPersistenceError,
                    SubmittedTaskCheckpointError,
                    GenerationReadinessError,
                    SemanticReviewUnavailableError,
                ):
                    raise
                except asyncio.TimeoutError:
                    result.errors.append(f"L{level}: 超时")
                    print(f"     ⚠ L{level} 超时")
                except Exception as e:
                    result.errors.append(f"L{level}: {str(e)}")
                    print(f"     ⚠ L{level} 异常: {str(e)[:200]}")

                await asyncio.sleep(5 * (attempt + 1))

        result.status = "failed"
        self._notify_progress(result)
        return result

    async def _poll_existing_task(self, task_id: str) -> dict:
        """Keep polling recovery separate from prompt compilation/submission.

        ``SeedanceAPI`` exposes ``poll_task``. The narrow fallback only supports
        older in-process test doubles; production never sends a placeholder
        prompt while resuming an existing provider task.
        """
        poll_task = getattr(self.api, "poll_task", None)
        if callable(poll_task):
            return await poll_task(task_id, timeout=config.GENERATION_TIMEOUT)
        return await self.api.generate(
            prompt="",
            duration=4,
            ratio="16:9",
            resolution=config.DEFAULT_RESOLUTION,
            model=config.SEEDANCE_MODEL,
            generate_audio=True,
            return_last_frame=self.api.supports_last_frame,
            image_urls=None,
            image_role=None,
            timeout=config.GENERATION_TIMEOUT,
            task_id=task_id,
            on_submitted=None,
        )

    @staticmethod
    def _compiled_contract_fingerprint(shot: dict) -> str:
        """Fingerprint the canonical prompt/review projection, never raw shot prose."""
        contract = compile_action_contract(shot)
        immutable = {
            "version": ACTION_CONTRACT_VERSION,
            "phase": contract.phase,
            "mode": contract.mode,
            "outcome_scope": contract.outcome_scope,
            "effect_motion": contract.effect_motion,
            "prompt_start_state": contract.prompt_start_state,
            "prompt_parts": contract.prompt_parts,
            "review_projection": contract.review_projection,
        }
        payload = json.dumps(
            immutable, sort_keys=True, ensure_ascii=False, default=str
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _reference_fingerprint(value: str) -> str:
        if not value.startswith(("http://", "https://", "data:")):
            path = Path(value)
            try:
                if path.is_file():
                    return hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                pass
        return hashlib.sha256(str(value).encode()).hexdigest()

    def _generation_provenance(
        self,
        shot: dict,
        image_urls: list[str],
        role: Optional[str],
        *,
        prompt_profile: str | None = None,
        prompt_fingerprint: str | None = None,
    ) -> dict:
        provenance = {
            "version": "generation-provenance-v3",
            "compiled_contract_version": ACTION_CONTRACT_VERSION,
            "compiled_contract_fingerprint": self._compiled_contract_fingerprint(shot),
            "image_role": role,
            "reference_fingerprints": [
                self._reference_fingerprint(value) for value in image_urls
            ],
        }
        if prompt_profile and prompt_fingerprint:
            provenance.update({
                "prompt_profile": prompt_profile,
                "prompt_fingerprint": prompt_fingerprint,
            })
        return provenance

    def _generation_provenance_path(self, shot_id: int, *, video_path: str | None = None) -> Path:
        if video_path:
            path = Path(video_path)
            return path.with_name(f"{path.stem}_generation.json")
        return self.output_dir / "shots" / f"shot_{shot_id:03d}_generation.json"

    def _write_generation_provenance(
        self, shot_id: int, provenance: dict, task_id: str | None
    ) -> None:
        payload = {**provenance, "provider_task_id": task_id}
        path = self._generation_provenance_path(shot_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _acceptance_context(
        self,
        shot: dict,
        image_urls: list[str],
        role: Optional[str],
        *,
        policy: str = "semantic_reviewed",
    ) -> dict:
        """Record the context under which a local take was accepted.

        This is deliberately separate from the immutable provider submission
        lineage. A later offline review may accept the same video against a new
        upstream tail; that must not rewrite what was originally submitted.
        """
        return {
            "version": "acceptance-context-v1",
            "policy": policy,
            "compiled_contract_version": ACTION_CONTRACT_VERSION,
            "compiled_contract_fingerprint": self._compiled_contract_fingerprint(shot),
            "semantic_evaluator_version": (
                SEMANTIC_REVIEW_VERSION if policy == "semantic_reviewed" else None
            ),
            "image_role": role,
            "reference_fingerprints": [
                self._reference_fingerprint(value) for value in image_urls
            ],
        }

    def _write_acceptance_context(
        self,
        shot_id: int,
        video_path: str,
        shot: dict,
        image_urls: list[str],
        role: Optional[str],
        *,
        policy: str = "semantic_reviewed",
    ) -> None:
        path = self._generation_provenance_path(shot_id, video_path=video_path)
        stored = self._read_generation_provenance(video_path)
        if not stored:
            # A local cache may predate remote lineage tracking.  Its acceptance
            # context is still authoritative for resume, but it must never be
            # mistaken for a provider submission record.
            stored = {
                "version": "acceptance-only-v1",
                "provider_task_id": None,
            }
        stored["acceptance_context"] = self._acceptance_context(
            shot, image_urls, role, policy=policy
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _provenance_matches(
        self, video_path: str, shot: dict, image_urls: list[str], role: Optional[str]
    ) -> bool:
        path = self._generation_provenance_path(
            int(shot["shot_id"]), video_path=video_path
        )
        if not path.is_file():
            return False
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = self._generation_provenance(shot, image_urls, role)
        return all(stored.get(key) == value for key, value in expected.items())

    def _pending_descriptor_matches(
        self,
        descriptor: Mapping[str, object],
        shot: dict,
        image_urls: list[str],
        role: Optional[str],
    ) -> bool:
        task_id = str(descriptor.get("task_id", "")).strip()
        prompt_profile = str(descriptor.get("prompt_profile", "")).strip()
        prompt_fingerprint = str(descriptor.get("prompt_fingerprint", "")).strip()
        contract_version = str(
            descriptor.get("compiled_contract_version", "")
        ).strip()
        contract_fingerprint = str(
            descriptor.get("compiled_contract_fingerprint", "")
        ).strip()
        if (
            not task_id
            or prompt_profile not in {"normal", "policy_safe"}
            or len(prompt_fingerprint) != 64
            or contract_version != ACTION_CONTRACT_VERSION
            or contract_fingerprint != self._compiled_contract_fingerprint(shot)
        ):
            return False
        path = self._generation_provenance_path(int(shot["shot_id"]))
        if not path.is_file():
            return False
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        expected = self._generation_provenance(
            shot,
            image_urls,
            role,
            prompt_profile=prompt_profile,
            prompt_fingerprint=prompt_fingerprint,
        )
        return (
            stored.get("provider_task_id") == task_id
            and all(stored.get(key) == value for key, value in expected.items())
        )

    def _restored_provenance_matches(
        self,
        restored: Mapping[str, object],
        video_path: str,
        shot: dict,
        image_urls: list[str],
        role: Optional[str],
    ) -> bool:
        stored = self._read_generation_provenance(video_path)
        acceptance_context = stored.get("acceptance_context")
        if not isinstance(acceptance_context, dict):
            return False
        policy = restored.get("acceptance_policy")
        if policy not in {"semantic_reviewed", "technical_only"}:
            return False
        expected_context = self._acceptance_context(
            shot, image_urls, role, policy=str(policy)
        )
        return (
            all(
                acceptance_context.get(key) == value
                for key, value in expected_context.items()
            )
            and restored.get("accepted_contract_version") == ACTION_CONTRACT_VERSION
            and restored.get("accepted_contract_fingerprint")
            == self._compiled_contract_fingerprint(shot)
            and (
                restored.get("semantic_evaluator_version") == SEMANTIC_REVIEW_VERSION
                if policy == "semantic_reviewed"
                else restored.get("semantic_evaluator_version") is None
            )
        )

    def _read_generation_provenance(self, video_path: str) -> dict:
        path = self._generation_provenance_path(0, video_path=video_path)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _preserve_incompatible_take(self, shot_id: int, video_path: str) -> str:
        source = Path(video_path)
        if not source.is_file():
            return video_path
        existing = len(list(
            source.parent.glob(f"shot_{shot_id:03d}_incompatible_*.mp4")
        ))
        rejected = source.with_name(
            f"shot_{shot_id:03d}_incompatible_{existing + 1}.mp4"
        )
        source.replace(rejected)
        provenance = self._generation_provenance_path(shot_id, video_path=str(source))
        if provenance.is_file():
            provenance.replace(
                self._generation_provenance_path(shot_id, video_path=str(rejected))
            )
        return str(rejected)

    def _build_image_refs(
        self,
        shot: dict,
        prev_last_frame: Optional[str],
        prev_shot: Optional[dict] = None,
    ) -> tuple[list[str], Optional[str]]:
        """按单一职责选择参考图，避免身份与起始状态共用一个 role。

        返回 (image_urls, role):
        ┌───────────────────────────────────────────────────────────┐
        │ seamless / 同场景切镜: 尾帧 first_frame，锁定真实起始状态 │
        │ 跨场景 intentional_cut: 角色图 reference_image            │
        │ 无可用职责资产: [] + None                                 │
        └───────────────────────────────────────────────────────────┘
        """
        continuity = shot.get("continuity_from_previous")
        if continuity == "seamless" and prev_last_frame:
            print("     [REF] 无缝续接: 尾帧 first_frame")
            return [prev_last_frame], "first_frame"

        # 收集本镜头涉及的角色参考帧 (本地文件路径)
        shot_chars = shot.get("characters", [])
        char_ref_paths = []
        for char_name in shot_chars:
            if char_name in self.character_refs:
                path = self.character_refs[char_name]
                if os.path.isfile(path):
                    char_ref_paths.append(path)

        if continuity == "intentional_cut":
            if prev_last_frame and _should_use_previous_tail_reference(
                shot,
                prev_shot,
                has_identity_reference=bool(char_ref_paths),
            ):
                print("     [REF] 同场景状态交接: 尾帧 first_frame（官方状态职责）")
                return [prev_last_frame], "first_frame"
            if char_ref_paths:
                print(
                    f"     [REF] 有意切镜: {len(char_ref_paths)} 张角色 reference_image"
                )
                return char_ref_paths, "reference_image"
            print("     [REF] 有意切镜: 无角色锚点，T2V 模式")
            return [], None

        # 旧分镜没有显式契约时，尾帧优先承担连续性职责。
        if prev_last_frame:
            print("     [REF] 兼容续接: 尾帧 first_frame")
            return [prev_last_frame], "first_frame"

        if char_ref_paths:
            print(f"     [REF] 角色锚定: {len(char_ref_paths)} 张 reference_image")
            return char_ref_paths, "reference_image"

        print("     [REF] T2V 模式: 无参考图")
        return [], None

    async def _review_take(
        self,
        video_path: str,
        shot: dict,
        *,
        previous_frame_path: str | None = None,
        previous_shot: dict | None = None,
        storyboard: dict | None = None,
    ):
        storyboard = storyboard or {}
        character_catalog = {
            str(character.get("name")): character
            for character in storyboard.get("characters", [])
            if character.get("name")
        }
        visible_characters = visible_character_names(shot, character_catalog)
        identity_paths = [
            self.character_refs[name]
            for name in visible_characters
            if name in self.character_refs
        ][:2]
        crop_entities = [
            name
            for name in visible_characters
            if name not in self.character_refs
            and name in character_catalog
            and character_catalog[name].get("reference_mode", "identity") == "identity"
        ]
        boundary_context = {}
        if previous_shot is not None:
            previous_state = previous_shot.get("observed_end_state", {})
            provenance = self._read_generation_provenance(video_path)
            boundary_context = {
                "same_scene": _scene_id(shot) == _scene_id(previous_shot),
                "previous_scene_id": _scene_id(previous_shot),
                "state_reference_role": provenance.get("image_role"),
                "previous_observed_end_state": (
                    dict(previous_state) if isinstance(previous_state, dict) else {}
                ),
            }
        try:
            return await asyncio.to_thread(
                self.semantic_reviewer.review,
                video_path,
                shot,
                previous_frame_path=previous_frame_path,
                identity_reference_paths=identity_paths,
                boundary_context=boundary_context,
                identity_crop_entities=crop_entities,
            )
        except SemanticReviewUnavailableError:
            raise
        except Exception as exc:
            raise SemanticReviewUnavailableError(
                f"Shot {shot.get('shot_id', '?')} 已生成，但语义验收暂不可用；"
                "已保留视频，恢复运行不会重复生成"
            ) from exc

    def _next_reference_depth(
        self,
        shot: dict,
        previous_shot: Optional[dict],
        previous_frame: Optional[str],
    ) -> int:
        if previous_shot is None or not previous_frame:
            return 0
        has_identity_reference = any(
            name in self.character_refs for name in shot.get("characters", [])
        )
        if not _should_use_previous_tail_reference(
            shot,
            previous_shot,
            has_identity_reference=has_identity_reference,
        ):
            return 0
        return int(previous_shot.get("output_reference_depth", 0)) + 1

    def _build_state_only_refs(
        self,
        shot: dict,
        prev_last_frame: Optional[str],
        prev_shot: Optional[dict],
    ) -> tuple[list[str], Optional[str]]:
        """Drop identity refs while preserving the active continuity mode."""
        if not prev_last_frame:
            print("     [REF] 隐私降级: 无可用尾帧, T2V 模式")
            return [], None
        if shot.get("continuity_from_previous") == "seamless":
            print("     [REF] 隐私降级: 仅尾帧 first_frame")
            return [prev_last_frame], "first_frame"
        if _should_use_previous_tail_reference(
            shot,
            prev_shot,
            has_identity_reference=False,
        ):
            print("     [REF] 隐私降级: 仅尾帧 first_frame")
            return [prev_last_frame], "first_frame"
        print("     [REF] 隐私降级: 跨场景或参考链已达上限, T2V 模式")
        return [], None

    @staticmethod
    def _has_removable_identity_refs(
        image_urls: list[str], role: Optional[str]
    ) -> bool:
        """Only reference_image carries identity assets that can be dropped."""
        return bool(image_urls) and role == "reference_image"

    @staticmethod
    def _is_privacy_failure(result: Mapping[str, object]) -> bool:
        declared_type = str(result.get("error_type", "")).strip()
        if declared_type == "privacy":
            return True
        if declared_type and declared_type != "unknown":
            return False
        message = str(result.get("error", "")).casefold()
        return any(keyword in message for keyword in (
            "real person", "privacy", "privacyinformation",
            "人脸", "真人", "肖像",
        ))

    @staticmethod
    def _provider_failure_detail(result: Mapping[str, object]) -> str:
        code = str(result.get("error_code", "")).strip()
        message = str(result.get("error", "unknown")).strip()
        return f"{code}: {message}" if code else message

    @staticmethod
    def _inject_copyright_boundary(prompt: str) -> str:
        return (
            "[Copyright boundary clarification: preserve only the subjects and visual "
            "requirements explicitly stated in the shot contract. Do not introduce any "
            "unstated franchise, copyrighted character, existing film scene, logo, brand, "
            "music, or signature visual motif. Use independently designed supporting "
            "costumes, settings, and props unless the contract explicitly says otherwise. "
            "Do not conceal or alter the user's intent.] "
            + prompt
        )

    @staticmethod
    def _has_state_reference(
        shot: dict,
        prev_shot: Optional[dict],
        prev_last_frame: Optional[str],
        image_urls: list[str],
        role: Optional[str],
    ) -> bool:
        if not prev_last_frame or not image_urls or image_urls[0] != prev_last_frame:
            return False
        if role == "first_frame":
            return True
        return role == "reference_image"

    def _inject_character_description(
        self, prompt: str, shot: dict, storyboard: dict
    ) -> str:
        """将角色外观描述注入 prompt (文字层面的双重保障)。

        即使有角色参考图, prompt 中也要包含角色描述,
        强化模型对角色特征的理解。
        """
        characters = storyboard.get("characters", [])
        shot_chars = shot.get("characters", [])
        if not characters or not shot_chars:
            return prompt

        char_descs = []
        for char in characters:
            if char.get("name") in shot_chars and char.get("description"):
                char_descs.append(f'{char["name"]}: {char["description"]}')

        if char_descs:
            desc_block = "; ".join(char_descs)
            prompt = f"[Character reference — {desc_block}] {prompt}"

        return prompt

    @staticmethod
    def _inject_shot_contract(
        prompt: str,
        shot: dict,
        *,
        has_observed_start: bool = False,
    ) -> str:
        """把一个主动作和明确起止状态编译为紧凑自然语言约束。"""
        start = VideoGenerator._state_summary(shot.get("start_state"))
        parts = []
        if shot.get("continuity_from_previous") == "seamless":
            parts.append(
                "start exactly from the supplied first frame and preserve its visible state"
            )
        elif has_observed_start:
            parts.append(
                "begin from the observed physical and action state in Image 1 "
                "while applying the current camera composition"
            )
        elif start:
            parts.append(f"start exactly with {start}")
        required_visible = [
            str(entity).strip()
            for entity in shot.get("required_visible_entities", [])
            if str(entity).strip()
        ]
        if required_visible:
            parts.append(
                "keep these required entities clearly visible: "
                + ", ".join(required_visible)
            )
        parts.extend(compile_action_contract(shot).prompt_parts)
        camera = shot.get("camera", {})
        positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
        composition_change = str(shot.get("composition_change", "")).strip()
        start_framing = str(camera.get("start_framing", "")).strip() if isinstance(camera, dict) else ""
        end_framing = str(camera.get("end_framing", "")).strip() if isinstance(camera, dict) else ""
        if composition_change in {"medium", "large"}:
            framing = start_framing or end_framing
            change_scope = "a clearly different shot size or angle" if composition_change == "medium" else "unmistakably different coverage"
            if framing:
                parts.append(
                    f"use a {framing} with {change_scope} from the supplied previous tail; "
                    "do not copy the previous framing"
                )
            else:
                parts.append(
                    f"use {change_scope} from the supplied previous tail; "
                    "do not copy the previous framing"
                )
        if positions:
            placement = ", ".join(
                f"{name}={position}" for name, position in positions.items()
            )
            parts.append(f"keep screen positions {placement}")
        blocking = shot.get("blocking", {})
        if isinstance(blocking, dict):
            for name, intent in blocking.items():
                if not isinstance(intent, dict):
                    continue
                details = [
                    f"frame position {intent.get('frame_position')}",
                    f"body oriented {intent.get('body_orientation')}",
                    f"facing {intent.get('facing_target')}",
                    f"eyeline on {intent.get('eyeline_target')}",
                    f"travel direction {intent.get('travel_direction')}",
                    f"action directed at {intent.get('action_target')}",
                ]
                present = [detail for detail in details if not detail.endswith((" None", " "))]
                if present:
                    parts.append(f"{name}: {', '.join(present)}")
        if not parts:
            return prompt
        return (
            f"[Shot contract — {'; '.join(parts)}; "
            "do not introduce another major action] "
            f"{prompt}"
        )

    @staticmethod
    def _inject_reference_scope(
        prompt: str,
        role: Optional[str],
        *,
        reference_count: int = 1,
        has_state_reference: bool = False,
    ) -> str:
        """Assign one non-overlapping responsibility to every reference image."""
        if role != "reference_image":
            return prompt
        if has_state_reference:
            identity_scope = ""
            if reference_count == 2:
                identity_scope = (
                    " Image 2 controls identity and appearance only; ignore its pose, "
                    "framing, background, and camera angle."
                )
            elif reference_count > 2:
                identity_scope = (
                    f" Images 2-{reference_count} control identity and appearance only; "
                    "ignore their pose, framing, background, and camera angle."
                )
            return (
                "[Reference scope — Image 1 is the accepted previous-shot tail and "
                "controls only the observed physical and scene state: environment layout, "
                "lighting, props, subject positions, pose, and action result; do not copy "
                "its camera framing or use it as canonical identity."
                f"{identity_scope} The current shot contract controls camera composition "
                f"and new action.] {prompt}"
            )
        return (
            "[Reference scope — use the supplied image for identity and appearance only; "
            "ignore its pose, framing, background, and camera angle; obey the current "
            f"shot contract and camera composition] {prompt}"
        )

    @staticmethod
    def _state_summary(state: object) -> str:
        if not isinstance(state, dict):
            return ""
        values = [
            str(state.get(key, "")).strip()
            for key in (
                "location", "subject", "action_phase", "camera",
                "screen_direction", "pose_and_gaze", "prop_state", "open_motion",
            )
        ]
        return ", ".join(value for value in values if value)

    def _extract_local_tail_frame(self, shot_id: int, video_path: str) -> str:
        """Persist the accepted take's actual end state for continuation/resume."""
        from tools.ffmpeg_ops import get_video_duration

        output_path = str(
            self.output_dir / "shots" / f"shot_{shot_id:03d}_lastframe.jpg"
        )
        duration = get_video_duration(video_path)
        return extract_frame(
            video_path, output_path, timestamp=max(0.0, duration - 0.1)
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _unique_rejected_takes(self, shot_id: int) -> list[Path]:
        """Return provider Takes, not filesystem copies left by interrupted resumes."""
        candidates = sorted(
            (self.output_dir / "shots").glob(
                f"shot_{shot_id:03d}_rejected_*.mp4"
            ),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        unique: list[Path] = []
        provider_ids: set[str] = set()
        content_hashes: set[str] = set()
        for path in candidates:
            provenance = self._read_generation_provenance(str(path))
            provider_id = str(provenance.get("provider_task_id") or "").strip()
            try:
                content_hash = self._file_sha256(path)
            except OSError:
                content_hash = ""
            if (
                provider_id and provider_id in provider_ids
            ) or (
                content_hash and content_hash in content_hashes
            ):
                continue
            unique.append(path)
            if provider_id:
                provider_ids.add(provider_id)
            if content_hash:
                content_hashes.add(content_hash)
        return unique

    def _rejected_provider_task_ids(self, shot_id: int) -> set[str]:
        task_ids = set()
        for path in (self.output_dir / "shots").glob(
            f"shot_{shot_id:03d}_rejected_*.mp4"
        ):
            task_id = self._read_generation_provenance(str(path)).get(
                "provider_task_id"
            )
            if task_id:
                task_ids.add(str(task_id))
        return task_ids

    def _preserve_rejected_take(self, shot_id: int, video_path: str) -> str:
        source = Path(video_path)
        existing_numbers = []
        prefix = f"shot_{shot_id:03d}_rejected_"
        for path in source.parent.glob(f"{prefix}*.mp4"):
            suffix = path.stem.removeprefix(prefix)
            if suffix.isdigit():
                existing_numbers.append(int(suffix))
        take_number = max(existing_numbers, default=0) + 1
        rejected = self.output_dir / "shots" / (
            f"shot_{shot_id:03d}_rejected_{take_number}.mp4"
        )
        source.replace(rejected)
        provenance = self._generation_provenance_path(
            shot_id, video_path=str(source)
        )
        if provenance.is_file():
            provenance.replace(
                self._generation_provenance_path(
                    shot_id, video_path=str(rejected)
                )
            )
        return str(rejected)

    async def _review_rejected_take(
        self,
        result: ShotResult,
        shot: dict,
        rejected_path: Path,
        *,
        previous_frame_path: str | None,
        previous_shot: dict | None,
        storyboard: dict,
    ) -> SemanticReview | None:
        qa = check_video_quality(str(rejected_path))
        result.local_path = str(rejected_path)
        result.quality_score = qa["quality_score"]
        result.technical_quality_score = qa["quality_score"]
        result.model_used = "recovered-local-take"
        if not qa["pass"]:
            result.errors.append(
                f"历史 rejected take 技术 QA 不通过: {qa['issues']}"
            )
            return None
        review = await self._review_take(
            str(rejected_path),
            shot,
            previous_frame_path=previous_frame_path,
            previous_shot=previous_shot,
            storyboard=storyboard,
        )
        result.semantic_accepted = review.accepted
        result.observed_end_state = review.observed_end_state
        return review

    def _promote_rejected_take(
        self,
        result: ShotResult,
        shot: dict,
        rejected_path: Path,
        review: SemanticReview,
        *,
        image_urls: list[str],
        image_role: Optional[str],
    ) -> ShotResult:
        self._register_identity_crops(
            shot_id=shot["shot_id"],
            video_path=str(rejected_path),
            crop_boxes=review.identity_crop_boxes,
        )
        canonical_path = (
            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
        )
        rejected_provenance = self._generation_provenance_path(
            shot["shot_id"], video_path=str(rejected_path)
        )
        provenance = self._read_generation_provenance(str(rejected_path))
        rejected_path.replace(canonical_path)
        if rejected_provenance.is_file():
            rejected_provenance.replace(
                self._generation_provenance_path(shot["shot_id"])
            )
        result.local_path = str(canonical_path)
        result.last_frame_url = self._extract_local_tail_frame(
            shot["shot_id"], str(canonical_path)
        )
        provider_task_id = provenance.get("provider_task_id")
        result.provider_task_id = (
            str(provider_task_id) if provider_task_id else None
        )
        result.semantic_accepted = True
        result.observed_end_state = review.observed_end_state
        result.accepted_contract_version = ACTION_CONTRACT_VERSION
        result.accepted_contract_fingerprint = self._compiled_contract_fingerprint(shot)
        result.semantic_evaluator_version = SEMANTIC_REVIEW_VERSION
        result.acceptance_policy = "semantic_reviewed"
        self._write_acceptance_context(
            shot["shot_id"],
            str(canonical_path),
            shot,
            image_urls,
            image_role,
        )
        result.status = "success"
        self._notify_progress(result)
        print("     ♻️ 使用当前验收器复核通过本地 take，不重新调用视频生成")
        return result

    async def _reassess_latest_rejected_take(
        self,
        result: ShotResult,
        shot: dict,
        rejected_path: Path,
        *,
        previous_frame_path: str | None,
        previous_shot: dict | None,
        storyboard: dict,
        image_urls: list[str],
        image_role: Optional[str],
    ) -> ShotResult:
        """Re-review local footage after evaluator changes without regenerating it."""
        review = await self._review_rejected_take(
            result,
            shot,
            rejected_path,
            previous_frame_path=previous_frame_path,
            previous_shot=previous_shot,
            storyboard=storyboard,
        )
        if review is None:
            result.status = "failed"
            result.errors.append(
                "语义重拍预算已达到上限"
            )
            self._notify_progress(result)
            return result
        if not review.accepted:
            result.status = "failed"
            result.errors.append(
                "按当前验收器复核仍不通过，语义重拍预算已达到上限: "
                + review.failure_reason
            )
            self._notify_progress(result)
            return result
        return self._promote_rejected_take(
            result,
            shot,
            rejected_path,
            review,
            image_urls=image_urls,
            image_role=image_role,
        )

    def _register_identity_crops(
        self,
        *,
        shot_id: int,
        video_path: str,
        crop_boxes: Mapping[str, tuple[float, float, float, float]],
    ) -> None:
        """Persist reviewer-approved identity crops from the reviewed midpoint."""
        import cv2
        from tools.ffmpeg_ops import get_video_duration

        valid_boxes = {}
        for name, box in crop_boxes.items():
            if (
                name in self.character_refs
                or not isinstance(box, (list, tuple))
                or len(box) != 4
            ):
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            if (
                0.0 <= x1 < x2 <= 1.0
                and 0.0 <= y1 < y2 <= 1.0
                and x2 - x1 >= 0.1
                and y2 - y1 >= 0.1
            ):
                valid_boxes[str(name)] = (x1, y1, x2, y2)
        if not valid_boxes:
            return

        midpoint_path = self.output_dir / "shots" / (
            f"shot_{shot_id:03d}_identity_midpoint.jpg"
        )
        try:
            extract_frame(
                video_path,
                str(midpoint_path),
                timestamp=get_video_duration(video_path) * 0.5,
            )
            frame = cv2.imread(str(midpoint_path))
            if frame is None or frame.size == 0:
                return
            height, width = frame.shape[:2]
            for name, (x1, y1, x2, y2) in valid_boxes.items():
                left = max(0, min(width - 1, round(x1 * width)))
                top = max(0, min(height - 1, round(y1 * height)))
                right = max(left + 1, min(width, round(x2 * width)))
                bottom = max(top + 1, min(height, round(y2 * height)))
                crop = frame[top:bottom, left:right]
                if crop.shape[0] < 48 or crop.shape[1] < 48:
                    continue

                filename = self._safe_character_filename(name)
                ref_path = self.output_dir / "character_refs" / f"{filename}.jpg"
                temporary = ref_path.with_suffix(".tmp.jpg")
                if not cv2.imwrite(str(temporary), crop):
                    temporary.unlink(missing_ok=True)
                    continue
                temporary.replace(ref_path)
                digest = hashlib.sha256(ref_path.read_bytes()).hexdigest()
                existing = self.character_ref_hashes.get(digest)
                if existing and existing != name:
                    ref_path.unlink(missing_ok=True)
                    print(
                        f"     ⚠ 拒绝重复角色参考: {name} 与 {existing} 使用了同一裁剪"
                    )
                    continue
                self.character_refs[name] = str(ref_path)
                self.character_ref_hashes[digest] = name
                print(f"     📸 已注册验收裁剪身份参考: {name} → {ref_path}")
        finally:
            midpoint_path.unlink(missing_ok=True)

    @staticmethod
    def _safe_character_filename(name: str) -> str:
        original = str(name)
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in original
        ).strip("._")
        digest = hashlib.sha256(original.encode()).hexdigest()[:12]
        if not safe:
            return digest
        return safe if safe == original else f"{safe}_{digest}"

    @staticmethod
    def _reconcile_start_state(shot: dict, previous_shot: Optional[dict]) -> None:
        if previous_shot is None or _scene_id(shot) != _scene_id(previous_shot):
            return
        observed = previous_shot.get("observed_end_state")
        if not isinstance(observed, dict) or not any(observed.values()):
            return
        planned = shot.get("start_state", {})
        planned = planned if isinstance(planned, dict) else {}
        shot["start_state"] = {
            key: str(observed.get(key) or planned.get(key, ""))
            for key in (
                "location", "subject", "action_phase", "camera",
                "screen_direction", "pose_and_gaze", "prop_state", "open_motion",
                "lighting",
            )
        }

    async def _extract_character_ref(
        self, shot: dict, video_path: str, storyboard: dict
    ) -> None:
        """Register one canonical identity from a single-subject source frame."""
        character_names = [
            item.get("name")
            for item in storyboard.get("characters", [])
            if item.get("name")
        ]
        shot_characters = visible_character_names(shot, character_names)
        if len(shot_characters) != 1:
            print("     ⚠ 跳过角色参考: 多主体整帧不能作为独立身份锚点")
            return

        char_name = shot_characters[0]
        character = next(
            (item for item in storyboard.get("characters", []) if item.get("name") == char_name),
            {},
        )
        if character.get("reference_mode", "identity") != "identity":
            print(f"     ⚠ 跳过角色参考: {char_name} 不是 identity 角色")
            return
        if char_name in self.character_refs:
            return

        filename = self._safe_character_filename(char_name)
        ref_path = str(self.output_dir / "character_refs" / f"{filename}.jpg")
        from tools.ffmpeg_ops import get_video_duration

        # Legacy fallback used only when semantic review is disabled.
        midpoint = get_video_duration(video_path) * 0.5
        extract_frame(video_path, ref_path, timestamp=midpoint)
        digest = hashlib.sha256(Path(ref_path).read_bytes()).hexdigest()
        existing = self.character_ref_hashes.get(digest)
        if existing and existing != char_name:
            print(
                f"     ⚠ 拒绝重复角色参考: {char_name} 与 {existing} 使用了同一画面"
            )
            Path(ref_path).unlink(missing_ok=True)
            return
        self.character_refs[char_name] = ref_path
        self.character_ref_hashes[digest] = char_name
        print(f"     📸 角色参考帧已提取: {char_name} → {ref_path}")

    @staticmethod
    def _inject_scene_continuity(
        prompt: str, current_shot: dict, prev_shot: Optional[dict]
    ) -> str:
        """注入上一镜头的环境承接信息, 帮助 Seedance 生成视觉一致的画面。

        策略 (保守, 避免 prompt 超长):
        - 仅在同场景连续镜头时注入
        - 仅注入: 光线条件 + 关键道具名称
        - 承接描述控制在 20 词以内
        - 不同场景的镜头不注入 (新场景有自己的世界)
        """
        if prev_shot is None:
            return prompt

        prev_scene = _scene_id(prev_shot)
        curr_scene = _scene_id(current_shot)

        # 不同场景 → 不注入 (新场景有自己的环境)
        if prev_scene != curr_scene:
            return prompt

        # 构建承接描述片段
        continuity_parts: list[str] = []

        # 1. 光线条件延续
        prev_lighting = prev_shot.get("lighting", "")
        if prev_lighting:
            # 取前 5 个词作为简洁的光线描述
            light_brief = " ".join(prev_lighting.split()[:5])
            continuity_parts.append(f"maintaining {light_brief}")

        # 2. 关键道具延续 (上一镜头的 key_props 中与当前镜头共有的)
        prev_props = set(
            p.lower().strip() for p in prev_shot.get("key_props", [])
        )
        curr_props = set(
            p.lower().strip() for p in current_shot.get("key_props", [])
        )
        shared_props = prev_props & curr_props
        if shared_props:
            # 只取前 3 个, 避免过长
            props_str = ", ".join(list(shared_props)[:3])
            continuity_parts.append(f"{props_str} still present")

        if not continuity_parts:
            return prompt

        continuity_hint = "; ".join(continuity_parts)
        return f"[Scene continuity — {continuity_hint}] {prompt}"

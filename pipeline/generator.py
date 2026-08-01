"""Stage 2: 视频生成 — 角色一致性 + 降级策略 + 即时下载"""

from __future__ import annotations

import asyncio
import hashlib
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
from pipeline.semantic_review import (
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
        self.resume_task_ids = dict(resume_task_ids or {})
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
        │ 同场景 intentional_cut: 尾帧状态 + 角色身份 reference_image   │
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
                    shot["end_state"] = result.observed_end_state

                semantic_label = (
                    "通过" if result.semantic_accepted is True
                    else "未启用" if result.semantic_accepted is None
                    else "未通过"
                )
                print(
                    f"     ✓ 完成 (技术质量: {result.technical_quality_score}, "
                    f"语义验收: {semantic_label}, 模型: {result.model_used})"
                )
                print(f"     [DEBUG] last_frame_url = {prev_last_frame}")
                print(f"     [DEBUG] character_refs = {list(self.character_refs.keys())}")

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

        restored = self.accepted_shot_artifacts.pop(shot["shot_id"], None)
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
            )
            if not result.last_frame_url or not Path(result.last_frame_url).is_file():
                result.last_frame_url = self._extract_local_tail_frame(
                    shot["shot_id"], result.local_path
                )
            self._notify_progress(result)
            print("     ♻️ 恢复已接受镜头，不重新生成或重新判定")
            return result

        resume_task_id = self.resume_task_ids.pop(shot["shot_id"], None)
        result = ShotResult(
            shot_id=shot["shot_id"],
            status="running",
            provider_task_id=resume_task_id,
        )
        self._notify_progress(result)
        semantic_retake_count = 0
        semantic_failure = ""
        rejected_takes = sorted(
            (self.output_dir / "shots").glob(
                f"shot_{shot['shot_id']:03d}_rejected_*.mp4"
            )
        )
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
                )
            result.status = "failed"
            result.errors.append(
                "语义验收已达到上限（原始 take + 1 次定向重拍），"
                "必须修改分镜契约后创建新运行"
            )
            self._notify_progress(result)
            return result

        # ─── 缓存检查: 已有本地文件则跳过 API (断点续传核心逻辑) ───
        cached_path = str(
            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
        )
        if Path(cached_path).exists() and Path(cached_path).stat().st_size > 0:
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
                        self._register_identity_crops(
                            shot_id=shot["shot_id"],
                            video_path=cached_path,
                            crop_boxes=review.identity_crop_boxes,
                        )
                    if not review.accepted:
                        reason = review.failure_reason or "镜头未满足动作与空间契约"
                        result.errors.append(f"缓存镜头语义验收不通过: {reason}")
                        rejected_path = self._preserve_rejected_take(
                            shot["shot_id"], cached_path, semantic_retake_count + 1
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
                        self._notify_progress(result)
                        print(
                            "     ⚠ 缓存镜头语义验收不通过，执行唯一一次定向重拍: "
                            f"{reason}"
                        )

                if cache_accepted:
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

        rate_limit_backoff = 0
        max_rate_limit_retries = 3

        use_refs = True  # 是否使用参考图 (失败后会关闭)
        skip_char_refs = False  # 隐私审核失败后, 丢弃角色参考帧但保留尾帧衔接
        requested_resolution = storyboard.get("resolution", config.DEFAULT_RESOLUTION)
        ensure_shot_ready(
            shot,
            previous_frame=prev_last_frame,
            previous_shot=prev_shot,
            character_refs=self.character_refs,
        )
        self._reconcile_start_state(shot, prev_shot)
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

                    # 构建 prompt (注入角色描述 + 环境承接, 双重保障)
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
                            "[Targeted retake — the previous take failed only because: "
                            f"{semantic_failure}. Correct that failure while preserving the "
                            "shot contract; do not add new actions.] " + prompt
                        )
                    if shot.get("negative_prompt"):
                        prompt += f". {shot['negative_prompt']}"

                    prompt = self._inject_reference_scope(
                        prompt,
                        role,
                        reference_count=len(image_urls),
                        has_state_reference=has_state_reference,
                    )

                    # 调用 API
                    def remember_submission(task_id: str) -> None:
                        result.provider_task_id = task_id
                        self._notify_progress(result)

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
                        task_id=resume_task_id,
                        on_submitted=remember_submission,
                    )
                    resume_task_id = None

                    if gen_result["status"] == "succeeded":
                        # ⚡ 立即下载
                        local_path = str(
                            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
                        )
                        await self.api.download_video(gen_result["video_url"], local_path)
                        result.local_path = local_path
                        result.last_frame_url = gen_result.get("last_frame_url")

                        # 质量检测
                        qa = check_video_quality(local_path)
                        result.quality_score = qa["quality_score"]
                        result.technical_quality_score = qa["quality_score"]

                        if not qa["pass"]:
                            result.errors.append(f"QA 不通过: {qa['issues']}")
                            print(f"     ⚠ QA 不通过: {qa['issues']}, 重试...")
                            continue

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
                                self._register_identity_crops(
                                    shot_id=shot["shot_id"],
                                    video_path=local_path,
                                    crop_boxes=review.identity_crop_boxes,
                                )
                            if not review.accepted:
                                reason = review.failure_reason or "镜头未满足动作与空间契约"
                                result.errors.append(f"语义验收不通过: {reason}")
                                rejected_path = self._preserve_rejected_take(
                                    shot["shot_id"], local_path, semantic_retake_count + 1
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
                        if not skip_char_refs:
                            skip_char_refs = True
                            result.errors.append(
                                f"L{level}: 隐私审核拒绝角色参考帧, 降级为尾帧衔接 + 文字描述"
                            )
                            print(f"     ⚠ 隐私审核: 角色参考帧含类真实人物, 改为尾帧衔接模式")
                            await asyncio.sleep(3)
                            continue
                        else:
                            # 尾帧也被拒绝 → 纯文本 T2V
                            use_refs = False
                            result.errors.append(
                                f"L{level}: 隐私审核再次拒绝, 退化为纯文本 T2V"
                            )
                            print(f"     ⚠ 隐私审核: 尾帧也被拒绝, 退化为纯文本 T2V")
                            await asyncio.sleep(3)
                            continue

                    elif gen_result.get("error_type") == "moderation":
                        result.errors.append(f"L{level}: 审核失败")
                        break

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
                        # 隐私类错误走专用降级路径
                        is_privacy = any(
                            kw in err_low for kw in [
                                "real person", "privacy", "sensitive",
                                "privacyinformation", "人脸", "真人", "肖像",
                            ]
                        )
                        if is_privacy and use_refs:
                            if not skip_char_refs:
                                skip_char_refs = True
                                result.errors.append(
                                    f"L{level}: 隐私审核 (错误消息), 降级为尾帧衔接"
                                )
                                print(f"     ⚠ 隐私审核 (错误消息): 降级为尾帧衔接模式")
                                await asyncio.sleep(3)
                                continue
                            else:
                                use_refs = False
                                result.errors.append(
                                    f"L{level}: 隐私审核再次拒绝, 退化为纯文本 T2V"
                                )
                                print(f"     ⚠ 隐私审核: 退化为纯文本 T2V")
                                await asyncio.sleep(3)
                                continue

                        is_ref_error = use_refs and any(
                            kw in err_low for kw in [
                                "download image", "image format", "image size",
                                "invalid image", "fetch image", "image url",
                                "cannot be mixed", "first_frame", "last_frame",
                                "reference_image", "首帧", "参考图",
                            ]
                        )
                        if is_ref_error:
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

    def _build_image_refs(
        self,
        shot: dict,
        prev_last_frame: Optional[str],
        prev_shot: Optional[dict] = None,
    ) -> tuple[list[str], Optional[str]]:
        """按单一职责选择参考图，避免身份与起始状态共用一个 role。

        返回 (image_urls, role):
        ┌───────────────────────────────────────────────────────────┐
        │ seamless: 尾帧 + first_frame，锁定真实起始状态            │
        │ 同场景 intentional_cut: 尾帧状态 + 角色图 reference_image │
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
                shot, prev_shot
            ):
                print(
                    "     [REF] 同场景切镜: 尾帧状态 + "
                    f"{len(char_ref_paths)} 张角色 reference_image"
                )
                return [prev_last_frame, *char_ref_paths], "reference_image"
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
            boundary_context = {
                "same_scene": _scene_id(shot) == _scene_id(previous_shot),
                "previous_scene_id": _scene_id(previous_shot),
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

    @staticmethod
    def _next_reference_depth(
        shot: dict,
        previous_shot: Optional[dict],
        previous_frame: Optional[str],
    ) -> int:
        if previous_shot is None or not previous_frame:
            return 0
        if not _should_use_previous_tail_reference(shot, previous_shot):
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
        if _should_use_previous_tail_reference(shot, prev_shot):
            print("     [REF] 隐私降级: 仅尾帧状态 reference_image")
            return [prev_last_frame], "reference_image"
        print("     [REF] 隐私降级: 跨场景或参考链已达上限, T2V 模式")
        return [], None

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
        return (
            role == "reference_image"
            and _should_use_previous_tail_reference(shot, prev_shot)
        )

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
        primary_action = str(shot.get("primary_action", "")).strip()
        start = VideoGenerator._state_summary(shot.get("start_state"))
        end = VideoGenerator._state_summary(shot.get("end_state"))
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
        if primary_action:
            parts.append(f"perform only this primary action: {primary_action}")
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
        geometry = shot.get("interaction_geometry", {})
        if isinstance(geometry, dict) and geometry.get("actor") and geometry.get("target"):
            actor = geometry["actor"]
            target = geometry["target"]
            geometry_parts = [f"interaction geometry {actor} toward {target}"]
            if geometry.get("must_share_frame"):
                geometry_parts.append("actor and target must share the frame")
            if geometry.get("line_of_action_visible"):
                geometry_parts.append("keep the line of action clearly visible")
            if geometry.get("occlusion_policy") == "none":
                geometry_parts.append("neither subject may be occluded")
            parts.append(", ".join(geometry_parts))
        if end:
            parts.append(f"finish with {end}")
        camera = shot.get("camera", {})
        positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
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
        action_beats = shot.get("action_beats", [])
        if isinstance(action_beats, list) and action_beats:
            compiled_beats = []
            for beat in action_beats:
                if not isinstance(beat, dict):
                    continue
                text = f"{beat.get('phase')}: {beat.get('actor')} {beat.get('action')}"
                if beat.get("target"):
                    text += f" toward {beat['target']}"
                if beat.get("visible_result"):
                    text += f", visibly resulting in {beat['visible_result']}"
                compiled_beats.append(text)
            if compiled_beats:
                parts.append("causal action phases " + "; ".join(compiled_beats))
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

    def _preserve_rejected_take(
        self, shot_id: int, video_path: str, take_number: int
    ) -> str:
        rejected = self.output_dir / "shots" / (
            f"shot_{shot_id:03d}_rejected_{take_number}.mp4"
        )
        Path(video_path).replace(rejected)
        return str(rejected)

    async def _reassess_latest_rejected_take(
        self,
        result: ShotResult,
        shot: dict,
        rejected_path: Path,
        *,
        previous_frame_path: str | None,
        previous_shot: dict | None,
        storyboard: dict,
    ) -> ShotResult:
        """Re-review local footage after evaluator changes without regenerating it."""
        qa = check_video_quality(str(rejected_path))
        result.local_path = str(rejected_path)
        result.quality_score = qa["quality_score"]
        result.technical_quality_score = qa["quality_score"]
        result.model_used = "recovered-local-take"
        if not qa["pass"]:
            result.status = "failed"
            result.errors.append(
                f"最后一次 rejected take 技术 QA 仍不通过，语义重拍预算已达到上限: "
                f"{qa['issues']}"
            )
            self._notify_progress(result)
            return result

        review = await self._review_take(
            str(rejected_path),
            shot,
            previous_frame_path=previous_frame_path,
            previous_shot=previous_shot,
            storyboard=storyboard,
        )
        result.semantic_accepted = review.accepted
        result.observed_end_state = review.observed_end_state
        if not review.accepted:
            result.status = "failed"
            result.errors.append(
                "按当前验收器复核仍不通过，语义重拍预算已达到上限: "
                + review.failure_reason
            )
            self._notify_progress(result)
            return result

        self._register_identity_crops(
            shot_id=shot["shot_id"],
            video_path=str(rejected_path),
            crop_boxes=review.identity_crop_boxes,
        )

        canonical_path = (
            self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}.mp4"
        )
        rejected_path.replace(canonical_path)
        result.local_path = str(canonical_path)
        result.last_frame_url = self._extract_local_tail_frame(
            shot["shot_id"], str(canonical_path)
        )
        result.provider_task_id = None
        result.status = "success"
        self._notify_progress(result)
        print("     ♻️ 使用当前验收器复核通过本地 take，不重新调用视频生成")
        return result

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

"""Stage 2: 视频生成 — 角色一致性 + 降级策略 + 即时下载"""

from __future__ import annotations

import asyncio
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
from pipeline.storyboard import _normalize_continuity_contract, _scene_id


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


class ProgressPersistenceError(RuntimeError):
    """Run state could not be persisted; continuing could duplicate paid work."""


class VideoGenerator:
    """视频生成引擎 — 角色一致性优先 + 画面衔接"""

    def __init__(
        self,
        output_dir: str,
        on_progress: Callable[[ShotResult], None] | None = None,
        resume_task_ids: Mapping[int, str] | None = None,
    ):
        self.api = SeedanceAPI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "shots").mkdir(exist_ok=True)
        (self.output_dir / "character_refs").mkdir(exist_ok=True)
        self.character_refs: dict[str, str] = {}  # name → 本地图片路径
        self.on_progress = on_progress
        self.resume_task_ids = dict(resume_task_ids or {})

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
        生成所有镜头 — 顺序执行 + 独立镜头预生成优化

        角色一致性策略:
        ┌─────────────────────────────────────────────────────────────────┐
        │ 首个合格角色镜头: 生成后提取清晰角色参考帧到本地                │
        │ seamless: first_frame (仅上一尾帧，锁定真实起始状态)            │
        │ intentional_cut: reference_image (仅角色参考帧，锁定身份)       │
        │                                                                 │
        │ 并行优化:                                                       │
        │ - 扫描后续镜头, 识别「独立镜头」(无角色+有意切镜 = 纯 T2V)     │
        │ - 独立镜头提前提交 API, 当流水线轮到时直接取结果               │
        │ - 保守策略: 只预生成确定无依赖的镜头, 保证正确性               │
        └─────────────────────────────────────────────────────────────────┘
        """
        shots = storyboard["shots"]
        self._normalize_continuity(shots)
        results: list[ShotResult] = [None] * len(shots)  # type: ignore
        prev_last_frame: Optional[str] = None
        prev_shot: Optional[dict] = None

        # 识别可预生成的独立镜头 (无角色 + 不依赖上一尾帧)
        independent_indices = self._find_independent_shots(shots)
        prefetch_tasks: dict[int, asyncio.Task] = {}

        for idx, shot in enumerate(shots):
            print(f"\n  🎬 生成 Shot {shot['shot_id']}...")

            # 如果当前镜头有预生成任务, 等待其结果
            if idx in prefetch_tasks:
                print(f"     ⚡ 预生成命中, 等待结果...")
                result = await prefetch_tasks.pop(idx)
            else:
                result = await self._generate_single_shot(
                    shot=shot,
                    prev_last_frame=prev_last_frame,
                    prev_shot=prev_shot,
                    storyboard=storyboard,
                )
            results[idx] = result

            if result.status == "success":
                prev_last_frame = result.last_frame_url

                print(f"     ✓ 完成 (质量: {result.quality_score}, 模型: {result.model_used})")
                print(f"     [DEBUG] last_frame_url = {prev_last_frame}")
                print(f"     [DEBUG] character_refs = {list(self.character_refs.keys())}")

                # 提取角色参考帧
                if shot.get("extract_character_ref"):
                    await self._extract_character_ref(shot, result.local_path)

                # 预提交后续独立镜头 (利用当前镜头下载/处理的空闲时间)
                for future_idx in independent_indices:
                    if future_idx > idx and future_idx not in prefetch_tasks:
                        future_shot = shots[future_idx]
                        print(f"     ⚡ 预提交独立镜头 Shot {future_shot['shot_id']} (T2V)")
                        prefetch_tasks[future_idx] = asyncio.create_task(
                            self._generate_single_shot(
                                shot=future_shot,
                                prev_last_frame=None,  # 独立镜头不依赖前帧
                                prev_shot=None,
                                storyboard=storyboard,
                            )
                        )
                        break  # 一次只预提交一个, 控制并发
            else:
                # 断链兜底: 保留上一张成功尾帧, 让后续镜头仍能衔接
                print(f"     ✗ 失败: {result.errors[-1] if result.errors else 'unknown'}")
                print(f"     ↪ 保留上一成功尾帧续接 (不退化为纯文本)")

            prev_shot = shot

        # 清理未消费的预生成任务
        for task in prefetch_tasks.values():
            task.cancel()

        return results

    @staticmethod
    def _normalize_continuity(shots: list[dict]) -> None:
        """生成入口再次收紧契约，覆盖旧工作区和外部分镜。"""
        for correction in _normalize_continuity_contract(shots):
            print(f"  [连续性校正] {correction}，改为 intentional_cut")

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
            # 条件 2: 剪辑契约不依赖上一镜头的尾帧
            continuity = shot.get("continuity_from_previous")
            if continuity == "seamless":
                continue
            if continuity in {None, "none"} and _scene_id(shot) == _scene_id(prev):
                continue
            # 条件 3: 非首镜 (首镜不需要预生成, 本来就是 T2V)
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

        resume_task_id = self.resume_task_ids.pop(shot["shot_id"], None)
        result = ShotResult(
            shot_id=shot["shot_id"],
            status="running",
            provider_task_id=resume_task_id,
        )
        self._notify_progress(result)

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
                result.model_used = "cached"

                # 从缓存视频提取尾帧, 供后续镜头衔接 (否则 last_frame_url=None 断链)
                lastframe_path = str(
                    self.output_dir / "shots" / f"shot_{shot['shot_id']:03d}_lastframe.jpg"
                )
                try:
                    from tools.ffmpeg_ops import get_video_duration
                    dur = get_video_duration(cached_path)
                    extract_frame(cached_path, lastframe_path, timestamp=max(0, dur - 0.1))
                    result.last_frame_url = lastframe_path
                except Exception:
                    result.last_frame_url = None

                print(f"     ♻️ 缓存命中: {cached_path} (质量: {qa['quality_score']})")

                # 缓存命中时也要提取角色参考 (否则后续镜头没有角色锚点)
                if shot.get("extract_character_ref") and not self.character_refs:
                    await self._extract_character_ref(shot, cached_path)

                self._notify_progress(result)
                return result
            else:
                print(f"     ⚠ 缓存文件 QA 不通过 ({qa['issues']}), 重新生成")

        rate_limit_backoff = 0
        max_rate_limit_retries = 3

        use_refs = True  # 是否使用参考图 (失败后会关闭)
        skip_char_refs = False  # 隐私审核失败后, 丢弃角色参考帧但保留尾帧衔接
        requested_resolution = storyboard.get("resolution", config.DEFAULT_RESOLUTION)
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
                    # 构建 prompt (注入角色描述 + 环境承接, 双重保障)
                    prompt = self._inject_character_description(
                        shot["prompt_en"], shot, storyboard
                    )
                    prompt = self._inject_scene_continuity(
                        prompt, shot, prev_shot
                    )
                    prompt = self._inject_shot_contract(prompt, shot)
                    if shot.get("negative_prompt"):
                        prompt += f". {shot['negative_prompt']}"

                    # 构建参考图列表 + 确定 role
                    if not use_refs:
                        image_urls, role = [], None
                    elif skip_char_refs:
                        # 隐私降级: 不传角色参考帧, 仅保留尾帧做画面衔接
                        if prev_last_frame and not os.path.isfile(prev_last_frame):
                            image_urls, role = [prev_last_frame], "first_frame"
                            print(f"     [REF] 隐私降级: 仅尾帧 first_frame (角色靠 prompt 描述)")
                        else:
                            image_urls, role = [], None
                            print(f"     [REF] 隐私降级: 无可用尾帧, T2V 模式")
                    else:
                        image_urls, role = self._build_image_refs(
                            shot, prev_last_frame
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

                        if not qa["pass"]:
                            result.errors.append(f"QA 不通过: {qa['issues']}")
                            print(f"     ⚠ QA 不通过: {qa['issues']}, 重试...")
                            continue

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

                except (ProgressPersistenceError, SubmittedTaskCheckpointError):
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
        self, shot: dict, prev_last_frame: Optional[str],
    ) -> tuple[list[str], Optional[str]]:
        """按单一职责选择参考图，避免身份与起始状态共用一个 role。

        返回 (image_urls, role):
        ┌───────────────────────────────────────────────────────────┐
        │ seamless: 尾帧 + first_frame，锁定真实起始状态            │
        │ intentional_cut: 角色图 + reference_image，锁定身份       │
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
    def _inject_shot_contract(prompt: str, shot: dict) -> str:
        """把一个主动作和明确起止状态编译为紧凑自然语言约束。"""
        primary_action = str(shot.get("primary_action", "")).strip()
        start = VideoGenerator._state_summary(shot.get("start_state"))
        end = VideoGenerator._state_summary(shot.get("end_state"))
        parts = []
        if shot.get("continuity_from_previous") == "seamless":
            parts.append(
                "start exactly from the supplied first frame and preserve its visible state"
            )
        elif start:
            parts.append(f"start exactly with {start}")
        if primary_action:
            parts.append(f"perform only this primary action: {primary_action}")
        if end:
            parts.append(f"finish with {end}")
        if not parts:
            return prompt
        return (
            f"[Shot contract — {'; '.join(parts)}; "
            "do not introduce another major action] "
            f"{prompt}"
        )

    @staticmethod
    def _state_summary(state: object) -> str:
        if not isinstance(state, dict):
            return ""
        values = [
            str(state.get(key, "")).strip()
            for key in ("location", "subject", "action_phase", "camera")
        ]
        return ", ".join(value for value in values if value)

    async def _extract_character_ref(self, shot: dict, video_path: str):
        """从生成的视频中提取角色参考帧, 存储到本地供后续镜头使用"""
        for char_name in shot.get("characters", ["main"]):
            ref_path = str(
                self.output_dir / "character_refs" / f"{char_name}.jpg"
            )
            extract_frame(video_path, ref_path)
            self.character_refs[char_name] = ref_path
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
        if (
            prev_scene != curr_scene
            or current_shot.get("continuity_from_previous") == "intentional_cut"
        ):
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

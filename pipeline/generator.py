"""Stage 2: 视频生成 — 角色一致性 + 降级策略 + 即时下载"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import config
from tools.seedance_api import SeedanceAPI, GenerationResult, GenerationStatus
from tools.frame_extractor import extract_frame, check_video_quality


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


class VideoGenerator:
    """视频生成引擎 — 角色一致性优先 + 画面衔接"""

    def __init__(self, output_dir: str):
        self.api = SeedanceAPI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "shots").mkdir(exist_ok=True)
        (self.output_dir / "character_refs").mkdir(exist_ok=True)
        self.character_refs: dict[str, str] = {}  # name → 本地图片路径

    async def generate_all(self, storyboard: dict) -> list[ShotResult]:
        """
        按顺序生成所有镜头

        角色一致性策略:
        ┌─────────────────────────────────────────────────────────────────┐
        │ Shot 1 (含角色): T2V 纯文本 → 提取角色参考帧到本地              │
        │ Shot 2+ (含角色): reference_image (角色参考帧 + 上一帧尾帧)     │
        │ Shot N  (无角色): first_frame (上一帧尾帧, I2V 强衔接)          │
        │                                                                 │
        │ 核心原则:                                                       │
        │ - 角色镜头用 reference_image 保证角色外观一致                    │
        │ - 非角色镜头用 first_frame 保证画面衔接                         │
        │ - 两种 role 不在同一镜头混用 (API 限制)                         │
        └─────────────────────────────────────────────────────────────────┘
        """
        results = []
        prev_last_frame: Optional[str] = None

        for shot in storyboard["shots"]:
            print(f"\n  🎬 生成 Shot {shot['shot_id']}...")

            result = await self._generate_single_shot(
                shot=shot,
                prev_last_frame=prev_last_frame,
                storyboard=storyboard,
            )
            results.append(result)

            if result.status == "success":
                prev_last_frame = result.last_frame_url

                print(f"     ✓ 完成 (质量: {result.quality_score}, 模型: {result.model_used})")
                print(f"     [DEBUG] last_frame_url = {prev_last_frame}")
                print(f"     [DEBUG] character_refs = {list(self.character_refs.keys())}")
            else:
                # 断链兜底: 保留上一张成功尾帧, 让后续镜头仍能衔接
                print(f"     ✗ 失败: {result.errors[-1] if result.errors else 'unknown'}")
                print(f"     ↪ 保留上一成功尾帧续接 (不退化为纯文本)")

        return results

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
        storyboard: dict
    ) -> ShotResult:
        """生成单个镜头 (含缓存 + 降级链 + 429 限流退避)"""

        result = ShotResult(shot_id=shot["shot_id"])

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
                result.last_frame_url = None
                print(f"     ♻️ 缓存命中: {cached_path} (质量: {qa['quality_score']})")

                # 缓存命中时也要提取角色参考 (否则后续镜头没有角色锚点)
                if shot.get("extract_character_ref") and not self.character_refs:
                    await self._extract_character_ref(shot, cached_path)

                return result
            else:
                print(f"     ⚠ 缓存文件 QA 不通过 ({qa['issues']}), 重新生成")

        rate_limit_backoff = 0
        max_rate_limit_retries = 3

        use_refs = True  # 是否使用参考图 (失败后会关闭)
        for level, deg_config in enumerate(config.DEGRADATION_CHAIN):
            if rate_limit_backoff >= max_rate_limit_retries:
                break

            for attempt in range(3):
                result.attempts += 1
                result.model_used = deg_config["model"]
                result.resolution_used = deg_config["resolution"]

                try:
                    # 构建 prompt (注入角色描述, 双重保障)
                    prompt = self._inject_character_description(
                        shot["prompt_en"], shot, storyboard
                    )
                    if shot.get("negative_prompt"):
                        prompt += f". {shot['negative_prompt']}"

                    # 构建参考图列表 + 确定 role
                    image_urls, role = self._build_image_refs(
                        shot, prev_last_frame
                    ) if use_refs else ([], None)

                    # 调用 API
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
                    )

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

                        # 提取角色参考帧 (Shot 1 标记了 extract_character_ref)
                        if shot.get("extract_character_ref"):
                            await self._extract_character_ref(shot, local_path)

                        result.status = "success"
                        return result

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

                except asyncio.TimeoutError:
                    result.errors.append(f"L{level}: 超时")
                    print(f"     ⚠ L{level} 超时")
                except Exception as e:
                    result.errors.append(f"L{level}: {str(e)}")
                    print(f"     ⚠ L{level} 异常: {str(e)[:200]}")

                await asyncio.sleep(5 * (attempt + 1))

        result.status = "failed"
        return result

    def _build_image_refs(
        self, shot: dict, prev_last_frame: Optional[str],
    ) -> tuple[list[str], Optional[str]]:
        """构建参考图列表 — 角色一致性优先策略

        返回 (image_urls, role):
        ┌───────────────────────────────────────────────────────────┐
        │ 有角色参考帧:                                             │
        │   → [角色参考帧, 上一帧尾帧] + "reference_image"          │
        │   角色参考帧是本地文件 (API 层自动转 base64)              │
        │   上一帧尾帧是远程 URL                                    │
        │   统一用 reference_image, 保证角色外观一致                │
        │                                                           │
        │ 无角色参考帧:                                             │
        │   → [上一帧尾帧] + "first_frame"                         │
        │   单图用 first_frame, I2V 强衔接                         │
        │                                                           │
        │ 什么都没有:                                               │
        │   → [] + None                                             │
        │   T2V 纯文本生成                                         │
        └───────────────────────────────────────────────────────────┘
        """
        # 收集本镜头涉及的角色参考帧 (本地文件路径)
        shot_chars = shot.get("characters", [])
        char_ref_paths = []
        for char_name in shot_chars:
            if char_name in self.character_refs:
                path = self.character_refs[char_name]
                if os.path.isfile(path):
                    char_ref_paths.append(path)

        if char_ref_paths:
            # ─── 角色模式: reference_image ───
            refs = list(char_ref_paths)  # 角色参考帧 (本地, 后续转 base64)
            # 追加上一帧尾帧用于场景连贯 (远程 URL)
            if prev_last_frame and not os.path.isfile(prev_last_frame):
                refs.append(prev_last_frame)
            print(f"     [REF] 角色模式: {len(refs)} 张 reference_image "
                  f"(角色: {len(char_ref_paths)}, 尾帧: {1 if prev_last_frame else 0})")
            return refs, "reference_image"

        # ─── 非角色模式: first_frame (I2V 强衔接) ───
        if prev_last_frame and not os.path.isfile(prev_last_frame):
            print(f"     [REF] 衔接模式: 1 张 first_frame")
            return [prev_last_frame], "first_frame"

        # ─── 无参考图: T2V ───
        print(f"     [REF] T2V 模式: 无参考图")
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

    async def _extract_character_ref(self, shot: dict, video_path: str):
        """从生成的视频中提取角色参考帧, 存储到本地供后续镜头使用"""
        for char_name in shot.get("characters", ["main"]):
            ref_path = str(
                self.output_dir / "character_refs" / f"{char_name}.jpg"
            )
            extract_frame(video_path, ref_path)
            self.character_refs[char_name] = ref_path
            print(f"     📸 角色参考帧已提取: {char_name} → {ref_path}")

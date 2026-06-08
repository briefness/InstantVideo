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
    """视频生成引擎 — 带角色一致性和降级链"""

    def __init__(self, output_dir: str):
        self.api = SeedanceAPI()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "shots").mkdir(exist_ok=True)
        (self.output_dir / "character_refs").mkdir(exist_ok=True)
        self.character_refs: dict[str, str] = {}  # name → image_url or local_path

    async def generate_all(self, storyboard: dict) -> list[ShotResult]:
        """
        按顺序生成所有镜头 (保持角色一致性)

        策略:
        - 角色锚点: 首个成功且含角色的镜头, 缓存其 last_frame_url 作为
          跨镜头稳定参考。后续所有含角色镜头都额外参考该锚点, 避免链式漂移。
        - 链式衔接: 同时用上一镜尾帧保证相邻镜头过渡平滑。
        - 断链兜底: 某镜失败时保留最近一张成功尾帧, 不清空 (避免后续全退化 T2V)。
        - 每个镜头生成后立即下载, 失败走降级链。
        """
        results = []
        prev_last_frame: Optional[str] = None
        character_anchor_url: Optional[str] = None  # 跨镜头角色锚点

        for shot in storyboard["shots"]:
            print(f"\n  🎬 生成 Shot {shot['shot_id']}...")

            result = await self._generate_single_shot(
                shot=shot,
                prev_last_frame=prev_last_frame,
                character_anchor_url=character_anchor_url,
                storyboard=storyboard,
            )
            results.append(result)

            if result.status == "success":
                prev_last_frame = result.last_frame_url

                # 设定角色锚点: 首个含角色且成功的镜头
                if character_anchor_url is None and self._shot_has_character(shot):
                    if result.last_frame_url:
                        character_anchor_url = result.last_frame_url
                        print(f"     🔒 角色锚点已锁定: Shot {shot['shot_id']}")

                print(f"     ✓ 完成 (质量: {result.quality_score}, 模型: {result.model_used})")
                print(f"     [DEBUG] last_frame_url = {prev_last_frame}")
                print(f"     [DEBUG] anchor = {character_anchor_url}")
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
        character_anchor_url: Optional[str], storyboard: dict
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
                # last_frame_url 无法恢复 (远程 URL 会过期), 但 generate_all
                # 的断链兜底会保留上一个成功镜头的值
                result.last_frame_url = None
                print(f"     ♻️ 缓存命中: {cached_path} (质量: {qa['quality_score']})")
                return result
            else:
                print(f"     ⚠ 缓存文件 QA 不通过 ({qa['issues']}), 重新生成")

        rate_limit_backoff = 0  # 429 连续退避计数器 (跨降级级别)
        max_rate_limit_retries = 3  # 429 最多重试 3 次

        use_refs = True  # 是否使用参考图 (失败后会关闭)
        for level, deg_config in enumerate(config.DEGRADATION_CHAIN):
            # 如果已触发 429 限流, 跳过降级 (降分辨率对配额无效)
            if rate_limit_backoff >= max_rate_limit_retries:
                break

            for attempt in range(3):  # 每级尝试 3 次 (含参考图失败后的无图重试)
                result.attempts += 1
                result.model_used = deg_config["model"]
                result.resolution_used = deg_config["resolution"]

                try:
                    # 构建 prompt
                    prompt = shot["prompt_en"]
                    if shot.get("negative_prompt"):
                        prompt += f". {shot['negative_prompt']}"

                    # 构建参考图列表
                    image_urls = self._build_image_refs(
                        shot, prev_last_frame, character_anchor_url
                    ) if use_refs else []

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
                            continue  # 重试

                        # 提取角色参考帧 (如果标记了)
                        if shot.get("extract_character_ref"):
                            await self._extract_character_ref(shot, local_path)

                        result.status = "success"
                        return result

                    elif gen_result.get("error_type") == "moderation":
                        result.errors.append(f"L{level}: 审核失败")
                        break  # 审核失败不重试

                    elif gen_result.get("error_type") == "rate_limit":
                        # ─── 429 限流: 指数退避, 不降级 ───
                        rate_limit_backoff += 1
                        if rate_limit_backoff >= max_rate_limit_retries:
                            result.errors.append(
                                f"L{level}: 配额耗尽, 已退避 {rate_limit_backoff} 次仍失败"
                            )
                            break  # 跳出当前级别, 外层也会因计数器跳出
                        wait_secs = 30 * (2 ** (rate_limit_backoff - 1))  # 30s, 60s, 120s
                        result.errors.append(
                            f"L{level}: 429 限流, 第 {rate_limit_backoff} 次退避 {wait_secs}s..."
                        )
                        print(f"     ⏳ 429 限流, 等待 {wait_secs}s 后重试...")
                        await asyncio.sleep(wait_secs)
                        continue  # 在当前级别重试, 不降级

                    else:
                        err_msg = gen_result.get("error", "unknown")
                        result.errors.append(f"L{level}: {err_msg}")
                        print(f"     ⚠ L{level} 错误: {err_msg[:200]}")

                        # 精确判断是否为参考图相关错误 (而非泛匹配 "image")
                        # 真实参考图错误: 图片下载失败/格式不支持/尺寸超限等
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
                            # 参考图本身有问题 → 关闭参考图, 当前级别无图重试
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

                # 重试前等待
                await asyncio.sleep(5 * (attempt + 1))

        # 全部失败
        result.status = "failed"
        return result

    def _build_image_refs(
        self, shot: dict, prev_last_frame: Optional[str],
        character_anchor_url: Optional[str] = None,
    ) -> list[str]:
        """构建参考图列表 — 双锚策略

        1. prev_last_frame: 上一镜尾帧 → 保证相邻镜头过渡平滑
        2. character_anchor_url: 角色锚点 → 保证跨镜头角色一致性

        API 层根据返回图片数量自动决定 role:
        - 单图 → first_frame (I2V 强衔接)
        - 多图 → 全部 reference_image (规避首帧/参考图混用限制)
        """
        refs = []

        # 上一镜尾帧 (远程 URL, 不是本地文件路径)
        if prev_last_frame and not os.path.isfile(prev_last_frame):
            refs.append(prev_last_frame)

        # 角色锚点 (和上一帧不同时才加, 避免传重复图)
        if character_anchor_url and character_anchor_url != prev_last_frame:
            if not os.path.isfile(character_anchor_url):
                refs.append(character_anchor_url)

        return refs

    async def _extract_character_ref(self, shot: dict, video_path: str):
        """从生成的视频中提取角色参考帧"""
        for char_name in shot.get("characters", ["main"]):
            ref_path = str(
                self.output_dir / "character_refs" / f"{char_name}.jpg"
            )
            extract_frame(video_path, ref_path)
            # TODO: 上传到 CDN 获取 URL (目前用本地路径)
            self.character_refs[char_name] = ref_path
            print(f"     📸 提取角色参考: {char_name} → {ref_path}")

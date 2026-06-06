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
        - 第一个含角色的镜头: T2V → 提取角色参考帧
        - 后续含角色的镜头: I2V (带角色参考图)
        - 每个镜头生成后立即下载
        - 失败走降级链
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
                prev_last_frame = None
                print(f"     ✗ 失败: {result.errors[-1] if result.errors else 'unknown'}")

        return results

    async def _generate_single_shot(
        self, shot: dict, prev_last_frame: Optional[str], storyboard: dict
    ) -> ShotResult:
        """生成单个镜头 (含降级链 + 429 限流退避)"""

        result = ShotResult(shot_id=shot["shot_id"])
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
                    image_urls = self._build_image_refs(shot, prev_last_frame) if use_refs else []

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
                                "首帧", "参考图",
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
        self, shot: dict, prev_last_frame: Optional[str]
    ) -> list[str]:
        """构建参考图列表 — 只用远程 URL，不传本地文件

        只传上一镜的 last_frame_url (远程 URL)
        暂不传本地角色参考帧 (base64 太大导致 API 拒绝)
        """
        refs = []

        # 只用远程 URL 的尾帧 (不是本地文件路径)
        if prev_last_frame and not os.path.isfile(prev_last_frame):
            refs.append(prev_last_frame)

        # 角色参考: 暂时只用 last_frame 保证连续性
        # TODO: 实现图片上传到 CDN 后再用角色参考

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

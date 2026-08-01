"""Seedance 2.0 API 封装 — 双路径: Volcengine Ark (首选) + fal.ai (备选)

Cherry-picked from OpenMontage seedance_video.py, simplified for single-project use.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import requests
from pathlib import Path
from collections.abc import Callable
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import config


class SubmittedTaskCheckpointError(RuntimeError):
    """A remote task exists, but its identity could not be persisted locally."""

    def __init__(self, task_id: str, cause: Exception):
        self.task_id = task_id
        super().__init__(f"远端任务 {task_id} 已创建，但本地状态保存失败: {cause}")


class GenerationStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MODERATION = "moderation"


@dataclass
class GenerationResult:
    """单个镜头的生成结果"""
    shot_id: int
    status: GenerationStatus
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    last_frame_url: Optional[str] = None
    audio_url: Optional[str] = None
    model_used: str = ""
    resolution_used: str = ""
    duration: float = 0
    cost_usd: float = 0
    error: Optional[str] = None
    attempt: int = 0


class SeedanceAPI:
    """Seedance 2.0 统一接口 — 自动选择最佳可用路径"""

    def __init__(self):
        self.ark_client = None
        self._init_clients()

    def _init_clients(self):
        """初始化火山引擎 Ark 客户端 (视频生成使用独立的 base URL)"""
        if config.ARK_API_KEY:
            try:
                from volcenginesdkarkruntime import Ark
                self.ark_client = Ark(
                    api_key=config.ARK_API_KEY_SEEDANCE,
                    base_url=config.ARK_BASE_URL_SEEDANCE,
                )
                print(f"  ✓ Volcengine Ark 连接就绪 (视频生成: {config.ARK_BASE_URL_SEEDANCE})")
            except ImportError:
                print("  ⚠ volcengine SDK 未安装: pip install 'volcengine-python-sdk[ark]'")
                raise
        else:
            raise RuntimeError("未设置 ARK_API_KEY。请在 .env 中配置。")

    @property
    def supports_last_frame(self) -> bool:
        """是否支持返回尾帧"""
        return True

    # ─── 主入口: 生成单个镜头 ───

    async def generate(
        self,
        prompt: str,
        duration: int = 5,
        ratio: str = "16:9",
        resolution: str = config.DEFAULT_RESOLUTION,
        model: str = config.SEEDANCE_MODEL,
        generate_audio: bool = True,
        return_last_frame: bool = True,
        image_urls: list[str] | None = None,
        image_role: str | None = None,
        timeout: int = 300,
        task_id: str | None = None,
        on_submitted: Callable[[str], None] | None = None,
    ) -> dict:
        """
        生成视频 — 通过火山引擎 Ark

        Args:
            image_urls: 参考图 URL 列表 (远程 URL 或本地文件路径)
            image_role: 参考图角色, 由调用方指定:
                - "first_frame": I2V 强衔接 (只能 1 张)
                - "reference_image": 主体/角色参考 (可多张)
                - None: 自动推断 (1张=first_frame, 多张=reference_image)

        Returns:
            {"status": "succeeded", "video_url": "...", "last_frame_url": "...", ...}
            {"status": "failed", "error": "...", "error_type": "moderation|timeout|unknown"}
        """
        if resolution not in config.SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Seedance 2.0 Mini 不支持分辨率: {resolution}")
        if ratio not in config.SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"Seedance 2.0 Mini 不支持画幅: {ratio}")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or not 4 <= duration <= 15
        ):
            raise ValueError("Seedance 2.0 Mini 时长必须为 4-15 秒的整数")

        return await self._generate_ark(
            prompt=prompt,
            duration=duration,
            ratio=ratio,
            resolution=resolution,
            model=model,
            generate_audio=generate_audio,
            return_last_frame=return_last_frame,
            image_urls=image_urls,
            image_role=image_role,
            timeout=timeout,
            task_id=task_id,
            on_submitted=on_submitted,
        )

    # ─── Volcengine Ark 路径 ───

    async def _generate_ark(
        self, prompt, duration, ratio, resolution, model,
        generate_audio, return_last_frame, image_urls, image_role, timeout,
        task_id=None, on_submitted=None,
    ) -> dict:
        """通过火山引擎 Ark 官方 API 生成"""
        if task_id is None:
            content = [{"type": "text", "text": prompt}]
            if image_urls:
                role = image_role or (
                    "first_frame" if len(image_urls) == 1 else "reference_image"
                )
                for url in image_urls:
                    actual_url = (
                        self._local_image_to_data_uri(url) if os.path.isfile(url) else url
                    )
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": actual_url},
                        "role": role,
                    })

            try:
                print(f"     [API] model={model}, duration={duration}, resolution={resolution}")
                print(f"     [API] content items: {len(content)} (types: {[c['type'] for c in content]})")
                if len(content) > 1:
                    print(f"     [API] image roles: {[c.get('role') for c in content if c['type']=='image_url']}")
                resp = self.ark_client.content_generation.tasks.create(
                    model=model,
                    content=content,
                    resolution=resolution,
                    ratio=ratio,
                    duration=duration,
                    generate_audio=generate_audio,
                    return_last_frame=return_last_frame,
                    watermark=False,
                )
                task_id = resp.id
            except Exception as exc:
                return self._failure_from_error(exc)

            if on_submitted:
                try:
                    on_submitted(task_id)
                except Exception as exc:
                    raise SubmittedTaskCheckpointError(task_id, exc) from exc
        else:
            print(f"     [API] 恢复远端任务: {task_id}")

        return await self._poll_ark(task_id, timeout)

    async def _poll_ark(self, task_id: str, timeout: int) -> dict:
        """Poll an existing Ark task so recovery never needs to resubmit it."""
        start_time = time.time()
        wait = 10
        consecutive_errors = 0
        last_poll_failure: dict | None = None

        while time.time() - start_time < timeout:
            try:
                result = self.ark_client.content_generation.tasks.get(task_id=task_id)
                consecutive_errors = 0
                status = result.status

                if status == "succeeded":
                    return {
                        "status": "succeeded",
                        "provider_task_id": task_id,
                        "video_url": result.content.video_url,
                        "last_frame_url": getattr(result.content, "last_frame_url", None),
                        "audio_url": getattr(result.content, "audio_url", None),
                    }
                elif status in ("failed", "expired", "cancelled"):
                    error_msg = getattr(result, "error", status)
                    msg_low = str(error_msg).lower()
                    if any(kw in msg_low for kw in [
                        "real person", "privacy", "sensitive",
                        "privacyinformation", "人脸", "真人", "肖像",
                    ]):
                        error_type = "privacy"
                    elif "content" in msg_low:
                        error_type = "moderation"
                    else:
                        error_type = "unknown"
                    return {
                        "status": "failed",
                        "provider_task_id": task_id,
                        "error": str(error_msg),
                        "error_type": error_type,
                    }

            except Exception as exc:
                failure = self._failure_from_error(exc)
                failure["status"] = "pending"
                failure["provider_task_id"] = task_id
                failure["error_type"] = "poll_error"
                last_poll_failure = failure
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    return failure
                await asyncio.sleep(0.25 * (2 ** (consecutive_errors - 1)))
                continue

            await asyncio.sleep(wait)
            wait = min(wait * 1.5, 60)

        return last_poll_failure or {
            "status": "pending",
            "provider_task_id": task_id,
            "error": f"Timeout ({timeout}s)",
            "error_type": "poll_timeout",
        }

    @staticmethod
    def _failure_from_error(exc: Exception) -> dict:
        error = str(exc)
        lowered = error.lower()
        if any(kw in lowered for kw in [
            "real person", "privacy", "sensitive",
            "privacyinformation", "人脸", "真人", "肖像",
        ]):
            error_type = "privacy"
        elif "content" in lowered and "moderation" in lowered:
            error_type = "moderation"
        elif "429" in error or "quotaexceeded" in lowered or "toomanyrequests" in lowered:
            error_type = "rate_limit"
        else:
            error_type = "unknown"
        return {"status": "failed", "error": error, "error_type": error_type}

    # ─── fal.ai 路径 ───

    async def _generate_fal(
        self, prompt, duration, ratio, resolution, generate_audio, image_urls, timeout
    ) -> dict:
        """通过 fal.ai 生成 (备选路径)"""

        # 确定 operation
        if image_urls:
            operation = "image-to-video" if len(image_urls) <= 2 else "reference-to-video"
        else:
            operation = "text-to-video"

        model_path = f"bytedance/seedance-2.0/{operation}"
        fal_url = f"https://queue.fal.run/{model_path}"

        payload = {
            "prompt": prompt,
            "duration": str(duration),
            "aspect_ratio": ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }

        if image_urls and operation == "image-to-video":
            payload["image_url"] = image_urls[0]
            if len(image_urls) > 1:
                payload["end_image_url"] = image_urls[1]
        elif image_urls and operation == "reference-to-video":
            payload["reference_image_urls"] = image_urls[:9]

        headers = {
            "Authorization": f"Key {config.FAL_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # Submit
            resp = requests.post(fal_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            request_id = data.get("request_id")

            if not request_id:
                # 同步返回
                video_url = data.get("video", {}).get("url")
                if video_url:
                    return {"status": "succeeded", "video_url": video_url, "last_frame_url": None}
                return {"status": "failed", "error": "No video in response", "error_type": "unknown"}

            # Poll
            status_url = f"https://queue.fal.run/{model_path}/requests/{request_id}/status"
            result_url = f"https://queue.fal.run/{model_path}/requests/{request_id}"
            start_time = time.time()
            wait = 10

            while time.time() - start_time < timeout:
                status_resp = requests.get(status_url, headers=headers, timeout=15)
                status_data = status_resp.json()
                status = status_data.get("status")

                if status == "COMPLETED":
                    result_resp = requests.get(result_url, headers=headers, timeout=15)
                    result_data = result_resp.json()
                    video_url = result_data.get("video", {}).get("url")
                    return {
                        "status": "succeeded",
                        "video_url": video_url,
                        "last_frame_url": None,  # fal.ai 不支持
                    }
                elif status == "FAILED":
                    error = status_data.get("error", "Unknown fal.ai error")
                    return {"status": "failed", "error": str(error), "error_type": "unknown"}

                await asyncio.sleep(wait)
                wait = min(wait * 1.5, 60)

            return {"status": "failed", "error": f"Timeout ({timeout}s)", "error_type": "timeout"}

        except Exception as e:
            return {"status": "failed", "error": str(e), "error_type": "unknown"}

    # ─── 工具方法 ───

    @staticmethod
    def _local_image_to_data_uri(filepath: str) -> str:
        """将本地图片转为 base64 data URI"""
        import mimetypes
        mime_type = mimetypes.guess_type(filepath)[0] or "image/jpeg"
        with open(filepath, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    async def download_video(self, video_url: str, save_path: str) -> str:
        """下载生成的视频到本地 (URL 24h 过期, 必须立即下载)"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(video_url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

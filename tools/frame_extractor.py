"""视频帧提取 — 角色参考帧 + 质量检测"""

from __future__ import annotations

import os
from pathlib import Path


def extract_frame(video_path: str, output_path: str, timestamp: float = None) -> str:
    """
    从视频中提取一帧作为角色参考图

    Args:
        video_path: 视频文件路径
        output_path: 输出 jpg 路径
        timestamp: 指定时间点 (秒)。None = 取中间帧
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if timestamp is not None:
        frame_idx = int(timestamp * fps)
    else:
        frame_idx = total_frames // 2

    frame_idx = max(0, min(frame_idx, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"无法从 {video_path} 提取帧 (idx={frame_idx})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def check_video_quality(video_path: str) -> dict:
    """
    自动检测视频质量 (模糊 + 闪烁)

    Returns:
        {
            "quality_score": 85,      # 0-100
            "pass": True,             # >= 60 通过
            "blur_ratio": 0.1,
            "flicker_ratio": 0.05,
            "issues": [],
        }
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    blur_count = 0
    flicker_count = 0
    prev_frame = None
    sample_interval = max(1, total_frames // 30)
    samples = 0

    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        samples += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 模糊检测
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            blur_count += 1

        # 闪烁检测
        if prev_frame is not None:
            diff = np.abs(gray.astype(float) - prev_frame.astype(float)).mean()
            if diff > 40:
                flicker_count += 1

        prev_frame = gray

    cap.release()

    blur_ratio = blur_count / max(samples, 1)
    flicker_ratio = flicker_count / max(samples - 1, 1)

    score = 100
    issues = []

    if blur_ratio > 0.3:
        score -= 30
        issues.append(f"模糊帧占比 {blur_ratio:.0%}")
    if flicker_ratio > 0.2:
        score -= 25
        issues.append(f"闪烁帧占比 {flicker_ratio:.0%}")

    return {
        "quality_score": max(0, score),
        "pass": score >= 60,
        "blur_ratio": blur_ratio,
        "flicker_ratio": flicker_ratio,
        "issues": issues,
    }

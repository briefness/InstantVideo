"""视频帧提取 — 智能参考帧选取 + 多维质量检测"""

from __future__ import annotations

import os
from pathlib import Path


def extract_frame(video_path: str, output_path: str, timestamp: float = None) -> str:
    """
    从视频中提取最佳帧作为角色参考图

    策略:
    - 指定 timestamp 时直接提取该时间点
    - 未指定时: 在视频前半段等间距采样 N 帧, 按清晰度 (Laplacian 方差) 排序,
      选最清晰的帧。前半段优先是因为角色通常在开头出现且姿态较稳定。

    Args:
        video_path: 视频文件路径
        output_path: 输出 jpg 路径
        timestamp: 指定时间点 (秒)。None = 智能选帧
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if timestamp is not None:
        # 指定时间点 → 直接提取
        frame_idx = int(timestamp * fps)
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"无法从 {video_path} 提取帧 (idx={frame_idx})")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return output_path

    # ─── 智能选帧: 前半段多帧采样 + 清晰度评分 ───
    # 取前 60% 的帧作为候选区间 (角色通常在视频前段出现且姿态稳定)
    search_end = int(total_frames * 0.6)
    search_end = max(search_end, min(total_frames, 10))  # 至少覆盖 10 帧

    num_samples = min(8, search_end)  # 采样 8 帧
    sample_interval = max(1, search_end // num_samples)

    candidates: list[tuple[int, float, any]] = []  # (frame_idx, sharpness, frame)

    for i in range(0, search_end, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Laplacian 方差 = 清晰度指标 (越高越清晰)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        candidates.append((i, sharpness, frame))

    cap.release()

    if not candidates:
        raise RuntimeError(f"无法从 {video_path} 提取任何帧")

    # 选最清晰的帧
    best = max(candidates, key=lambda x: x[1])
    print(f"     [FRAME] 采样 {len(candidates)} 帧, "
          f"选中 idx={best[0]} (清晰度={best[1]:.0f}, "
          f"最差={min(c[1] for c in candidates):.0f})")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, best[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def check_video_quality(video_path: str) -> dict:
    """
    多维视频质量检测

    检测维度:
    1. 模糊 (Laplacian 方差 < 50)
    2. 闪烁 (帧间亮度差 > 40)
    3. 色彩异常 (帧间颜色直方图相关度骤降)
    4. 时序突变 (帧间结构相似度骤降, 检测物体突然出现/消失)

    Returns:
        {
            "quality_score": 85,      # 0-100
            "pass": True,             # >= 60 通过
            "blur_ratio": 0.1,
            "flicker_ratio": 0.05,
            "color_anomaly_ratio": 0.02,
            "temporal_break_ratio": 0.01,
            "issues": [],
        }
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    blur_count = 0
    flicker_count = 0
    color_anomaly_count = 0
    temporal_break_count = 0
    prev_gray = None
    prev_hist = None
    sample_interval = max(1, total_frames // 30)
    samples = 0

    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        samples += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 模糊检测 (Laplacian 方差)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            blur_count += 1

        if prev_gray is not None:
            # 2. 闪烁检测 (帧间亮度差)
            diff = np.abs(gray.astype(float) - prev_gray.astype(float)).mean()
            if diff > 40:
                flicker_count += 1

            # 3. 色彩异常检测 (帧间颜色直方图相关度)
            # 用 HSV 色相通道的直方图做比较, 对光线变化更鲁棒
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32],
                                [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                # 相关系数: 1.0=完全一致, 0=完全不同, <0=反相关
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < 0.7:
                    color_anomaly_count += 1
            prev_hist = hist

            # 4. 时序一致性 (帧间结构相似度 — SSIM 简化版)
            # 用缩小到 64x64 的灰度图做快速结构对比
            small_curr = cv2.resize(gray, (64, 64))
            small_prev = cv2.resize(prev_gray, (64, 64))
            # 归一化互相关作为结构相似度代理
            ncc = np.corrcoef(
                small_curr.flatten().astype(float),
                small_prev.flatten().astype(float),
            )[0, 1]
            # NCC < 0.5 表示帧间结构剧变 (物体突然出现/消失/形变)
            if ncc < 0.5:
                temporal_break_count += 1

        prev_gray = gray

    cap.release()

    compared_pairs = max(samples - 1, 1)
    blur_ratio = blur_count / max(samples, 1)
    flicker_ratio = flicker_count / compared_pairs
    color_anomaly_ratio = color_anomaly_count / compared_pairs
    temporal_break_ratio = temporal_break_count / compared_pairs

    score = 100
    issues = []

    if blur_ratio > 0.3:
        score -= 30
        issues.append(f"模糊帧占比 {blur_ratio:.0%}")
    if flicker_ratio > 0.2:
        score -= 25
        issues.append(f"闪烁帧占比 {flicker_ratio:.0%}")
    if color_anomaly_ratio > 0.15:
        score -= 20
        issues.append(f"色彩异常占比 {color_anomaly_ratio:.0%}")
    if temporal_break_ratio > 0.1:
        score -= 15
        issues.append(f"时序突变占比 {temporal_break_ratio:.0%}")

    return {
        "quality_score": max(0, score),
        "pass": score >= 60,
        "blur_ratio": blur_ratio,
        "flicker_ratio": flicker_ratio,
        "color_anomaly_ratio": color_anomaly_ratio,
        "temporal_break_ratio": temporal_break_ratio,
        "issues": issues,
    }

"""音乐 BPM 分析 & 节拍检测 — 用于镜头卡点对齐"""

from __future__ import annotations
from typing import Optional


def analyze_beats(audio_path: str) -> dict:
    """
    分析音乐的 BPM、节拍点和能量包络

    Returns:
        {
            "bpm": 120.0,
            "beat_times": [0.5, 1.0, 1.5, ...],     # 所有节拍时间点 (秒)
            "downbeat_times": [0.5, 2.5, 4.5, ...],  # 强拍 (每 4 拍)
            "beat_interval": 0.5,                     # 每拍间隔 (秒)
            "total_duration": 30.0,
            "climax_time": 18.5,                      # 能量最高点
        }
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050)

    # BPM + 节拍检测
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # 强拍 (4/4 拍每小节第一拍)
    downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times

    # 能量分析 → 找高潮段
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(range(len(rms)), sr=sr)
    # 平滑后找峰值
    smoothed = np.convolve(rms, np.ones(50) / 50, mode="same")
    climax_idx = int(np.argmax(smoothed))
    climax_time = float(rms_times[climax_idx]) if climax_idx < len(rms_times) else 0

    return {
        "bpm": float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]),
        "beat_times": beat_times.tolist(),
        "downbeat_times": downbeat_times.tolist(),
        "beat_interval": 60.0 / (float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])),
        "total_duration": float(len(y) / sr),
        "climax_time": climax_time,
    }


def align_durations_to_beats(
    beat_analysis: dict,
    num_shots: int,
    min_duration: int = 4,
    max_duration: int = 15,
) -> list[int]:
    """
    根据音乐节拍计算每个镜头的最佳时长 (对齐到节拍点)

    策略: 每个镜头的切换点落在 beat 上, 时长为 beat_interval 的整数倍

    Returns:
        [5, 5, 8, 5, 4, 3]  # 每个镜头的时长 (秒, 整数)
    """
    beat_interval = beat_analysis["beat_interval"]
    total_duration = beat_analysis["total_duration"]

    # 每镜头平均多少拍
    total_beats = len(beat_analysis["beat_times"])
    avg_beats_per_shot = max(
        int(min_duration / beat_interval),
        total_beats // num_shots,
    )

    durations = []
    for i in range(num_shots):
        # 以 beat_interval 的整数倍为时长
        raw_duration = avg_beats_per_shot * beat_interval
        # 限制到 Seedance 允许范围
        clamped = max(min_duration, min(max_duration, round(raw_duration)))
        durations.append(clamped)

    # 微调: 如果总时长超出音乐时长, 缩短后面的镜头
    while sum(durations) > total_duration and any(d > min_duration for d in durations):
        for i in range(len(durations) - 1, -1, -1):
            if durations[i] > min_duration:
                durations[i] -= 1
                break

    return durations

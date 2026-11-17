"""连续像素只提供重播反证，未命中不能证明语义推进。

比较整个已用片段与当前开头，避免只看尾帧漏掉原片中段重演。
保守限定近似复制的动态序列，不将相同背景、静止保持或单帧相似当重播。
"""
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np

VERSION = 'dynamic-prefix-reuse-v1'
FPS = 4


def compare_sequences(previous, current) -> dict:
    arrays = [np.asarray(a, dtype=np.float32) for a in (previous, current)]
    if any(a.ndim != 3 or a.shape[1:] != (36, 64) or len(a) > 120
           or not np.isfinite(a).all() or np.any((a < 0) | (a > 256)) for a in arrays):
        raise ValueError('重播测量采样无效')
    previous, current = arrays
    result = {'version': VERSION, 'status': 'inconclusive', 'sample_fps': FPS}
    # 2秒连续动态相似证据；允许单个视频帧的首帧量化差，不搜索任意当前片段。
    for start in range(max(0, len(previous) - 7)):
        n = min(len(previous) - start, len(current))
        if n < 8:
            continue
        a, b = previous[start:start+n], current[:n]
        close = np.mean(np.abs(a-b), axis=(1, 2)) <= 5
        count = next((i for i, match in enumerate(close) if not match), n)
        if count < 8:
            continue
        da, db = np.diff(a[:count], axis=0), np.diff(b[:count], axis=0)
        energy = min(float(np.abs(da).mean()), float(np.abs(db).mean()))
        denominator = float(np.linalg.norm(da) * np.linalg.norm(db))
        correlation = float(np.sum(da*db)/denominator) if denominator else 0.
        if energy >= 2 and correlation >= .9:
            return {**result, 'status': 'replayed_prefix', 'previous_start_seconds': start/FPS,
                    'current_start_seconds': 0., 'matched_seconds': count/FPS,
                    'motion_correlation': round(correlation, 6), 'motion_energy': round(energy, 6)}
    return result


def measure_reuse(previous, current, previous_duration: float, current_duration: float) -> dict:
    frames, hashes = [], []
    for value, duration in ((previous, previous_duration), (current, current_duration)):
        path = Path(value)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000_000
                or type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 30):
            raise ValueError('重播测量缺少有效有界原片')
        with path.open('rb') as stream:
            hashes.append(hashlib.file_digest(stream, 'sha256').hexdigest())
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-t', str(duration),
            '-vf', 'fps=4,scale=64:36,format=gray', '-frames:v', '120', '-f', 'rawvideo', '-'],
            capture_output=True, check=True, timeout=30).stdout
        if not raw or len(raw) % (36*64):
            raise ValueError('重播测量视频解码不完整')
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64))
    return {**compare_sequences(*frames), 'previous_sha256': hashes[0], 'current_sha256': hashes[1],
            'previous_duration': previous_duration, 'current_duration': current_duration}

"""Check encoded reference pixels against the deterministic planned poses offline."""
from __future__ import annotations

import subprocess

import numpy as np
from PIL import Image

from pipeline.motion_plan import MotionPlan
from tools.motion_renderer import draw_motion_frame


def _joint_coverage(actual: np.ndarray, plan: MotionPlan, sample: dict) -> float:
    """Independent geometric oracle: projected joints must have their role's color."""
    height, width, _ = actual.shape
    red, green, blue = (actual[:, :, i] for i in range(3))
    masks = ((green - red > 30) & (blue - red > 25),
             (red - blue > 60) & (red - green > 15) & (green - blue > 15))
    coverage = []
    for role, mask in zip(('actor', 'target'), masks):
        points = sample['entities'][sample['roles'][role]]
        visible = 0
        for x, y in points.values():
            px = round(x * min(plan.width, plan.height) * width / plan.width)
            py = round(y * min(plan.width, plan.height) * height / plan.height)
            visible += bool(mask[max(0, py - 2):min(height, py + 3),
                                 max(0, px - 2):min(width, px + 3)].any())
        coverage.append(visible / len(points))
    return min(coverage)


def verify_rendered_motion(payload: bytes, plan: MotionPlan, compiled: dict) -> dict:
    """Certify sampled rendering fidelity, never generated-footage semantics or physics."""
    if not payload or len(payload) > 32 * 1024 * 1024:
        raise ValueError('动作画面校验需要有界的本地视频')
    if compiled['diagnostics']['status'] != 'valid_geometry':
        raise ValueError('动作画面来源的几何检查未通过')
    fps = max(24, plan.fps)
    count = round(compiled['frame_count'] * fps / plan.fps)
    # 均匀覆盖运动过程，并保留接触/反应前后帧，不能只看开始与结尾。
    indices = {round(index * (count - 1) / 15) for index in range(16)}
    for frame in (plan.contact.frame, plan.contact.reaction_frame):
        center = round(frame * fps / plan.fps)
        indices.update(max(0, min(count - 1, center + offset)) for offset in (-1, 0, 1))
    indices = sorted(indices)
    scale = 256 / max(plan.width, plan.height)
    width, height = max(1, round(plan.width * scale)), max(1, round(plan.height * scale))
    selection = '+'.join(f'eq(n,{index})' for index in indices)
    try:
        result = subprocess.run([
            'ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', 'pipe:0',
            '-vf', f"fps={fps},select='{selection}',scale={width}:{height}:flags=area",
            '-frames:v', str(len(indices)), '-fps_mode', 'passthrough', '-an',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1',
        ], input=payload, capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        raise ValueError('动作参考画面无法解码核验') from None
    if len(result.stdout) != len(indices) * width * height * 3:
        raise ValueError('动作参考画面采样不完整')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(indices), height, width, 3)
    errors, coverages = [], []
    background = np.array([16, 24, 32], dtype=np.int16)
    for index, actual in zip(indices, frames):
        source_frame = min(compiled['frame_count'] - 1, index * plan.fps // fps)
        expected = np.asarray(draw_motion_frame(plan, compiled['frames'][source_frame], annotated=False)
                              .resize((width, height), Image.Resampling.BOX), dtype=np.int16)
        actual = actual.astype(np.int16)
        coverage = _joint_coverage(actual, plan, compiled['frames'][source_frame])
        coverages.append(coverage)
        if coverage < .75:
            raise ValueError(f'动作参考画面的人物关节或角色颜色与计划不符（第 {index} 帧）')
        # 分别比较计划主体和实际前景。并集平均会被改色背景稀释，漏掉黑屏。
        expected_mask = np.max(np.abs(expected - background), axis=2) > 30
        actual_mask = np.max(np.abs(actual - background), axis=2) > 30
        if not np.any(expected_mask):
            raise ValueError('动作参考画面没有可核验的主体')
        difference = np.abs(actual - expected)
        error = max(float(difference[expected_mask].mean()),
                    float(difference[actual_mask].mean()) if np.any(actual_mask) else 0.)
        errors.append(error)
        if error > 28:
            raise ValueError(f'动作参考画面与计划运动不符（第 {index} 帧）；禁止使用')
    return {'version': 'render-fidelity-v1', 'status': 'verified_sampled_render',
            'sampled_frames': indices, 'fps': fps, 'max_foreground_mae': round(max(errors), 3),
            'min_joint_coverage': round(min(coverages), 3),
            'acceptance_kind': 'planned_render_only'}

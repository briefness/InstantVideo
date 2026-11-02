"""Offline reference validation and immutable submission identity for motion input."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from pipeline.models import MotionReferenceSpec
from pipeline.motion_plan import MotionPlan, compile_motion_previs, motion_source_binding, supported_motion_shot
from tools.motion_media_validation import verify_rendered_motion
from tools.reference_video_url import MAX_VIDEO_REFERENCE_BYTES, validate_public_video_url


MOTION_MODELS = frozenset({
    "doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-2-0-260128",
    # Agent Plan 官方模型名独立留存，不推断或替换为数据面固定版本号。
    "doubao-seedance-2.0-mini", "doubao-seedance-2.0-fast", "doubao-seedance-2.0",
})
MAX_REFERENCE_BYTES = MAX_VIDEO_REFERENCE_BYTES


def validate_video_reference(asset_uri: str, model: str, image_role: str | None,
                             image_count: int) -> None:
    if model not in MOTION_MODELS:
        raise ValueError("动作参考需要已核验的 Seedance 2.0 模型版本")
    if isinstance(asset_uri, str) and asset_uri.startswith("https://"):
        validate_public_video_url(asset_uri)
    elif not isinstance(asset_uri, str) or not re.fullmatch(r"asset://[A-Za-z0-9_-]{1,128}", asset_uri):
        raise ValueError("动作参考须绑定公网 HTTPS 链接或 asset:// 素材 ID；不能提交本地路径或 Base64")
    if image_role in {"first_frame", "last_frame"} or (
        image_count and image_role != "reference_image"
    ):
        raise ValueError("动作视频参考不能与 first_frame / last_frame 混用")
    if not 0 <= image_count <= 9:
        raise ValueError("动作视频最多搭配 9 张身份参考图")


def motion_reference_context(shot: dict) -> dict | None:
    raw = shot.get("motion_reference")
    if raw is None:
        return None
    spec = MotionReferenceSpec.model_validate(raw)
    context = {
        "version": "motion-reference-v1", "input_mode": "reference_video",
        "sha256": spec.sha256, "source_sha256": spec.source_sha256,
        "model": spec.model, "asset_uri": spec.asset_uri,
    }
    # 不给旧 asset:// 记录添加空字段，保留已有任务的精确恢复兼容性。
    if spec.video_url is not None:
        context["video_url"] = spec.video_url
    return context


def read_reference_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("动作参考必须是普通本地 MP4 文件")
    with path.open("rb") as handle:
        payload = handle.read(MAX_REFERENCE_BYTES + 1)
    if not payload or len(payload) > MAX_REFERENCE_BYTES:
        raise ValueError("动作参考不能为空或超过 32 MiB")
    return payload


def probe_motion_reference(payload: bytes, duration: int, ratio: str) -> dict:
    # 从已哈希的字节核验，不让文件在 hash 与 ffprobe 之间被替换。
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", "pipe:0",
        ], input=payload, capture_output=True, check=True, timeout=15)
        data = json.loads(result.stdout)
        streams = data["streams"]
        video = streams[0]
        width, height = video["width"], video["height"]
        numerator, denominator = map(int, video["avg_frame_rate"].split("/"))
        fps = numerator / denominator
        rw, rh = map(int, ratio.split(":"))
        valid = (
            len(streams) == 1 and video["codec_type"] == "video"
            and video["codec_name"] == "h264" and "mp4" in data["format"]["format_name"]
            and 300 <= min(width, height) and max(width, height) <= 6000
            and 407696 <= width * height <= 8295044 and 0.4 <= width / height <= 2.5
            and 24 <= fps <= 60 and math.isclose(width / height, rw / rh, rel_tol=0.005)
            and math.isclose(float(video["duration"]), duration, abs_tol=1 / fps)
            and int(video["nb_frames"]) == round(duration * fps)
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError,
            TypeError, ZeroDivisionError):
        raise ValueError("无法核验动作参考视频；请重新离线导出") from None
    if not valid:
        raise ValueError("动作参考的格式、画幅、帧率、时长或音轨不符合本地接入约定")
    return video


def prepare_motion_reference(shot: dict, ratio: str, workspace: Path, model: str) -> dict | None:
    context = motion_reference_context(shot)
    if context is None:
        return None
    spec = MotionReferenceSpec.model_validate(shot["motion_reference"])
    supported_motion_shot(shot, ratio)
    if model != spec.model:
        raise ValueError("动作参考绑定的模型与当前模型不同；请重新准备工作区")
    if motion_source_binding(shot, ratio) != spec.source_sha256:
        raise ValueError("动作参考与当前分镜不匹配；请重新预演并导出")
    root = workspace.resolve()
    relative = Path(spec.path)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
        raise ValueError("动作参考必须位于当前工作区内")
    payload = read_reference_bytes(path)
    if hashlib.sha256(payload).hexdigest() != spec.sha256:
        raise ValueError("动作参考文件指纹变化；禁止复用或提交")
    probe_motion_reference(payload, shot["duration"], ratio)
    # 视频哈希只证明字节一致；保留原数值计划才能验证参考画面真正执行了动作。
    plan_path = path.parent / 'motion_plan.json'
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError('动作参考缺少本地数值运动计划；请从原预演包离线重新准备，不会自动生成')
    from tools.motion_previs import read_previs_json
    plan = MotionPlan.model_validate(read_previs_json(plan_path))
    compiled = compile_motion_previs(shot, ratio, plan)
    verify_rendered_motion(payload, plan, compiled)
    return context


def ensure_motion_lineage(shot: dict, video_path: str | Path) -> None:
    path = Path(video_path)
    lineage = path.with_name(f"{path.stem}_generation.json")
    if not lineage.exists() and not path.exists():
        return
    expected = motion_reference_context(shot)
    try:
        stored = json.loads(lineage.read_text(encoding="utf-8")) if lineage.exists() else {}
        if not isinstance(stored, dict):
            raise ValueError
    except (OSError, ValueError):
        raise ValueError("无法核验已有镜头的动作参考提交记录") from None
    if stored.get("motion_reference") != expected:
        raise ValueError("已有任务的动作参考、模型或输入模式变化；禁止复用、重新验收或隐式重提交")


def validate_workspace_motion(storyboard: dict, workspace: Path, model: str, *,
                              accepted_artifacts: dict | None = None) -> None:
    shots = storyboard["shots"]
    for shot in shots:
        if shot.get("motion_reference") is not None and len(shots) != 1:
            raise ValueError("当前动作参考仅支持单镜头独立工作区，不能替代连续镜头的尾帧")
        prepare_motion_reference(shot, storyboard.get("aspect_ratio", "16:9"), workspace, model)
        ensure_motion_lineage(shot, workspace / "shots" / f"shot_{shot['shot_id']:03d}.mp4")
        accepted = (accepted_artifacts or {}).get(shot["shot_id"])
        if accepted and accepted.get("local_path"):
            ensure_motion_lineage(shot, accepted["local_path"])


def motion_prompt_suffix(shot: dict) -> str:
    geometry = shot["interaction_geometry"]
    return (
        f" Use the reference video only for timing and body motion: the teal skeleton is {geometry['actor']}; "
        f"the orange skeleton is {geometry['target']}. Preserve the contracted contact-before-reaction order. "
        "Render the described characters and scene, without skeleton lines, labels or diagnostic overlays."
    )

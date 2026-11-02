"""Bounded local rendering of already validated planned skeleton coordinates."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.motion_plan import BONES, MotionPlan


def draw_motion_frame(plan: MotionPlan, sample: dict, *, annotated: bool = True) -> Image.Image:
    image = Image.new("RGB", (plan.width, plan.height), "#101820")
    draw = ImageDraw.Draw(image)
    unit = min(plan.width, plan.height)
    if annotated:
        for x in range(0, plan.width, max(1, round(unit / 10))):
            draw.line((x, 38, x, plan.height), fill="#1c2934")
        for y in range(38, plan.height, max(1, round(unit / 10))):
            draw.line((0, y, plan.width, y), fill="#1c2934")
    # 角色由合约绑定，不能让数组顺序改变输入视频与提示词中的颜色含义。
    ordered = [next(rig for rig in plan.rigs if rig.entity == sample["roles"][role])
               for role in ("actor", "target")]
    for index, rig in enumerate(ordered):
        color = ("#54d9c1", "#ffb363")[index]
        joints = sample["entities"][rig.entity]
        pixels = {name: (point[0] * unit, point[1] * unit) for name, point in joints.items()}
        for name, (parent, _, _) in BONES.items():
            draw.line((*pixels[parent], *pixels[name]), fill=color, width=max(2, round(unit * 0.012)))
        for name, (x, y) in pixels.items():
            radius = unit * (0.022 if name == "head" else 0.006)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        x, y = pixels["head"]
        direction = 1 if rig.facing == "right" else -1
        # 圆头无法表达侧向。鼻部轮廓显式编码朝向，仍是二维参考而非解剖模型。
        draw.polygon([(round(px), round(py)) for px, py in
                     [(x + direction * unit * 0.013, y - unit * 0.012),
                      (x + direction * unit * 0.04, y),
                      (x + direction * unit * 0.013, y + unit * 0.012)]], fill=color)
        if annotated:
            draw.text((x - 4, y - unit * 0.065), "AB"[index], fill=color)
    if not annotated:
        return image
    draw.rectangle((0, 0, plan.width, 35), fill="#101820")
    draw.text((12, 8), "PLANNED MOTION / NOT GENERATED FOOTAGE", fill="#edf3f6")
    frame = sample["frame"]
    phase = "PREPARE" if frame < plan.contact.frame else (
        "CONTACT" if frame < plan.contact.reaction_frame else "REACT / SETTLE"
    )
    draw.text((12, plan.height - 20),
              f"{frame:03d}  {frame / plan.fps:.3f}s  {phase}", fill="#edf3f6")
    return image


def render_motion_media(plan: MotionPlan, compiled: dict, directory: Path) -> dict:
    """Render to a private staging folder; verify encoded frames independently."""
    if compiled["diagnostics"]["status"] != "valid_geometry":
        raise ValueError("几何检查未通过，不能导出预演视频")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("预演导出需要本地 ffmpeg 和 ffprobe")
    frames_dir = directory / "frames"
    frames_dir.mkdir()
    for sample in compiled["frames"]:
        draw_motion_frame(plan, sample).save(frames_dir / f"{sample['frame']:06d}.png")
    keyframes = [0, plan.contact.frame, plan.contact.reaction_frame, compiled["frame_count"] - 1]
    sheet = Image.new("RGB", (plan.width * 2, plan.height * 2))
    for index, frame in enumerate(keyframes):
        image = draw_motion_frame(plan, compiled["frames"][frame])
        sheet.paste(image, ((index % 2) * plan.width, (index // 2) * plan.height))
        image.save(directory / f"keyframe_{frame:03d}.png")
    sheet.save(directory / "contact_sheet.png")
    preview = _encode_motion_frames(plan, compiled, frames_dir, directory / "preview.mp4")
    for sample in compiled["frames"]:
        draw_motion_frame(plan, sample, annotated=False).save(
            frames_dir / f"{sample['frame']:06d}.png"
        )
    # 视频参考另行满足输入像素/帧率下限，保留同一时间轴与画幅。
    scale = max(1, math.sqrt(409600 / (plan.width * plan.height)),
                300 / min(plan.width, plan.height))
    width, height = (math.ceil(value * scale / 2) * 2 for value in (plan.width, plan.height))
    reference = _encode_motion_frames(
        plan, compiled, frames_dir, directory / "reference.mp4",
        width=width, height=height, fps=max(24, plan.fps),
    )
    from tools.motion_media_validation import verify_rendered_motion
    reference['pixel_verification'] = verify_rendered_motion(
        (directory / 'reference.mp4').read_bytes(), plan, compiled)
    shutil.rmtree(frames_dir)
    return {**preview, "reference_video": reference}


def _encode_motion_frames(plan, compiled, frames_dir, destination, *, width=None, height=None, fps=None):
    width, height, fps = width or plan.width, height or plan.height, fps or plan.fps
    count = round(compiled["frame_count"] * fps / plan.fps)
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1",
            "-framerate", str(plan.fps), "-start_number", "0",
            "-i", str(frames_dir / "%06d.png"),
            "-vf", f"scale={width}:{height},fps={fps}",
            "-frames:v", str(count), "-an",
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(destination),
        ], check=True, capture_output=True, timeout=90)
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
            "-of", "json", str(destination),
        ], check=True, capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("预演编码或验证超时；未发布输出，请缩小画布后重试") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("预演编码或验证失败；请检查本地 FFmpeg 的 libx264 支持") from exc
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        matches = (
            stream["width"] == width and stream["height"] == height
            and int(stream["nb_frames"]) == count
            and denominator > 0 and numerator / denominator == fps
            and math.isclose(float(stream["duration"]), compiled["frame_count"] / plan.fps,
                             abs_tol=1 / plan.fps)
        )
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        raise RuntimeError("无法核验编码结果，未发布输出") from None
    if not matches:
        raise RuntimeError("编码结果与预演时间轴不一致，未发布输出")
    return stream

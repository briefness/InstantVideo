"""Prepare a single-shot production workspace offline, without submitting a task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from pipeline.models import MotionReferenceSpec, RunOptions, validate_storyboard
from pipeline.motion_plan import MotionPlan, compile_motion_previs, motion_source_binding
from pipeline.motion_reference import (
    MOTION_MODELS, prepare_motion_reference, probe_motion_reference, read_reference_bytes,
)
from pipeline.readiness import ensure_storyboard_ready
from pipeline.run_state import RunWorkspace
from pipeline.storyboard import _apply_coverage_defaults
from tools.motion_previs import read_previs_json
from tools.motion_media_validation import verify_rendered_motion


def prepare_motion_run(storyboard: dict, package: Path, output_root: Path, *,
                       model: str, asset_uri: str | None = None,
                       paid_take_budget: int = 0, video_url: str | None = None) -> RunWorkspace:
    """Snapshot checked bytes; asset binding is explicit and never performs an upload."""
    normalized = validate_storyboard(storyboard)
    if len(normalized["shots"]) != 1:
        raise ValueError("动作参考目前只支持单镜头独立工作区")
    _apply_coverage_defaults(normalized["shots"])
    normalized = validate_storyboard(normalized)
    ensure_storyboard_ready(normalized)
    shot = normalized["shots"][0]
    if shot.get("motion_reference") is not None:
        raise ValueError("分镜已绑定动作参考；请使用未绑定的源分镜准备新工作区")
    ratio = normalized["aspect_ratio"]
    manifest = read_previs_json(package / "manifest.json")
    if manifest.get("state_kind") != "planned_only" or manifest.get("version") != "motion-previs-v1":
        raise ValueError("不是支持的离线预演包")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("预演包缺少文件指纹清单")
    geometry = shot["interaction_geometry"]
    if manifest.get("rig_labels") != {"A": geometry["actor"], "B": geometry["target"]}:
        raise ValueError("参考素材角色颜色与合约不匹配；请重新导出，不能仅修改标签")
    # 仅读取固定文件名；manifest 不能指挥读取其他路径。
    video = read_reference_bytes(package / "reference.mp4")
    plan_data = read_reference_bytes(package / "motion_plan.json")
    if len(plan_data) > 2 * 1024 * 1024:
        raise ValueError("预演计划不得超过 2 MiB")
    for name, payload in (("reference.mp4", video), ("motion_plan.json", plan_data)):
        if manifest.get("files", {}).get(name) != hashlib.sha256(payload).hexdigest():
            raise ValueError("预演包指纹不匹配；请重新导出")
    plan = MotionPlan.model_validate(json.loads(plan_data))
    compiled = compile_motion_previs(shot, ratio, plan)
    if compiled["diagnostics"]["issues"] or manifest.get("source_sha256") != motion_source_binding(shot, ratio):
        raise ValueError("预演包与生产分镜不一致")
    probe_motion_reference(video, shot["duration"], ratio)
    verify_rendered_motion(video, plan, compiled)
    reference = MotionReferenceSpec(
        path="motion_refs/reference.mp4", sha256=hashlib.sha256(video).hexdigest(),
        source_sha256=plan.source_sha256, model=model, asset_uri=asset_uri, video_url=video_url,
    )
    options = RunOptions(
        request="已确认的单镜头动作预演分镜", resolution=normalized["resolution"],
        aspect_ratio=ratio, style=normalized["style"], paid_take_budget=paid_take_budget,
        take_policy='completion_first',
    )
    workspace = RunWorkspace.create(output_root, options)
    try:
        destination = workspace.path / reference.path
        destination.parent.mkdir()
        with destination.open("xb") as handle:
            handle.write(video)
        with (destination.parent / 'motion_plan.json').open('xb') as handle:
            handle.write(plan_data)
        shot["motion_reference"] = reference.model_dump(mode="json")
        saved = workspace.save_storyboard(normalized)
        prepare_motion_reference(saved["shots"][0], ratio, workspace.path, model)
    except Exception:
        workspace.mark_failed("离线动作参考工作区准备失败；未提交任何任务")
        raise
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线准备动作参考工作区；不上传、不生成")
    parser.add_argument("--storyboard", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MOTION_MODELS), default="doubao-seedance-2-0-mini-260615")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--asset-uri", help="人工确认上传相同 SHA-256 文件后取得的素材 ID")
    source.add_argument("--video-url", help="无查询参数的公网 HTTPS 文件直链；提交生成前核验远端 SHA-256")
    parser.add_argument("--paid-take-budget", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        workspace = prepare_motion_run(
            read_previs_json(args.storyboard), args.package, args.output_root,
            model=args.model, asset_uri=args.asset_uri, paid_take_budget=args.paid_take_budget,
            video_url=args.video_url,
        )
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        print(f"准备失败，输入结构无效：{fields}；未发起远端调用。")
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"准备失败：{exc}；未发起远端调用。")
        return 2
    print(f"工作区已离线准备：{workspace.path}")
    print(f"付费 take 上限：{workspace.options.paid_take_budget}；本次远端调用：0")
    if args.video_url:
        print("公网链接已记录，尚未访问或验证远端文件；创建生成任务前将比对 SHA-256。")
    elif not args.asset_uri:
        print("尚未绑定上传素材，不能提交生成；请保留 reference.mp4 及其 SHA-256。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

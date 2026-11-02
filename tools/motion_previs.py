"""Offline CLI: inspect a shot, validate a motion plan, export a review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

from pipeline.causality import ACTION_CONTRACT_VERSION, compile_action_contract
from pipeline.models import CharacterSpec
from pipeline.motion_plan import (
    MOTION_PLAN_VERSION, MotionPlan, compile_motion_previs,
    motion_source_binding, supported_motion_shot,
)
from tools.motion_renderer import render_motion_media


def read_previs_json(path: Path) -> dict:
    # 在分配内存/解析前限制文件体积；不读取远端 URL，不执行外部脚本。
    if not path.is_file():
        raise ValueError("预演输入必须是本地普通文件")
    with path.open("rb") as handle:
        payload = handle.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("预演输入 JSON 不得超过 2 MiB")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON 对象不能包含重复字段")
            result[key] = value
        return result

    try:
        data = json.loads(payload, object_pairs_hook=unique_object)
    except RecursionError:
        raise ValueError("JSON 嵌套过深") from None
    if not isinstance(data, dict):
        raise ValueError("JSON 根节点必须是对象")
    return data


def select_motion_shot(storyboard: dict, shot_id: int) -> tuple[dict, str]:
    shots = storyboard.get("shots")
    if not isinstance(shots, list) or any(not isinstance(shot, dict) for shot in shots):
        raise ValueError("分镜 shots 必须是对象数组")
    ids = [shot.get("shot_id") for shot in shots]
    if any(type(value) is not int or value <= 0 for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("shot_id 必须为唯一的正整数")
    selected = [shot for shot in shots if shot["shot_id"] == shot_id]
    if not selected:
        raise ValueError("找不到指定镜头")
    ratio = storyboard.get("ratio", storyboard.get("aspect_ratio", "16:9"))
    actor, target = supported_motion_shot(selected[0], ratio)
    characters = storyboard.get("characters", [])
    if not isinstance(characters, list):
        raise ValueError("characters 必须是数组")
    for raw_character in characters:
        character = CharacterSpec.model_validate(raw_character)
        if character.name in {actor, target} and (
            character.reference_mode == "group"
            or character.mobility not in {"bipedal", "unspecified"}
        ):
            raise ValueError("当前骨架仅支持单个双足角色，不能代替群体或其他运动结构")
    return selected[0], ratio


def build_motion_previs(storyboard: dict, plan: MotionPlan, output: Path) -> Path:
    """Publish a complete local package without touching production workspace state."""
    shot, ratio = select_motion_shot(storyboard, plan.shot_id)
    compiled = compile_motion_previs(shot, ratio, plan)
    if compiled["diagnostics"]["issues"]:
        raise ValueError("预演检查失败：" + "；".join(compiled["diagnostics"]["issues"]))
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("输出目录已存在；请选择新目录以保留上一版")
    if not output.parent.is_dir():
        raise ValueError("输出目录的父目录不存在")
    contract = compile_action_contract(shot)
    actor = contract.canonical_geometry["actor"]
    target = contract.canonical_geometry["target"]
    protocol = {
        "status": "prepared_offline",
        "generation_quality": "not_evaluated",
        "paid_take_budget": 0,
        "source_sha256": plan.source_sha256,
        "arms": [
            {"name": "baseline", "input": "existing Action Contract pipeline"},
            {"name": "motion_reference", "input": "reference.mp4",
             "submission_ready": False,
             "blocker": "offline only; bind an uploaded provider asset and authorize a paid budget before submission"},
        ],
        "repetitions_per_arm": 3,
        "execution_order": ["baseline", "motion_reference", "motion_reference",
                            "baseline", "baseline", "motion_reference"],
        "controls": ["same model/version, base prompt, identity references, duration, ratio, resolution",
                     "treatment adds motion video AND its fixed grounding instruction; not a video-only causal test",
                     "log seeds where supported; retain every failed take"],
        "review": {
            "method": "independent blinded review of every raw take; frame-indexed observations",
            "actor": actor, "target": target,
            "contract_projection": contract.review_projection,
            "planned_contact_seconds": plan.contact.frame / plan.fps,
            "planned_reaction_seconds": plan.contact.reaction_frame / plan.fps,
            "checks": ["contact before target reaction", "correct participant and permitted scope",
                       "identity and screen direction preserved", "contracted visible endpoint"],
            "costs": ["preparation minutes", "manual correction minutes", "all paid takes",
                      "elapsed time", "total cost per accepted take (undefined when none accepted)"],
            "decision": "three pairs are a pilot, not proof; report raw counts and uncertainty",
            "stop": "stop at authorized budget, unsupported inputs, or failed provenance checks",
        },
        "limitations": ["2D blocking only; no collision dynamics, anatomy, foot-slip or occlusion proof",
                        "target must hold still before contact; independent motion unsupported",
                        "preview is planned state, not observed or canonical state",
                        "keypoint JSON is a neutral format, not a Wan/ATI-compatible control file"],
    }
    with tempfile.TemporaryDirectory(prefix=".motion-previs-", dir=output.parent) as temp:
        staging = Path(temp)
        encoded = render_motion_media(plan, compiled, staging)
        payloads = {
            "motion_plan.json": plan.model_dump(mode="json"),
            "source_shot.json": {"ratio": ratio, "shots": [shot]},
            "keypoints.json": {key: value for key, value in compiled.items() if key != "diagnostics"},
            "diagnostics.json": {**compiled["diagnostics"], "encoded_video": encoded},
            "experiment.json": protocol,
        }
        for filename, payload in payloads.items():
            (staging / filename).write_text(json.dumps(
                payload, ensure_ascii=False, indent=2, allow_nan=False,
            ) + "\n", encoding="utf-8")
        manifest = {
            "version": MOTION_PLAN_VERSION,
            "state_kind": "planned_only",
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "source_sha256": plan.source_sha256,
            "rig_labels": {"A": actor, "B": target},
            "render_contract": "role-bound-profile-v1",
            "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                      for file in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # mkdir 独占目标；硬链接不覆盖文件；最后发布 manifest 才表示完整包。
        output.mkdir()
        published = []
        try:
            files = sorted(path for path in staging.iterdir() if path.name != "manifest.json")
            for file in [*files, staging / "manifest.json"]:
                destination = output / file.name
                os.link(file, destination)
                published.append(destination)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            if not any(output.iterdir()):
                output.rmdir()
            raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="动作预演：离线检查与导出，不调用付费服务")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "validate", "render"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--storyboard", type=Path, required=True)
        if command == "inspect":
            subparser.add_argument("--shot", type=int, required=True)
        else:
            subparser.add_argument("--plan", type=Path, required=True)
        if command == "render":
            subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        storyboard = read_previs_json(args.storyboard)
        if args.command == "inspect":
            shot, ratio = select_motion_shot(storyboard, args.shot)
            print(json.dumps({
                "shot_id": args.shot, "duration": shot["duration"], "ratio": ratio,
                "source_sha256": motion_source_binding(shot, ratio),
                "contract_projection": compile_action_contract(shot).review_projection,
            }, ensure_ascii=False, indent=2))
            return 0
        plan = MotionPlan.model_validate(read_previs_json(args.plan))
        shot, ratio = select_motion_shot(storyboard, plan.shot_id)
        if args.command == "validate":
            diagnostics = compile_motion_previs(shot, ratio, plan)["diagnostics"]
            print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
            return 2 if diagnostics["issues"] else 0
        output = build_motion_previs(storyboard, plan, args.output)
        print(f"预演已导出：{output}\n几何检查通过；生成质量尚未验证；付费调用：0")
        return 0
    except ValidationError as exc:
        # 不打印第三方输入值，以免把附带私密资料写入日志。
        locations = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        print(f"输入结构无效，请检查字段：{locations}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"无法完成动作预演：{exc}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

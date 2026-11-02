"""Synthetic reviewer practice clips; never production references or measured model output."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import ImageDraw

from pipeline.motion_assessment import (
    REVIEW_INSTRUCTIONS, REVIEW_VERSION, FRAME_TOLERANCE, assess_document,
    blank_rating, compare_assessments, contract_digest, motion_review_contract,
)
from pipeline.motion_experiment import file_digest, write_new_json
from pipeline.motion_plan import MotionPlan, compile_motion_previs
from pipeline.motion_reference import read_reference_bytes
from pipeline.motion_review import _frame_count
from tools.motion_previs import read_previs_json
from tools.motion_renderer import _encode_motion_frames, draw_motion_frame

CASES = {
    "contact_then_reaction": "pass",
    "premature_reaction": "fail",
    "missed_contact": "fail",
    "no_reaction": "fail",
    "roles_swapped": "fail",
    "contact_obscured": "unjudgeable",
}


def calibration_frames(frames: list[dict], case: str) -> list[dict]:
    """Perturb a single fixed teaching example; preserve original data."""
    if case not in CASES:
        raise ValueError("未知校准案例")
    result = copy.deepcopy(frames)
    for sample in result:
        entities = sample["entities"]
        frame = sample["frame"]
        if case == "premature_reaction" and 12 < frame < 36:
            shift = 0.10 * math.sin(math.pi * (frame - 12) / 24)
            for point in entities["defender"].values():
                point[0] += shift
        elif case == "missed_contact":
            for point in entities["defender"].values():
                point[0] += 0.18
        elif case == "no_reaction":
            entities["defender"] = copy.deepcopy(frames[0]["entities"]["defender"])
        elif case == "roles_swapped":
            entities["attacker"], entities["defender"] = entities["defender"], entities["attacker"]
    return result


def build_calibration(output: Path) -> dict:
    example = Path(__file__).resolve().parents[1] / "examples/contact_previs"
    source = read_previs_json(example / "storyboard.json")
    plan = MotionPlan.model_validate(read_previs_json(example / "motion.json"))
    compiled = compile_motion_previs(source["shots"][0], source["aspect_ratio"], plan)
    if (compiled["diagnostics"]["issues"] or compiled["frame_count"] != 96 or plan.fps != 24
            or plan.contact.frame != 48 or plan.contact.reaction_frame != 55
            or [rig.entity for rig in plan.rigs] != ["attacker", "defender"]):
        raise ValueError("教学样例已改变，须重新设计和核验校准反例")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("输出须为已有父目录下的新目录")
    contract = motion_review_contract(source["shots"][0])
    mapping, ratings = [], []
    with tempfile.TemporaryDirectory(prefix=".calibration-", dir=output.parent) as temporary:
        staging = Path(temporary)
        reviewer = staging / "reviewer"
        reviewer.mkdir()
        for case, expected in CASES.items():
            identifier = uuid.uuid4().hex
            transformed = calibration_frames(compiled["frames"], case)
            with tempfile.TemporaryDirectory(dir=staging) as frame_directory:
                for sample in transformed:
                    image = draw_motion_frame(plan, sample, annotated=False)
                    if case == "contact_obscured" and 40 <= sample["frame"] <= 65:
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, plan.width, plan.height), fill="#101820")
                    image.save(Path(frame_directory) / f"{sample['frame']:06d}.png")
                clip = reviewer / f"{identifier}.mp4"
                # 故意错误的教学片段只复用底层编码器，不伪装为通过几何检查的生产预演。
                _encode_motion_frames(plan, compiled, Path(frame_directory), clip)
            digest = file_digest(clip)
            mapping.append({"review_id": identifier, "case": case, "expected": expected,
                            "sha256": digest, "frames": compiled["frame_count"]})
            ratings.append(blank_rating(identifier, digest, compiled["frame_count"]))
        random.SystemRandom().shuffle(ratings)
        write_new_json(reviewer / "ratings.json", {
            "version": REVIEW_VERSION, "reviewer_id": "", "contract": contract,
            "contract_sha256": contract_digest(contract), "ratings": ratings,
            "instructions": REVIEW_INSTRUCTIONS + " 本包为二维合成教学片段，青色是 attacker，橙色是 defender；不用于衡量真实模型能力。",
        })
        write_new_json(staging / "calibration_manifest.json", {
            "version": "motion-calibration-v1", "state_kind": "synthetic_calibration_only",
            "contract": contract, "mapping": mapping,
            "source_files": {name: file_digest(example / name) for name in ("storyboard.json", "motion.json")},
            "note": "答案是构造标签，尚未由独立人员校验；六个案例只用于校准练习，不是保留测试集或生成质量证据。",
        })
        output.mkdir()
        published = []
        try:
            (output / "reviewer").mkdir()
            for file in reviewer.iterdir():
                destination = output / "reviewer" / file.name
                os.link(file, destination)
                published.append(destination)
            os.link(staging / "calibration_manifest.json", output / "calibration_manifest.json")
        except BaseException:
            for file in published:
                file.unlink(missing_ok=True)
            (output / "reviewer").rmdir()
            output.rmdir()
            raise
    return {"status": "prepared_offline", "clips": len(mapping), "remote_calls": 0,
            "output": str(output.resolve()), "human_validation": "not_evaluated"}


def score_calibration(directory: Path, paths: list[Path] | None = None) -> dict:
    manifest = read_previs_json(directory / "calibration_manifest.json")
    if manifest.get("version") != "motion-calibration-v1" or manifest.get("state_kind") != "synthetic_calibration_only":
        raise ValueError("不支持的校准包")
    entries = manifest.get("mapping")
    if not isinstance(entries, list) or len(entries) != len(CASES):
        raise ValueError("校准案例缺失")
    lookup, cases = {}, set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("校准映射无效")
        identifier, case = entry.get("review_id"), entry.get("case")
        if (not isinstance(identifier, str) or len(identifier) != 32
                or any(c not in "0123456789abcdef" for c in identifier)
                or identifier in lookup or not isinstance(case, str) or case not in CASES or case in cases
                or entry.get("expected") != CASES[case]):
            raise ValueError("校准映射重复或已改变")
        clip = directory / "reviewer" / f"{identifier}.mp4"
        read_reference_bytes(clip)
        if file_digest(clip) != entry.get("sha256") or _frame_count(clip) != entry.get("frames"):
            raise ValueError("校准视频或帧数已改变")
        lookup[identifier] = entry
        cases.add(case)
    paths = paths if paths is not None else [directory / "reviewer/ratings.json"]
    if not 1 <= len(paths) <= 2:
        raise ValueError("仅支持一或两份校准评分")
    documents = [assess_document(read_previs_json(path), lookup, manifest["contract"]) for path in paths]
    if any(doc["version"] not in ("motion-review-v2", REVIEW_VERSION) for doc in documents):
        raise ValueError("校准需要新版评审表")
    results = []
    for doc in documents:
        rows = []
        for identifier, entry in lookup.items():
            observed = doc["verdicts"].get(identifier, "unreviewed")
            correct = observed == entry["expected"] if observed != "unreviewed" else None
            if entry["case"] == "contact_then_reaction" and observed == "pass":
                rating = doc["ratings"][identifier]
                correct = (abs(rating["contact_frame"] - 48) <= FRAME_TOLERANCE
                           and abs(rating["reaction_frame"] - 55) <= FRAME_TOLERANCE)
            rows.append({"review_id": identifier, "case": entry["case"], "expected": entry["expected"],
                         "observed": observed, "correct": correct})
        results.append({"reviewer_id": doc["reviewer_id"], "assessment_version": doc["version"], "cases": rows,
                        "answered": sum(row["correct"] is not None for row in rows),
                        "correct": sum(row["correct"] is True for row in rows),
                        "false_acceptances": sum(row["observed"] == "pass" and row["expected"] != "pass" for row in rows)})
    return {"status": "practice_report", "reviewers": results,
            "agreement": compare_assessments(*documents, list(lookup)) if len(documents) == 2 else None,
            "generation_quality": "not_evaluated", "independent_human_validation": "not_evaluated",
            "note": "构造答案仅用于教学核对；不证明评审者可靠或生成效果提升。请勿把本报告交给尚未完成标注的评审者。"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线评审校准：合成教学片段，不调用模型")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--ratings-file", action="append", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_calibration(args.output) if args.command == "build" else score_calibration(args.bundle, args.ratings_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError, subprocess.SubprocessError):
        print("校准未完成：检查输入、评审状态及本地媒体工具；不会覆盖原目录或调用模型。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

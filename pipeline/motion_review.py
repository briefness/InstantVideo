"""Opaque-label raw-take exports and explicit, frame-indexed human observations."""

from __future__ import annotations

import json
import random
import re
import subprocess
import uuid
from pathlib import Path

from pipeline.motion_experiment import file_digest, load_experiment, write_new_json
from pipeline.motion_assessment import (
    REVIEW_VERSION, REVIEW_INSTRUCTIONS, assess_document, blank_rating,
    compare_assessments, contract_digest, motion_review_contract,
)
from pipeline.motion_reference import read_reference_bytes
from tools.motion_previs import read_previs_json


def _take_path(workspace, take) -> Path:
    relative = Path(take.local_path)
    path = workspace.path / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(workspace.path):
        raise ValueError("Take 路径越界")
    return path


def _frame_count(path: Path) -> int:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
    ], capture_output=True, check=True, timeout=30)
    count = int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"])
    if not 1 <= count <= 900:
        raise ValueError("评审视频帧数超限")
    return count


def export_review_bundle(root: Path, output: Path) -> dict:
    _, loaded = load_experiment(root)
    output.mkdir(exist_ok=False)
    reviewer = output / "reviewer"
    reviewer.mkdir()
    mapping, observations, unavailable = [], [], []
    for sample, workspace in loaded:
        for state in workspace.manifest.shots.values():
            for take in state.take_history:
                path = _take_path(workspace, take)
                try:
                    payload = read_reference_bytes(path)
                    count = _frame_count(path)
                except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
                    unavailable.append({"sample_id": sample["sample_id"], "take_id": take.take_id, "reason": "无法逐帧核验"})
                    continue
                review_id = uuid.uuid4().hex
                clip = reviewer / f"{review_id}.mp4"
                clip.write_bytes(payload)
                digest = file_digest(clip)
                mapping.append({"review_id": review_id, "sample_id": sample["sample_id"],
                                "arm": sample["arm"], "take_id": take.take_id,
                                "sha256": digest, "frames": count})
                observations.append(blank_rating(review_id, digest, count))
    random.SystemRandom().shuffle(observations)
    write_new_json(output / "private_mapping.json", {
        "experiment_sha256": file_digest(root / "experiment_manifest.json"),
        "mapping": mapping, "unavailable": unavailable,
    })
    contract = motion_review_contract(loaded[0][1].load_storyboard()["shots"][0])
    write_new_json(reviewer / "ratings.json", {
        "version": REVIEW_VERSION, "reviewer_id": "", "instructions": REVIEW_INSTRUCTIONS,
        "contract": contract, "contract_sha256": contract_digest(contract),
        "ratings": observations,
    })
    return {"status": "exported", "clips": len(mapping), "unavailable": unavailable,
            "reviewer_directory": str(reviewer.resolve()), "quality": "not_evaluated"}


def summarize_reviews(root: Path, directory: Path, *, ratings_files: list[Path] | None = None) -> dict:
    _, loaded = load_experiment(root)
    mapping = read_previs_json(directory / "private_mapping.json")
    if mapping.get("experiment_sha256") != file_digest(root / "experiment_manifest.json"):
        raise ValueError("盲评包不属于此实验")
    entries = mapping.get("mapping")
    if not isinstance(entries, list):
        raise ValueError("评审映射无效")
    originals = {(sample["sample_id"], take.take_id): (sample, workspace, take)
                 for sample, workspace in loaded for state in workspace.manifest.shots.values()
                 for take in state.take_history}
    lookup, mapped_takes = {}, set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("review_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", entry["review_id"]):
            raise ValueError("评审 ID 无效")
        if not isinstance(entry.get("sample_id"), str) or not isinstance(entry.get("take_id"), str):
            raise ValueError("评审 Take 无效")
        key = (entry["sample_id"], entry["take_id"])
        if key not in originals or key in mapped_takes or entry["review_id"] in lookup:
            raise ValueError("评审 Take 未知或重复")
        sample, workspace, take = originals[key]
        clip = directory / "reviewer" / f"{entry['review_id']}.mp4"
        # 映射与评分文件都是外部输入；必须回到原始账本与媒体核对。
        read_reference_bytes(clip)
        original = _take_path(workspace, take)
        read_reference_bytes(original)
        if (entry.get("arm") != sample["arm"] or file_digest(original) != entry.get("sha256")
                or file_digest(clip) != entry.get("sha256") or _frame_count(clip) != entry.get("frames")):
            raise ValueError("评审映射或视频指纹变化")
        mapped_takes.add(key)
        lookup[entry["review_id"]] = entry
    paths = ratings_files if ratings_files is not None else [directory / "reviewer/ratings.json"]
    if not 1 <= len(paths) <= 2:
        raise ValueError("本次仅支持一份评审或两份独立评审对照")
    contract = motion_review_contract(loaded[0][1].load_storyboard()["shots"][0])
    documents = [assess_document(read_previs_json(path), lookup, contract) for path in paths]
    comparison = compare_assessments(*documents, list(lookup)) if len(documents) == 2 else None
    verdicts = comparison["verdicts"] if comparison else documents[0]["verdicts"]
    counts = {arm: {"planned": sum(sample["arm"] == arm for sample, _ in loaded),
                    "reviewed": 0, "passed": 0} for arm in ("baseline", "motion_reference")}
    breakdown = {key: 0 for key in ("pass", "fail", "unreviewed", "unjudgeable", "disputed")}
    for identifier, entry in lookup.items():
        verdict = verdicts.get(identifier, "unreviewed")
        breakdown[verdict] += 1
        result = counts[entry["arm"]]
        result["reviewed"] += verdict in {"pass", "fail"}
        result["passed"] += verdict == "pass"
    return {"arms": counts, "decision": "pilot_observations_only" if any(v["reviewed"] for v in counts.values()) else "not_evaluated",
            "assessment_versions": [doc["version"] for doc in documents],
            "judgments": breakdown, "agreement": comparison,
            "note": "保留未生成、失败及未评审样本；不以技术通过代替人工判断，不据此宣称统计显著"}

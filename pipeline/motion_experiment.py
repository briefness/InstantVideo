"""Frozen paired trials and read-only reports derived from ordinary run workspaces."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

import config
from pipeline.models import RunOptions
from pipeline.motion_reference import validate_workspace_motion
from pipeline.production_analytics import derive_production_statistics
from pipeline.motion_reference import MOTION_MODELS
from pipeline.readiness import ensure_storyboard_ready
from pipeline.run_state import RunWorkspace
from tools.motion_previs import read_previs_json
from tools.prepare_motion_run import prepare_motion_run


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_experiment(storyboard: dict, package: Path, output: Path, *, model: str,
                       repetitions: int = 3, paid_take_budget: int = 0,
                       video_url: str | None = None, asset_uri: str | None = None) -> Path:
    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("试验限定每组 1–3 个样本")
    if type(paid_take_budget) is not int or paid_take_budget not in (0, 2 * repetitions):
        raise ValueError("预算只能为 0 或样本总数；每个独立样本最多 1 个付费 Take")
    if not output.parent.is_dir():
        raise ValueError("实验输出的父目录不存在")
    output.mkdir(exist_ok=False)
    samples = []
    # manifest 最后发布；中断的目录没有 manifest，不能误作可执行实验。
    for pair in range(repetitions):
        motion = prepare_motion_run(
            storyboard, package, output / "runs", model=model, video_url=video_url,
            asset_uri=asset_uri, paid_take_budget=1 if paid_take_budget else 0,
        )
        normalized = motion.load_storyboard()
        baseline_storyboard = copy.deepcopy(normalized)
        baseline_storyboard["shots"][0].pop("motion_reference", None)
        baseline = RunWorkspace.create(output / "runs", RunOptions.model_validate(motion.options.model_dump()))
        baseline.save_storyboard(baseline_storyboard)
        arms = [("baseline", baseline), ("motion_reference", motion)]
        if pair % 2:
            arms.reverse()
        for arm, workspace in arms:
            samples.append({
                "sample_id": uuid.uuid4().hex, "pair": pair + 1, "arm": arm,
                "workspace": str(workspace.path.relative_to(output.resolve())),
                "storyboard_sha256": file_digest(workspace.path / "storyboard.json"),
                "options": workspace.options.model_dump(mode="json"),
            })
    manifest = {
        "version": "motion-experiment-v1", "model": model,
        "paid_take_budget": paid_take_budget, "samples": samples,
        "hypothesis": "motion video plus grounding instruction versus baseline; not video-only causality",
        "evaluation": "blinded human review of all raw takes; technical acceptance is not semantic success",
        "postprocessing": False, "remote_semantic_review": False,
    }
    write_new_json(output / "experiment_manifest.json", manifest)
    return output.resolve()


def load_experiment(root: Path) -> tuple[dict, list[tuple[dict, RunWorkspace]]]:
    root = root.resolve()
    manifest = read_previs_json(root / "experiment_manifest.json")
    samples = manifest.get("samples")
    if manifest.get("version") != "motion-experiment-v1" or not isinstance(samples, list) or len(samples) not in (2, 4, 6):
        raise ValueError("无效或未完成的实验清单")
    if (not isinstance(manifest.get("model"), str) or manifest["model"] not in MOTION_MODELS or manifest.get("postprocessing") is not False
            or manifest.get("remote_semantic_review") is not False):
        raise ValueError("实验模型或评审流程不受支持")
    budget = manifest.get("paid_take_budget")
    if type(budget) is not int or budget not in (0, len(samples)):
        raise ValueError("实验预算与样本数量不一致")
    loaded, seen, pairs = [], set(), {}
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("arm") not in ("baseline", "motion_reference"):
            raise ValueError("实验样本结构无效")
        if (not isinstance(sample.get("sample_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", sample["sample_id"])
                or not isinstance(sample.get("workspace"), str) or not sample["workspace"]):
            raise ValueError("实验样本 ID 或工作区无效")
        relative = Path(sample.get("workspace", ""))
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root) or path.resolve() in seen:
            raise ValueError("实验工作区越界或重复")
        seen.add(path.resolve())
        if file_digest(path / "storyboard.json") != sample.get("storyboard_sha256"):
            raise ValueError("实验分镜已改变；请准备新的实验")
        workspace = RunWorkspace.resume(path)
        if workspace.options.model_dump(mode="json") != sample.get("options") or workspace.options.paid_take_budget != (1 if budget else 0):
            raise ValueError("工作区参数或预算已改变")
        storyboard = workspace.load_storyboard()
        ensure_storyboard_ready(storyboard)
        if len(storyboard["shots"]) != 1:
            raise ValueError("实验必须为单镜头")
        reference = storyboard["shots"][0].get("motion_reference")
        if bool(reference) != (sample["arm"] == "motion_reference"):
            raise ValueError("实验组与参考输入不一致")
        validate_workspace_motion(storyboard, path, manifest["model"], accepted_artifacts=workspace.accepted_shot_artifacts())
        clean = copy.deepcopy(storyboard)
        clean["shots"][0].pop("motion_reference", None)
        # observed state is outcome, not an experimental input.
        clean["shots"][0].pop("observed_end_state", None)
        pair = sample.get("pair")
        if type(pair) is not int or not 1 <= pair <= len(samples) // 2:
            raise ValueError("样本配对编号无效")
        paired = pairs.setdefault(pair, {})
        if sample["arm"] in paired:
            raise ValueError("配对内实验组重复")
        paired[sample["arm"]] = clean
        loaded.append((sample, workspace))
    if len({sample.get("sample_id") for sample in samples}) != len(samples):
        raise ValueError("样本 ID 重复")
    if len(pairs) != len(samples) // 2 or any(p.get("baseline") != p.get("motion_reference") for p in pairs.values()):
        raise ValueError("两组除动作参考之外的输入不一致")
    if any(pair["baseline"] != pairs[1]["baseline"] for pair in pairs.values()):
        raise ValueError("重复实验的基础输入不一致")
    return manifest, loaded


def experiment_preflight(root: Path) -> dict:
    manifest, loaded = load_experiment(root)
    blockers = []
    if config.SEEDANCE_MODEL != manifest["model"]:
        blockers.append("设置 SEEDANCE_MODEL 为实验冻结的模型版本")
    if config.SEMANTIC_REVIEW_ENABLED:
        blockers.append("设置 SEMANTIC_REVIEW_ENABLED=false；试验采用人工盲评，避免额外模型审查费用")
    if not config.ARK_API_KEY or not config.ARK_API_KEY_SEEDANCE:
        blockers.append("尚未配置视频 API 凭据；不要将密钥写入实验文件")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        blockers.append("需要安装 ffmpeg 与 ffprobe")
    if not manifest["paid_take_budget"]:
        blockers.append("当前是零预算演练；真实试验须另建明确预算的实验")
    for sample, workspace in loaded:
        shot = workspace.load_storyboard()["shots"][0]
        reference = shot.get("motion_reference") or {}
        if sample["arm"] == "motion_reference" and not (reference.get("video_url") or reference.get("asset_uri")):
            blockers.append("动作参考尚未绑定已托管文件")
        if any(item.status == "reserved" for item in workspace.manifest.paid_take_budget.reservations):
            blockers.append("存在提交结果未知的预算预留，必须核对原任务后再继续")
        recorded_tasks = {state.provider_task_id: state for state in workspace.manifest.shots.values()
                          if state.provider_task_id}
        for item in workspace.manifest.paid_take_budget.reservations:
            if item.status in {"submitted", "reconciled"}:
                state = recorded_tasks.get(item.provider_task_id)
                if state is None or (item.status == "submitted" and state.status.value != "running"):
                    blockers.append("已确认付费任务缺少完整状态记录；核对原任务后再继续")
        reconciled_ids = {item.provider_task_id for item in workspace.manifest.paid_take_budget.reservations
                          if item.status == "reconciled" and item.provider_task_id}
        # 已记账的技术拒绝是实验结果；执行器会跳过，不沿用普通生产流程的重试入口。
        unresolved_local = any(task_id not in reconciled_ids
                               for task_id in workspace.terminal_materialization_tasks().values())
        if workspace.unresolved_legacy_provider_tasks() or unresolved_local:
            blockers.append("存在缺少提交记录或本地物化失败的任务，禁止重新提交")
    return {
        "status": "blocked" if blockers else "ready_for_explicit_execution",
        "model": manifest["model"], "sample_count": len(loaded),
        "paid_take_budget": manifest["paid_take_budget"], "blockers": list(dict.fromkeys(blockers)),
        "network_checked": False, "account_entitlement_checked": False,
        "generation_quality": "not_evaluated",
    }


def experiment_report(root: Path) -> dict:
    manifest, loaded = load_experiment(root)
    rows = []
    for sample, workspace in loaded:
        ledger = workspace.manifest.paid_take_budget.reservations
        rows.append({
            "sample_id": sample["sample_id"], "pair": sample["pair"], "arm": sample["arm"],
            "status": ("prepared" if not ledger and workspace.manifest.stage == "storyboard_ready"
                       else workspace.manifest.status.value),
            "workspace_stage": workspace.manifest.stage,
            "submitted_tasks": len({item.provider_task_id for item in ledger if item.provider_task_id}),
            "unresolved_reservations": sum(item.status == "reserved" for item in ledger),
            "statistics": derive_production_statistics(workspace.manifest),
            "prompt_fingerprints": sorted({a.fingerprint for state in workspace.manifest.shots.values() for a in state.prompt_attempts}),
        })
    return {"version": manifest["version"], "hypothesis": manifest["hypothesis"],
            "generation_quality": "not_evaluated", "cost_currency": None,
            "samples": rows, "note": "技术验收不代表动作正确；费用需对账；小样本不能证明普遍改善"}

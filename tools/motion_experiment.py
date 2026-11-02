"""Offline trial preparation/check/report; explicit execution reuses the production generator."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from inspect import signature
from pathlib import Path

from pipeline.models import RunStatus
from pipeline.motion_experiment import (
    experiment_preflight, experiment_report, load_experiment, prepare_experiment,
)
from pipeline.motion_reference import MOTION_MODELS
from tools.motion_previs import read_previs_json


async def execute_experiment(root: Path) -> dict:
    # 本机独占执行；进程退出时 OS 自动释放，避免崩溃留下永久锁。
    import fcntl
    from pipeline.generator import ProgressPersistenceError, RemoteTaskPendingError, VideoGenerator

    with (root / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("此实验已有执行进程") from None
        preflight = experiment_preflight(root)
        if preflight["blockers"]:
            return preflight
        _, loaded = load_experiment(root)
        for _, workspace in loaded:
            if workspace.accepted_shot_artifacts():
                continue
            pending = workspace.resumable_pending_tasks()
            spent = any(item.status in {"submitted", "reconciled"}
                        for item in workspace.manifest.paid_take_budget.reservations)
            if spent and not pending:
                # 一个样本一次付费；失败也保留，不用补拍挑选更好的样本。
                continue

            def record(result):
                # ShotResult 和账本共享命名；过滤仅用于显示的字段，完整保存公开账本字段。
                fields = signature(workspace.record_shot).parameters
                workspace.record_shot(**{key: value for key, value in asdict(result).items() if key in fields})

            generator = VideoGenerator(
                str(workspace.path), on_progress=record,
                reserve_paid_take=workspace.reserve_paid_take,
                confirm_paid_take_submission=workspace.confirm_paid_take_submission,
                reconcile_paid_take=workspace.reconcile_paid_take,
                release_unsubmitted_paid_take=workspace.release_unsubmitted_paid_take,
                resume_tasks=pending, accepted_shot_artifacts=workspace.accepted_shot_artifacts(),
            )
            if generator.api.ark_client is not None:
                # 创建请求超时可能已经收费；实验禁止 SDK 在账本之外隐式重试。
                generator.api.ark_client.max_retries = 0
            workspace.checkpoint("raw_take_generation", RunStatus.running)
            try:
                results = await generator.generate_all(workspace.load_storyboard())
            except RemoteTaskPendingError:
                workspace.mark_interrupted("远端任务未完成；再次执行只轮询原任务")
                return {"status": "pending", "report": experiment_report(root)}
            except RuntimeError as exc:
                states = workspace.manifest.shots.values()
                ledger = workspace.manifest.paid_take_budget.reservations
                if (not isinstance(exc, ProgressPersistenceError)
                        and all(state.status.value == "failed" for state in states)
                        and any(item.status == "reconciled" for item in ledger)
                        and not any(item.status == "reserved" for item in ledger)):
                    workspace.mark_failed("样本失败；保留原始结果，不追加付费重拍")
                    continue
                workspace.mark_interrupted("生成中断；核对原工作区后恢复")
                raise
            except BaseException:
                workspace.mark_interrupted("执行中断；保留任务与预算记录，恢复前检查原工作区")
                raise
            if all(result.status == "success" for result in results):
                workspace.checkpoint("raw_take_ready", RunStatus.succeeded)
            else:
                workspace.mark_failed("样本失败；保留原始结果，不追加付费重拍")
                # 没有任务 ID 的输入/传输失败属于准备问题，停止整批以免重复失败。
                if not any(item.provider_task_id for item in workspace.manifest.paid_take_budget.reservations):
                    return {"status": "blocked", "report": experiment_report(root)}
        return {"status": "finished", "report": experiment_report(root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="动作对照实验：默认离线；run --execute 才会调用真实服务")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--storyboard", type=Path, required=True)
    prepare.add_argument("--package", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--model", choices=sorted(MOTION_MODELS), default="doubao-seedance-2-0-mini-260615")
    prepare.add_argument("--repetitions", type=int, default=3)
    prepare.add_argument("--paid-take-budget", type=int, default=0)
    source = prepare.add_mutually_exclusive_group()
    source.add_argument("--video-url")
    source.add_argument("--asset-uri")
    for name in ("check", "run", "report", "review-export"):
        command = sub.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
        if name == "run":
            command.add_argument("--execute", action="store_true")
        if name == "review-export":
            command.add_argument("--output", type=Path, required=True)
        if name == "report":
            command.add_argument("--review-dir", type=Path)
            command.add_argument("--ratings-file", type=Path, action="append",
                                 help="指定一或两份独立评分文件；需同时指定 --review-dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare_experiment(read_previs_json(args.storyboard), args.package, args.output,
                model=args.model, repetitions=args.repetitions, paid_take_budget=args.paid_take_budget,
                video_url=args.video_url, asset_uri=args.asset_uri)
            result = {"status": "prepared_offline", "experiment": str(path), "remote_calls": 0}
        elif args.command == "run" and args.execute:
            result = asyncio.run(execute_experiment(args.experiment))
        elif args.command in {"run", "check"}:
            result = experiment_preflight(args.experiment)
            result["executed"] = False
        elif args.command == "review-export":
            from pipeline.motion_review import export_review_bundle
            result = export_review_bundle(args.experiment, args.output)
        else:
            if args.ratings_file and not args.review_dir:
                raise ValueError("--ratings-file 需要 --review-dir")
            result = experiment_report(args.experiment)
            if args.review_dir:
                from pipeline.motion_review import summarize_reviews
                result["human_review"] = summarize_reviews(args.experiment, args.review_dir, ratings_files=args.ratings_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") in {"blocked", "pending"} else 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        # 外部服务异常可能带 URL 凭据；结构化错误详情留在既有工作区。
        print(f"实验未完成（{type(exc).__name__}），请检查输入文件与工作区状态；不会自动追加预算。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

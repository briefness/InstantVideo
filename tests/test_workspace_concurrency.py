"""Offline cross-instance/process regression of durable run ownership."""
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import pytest

from pipeline.models import RunOptions
from pipeline.run_state import PaidTakeBudgetExhaustedError, RunWorkspace


def _reserve(path, start, results):
    workspace = RunWorkspace.resume(path)
    start.wait(10)
    try:
        results.put(workspace.reserve_paid_take(1))
    except PaidTakeBudgetExhaustedError:
        results.put("exhausted")


def test_stale_instance_cannot_reserve_same_authorization(tmp_path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    first, second = RunWorkspace.resume(workspace.path), RunWorkspace.resume(workspace.path)
    first.reserve_paid_take(1)
    with pytest.raises(PaidTakeBudgetExhaustedError):
        second.reserve_paid_take(1)


@pytest.mark.parametrize("mutation", [
    lambda w: w.checkpoint("postprocessing"),
    lambda w: w.mark_failed("offline failure"),
    lambda w: w.mark_interrupted(),
    lambda w: w.record_shot(shot_id=2, status="running"),
    lambda w: w.mark_succeeded("final.mp4"),
])
def test_stale_writers_preserve_other_shots_and_paid_ledger(tmp_path, mutation):
    first = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    stale = RunWorkspace.resume(first.path)
    reservation = first.reserve_paid_take(1)
    first.confirm_paid_take_submission(reservation, "offline-provider-task")
    first.record_shot(shot_id=1, status="failed", provider_task_id="offline-provider-task")
    mutation(stale)
    latest = RunWorkspace.resume(first.path).manifest
    assert latest.paid_take_budget.reservations[0].provider_task_id == "offline-provider-task"
    assert latest.shots["1"].provider_task_id == "offline-provider-task"


def test_processes_share_a_single_budget(tmp_path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [context.Process(target=_reserve, args=(workspace.path, start, results)) for _ in range(3)]
    try:
        for process in processes:
            process.start()
        start.set()
        received = [results.get(timeout=20) for _ in processes]
        assert sorted(received) == ["1:1", "exhausted", "exhausted"]
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
    assert len(RunWorkspace.resume(workspace.path).manifest.paid_take_budget.reservations) == 1


@pytest.mark.asyncio
async def test_formal_run_excludes_second_executor_and_releases_after_cancel(tmp_path, monkeypatch):
    from pipeline.orchestrator import VideoPipeline
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline"))
    first, second = VideoPipeline(run_workspace=workspace), VideoPipeline.from_workspace(workspace.path)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def work(_):
        entered.set()
        await finish.wait()
        return "done"

    monkeypatch.setattr(first, "_resuming", False)
    monkeypatch.setattr(second, "_resuming", False)
    monkeypatch.setattr(first, "_run", work)
    other_work = AsyncMock(return_value="done")
    monkeypatch.setattr(second, "_run", other_work)
    active = asyncio.create_task(first.run())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(RuntimeError, match="正在运行"):
            await second.run()
        other_work.assert_not_awaited()
        assert RunWorkspace.resume(workspace.path).manifest.error is None
    finally:
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
    assert await second.run() == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [False, True])
async def test_crash_between_submission_checkpoints_cannot_start_a_second_take(tmp_path, monkeypatch, confirmed):
    from pipeline.orchestrator import VideoPipeline
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=2))
    reservation = workspace.reserve_paid_take(1)
    if confirmed:
        workspace.confirm_paid_take_submission(reservation, "offline-task")
    resumed = VideoPipeline.from_workspace(workspace.path)
    monkeypatch.setattr(resumed, "_resuming", False)
    work = AsyncMock(return_value="unexpected fresh submission")
    monkeypatch.setattr(resumed, "_run", work)
    with pytest.raises(RuntimeError, match="提交结果尚未确认"):
        await resumed.run()
    work.assert_not_awaited()
    assert len(RunWorkspace.resume(workspace.path).manifest.paid_take_budget.reservations) == 1


def test_stale_private_save_cannot_bypass_transaction(tmp_path):
    first = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    stale = RunWorkspace.resume(first.path)
    first.reserve_paid_take(1)
    with pytest.raises(RuntimeError, match="运行状态已变化"):
        stale._save_manifest()
    assert len(RunWorkspace.resume(first.path).manifest.paid_take_budget.reservations) == 1


def test_failed_atomic_publication_rolls_back_and_cleans_temp_files(tmp_path, monkeypatch):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    with monkeypatch.context() as patch:
        patch.setattr("pipeline.run_state.os.replace", Mock(side_effect=OSError("offline disk failure")))
        with pytest.raises(OSError, match="offline disk failure"):
            workspace.reserve_paid_take(1)
    assert workspace.manifest.paid_take_budget.reservations == []
    assert RunWorkspace.resume(workspace.path).manifest.paid_take_budget.reservations == []
    assert not list(workspace.path.glob("*.tmp"))
    assert workspace.reserve_paid_take(1) == "1:1"


@pytest.mark.parametrize("shared_instance", [True, False])
def test_threads_serialize_mutations_without_losing_reservations(tmp_path, shared_instance):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=12))
    workers = [workspace if shared_instance else RunWorkspace.resume(workspace.path) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        reservations = list(pool.map(lambda worker: worker.reserve_paid_take(1), workers))
    assert len(set(reservations)) == 12
    assert len(RunWorkspace.resume(workspace.path).manifest.paid_take_budget.reservations) == 12


@pytest.mark.parametrize("lock_name", [".manifest.lock", ".execution.lock"])
def test_workspace_lock_rejects_symlink(tmp_path, lock_name):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline"))
    target = tmp_path / "unrelated"
    target.write_text("unchanged")
    lock = workspace.path / lock_name
    lock.unlink(missing_ok=True)
    lock.symlink_to(target)
    with pytest.raises(OSError):
        if lock_name == ".manifest.lock":
            workspace.checkpoint("must not persist")
        else:
            with workspace.execution():
                pytest.fail("unsafe execution lock")
    assert target.read_text() == "unchanged"


def test_known_pending_task_remains_pollable(tmp_path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="offline", paid_take_budget=1))
    reservation = workspace.reserve_paid_take(1)
    workspace.confirm_paid_take_submission(reservation, "offline-task")
    workspace.record_shot(shot_id=1, status="running", provider_task_id="offline-task",
                          prompt_profile="normal", prompt_fingerprint="a" * 64,
                          compiled_contract_version="action-contract-v4",
                          compiled_contract_fingerprint="b" * 64)
    resumed = RunWorkspace.resume(workspace.path)
    resumed.ensure_recoverable_submissions()
    assert resumed.resumable_pending_tasks()[1]["task_id"] == "offline-task"

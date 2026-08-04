"""One-click orchestration recovery tests."""

from pathlib import Path

import pytest

import config
from pipeline.generator import RemoteTaskPendingError, ShotResult
from pipeline.models import RunOptions, RunStatus
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SemanticReviewUnavailableError


def storyboard() -> dict:
    return {
        "title": "Test",
        "mood": "cinematic",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "shots": [
            {
                "shot_id": 1,
                "duration": 5,
                "scene_description": "A test scene",
                "prompt_en": "A cinematic test scene with stable motion",
            }
        ],
    }


class FailedGenerator:
    def __init__(self, *_args, on_progress=None, **_kwargs):
        self.on_progress = on_progress

    async def generate_all(self, _storyboard):
        result = ShotResult(shot_id=1, status="failed", errors=["generation failed"])
        if self.on_progress:
            self.on_progress(result)
        return [result]


class PendingGenerator:
    def __init__(self, *_args, on_progress=None, **_kwargs):
        self.on_progress = on_progress

    async def generate_all(self, _storyboard):
        result = ShotResult(
            shot_id=1,
            status="running",
            provider_task_id="ark-task-running",
        )
        if self.on_progress:
            self.on_progress(result)
        raise RemoteTaskPendingError(result, Path("/tmp/test-run"))


class ReviewPendingGenerator:
    def __init__(self, *_args, on_progress=None, **_kwargs):
        self.on_progress = on_progress

    async def generate_all(self, _storyboard):
        result = ShotResult(
            shot_id=1,
            status="running",
            provider_task_id="ark-task-generated",
            local_path="shots/shot_001.mp4",
        )
        if self.on_progress:
            self.on_progress(result)
        raise SemanticReviewUnavailableError("语义验收暂不可用，视频已保留")


@pytest.mark.asyncio
async def test_failed_shot_stops_before_postprocessing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", lambda **_kwargs: storyboard())
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", FailedGenerator)
    normalize_calls: list[str] = []
    monkeypatch.setattr(
        "pipeline.orchestrator.ffmpeg_ops.normalize_video",
        lambda *_args, **_kwargs: normalize_calls.append("called"),
    )

    pipeline = VideoPipeline()
    with pytest.raises(RuntimeError, match="可使用 --resume 继续"):
        await pipeline.run("5秒测试视频")

    assert normalize_calls == []
    assert pipeline.run_workspace.manifest.status.value == "failed"


@pytest.mark.asyncio
async def test_pending_remote_task_interrupts_run_instead_of_failing(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        "pipeline.orchestrator.generate_storyboard", lambda **_kwargs: storyboard()
    )
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", PendingGenerator)

    pipeline = VideoPipeline()
    with pytest.raises(RemoteTaskPendingError, match="--resume"):
        await pipeline.run("5秒测试视频")

    assert pipeline.run_workspace.manifest.status == RunStatus.interrupted
    assert pipeline.run_workspace.manifest.shots["1"].status.value == "running"


@pytest.mark.asyncio
async def test_semantic_review_outage_interrupts_without_resubmission(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        "pipeline.orchestrator.generate_storyboard", lambda **_kwargs: storyboard()
    )
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", ReviewPendingGenerator)

    pipeline = VideoPipeline()
    with pytest.raises(SemanticReviewUnavailableError, match="视频已保留"):
        await pipeline.run("5秒测试视频")

    assert pipeline.run_workspace.manifest.status == RunStatus.interrupted
    assert pipeline.run_workspace.resumable_provider_tasks() == {
        1: "ark-task-generated"
    }


@pytest.mark.asyncio
async def test_resume_skips_storyboard_generation(tmp_path: Path, monkeypatch):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="5秒测试视频"))
    workspace.save_storyboard(storyboard())
    monkeypatch.setattr(
        "pipeline.orchestrator.generate_storyboard",
        lambda **_kwargs: pytest.fail("resume must not regenerate storyboard"),
    )
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", FailedGenerator)

    pipeline = VideoPipeline.from_workspace(workspace.path)
    with pytest.raises(RuntimeError, match="可使用 --resume 继续"):
        await pipeline.run()


@pytest.mark.asyncio
async def test_resume_blocks_legacy_remote_task_without_submission_descriptor(
    tmp_path: Path, monkeypatch
):
    """An old running task must never be mistaken for a fresh paid submission."""
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="5秒测试视频"))
    workspace.save_storyboard(storyboard())
    workspace.record_shot(
        shot_id=1,
        status="running",
        provider_task_id="legacy-provider-task",
        attempts=1,
    )

    class GeneratorMustNotStart:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("legacy unresolved task must block before generator startup")

    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", GeneratorMustNotStart)

    pipeline = VideoPipeline.from_workspace(workspace.path)
    with pytest.raises(RuntimeError, match="缺少不可变提交描述"):
        await pipeline.run()

    assert pipeline.run_workspace.resumable_pending_tasks() == {}
    assert pipeline.run_workspace.resumable_provider_tasks() == {
        1: "legacy-provider-task"
    }


@pytest.mark.asyncio
async def test_resume_blocks_terminal_local_qa_failure_without_new_submission(
    tmp_path: Path, monkeypatch
):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="5秒测试视频"))
    workspace.save_storyboard(storyboard())
    failed_take = workspace.path / "shots" / "shot_001.mp4"
    failed_take.parent.mkdir(exist_ok=True)
    failed_take.write_bytes(b"unusable local take")
    workspace.record_shot(
        shot_id=1,
        status="failed",
        provider_task_id="already-paid-task",
        local_path=str(failed_take),
        errors=["远端任务已成功，但本地技术 QA 异常"],
    )

    class GeneratorMustNotStart:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("terminal materialization failure must block before generation")

    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", GeneratorMustNotStart)

    pipeline = VideoPipeline.from_workspace(workspace.path)
    with pytest.raises(RuntimeError, match="本地物化或技术 QA 失败"):
        await pipeline.run()


@pytest.mark.asyncio
async def test_completed_resume_returns_without_running_pipeline(tmp_path: Path, monkeypatch):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="5秒测试视频"))
    workspace.save_storyboard(storyboard())
    final_path = workspace.path / "final.mp4"
    final_path.write_bytes(b"video")
    workspace.mark_succeeded(str(final_path))

    pipeline = VideoPipeline.from_workspace(workspace.path)
    monkeypatch.setattr(
        pipeline,
        "_run",
        lambda _request: pytest.fail("completed run must not execute stages"),
    )

    assert await pipeline.run() == str(final_path)


@pytest.mark.asyncio
async def test_resume_passes_accepted_shots_as_canonical_generator_inputs(
    tmp_path: Path, monkeypatch
):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="5秒测试视频"))
    workspace.save_storyboard(storyboard())
    shot_path = workspace.path / "shots" / "shot_001.mp4"
    shot_path.parent.mkdir(parents=True)
    shot_path.write_bytes(b"accepted")
    workspace.record_shot(
        shot_id=1,
        status="success",
        local_path=str(shot_path),
        semantic_accepted=True,
        observed_end_state={"location": "test scene"},
    )
    captured = {}

    class CapturingGenerator(FailedGenerator):
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)
            super().__init__(*_args, **kwargs)

    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", CapturingGenerator)

    pipeline = VideoPipeline.from_workspace(workspace.path)
    with pytest.raises(RuntimeError, match="可使用 --resume 继续"):
        await pipeline.run()

    accepted = captured["accepted_shot_artifacts"]
    assert accepted[1]["local_path"] == str(shot_path)
    assert accepted[1]["semantic_accepted"] is True


def test_shot_timeline_uses_actual_media_durations():
    pipeline = VideoPipeline()
    board = {
        "shots": [
            {"shot_id": 1, "duration": 5},
            {"shot_id": 2, "duration": 5},
            {"shot_id": 3, "duration": 5},
        ]
    }

    starts = pipeline._calc_shot_start_times(
        board,
        transitions=[("cut", 0.0), ("crossfade", 0.5)],
        media_durations={1: 4.25, 2: 6.0, 3: 4.8},
    )

    assert starts == {1: 0.0, 2: 4.25, 3: 9.75}

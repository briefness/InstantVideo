"""One-click orchestration recovery tests."""

from pathlib import Path

import pytest

import config
from pipeline.generator import ShotResult
from pipeline.models import RunOptions
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace


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

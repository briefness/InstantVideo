"""Explicit resume shares the original two-call planning limit; no network."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import pipeline.motion_preparation as preparation
from test_motion_preparation import inputs, offline, NumericPlanner  # noqa: F401


def _unavailable():
    create = Mock(side_effect=TimeoutError("offline injected timeout"))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


def test_transient_planner_failure_recovers_in_same_workspace(inputs, tmp_path):
    board, raw = inputs
    unavailable, calls = _unavailable()
    first = preparation.prepare_motion_previsualization(board, tmp_path, client=unavailable)
    assert first["status"] == "failed" and calls.call_count == 1
    healthy = NumericPlanner(raw)
    second = preparation.prepare_motion_previsualization(board, tmp_path, client=healthy)
    assert second["status"] == "prepared_unbound", second
    assert len(healthy.calls) == 1
    assert second["shots"][0]["planning_attempts"] == 2
    assert not second["shots"][0].get("retryable")


def test_repeated_resume_never_resets_planner_call_limit(inputs, tmp_path):
    board, _ = inputs
    unavailable, calls = _unavailable()
    for _ in range(5):
        result = preparation.prepare_motion_previsualization(board, tmp_path, client=unavailable)
        assert result["status"] == "failed"
    assert calls.call_count == 2
    assert result["shots"][0]["planning_attempts"] == 2
    assert result["shots"][0]["retryable"] is False


@pytest.mark.parametrize("raw", ['{"rigs":', '{"rigs": [], "contact": {}, "command": "untrusted"}'])
def test_invalid_output_is_not_reclassified_as_network_failure(inputs, tmp_path, raw):
    board, valid = inputs
    first = preparation.prepare_motion_previsualization(board, tmp_path, client=NumericPlanner(raw))
    healthy = NumericPlanner(valid)
    second = preparation.prepare_motion_previsualization(board, tmp_path, client=healthy)
    assert first["status"] == second["status"] == "failed"
    assert not healthy.calls


def test_unknown_interrupted_attempt_does_not_silently_retry(inputs, tmp_path):
    board, raw = inputs
    unavailable, _ = _unavailable()
    preparation.prepare_motion_previsualization(board, tmp_path, client=unavailable)
    path = tmp_path / "motion_preparation.json"
    value = json.loads(path.read_text())
    value["shots"][0].update(status="preparing", retryable=False)
    path.write_text(json.dumps(value))
    planner = NumericPlanner(raw)
    assert preparation.prepare_motion_previsualization(board, tmp_path, client=planner)["status"] == "failed"
    assert not planner.calls


@pytest.mark.asyncio
async def test_formal_resume_recovers_transient_preparation_without_video_call(inputs, tmp_path, monkeypatch):
    import config
    from copy import deepcopy
    from pipeline.orchestrator import MotionPreparationRequiredError, VideoPipeline
    board, raw = inputs
    unavailable, calls = _unavailable()
    healthy = NumericPlanner(raw)
    clients = iter([unavailable, healthy])
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.motion_preparation.OpenAI", lambda **kw: next(clients))
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", lambda **kw: deepcopy(board))
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", lambda *a, **kw: pytest.fail("video provider reached"))
    pipeline = VideoPipeline(motion_control="reference_motion", paid_take_budget=0)
    with pytest.raises(MotionPreparationRequiredError, match="failed"):
        await pipeline.run("4秒训练动作")
    for _ in range(2):
        resumed = VideoPipeline.from_workspace(pipeline.workspace)
        with pytest.raises(MotionPreparationRequiredError, match="prepared_unbound"):
            await resumed.run()
    assert calls.call_count == 1 and len(healthy.calls) == 1
    assert resumed.run_workspace.manifest.paid_take_budget.reservations == []


def test_invalid_package_clears_previous_network_retry_permission(inputs, tmp_path):
    board, raw = inputs
    unavailable, _ = _unavailable()
    first = preparation.prepare_motion_previsualization(board, tmp_path, client=unavailable)
    package = tmp_path / first["shots"][0]["package"]
    package.parent.mkdir()
    package.write_text("invalid interrupted package")
    healthy = NumericPlanner(raw)
    failed = preparation.prepare_motion_previsualization(board, tmp_path, client=healthy)
    assert failed["status"] == "failed" and not healthy.calls
    assert failed["shots"][0]["retryable"] is False

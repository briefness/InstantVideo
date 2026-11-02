"""Formal orchestration: SDK doubles, real preparation/media, no paid service."""
import json
from copy import deepcopy

import httpx
import pytest

import config
from pipeline.models import DirectorIntentSpec, RunOptions
from pipeline.orchestrator import VideoPipeline, MotionPreparationRequiredError
from pipeline.run_state import RunWorkspace
from pipeline.motion_reference import probe_motion_reference
from test_motion_preparation import EXAMPLE, NumericPlanner
from test_action_sequence import sequence_story
from web_server import _parse_create_payload, _run_summary


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("offline entrypoint attempted network access")
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(httpx.AsyncClient, "send", reject)
    monkeypatch.setattr(config, "SEEDANCE_MODEL", "doubao-seedance-2-0-mini-260615")


@pytest.mark.asyncio
async def test_formal_run_prepares_real_video_then_resumes_without_replanning(tmp_path, monkeypatch):
    board = json.loads((EXAMPLE / "storyboard.json").read_text())
    raw = json.loads((EXAMPLE / "motion.json").read_text())
    client = NumericPlanner({k: raw[k] for k in ("rigs", "contact")})
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", lambda **kw: deepcopy(board))
    monkeypatch.setattr("pipeline.motion_preparation.OpenAI", lambda **kw: client)
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", lambda *a, **kw: pytest.fail("paid generator reached"))
    pipeline = VideoPipeline(motion_control="reference_motion", paid_take_budget=0)
    with pytest.raises(MotionPreparationRequiredError, match="prepared_unbound"):
        await pipeline.run("4秒训练动作，使用建模动作参考")
    assert pipeline.run_workspace.manifest.status.value == "interrupted"
    assert pipeline.run_workspace.options.motion_control == "reference_motion"
    report = json.loads((pipeline.workspace / "motion_preparation.json").read_text())
    media = pipeline.workspace / report["shots"][0]["reference_path"]
    assert int(probe_motion_reference(media.read_bytes(), 4, "16:9")["nb_frames"]) == 96
    before = media.read_bytes()
    assert len(client.calls) == 1
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    with pytest.raises(MotionPreparationRequiredError, match="prepared_unbound"):
        await resumed.run()
    assert len(client.calls) == 1 and media.read_bytes() == before
    assert resumed.run_workspace.options.paid_take_budget == 0


@pytest.mark.asyncio
async def test_unsupported_sequence_never_degrades_to_text_or_calls_numeric_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", lambda **kw: sequence_story())
    monkeypatch.setattr("pipeline.motion_preparation.OpenAI", lambda **kw: pytest.fail("unsupported model call"))
    monkeypatch.setattr("pipeline.orchestrator.VideoGenerator", lambda *a, **kw: pytest.fail("silent text fallback"))
    pipeline = VideoPipeline(motion_control="reference_motion", paid_take_budget=0)
    with pytest.raises(MotionPreparationRequiredError, match="unsupported"):
        await pipeline.run("5秒连续交互")
    report = json.loads((pipeline.workspace / "execution_control.json").read_text())
    assert report["reference_required"] is True and report["status"] == "unsupported"


@pytest.mark.parametrize("mode", ["auto", "text_guided"])
def test_frozen_semantic_requirement_cannot_be_overridden_by_text_mode(tmp_path, monkeypatch, mode):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="用动作参考", motion_control=mode))
    pipeline = VideoPipeline(run_workspace=workspace)
    board = sequence_story()
    board["production_plan"] = {"director_intent": {"motion_control": "reference_motion"}}
    monkeypatch.setattr("pipeline.motion_preparation.OpenAI", lambda **kw: pytest.fail("unsupported model call"))
    with pytest.raises(MotionPreparationRequiredError):
        pipeline._prepare_execution_inputs(board)


def test_default_records_actual_text_capability_without_preparation(tmp_path, monkeypatch):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="交互动作"))
    pipeline = VideoPipeline(run_workspace=workspace)
    monkeypatch.setattr("pipeline.orchestrator.prepare_motion_previsualization", lambda *a: pytest.fail("unexpected preparation"))
    pipeline._prepare_execution_inputs(sequence_story())
    report = json.loads((workspace.path / "execution_control.json").read_text())
    assert report["shots"] == [{"shot_id": 1, "mode": "text_guided"}]
    assert report["reference_required"] is False


@pytest.mark.parametrize("mode", ["auto", "text_guided", "reference_motion"])
def test_web_and_resume_preserve_control_requirement(tmp_path, mode):
    options = _parse_create_payload({"request": "动作验证", "motion_control": mode})
    workspace = RunWorkspace.create(tmp_path, options)
    assert VideoPipeline.from_workspace(workspace.path).motion_control == mode
    assert _run_summary(workspace.path)["options"]["motion_control"] == mode


@pytest.mark.parametrize("mode", [None, "3d_guaranteed", True, {}])
def test_web_rejects_unknown_control_mode(mode):
    with pytest.raises(ValueError):
        _parse_create_payload({"request": "动作验证", "motion_control": mode})


def test_motion_requirement_needs_source_evidence():
    base = {"source_request": "先用骨架预演动作", "content_focus": "action", "pacing": "balanced",
            "intent_evidence": "先用骨架预演动作", "motion_control": "reference_motion"}
    with pytest.raises(ValueError, match="原文依据"):
        DirectorIntentSpec.model_validate(base)
    intent = DirectorIntentSpec.model_validate({**base, "motion_control_evidence": "骨架预演动作"})
    assert intent.motion_control == "reference_motion"


def test_semantic_intent_retains_explicit_motion_requirement():
    from pipeline.director_intent import resolve_director_intent
    from test_director_intent import Responses, intent_reply
    request = "先用骨架预演动作"
    intent = resolve_director_intent(Responses({**intent_reply(request, pacing="balanced"),
        "motion_control": "reference_motion", "motion_control_evidence": "骨架预演动作"}), request)
    assert intent["motion_control"] == "reference_motion"


def test_cli_passes_control_requirement(monkeypatch):
    import main
    from types import SimpleNamespace
    seen = {}
    async def run(request):
        seen["request"] = request
        return "offline.mp4"
    def factory(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(run=run, delivery_mode=kwargs['delivery_mode'])
    monkeypatch.setattr(main, "VideoPipeline", factory)
    monkeypatch.setattr("sys.argv", ["main.py", "动作验证", "--motion-control", "reference_motion", "--paid-take-budget", "0"])
    main.main()
    assert seen["motion_control"] == "reference_motion" and seen["paid_take_budget"] == 0
    assert seen['delivery_mode'] == 'reviewed'


def test_cli_cannot_replace_control_requirement_on_resume(monkeypatch):
    import main
    monkeypatch.setattr("sys.argv", ["main.py", "--resume", "unused", "--motion-control", "text_guided"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2

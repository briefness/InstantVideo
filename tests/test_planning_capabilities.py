"""Use structured execution constraints before spending creative planning calls."""
import pytest

from pipeline.storyboard import generate_storyboard
from test_director_intent import intent_reply, Responses


@pytest.mark.parametrize("mode,semantic,duration,pacing", [
    ("reference_motion", False, 15, "balanced"),
    ("auto", True, 15, "balanced"),
    ("text_guided", True, 4, "dense"),
    ("reference_motion", False, 4, "dense"),
])
def test_incompatible_motion_plan_stops_before_creative_model(monkeypatch, mode, semantic, duration, pacing):
    request = "通过动作参考完成训练场景"
    response = intent_reply(request, pacing=pacing)
    if semantic:
        response.update(motion_control="reference_motion", motion_control_evidence="动作参考")
    client = Responses(response)
    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kw: client)
    with pytest.raises(ValueError, match="动作参考执行能力"):
        generate_storyboard(request, target_duration=duration, motion_control=mode)
    assert len(client.calls) == 1


def test_supported_plan_receives_control_constraints_without_rewriting_intent(monkeypatch):
    request = "训练者推开对手的防守"
    client = Responses(intent_reply(request, pacing="balanced"))
    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kw: client)

    def inspect_draft(client, system, prompt, plan):
        assert "reference_motion" in prompt and "固定机位" in prompt
        assert "不得删减" in prompt
        assert plan["director_intent"]["source_request"] == request
        raise RuntimeError("draft reached")

    monkeypatch.setattr("pipeline.storyboard._generate_storyboard_draft", inspect_draft)
    with pytest.raises(RuntimeError, match="draft reached"):
        generate_storyboard(request, target_duration=4, motion_control="reference_motion")


@pytest.mark.parametrize("mode", ["unknown", None, {}, [], True])
def test_invalid_execution_mode_stops_before_any_model(monkeypatch, mode):
    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kw: pytest.fail("unexpected model client"))
    with pytest.raises(ValueError, match="动作控制"):
        generate_storyboard("训练场景", motion_control=mode)


@pytest.mark.asyncio
async def test_official_entry_passes_explicit_control_to_planner(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import config
    from pipeline.orchestrator import VideoPipeline
    planner = Mock(side_effect=ValueError("offline planning seam"))
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("pipeline.orchestrator.generate_storyboard", planner)
    with pytest.raises(ValueError, match="offline planning seam"):
        await VideoPipeline(motion_control="reference_motion", paid_take_budget=0).run("4秒训练动作")
    assert planner.call_args.kwargs["motion_control"] == "reference_motion"

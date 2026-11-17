"""单次规划限制覆盖旧分镜路径，不能只限制视频重拍。"""
import json

import pytest

from pipeline.models import RunOptions
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.storyboard import generate_storyboard, _generate_storyboard_draft
from test_director_intent import Responses, draft
from test_storyboard_llm_response import _client


def frozen_legacy_intent(monkeypatch, brief):
    """旧版本已冻结解释的测试接缝，不经过新解释器，不修改实际运行。"""
    from pipeline.models import DirectorIntentSpec
    value = DirectorIntentSpec(source_request=brief, content_focus='balanced',
                               pacing='balanced', intent_evidence=brief).model_dump()
    monkeypatch.setattr('pipeline.storyboard.resolve_director_intent', lambda *_a, **_kw: dict(value))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('unexpected network'))


@pytest.mark.parametrize('duration', [12, 60])
def test_malformed_draft_stops_without_json_repair_and_preserves_response(duration):
    from pipeline.production_plan import build_production_plan
    client, calls = _client('{malformed')
    records = {}
    with pytest.raises(ValueError, match='未重新调用模型'):
        _generate_storyboard_draft(client, 'system', 'brief', build_production_plan('balanced', duration),
            allow_repair=False, save_diagnostic=records.__setitem__)
    assert len(calls.calls) == 1
    assert len(records) == 1 and next(iter(records.values()))['content'] == '{malformed'


def test_invalid_draft_does_not_trigger_contract_rewrite(monkeypatch):
    brief = '记录日常训练。'
    # 独立错误样本：摄影机运动不能替代主体表演，不依赖编译器误改 coverage。
    invalid = draft()
    invalid['shots'][0]['primary_action'] = 'the camera pans across the room'
    frozen_legacy_intent(monkeypatch, brief)
    client = Responses(invalid)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    records = {}
    with pytest.raises(ValueError, match='规划修正已关闭'):
        generate_storyboard(brief, target_duration=15, allow_repair=False, save_diagnostic=records.__setitem__)
    assert len(client.calls) == 1
    assert records['planning_check_1.json']['critical']
    assert 'planning_legacy_response.json' in records and 'planning_draft_2.json' not in records


def test_no_repair_policy_is_frozen_and_legacy_default_stays_omitted(tmp_path):
    assert 'planning_repair' not in RunOptions(request='existing').model_dump()
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='new', planning_repair=False, paid_take_budget=0))
    restored = VideoPipeline.from_workspace(workspace.path)
    assert restored.planning_repair is False
    assert restored.run_workspace.options.model_dump()['planning_repair'] is False


@pytest.mark.parametrize('invalid', ['false', 0, None])
def test_policy_rejects_coerced_values(invalid):
    with pytest.raises(ValueError):
        RunOptions(request='test', planning_repair=invalid)


@pytest.mark.asyncio
async def test_formal_plan_only_stops_before_video_and_retains_failed_draft(tmp_path, monkeypatch):
    import config
    brief = '15秒记录日常训练。'
    invalid = draft()
    invalid['shots'][0]['primary_action'] = 'the camera pans across the room'
    frozen_legacy_intent(monkeypatch, brief)
    client = Responses(invalid)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **_: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, planning_repair=False)
    with pytest.raises(ValueError, match='规划修正已关闭'):
        await pipeline.run(brief)
    assert len(client.calls) == 1
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    assert json.loads((pipeline.workspace / 'planning_check_1.json').read_text())['critical']
    restored = VideoPipeline.from_workspace(pipeline.workspace)
    restored.plan_only = True
    with pytest.raises(ValueError, match='规划修正已关闭'):
        await restored.run()
    assert len(client.calls) == 1

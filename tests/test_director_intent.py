"""Semantic-service boundary replay: no live model or provider calls."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.director_intent import resolve_director_intent
from pipeline.production_plan import build_production_plan, production_plan_issues
from pipeline.storyboard import generate_storyboard


class Responses:
    def __init__(self, *values):
        responses = []
        for value in values:
            if isinstance(value, dict) and value.get('shots') and 'action_plan' in value['shots'][0]:
                responses.append({k: v for k, v in value.items() if k != 'shots'})
                from authoring_cases import contextual_shot_fixture, SyntheticAuthoredResponse
                response_type = SyntheticAuthoredResponse if 'locations' in value else dict
                responses.extend(response_type(shot={k: v for k, v in (
                    contextual_shot_fixture(shot) if 'locations' in value else shot).items() if k != 'shot_id'})
                    for shot in value['shots'])
            else:
                responses.append(value)
        self.values = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        from planning_protocol_cases import wire_verdict
        from authoring_cases import SyntheticAuthoredResponse, scheduled_shot_fixture
        value = next(self.values)
        if (kwargs.get('response_format', {}).get('json_schema', {}).get('name') == 'director_intent'
                and isinstance(value, dict) and 'completion_goals' not in value):
            # 旧离线服务用整项请求作为一个严格目标；新语义分解有专门标注用例。
            value = {**value, 'completion_goals': [{'goal': value.get('ending_requirement') or value['intent_evidence'],
                'basis': 'user', 'evidence': value['intent_evidence']}]}
        if isinstance(value, SyntheticAuthoredResponse):
            context = json.loads(kwargs['messages'][-1]['content'])
            if context.get('authoring_protocol') in {'story-context-v5-entity-results', 'story-context-v6-candidate-results',
                                                    'story-context-v7-execution-timing', 'story-context-v8-action-semantics',
                                                    'story-context-v9-temporal-relations'}:
                value = {'shot': scheduled_shot_fixture(value['shot'], context['slot'],
                    execution_timing=context['authoring_protocol'] in {
                        'story-context-v7-execution-timing', 'story-context-v8-action-semantics',
                        'story-context-v9-temporal-relations'})}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=json.dumps(wire_verdict(value, kwargs)), refusal=None))])


def intent_reply(request, focus='action', pacing='dense'):
    return {'content_focus': focus, 'pacing': pacing, 'intent_evidence': request,
            'ending_requirement': '', 'ending_evidence': '',
            'completion_goals': [{'goal': request, 'basis': 'user', 'evidence': request}]}


@pytest.fixture(autouse=True)
def no_external_http(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx.Client, 'send', lambda *_a, **_k: pytest.fail('offline test attempted HTTP'))


@pytest.mark.parametrize('brief,focus,pacing', [
    ('武打动作要多', 'action', 'dense'),
    ('武侠动作短片，密集招式和攻防衔接', 'action', 'dense'),
    ('不要慢节奏为主，要密集格斗', 'action', 'dense'),
    ('不是产品广告，是动作视频，交锋要持续', 'action', 'dense'),
    ('武侠氛围为主，人物静静对峙，不要连续打斗', 'action', 'atmospheric'),
    ('展示运动鞋，格斗演员只是背景，突出鞋底材质', 'product', 'balanced'),
    ('A sustained exchange of counters and evasions, with very little posing', 'action', 'dense'),
])
def test_whole_request_reaches_semantic_boundary_without_keyword_override(brief, focus, pacing):
    # 人工标注的服务响应只验证协议和传递，不冒充离线语言理解精度评测。
    client = Responses(intent_reply(brief, focus, pacing))
    intent = resolve_director_intent(client, brief)
    assert json.loads(client.calls[0]['messages'][1]['content'])['request'] == brief
    plan = build_production_plan(intent['content_focus'], 15, intent=intent)
    assert plan['content_focus'] == focus
    assert plan['director_intent']['pacing'] == pacing
    if pacing == 'dense':
        assert plan['min_interaction_seconds'] == 10.5
        assert sum(slot['min_interaction_seconds'] for slot in plan['slots']) == 0
        assert all(slot['allowed_effect_phases'] == ['active'] for slot in plan['slots'])
        assert plan['slots'][-1]['framing_family'] == 'wide'


@pytest.mark.parametrize('changes', [
    {'intent_evidence': 'an invented quote'}, {'content_focus': 'unknown'},
    {'pacing': 'dense', 'content_focus': 'product'}, {'source_request': 'replace input'},
    {'ending_requirement': 'invented ending'}, {'intent_evidence': ''},
    {'unexpected': True},
])
def test_invalid_intent_stops_without_keyword_fallback(changes):
    request = '密集的交手与反制'
    with pytest.raises(ValueError, match='未按关键词'):
        resolve_director_intent(Responses({**intent_reply(request), **changes}), request)


def draft():
    base = json.loads((Path(__file__).resolve().parents[1] / 'examples/contact_previs/storyboard.json').read_text())
    original = base['shots'][0]
    base.update(title='训练交手', story_arc={'goal': 'complete the training exchange',
        'stakes': 'the guard remains untested', 'turning_point': 'contact tests the guard',
        'resolution': 'the guard yields and remains lowered'})
    base['shots'] = []
    for index in range(3):
        shot = deepcopy(original)
        shot.update(shot_id=index + 1, duration=5, scene_id='training_roof',
            mood=['focused', 'urgent', 'resolved'][index],
            prompt_en=' '.join(['A coherent cinematic training exchange with readable palm contact and a visible response.'] * 8),
            narrative_beat={'function': 'progress', 'state_before': f'state {index}',
                'state_change': 'the defender guard visibly yields', 'state_after': f'state {index + 1}'},
            start_state={'location': 'training roof', 'subject': 'attacker and defender',
                         'action_phase': 'both ready', 'camera': 'wide shot'},
            end_state={'location': 'training roof', 'subject': 'attacker and defender',
                       'action_phase': 'guard yielded', 'camera': 'wide shot'})
        beat = deepcopy(shot['action_beats'][0])
        shot['action_beats'] = [
            {**beat, 'phase': 'trigger', 'action': 'closes the remaining gap', 'visible_result': 'palms are in range',
             'start_seconds': 0., 'end_seconds': 1., 'activity': 'transition'},
            {**beat, 'phase': 'peak', 'start_seconds': 1., 'end_seconds': 2.5, 'activity': 'interaction',
             'interaction_geometry': deepcopy(shot['interaction_geometry']), 'effect_at_seconds': 1.5, 'reaction_at_seconds': 2.},
            {**beat, 'phase': 'peak', 'start_seconds': 2.5, 'end_seconds': 4.5, 'activity': 'interaction',
             'interaction_geometry': deepcopy(shot['interaction_geometry']), 'effect_at_seconds': 3., 'reaction_at_seconds': 3.5},
            {**beat, 'phase': 'aftermath', 'action': 'holds the ending guard',
             'start_seconds': 4.5, 'end_seconds': 5., 'activity': 'hold'}]
        base['shots'].append(shot)
    return base


def test_formal_planner_corrects_time_budget_without_manual_storyboard_patch(monkeypatch):
    from authoring_cases import scene_authoring_from_legacy
    from pipeline.action_timing import action_timing_issues
    request = '15秒武侠动作短片，密集招式和攻防衔接'
    bad, corrected = draft(), draft()
    for shot in bad['shots']:
        shot['action_beats'][0]['end_seconds'] = 2.
        shot['action_beats'][1]['start_seconds'] = 2.
        shot['action_beats'][1]['effect_at_seconds'] = 2.1
        shot['action_beats'][1]['reaction_at_seconds'] = 2.2
    # The new input has no absolute timestamps to repair; the compiler owns allocation.
    client = Responses(intent_reply(request), scene_authoring_from_legacy(bad))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    result = generate_storyboard(request, target_duration=15)
    assert len(client.calls) == 5  # 意图 → 全片身份 → 三个固定槽位；无重写
    assert result['content_focus'] == 'action'
    assert [s['duration'] for s in result['shots']] == [5, 5, 5]
    assert all(not action_timing_issues(s) for s in result['shots'])
    assert not production_plan_issues(result)
    changed = deepcopy(result)
    changed['production_plan']['min_interaction_seconds'] = 0
    assert production_plan_issues(changed)


def test_formal_planner_cannot_bypass_invalid_intent(monkeypatch):
    client = Responses({'content_focus': 'action'})
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    with pytest.raises(ValueError, match='未按关键词'):
        generate_storyboard('15秒密集交锋', target_duration=15)
    assert len(client.calls) == 1


@pytest.mark.parametrize('finish,raw,refusal', [
    ('length', '{}', None), ('stop', '{broken', None), ('stop', '[]', None),
    ('stop', '{}', 'refused'), ('stop', None, None), ('stop', 'x' * 20001, None),
])
def test_unusable_semantic_response_is_not_replaced_by_keyword_guess(finish, raw, refusal):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=raw, refusal=refusal))]))))
    with pytest.raises(ValueError, match='未按关键词'):
        resolve_director_intent(client, '武打动作要多')


@pytest.mark.asyncio
async def test_official_pipeline_stops_unrepaired_timing_before_video_stage(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    brief = '15秒武侠动作短片，密集招式和攻防衔接'
    bad = draft()
    for shot in bad['shots']:
        shot['action_beats'][0]['end_seconds'] = 2.
        shot['action_beats'][1]['start_seconds'] = 2.
        shot['action_beats'][1]['effect_at_seconds'] = 2.1
        shot['action_beats'][1]['reaction_at_seconds'] = 2.2
    client = Responses(intent_reply(brief), bad, deepcopy(bad))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator',
                        lambda *_a, **_k: pytest.fail('invalid timing reached video stage'))
    pipeline = VideoPipeline(paid_take_budget=0)
    with pytest.raises(ValueError, match='创作响应契约不满足'):
        await pipeline.run(brief)
    assert len(client.calls) == 2
    assert (pipeline.workspace / 'run_manifest.json').is_file()


def test_ending_is_reserved_for_final_slot_in_batched_plan():
    from pipeline.production_plan import format_production_plan
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_policy_safe_prompt
    brief = '持续交手，最后对手倒地不再起身'
    intent = {**intent_reply(brief), 'source_request': brief,
              'ending_requirement': '对手倒地不再起身', 'ending_evidence': '最后对手倒地不再起身'}
    plan = build_production_plan('action', 60, intent=intent)
    assert all(not s.get('required_outcome') for s in plan['slots'][:-1])
    assert '本批不含全片结局槽位' in format_production_plan({**plan, 'slots': plan['slots'][:3]})
    assert 'Shot 10 必须因果可见地实现用户结局' in format_production_plan({**plan, 'slots': plan['slots'][-3:]})
    shot = draft()['shots'][-1]
    shot['production_slot'] = plan['slots'][-1]
    contract = compile_action_contract(shot)
    assert contract.review_projection['required_final_outcome'] == intent['ending_requirement']
    assert intent['ending_requirement'] in compile_policy_safe_prompt(
        shot, has_state_reference=False, image_role=None, reference_count=0)

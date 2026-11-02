"""Executable authoring contract: raw replay, invariants and official SDK seam."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units, compile_authored_storyboard
from pipeline.action_timing import action_timing_issues, sequence_events
from pipeline.production_plan import build_production_plan
from pipeline.storyboard import generate_storyboard, _compile_storyboard_contract
from pipeline.readiness import storyboard_readiness_issues
from test_director_intent import draft, Responses, intent_reply
from authoring_cases import authoring_from_legacy


def plan_for(duration=15):
    return build_production_plan('action', duration, intent={**intent_reply('continuous action'), 'source_request': 'continuous action'})


@pytest.mark.parametrize('duration', [10, 15, 36, 60, 150])
def test_payoff_budget_preserves_total_actions_without_counting_a_fall_as_new_attack(duration):
    request = 'continuous action, ends with the defender falling'
    intent = {**intent_reply(request), 'source_request': request,
              'ending_requirement': 'defender falls and remains down', 'ending_evidence': 'defender falling'}
    old = build_production_plan('action', duration, intent=intent, contract_version='production-plan-v4-events')
    new = build_production_plan('action', duration, intent=intent)
    assert new['slots'][-1]['min_interaction_events'] == 1
    assert old['slots'][-1]['min_interaction_events'] >= 2
    for field in ('duration', 'min_interaction_events', 'min_interaction_seconds', 'result_hold_seconds'):
        assert sum(s[field] for s in old['slots']) == sum(s[field] for s in new['slots'])
    assert max(s['min_interaction_events'] for s in new['slots']) <= 12


def test_structured_bilingual_action_is_not_measured_by_ascii_spaces(monkeypatch):
    raw = authoring_from_legacy(draft())
    for shot in raw['shots']:
        shot['visual_description'] = '暮色下的训练屋脊，灰瓦与远处宫墙层次清晰，冷色散射光勾勒两名训练者的衣料与兵器。'
        for unit in shot['action_plan']:
            unit['action'] = '训练者顺着前一步的重心移动，完成该阶段的掌势变化'
            unit['result'] = '对手的防守位置随力道改变，双方脚步和姿势持续可见'
    client = Responses(intent_reply('continuous action'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    assert generate_storyboard('continuous action', target_duration=15)['shots']
    assert len(client.calls) == 5


def test_single_payoff_cause_includes_complete_fall_without_synthetic_second_effect(monkeypatch):
    request = '15秒持续交锋，最后对手倒下不再起身'
    intent = {**intent_reply(request), 'ending_requirement': '对手倒下不再起身', 'ending_evidence': '对手倒下不再起身'}
    raw = authoring_from_legacy(draft())
    raw['story_arc']['resolution'] = intent['ending_evidence']
    middle = raw['shots'][1]['action_plan']
    redirect = deepcopy(middle[2])
    redirect.update(id='redirect', action='redirects the returning palm outward', result='the guard opens to one side', response='deflection')
    middle.insert(3, redirect)
    ending = raw['shots'][2]['action_plan']
    ending.pop(2)
    ending[1].update(action='pushes through the open guard', result='the defender loses balance, falls and remains down')
    ending[-1].update(action='lowers the palm and stands', result='the defender remains down while the attacker stands')
    client = Responses(intent, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard(request, target_duration=15)
    assert [len(sequence_events(s)) for s in result['shots']] == [2, 3, 1]
    assert len(client.calls) == 5
    assert 'falls and remains down' in result['shots'][-1]['action_beats'][1]['visible_result']
    assert not storyboard_readiness_issues(result)


def test_semantic_units_compile_without_duplicate_authority_or_model_repair(monkeypatch):
    raw = authoring_from_legacy(draft())
    original = deepcopy(raw)
    client = Responses(intent_reply('continuous action'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    compiled = generate_storyboard('continuous action', target_duration=15)
    assert raw == original and len(client.calls) == 5
    assert not storyboard_readiness_issues(compiled)
    assert [s['duration'] for s in compiled['shots']] == [5, 5, 5]
    for source, shot in zip(raw['shots'], compiled['shots']):
        assert len(sequence_events(shot)) == 2
        for unit, beat in zip(source['action_plan'], shot['action_beats']):
            assert (unit['actor'], unit['target'], unit['action'], unit['result']) == (
                beat['actor'], beat['target'], beat['action'], beat['visible_result'])


def test_program_requests_each_fixed_slot_instead_of_asking_model_to_return_a_shot_count(monkeypatch):
    raw = authoring_from_legacy(draft())
    metadata = {k: v for k, v in raw.items() if k != 'shots'}
    responses = [{'shot': {k: v for k, v in s.items() if k != 'shot_id'}} for s in raw['shots']]
    client = Responses(intent_reply('continuous action'), metadata, *responses)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard('continuous action', target_duration=15)
    assert [s['shot_id'] for s in result['shots']] == [1, 2, 3]
    assert len(client.calls) == 5
    for i, call in enumerate(client.calls[2:]):
        context = json.loads(call['messages'][1]['content'])
        assert context['slot']['shot_id'] == i + 1
        assert context['is_final'] is (i == 2)
        assert context['previous_shot'] is None if i == 0 else context['previous_shot']['shot_id'] == i
        assert all(c['name'] in call['messages'][0]['content'] for c in raw['characters'])


def test_consistent_tone_does_not_override_visible_story_progress(monkeypatch):
    raw = authoring_from_legacy(draft())
    for shot in raw['shots']:
        shot['mood'] = 'tense'
    client = Responses(intent_reply('continuous action'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard('continuous action', target_duration=15)
    assert len({s['narrative_beat']['state_after'] for s in result['shots']}) == 3
    assert len(client.calls) == 5


@pytest.mark.parametrize('duration', [4, 5, 6, 8, 15])
@pytest.mark.parametrize('weights', [[1., 1., 1., 1.], [.001, 100., .1, .001]])
def test_integer_allocation_never_creates_gaps_or_loses_result_budget(duration, weights):
    raw = authoring_from_legacy(draft())['shots'][-1]
    for unit, weight in zip(raw['action_plan'], weights):
        unit['weight'] = weight
    slot = {**plan_for()['slots'][-1], 'duration': duration, 'min_interaction_seconds': duration * .7}
    beats = compile_action_units(AuthoredShot.model_validate(raw), slot)
    shot = {'shot_id': 3, 'duration': duration, 'production_slot': slot, 'action_beats': beats}
    assert not action_timing_issues(shot)
    assert beats[-1]['end_seconds'] == duration
    assert beats[-1]['end_seconds'] - beats[-1]['start_seconds'] >= .5


def test_cross_shot_carry_and_linked_overlap_have_one_unambiguous_scope():
    raw = authoring_from_legacy(draft())
    units = raw['shots'][0]['action_plan']
    units.pop(0)
    units[0]['carry'] = 'continues the preceding shot momentum'
    units[1].update(carry='continues the same rotation', overlap=True)
    compiled = compile_authored_storyboard(raw, plan_for())
    shot = compiled['shots'][0]
    assert shot['start_state']['open_motion'] == units[0]['carry']
    assert 'flow_from_previous' not in shot['action_beats'][0]
    assert shot['action_beats'][1]['start_seconds'] < shot['action_beats'][0]['end_seconds']
    assert not action_timing_issues({**shot, 'production_slot': plan_for()['slots'][0]} )


@pytest.mark.parametrize('change', [
    lambda s: s['action_plan'][1].update(phase='aftermath'),
    lambda s: s['action_plan'][1].update(start_seconds=0),
    lambda s: s['action_plan'][1].update(interaction_geometry={'actor': 'other'}),
    lambda s: s['action_plan'][0].update(effect={'mode': 'direct_contact'}),
    lambda s: s['action_plan'][1].update(weight=float('nan')),
    lambda s: s['action_plan'][1].update(weight=True),
    lambda s: s['action_plan'][2].update(id=s['action_plan'][1]['id']),
    lambda s: s['action_plan'][1].update(actor='unregistered'),
    lambda s: s['action_plan'][0].update(actor='attacker and defender'),
    lambda s: s['action_plan'].pop(2),
    lambda s: s['action_plan'][1].update(overlap=True),
])
def test_ambiguous_or_infeasible_authoring_stops_without_rewriting(change, monkeypatch):
    raw = authoring_from_legacy(draft())
    change(raw['shots'][0])
    client = Responses(intent_reply('continuous action'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    with pytest.raises(ValueError, match='创作稿'):
        generate_storyboard('continuous action', target_duration=15)
    assert len(client.calls) == 3  # metadata + first invalid slot; no later slot or retry


RECORDS = json.loads((Path(__file__).parent / 'fixtures/real_planning_failures.json').read_text())['records']
REPLAY_EXPECTED = [
    'conflicting event ownership', 'conflicting event ownership',
    'effect.motion', 'compiled', 'conflicting event ownership', 'compiled',
]


def test_latest_real_missing_shot_and_joined_actor_are_rejected_without_mutation():
    record = json.loads((Path(__file__).parent / 'fixtures/real_authoring_failure.json').read_text())
    raw = record['draft']
    original = deepcopy(raw)
    plan = plan_for()
    assert len(raw['shots']) == 2 and len(plan['slots']) == 3
    with pytest.raises(ValueError, match='镜头数'):
        compile_authored_storyboard(raw, plan)
    # Isolate the first raw slot, without correcting its combined actor identifier.
    with pytest.raises(ValueError, match='未注册的动作参与者'):
        compile_authored_storyboard({**raw, 'shots': raw['shots'][:1]}, {**plan, 'slots': plan['slots'][:1]})
    assert raw == original


def test_single_slot_response_schema_matches_the_requested_envelope(monkeypatch):
    client = Responses(intent_reply('continuous action'), authoring_from_legacy(draft()))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    generate_storyboard('continuous action', target_duration=15)
    schema = json.loads(client.calls[2]['messages'][0]['content'].split('\n')[-1])
    assert schema['required'] == ['shot'] and not schema['additionalProperties']
    assert 'shot_id' not in schema['properties']['shot']['properties']
    assert '$defs' in schema and '$defs' not in schema['properties']['shot']
    assert all(v['properties']['actor']['enum'] == ['attacker', 'defender']
               for v in schema['$defs']['InteractionUnit']['oneOf'])


@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['source'])
def test_real_raw_failure_replay_preserves_meaning_or_rejects_ambiguity(record):
    raw = record['draft']
    original = deepcopy(raw)
    expected = REPLAY_EXPECTED[RECORDS.index(record)]
    try:
        authored = authoring_from_legacy(raw)
        compiled = compile_authored_storyboard(authored, plan_for())
    except ValueError as exc:
        assert expected != 'compiled' and expected in str(exc)
    else:
        assert expected == 'compiled'
        persisted = _compile_storyboard_contract(compiled, '16:9', '720p', 'cinematic', production_plan=plan_for())
        assert not storyboard_readiness_issues(persisted)
        assert all(not action_timing_issues({**shot, 'production_slot': slot})
                   for shot, slot in zip(compiled['shots'], plan_for()['slots']))
        for source, shot in zip(raw['shots'], compiled['shots']):
            for before, after in zip(source['action_beats'], shot['action_beats']):
                assert before['action'] == after['action'] and before['visible_result'] == after['visible_result']
                assert before['actor'] == after['actor'] and before.get('target', '') == after['target']
    assert raw == original


def test_long_event_authoring_keeps_one_spine_and_no_execution_schema_in_batches(monkeypatch):
    raw = authoring_from_legacy(draft())
    raw['shots'] += deepcopy(raw['shots'])
    for i, shot in enumerate(raw['shots'], 1):
        shot['shot_id'] = i
        extra = deepcopy(shot['action_plan'][2])
        extra.update(id='release_and_redirect', action='releases the wrist and redirects the returning palm',
                     result='the partner follows the redirection into a lower guard', response='deflection')
        shot['action_plan'].insert(3, extra)
    client = Responses(intent_reply('continuous action'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard('continuous action', target_duration=36)
    assert len(client.calls) == 8 and len(result['shots']) == 6
    assert sum(s['duration'] for s in result['shots']) == 36
    for call in client.calls[2:]:
        prompt = call['messages'][1]['content']
        assert 'effect_at_seconds' not in prompt and 'action_beats' not in prompt


def test_bad_json_never_triggers_a_repair_call_in_semantic_authoring(monkeypatch):
    from types import SimpleNamespace
    client = Responses(intent_reply('continuous action'))
    create = client.create
    def malformed(**kwargs):
        if not client.calls:
            return create(**kwargs)
        client.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{bad', refusal=None))])
    client.create = malformed
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    with pytest.raises(ValueError, match='未重新调用模型'):
        generate_storyboard('continuous action', target_duration=15)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_official_entry_archives_failed_semantic_input_without_video_or_rewrite(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    raw = authoring_from_legacy(draft())
    raw['shots'][0]['action_plan'][1]['effect']['source'] = ''
    client = Responses(intent_reply('15秒连续交互'), raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('unexpected video'))
    pipeline = VideoPipeline(paid_take_budget=0)
    with pytest.raises(ValueError, match='创作稿未通过编译'):
        await pipeline.run('15秒连续交互')
    assert len(client.calls) == 3
    # 创建边界在汇总稿形成前拒绝；原始响应仍完整留存，不能把缺失汇总文件当成丢稿。
    saved = json.loads((pipeline.workspace / 'planning_shot_001.json').read_text())
    response = json.loads((pipeline.workspace / 'planning_shot_001_response.json').read_text())
    assert {'shot_id': 1, **saved['shot']} == raw['shots'][0]
    assert json.loads(response['content']) == saved
    assert json.loads((pipeline.workspace / 'planning_check.json').read_text())['critical']
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_official_authoring_to_real_export_and_resume(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.generator import ShotResult
    from tools import ffmpeg_ops
    from test_director_export import clip
    from test_planning_semantics import verdict_for
    media = tmp_path / 'synthetic.mp4'
    clip(media)
    client = Responses(intent_reply('15秒连续交互'), authoring_from_legacy(draft()), verdict_for(draft()))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    class Generated:
        def __init__(self, *a, on_progress=None, **kw):
            self.progress = on_progress
        async def generate_all(self, storyboard):
            assert not storyboard_readiness_issues(storyboard)
            results = []
            for shot in storyboard['shots']:
                result = ShotResult(shot_id=shot['shot_id'], status='success', local_path=str(media))
                self.progress(result)
                results.append(result)
            return results
    async def no_voice(*a):
        return []
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no_luts')
    monkeypatch.setitem(config.SEEDANCE_OUTPUT_DIMENSIONS['480p'], '16:9', '320:180')
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', Generated)
    monkeypatch.setattr('pipeline.orchestrator.synthesize_voiceover', no_voice)
    monkeypatch.setattr(VideoPipeline, '_auto_select_music', lambda *a: None)
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.export_for_platform', lambda src, p, dst: shutil.copyfile(src, dst))
    validate = ffmpeg_ops.validate_publish_ready
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.validate_publish_ready', lambda *a, platform=None, **kw: validate(*a, **kw))
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=0)
    output = await pipeline.run('15秒连续交互')
    info = ffmpeg_ops.get_video_info(output)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    assert int(stream['nb_frames']) == 360 and float(stream['duration']) == pytest.approx(15)
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert len(client.calls) == 6
    assert (pipeline.workspace / 'planning_authoring.json').is_file()
    assert (pipeline.workspace / 'planning_compiled.json').is_file()
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)

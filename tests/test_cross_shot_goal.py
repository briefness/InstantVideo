"""跨镜目标使用原始审查重放；不修原稿、不把合成观察当成真实视觉通过。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline import goal_allocation as allocation
from pipeline.completion_intent import source_goals, validate_goal_coverage
from pipeline.planning_review_protocol import review_targets
from pipeline.result_admission import apply_result_admission
from pipeline.causality import compile_action_contract
from test_goal_allocation import offline


def capture():
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_cross_shot_goal.json').read_text())
    for record in fixture['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    return fixture, json.loads(fixture['files']['storyboard.json']['text']), json.loads(
        fixture['files']['planning_semantic_review.json']['text'])


def admitted(story, judgments):
    targets = review_targets(story)
    units = [{'shot_id': t['shot_id'], 'beat_index': t['beat_index']} for t in targets.values()]
    raw = allocation.decode_allocation({'units': units, 'goal_allocation': judgments},
        source_goals(story), targets, storyboard=story)
    return apply_result_admission(story, raw['units'])


def test_original_review_cross_shot_goal_reaches_bounded_execution_without_rewriting():
    _, story, review = capture()
    original = deepcopy((story, review))
    bound = admitted(story, review['verdict']['goal_allocation'])
    validate_goal_coverage(bound)
    quote = source_goals(story)['/intent/completion_goals/1']['goal']
    contracts = [compile_action_contract(s) for s in bound['shots']]
    for i, local in [(0, [2]), (1, [1])]:
        criteria = contracts[i].review_projection['completion_contract']['criteria']
        item = next(c for c in criteria.values() if c.get('action') == quote)
        assert item['beat_ids'] == local
        assert item['execution_scope']['shot_id'] == i + 1
        assert [u['shot_id'] for u in item['execution_scope']['units']] == [1, 2]
        assert 'only this shot' in '\n'.join(contracts[i].prompt_parts)
    assert (story, review) == original


def test_old_protocol_still_rejects_original_cross_shot_record():
    _, story, review = capture()
    targets = review_targets(story)
    units = [{'shot_id': t['shot_id'], 'beat_index': t['beat_index']} for t in targets.values()]
    with pytest.raises(ValueError, match='同一镜'):
        allocation.decode_allocation({'units': units, 'goal_allocation': review['verdict']['goal_allocation']},
            source_goals(story), targets)


@pytest.mark.parametrize('mutation', ['missing_phase', 'changed_scope', 'changed_source', 'reversed', 'unknown', 'duplicate'])
def test_cross_shot_integrity_cannot_drop_or_invent_a_phase(mutation):
    _, story, review = capture()
    judgments = deepcopy(review['verdict']['goal_allocation'])
    part = judgments['/intent/completion_goals/1']['parts'][0]
    if mutation == 'reversed':
        part['units'].reverse()
    elif mutation == 'unknown':
        part['units'].append('900:0')
    elif mutation == 'duplicate':
        part['units'].append('2:0')
    if mutation in ('reversed', 'unknown', 'duplicate'):
        with pytest.raises(ValueError):
            admitted(story, judgments)
        return
    bound = admitted(story, judgments)
    if mutation == 'missing_phase':
        bound['shots'][0]['action_beats'][1]['completion_intent']['requirements'] = []
    elif mutation == 'changed_scope':
        bound['shots'][1]['action_beats'][0]['completion_intent']['requirements'][0]['scope'][0]['action'] = 'invented'
    else:
        bound['shots'][0]['action_beats'][1]['action'] = 'invented'
    with pytest.raises(ValueError, match='范围|来源|承接'):
        validate_goal_coverage(bound)


@pytest.mark.parametrize('goal,phases,optional', [
    ('Courier hands over a parcel and receiver signs the receipt.',
     ['Courier hands the parcel to receiver.', 'Receiver signs the receipt.'], 'Courier adjusts a hat.'),
    ('Dancers exchange places and finish together.',
     ['Dancers exchange places.', 'Dancers finish side by side.'], 'A dancer smiles.'),
    ('Technician disconnects power then replaces the fuse.',
     ['Technician disconnects power.', 'Technician replaces the fuse.'], 'Technician wipes a sleeve.'),
    ('The rotating lamp reveals its profile and then its translucent shade.',
     ['The lamp rotates to show its profile.', 'The translucent shade glows visibly.'], 'A reflection moves.'),
])
def test_different_scenes_share_scopes_without_promoting_incidental_performance(goal, phases, optional):
    from pipeline.result_observations import completion_contract
    from test_goal_allocation import evidence
    story = {'production_plan': {'director_intent': {'authoring_protocol': 'scene-units-v1',
        'source_request': goal, 'completion_goals': [{'goal': goal, 'basis': 'user', 'evidence': goal}]}},
        'shots': [{'shot_id': sid, 'result_admission': 'review-required-v1',
            'production_slot': {'choreography': 'flexible'}, 'action_beats': [
                {'actor': 'subject', 'activity': 'transition', 'action': action, 'visible_result': action},
                {'actor': 'subject', 'activity': 'hold', 'action': optional, 'visible_result': optional}]}
            for sid, action in zip((8, 3), phases)]}
    ref = '/intent/completion_goals/0'
    bound = admitted(story, {ref: {'parts': [evidence(goal, '8:0', '3:0')]}})
    for shot in bound['shots']:
        criteria = completion_contract(shot)['criteria']
        function = next(c for c in criteria.values() if c['kind'] == 'completion')
        assert function['action'] == goal
        assert function['beat_ids'] == [1]
        assert function['execution_scope']['shot_id'] == shot['shot_id']
        assert [u['action'] for u in function['execution_scope']['units']] == phases
        assert criteria['beat:2/completion']['supports'] == []
    # 非相邻编号按时间线顺序；删除第一阶段不能靠最终结果仍有同一目标来遮蔽。
    bound['shots'][0]['action_beats'][0]['completion_intent']['requirements'] = []
    with pytest.raises(ValueError, match='承接'):
        validate_goal_coverage(bound)


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
@pytest.mark.parametrize('status', ['satisfied', 'violated', 'unknown'])
def test_joint_work_keeps_local_completion_causality_and_methods(topic, status):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_goal_allocation import evidence
    from test_goal_authority import annotated_snapshot
    from test_functional_results import format_for
    from pipeline.result_observations import decode_observations
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    _, _, raw, plan = scenario_story(topic)
    goal = '; '.join(u['action'] for s in raw['shots'] for u in s['action_plan'] if u['kind'] == 'interaction')
    plan['director_intent'].update(authoring_protocol='scene-units-v1',
        source_request=plan['director_intent']['source_request'] + goal,
        completion_goals=[{'goal': goal, 'basis': 'user', 'evidence': goal}])
    story = compile_story(raw, plan)
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    keys = [f"{s['shot_id']}:{i}" for s in story['shots'] for i, b in enumerate(s['action_beats'])
            if b['activity'] == 'interaction']
    bound = admitted(story, {'/intent/completion_goals/0': {'parts': [evidence(goal, *keys)]}})
    for shot in bound['shots']:
        contract = compile_action_contract(shot)
        catalog = contract.review_projection['completion_contract']
        assert any(c.get('action') == goal for c in catalog['criteria'].values())
        schema = format_for(shot)['json_schema']['schema']
        value = annotated_snapshot(shot)
        for key in value['observations']:
            c = catalog['criteria'][key]
            value['observations'][key]['status'] = status if c['kind'] == 'completion' else 'violated'
        if status == 'unknown':
            with pytest.raises(ValueError, match='观察不足'):
                decode_observations(value, catalog, schema, 25)
        else:
            result = decode_observations(value, catalog, schema, 25)
            assert result['accepted'] is (status == 'satisfied')
            assert all(e['result_visible'] is (status == 'satisfied') for e in result['action_event_evidence'])
        for prompt in (compile_normal_provider_prompt(shot, bound, has_observed_start=True),
                       compile_policy_safe_prompt(shot, storyboard=bound, has_state_reference=True,
                                                  image_role='reference_image', reference_count=1)):
            # 原始功能目标保留为上下文，本镜范围必须单独限定；不再用局部编舞替换目标。
            assert f'Joint goal context: {goal}' in prompt
            assert "Execute only this shot's contribution in beats" in prompt
            assert all(b['action'] in prompt for b in shot['action_beats'] if b['activity'] == 'interaction')
            local_actions = {b['action'] for b in shot['action_beats']}
            executable_prompt = prompt.replace(goal, '')
            assert all(b['action'] not in executable_prompt for other in bound['shots']
                       if other['shot_id'] != shot['shot_id'] for b in other['action_beats']
                       if b['activity'] == 'interaction' and b['action'] not in local_actions)


def test_real_legacy_cache_stays_failed_without_rewriting_or_retry(tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from pipeline.run_state import RunWorkspace
    _, story, review = capture()
    RunWorkspace._atomic_write_json(tmp_path / 'planning_semantic_review.json', review)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: pytest.fail('no retry allowed'))
    for _ in range(2):
        with pytest.raises(ValueError, match='同一镜'):
            ensure_planning_semantics(story, tmp_path, lambda *_: pytest.fail('do not rewrite source'))
    assert json.loads((tmp_path / 'planning_semantic_review.json').read_text()) == review


def test_continuation_keeps_the_same_scope_and_actual_source_actions(tmp_path):
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    from test_director_intent import Responses
    _, story, review = capture()
    shot = admitted(story, review['verdict']['goal_allocation'])['shots'][1]
    _, previous, tail = scenario(tmp_path)
    previous['scene_id'] = shot['scene_id']
    sdk = Responses({'compatible': True, 'entry_action': 'Continue from actual state.',
                     'reason': 'offline annotation'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    context = json.loads(sdk.calls[0]['messages'][-1]['content'])
    assert 'only this shot' in context['completion_instruction']
    assert 'only this shot' in context['completion_scope_instruction']
    assert 'only this shot' in context['planned_shot']['action_beats'][0]['action']
    assert result['action_beats'] == shot['action_beats']
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk) == result
    assert len(sdk.calls) == 1


def test_repeated_function_retains_distinct_joint_execution_groups():
    from test_goal_allocation import evidence
    from pipeline.result_observations import completion_contract
    goal = 'Partners repeatedly exchange places and finish side by side.'
    story = {'production_plan': {'director_intent': {'authoring_protocol': 'scene-units-v1',
        'source_request': goal, 'completion_goals': [{'goal': goal, 'basis': 'user', 'evidence': goal}]}},
        'shots': [{'shot_id': i, 'result_admission': 'review-required-v1',
            'production_slot': {'choreography': 'flexible'}, 'action_beats': [
                {'actor': 'partners', 'activity': 'transition', 'action': action, 'visible_result': action}]}
            for i, action in enumerate(['Partners exchange places.', 'Partners finish side by side.'] * 2, 1)]}
    ref = '/intent/completion_goals/0'
    bound = admitted(story, {ref: {'parts': [evidence(goal, '1:0', '2:0'), evidence(goal, '3:0', '4:0')]}})
    keys = [next(k for k, c in completion_contract(s)['criteria'].items() if c['kind'] == 'completion')
            for s in bound['shots']]
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
    bound['shots'][0]['action_beats'][0]['completion_intent']['requirements'] = []
    with pytest.raises(ValueError, match='承接'):
        validate_goal_coverage(bound)

"""生成端空间与动作装配：真实失败不改判，所有服务调用禁止。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_goal_allocation import offline
from test_staging_direction import compile_story
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_generation_preview


def captured():
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_generation_spatial_failure.json').read_text())
    records = {}
    for path, item in fixture['files'].items():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
        records[path] = json.loads(item['text'])
    return fixture, records


def recompiled():
    from pipeline.result_admission import apply_result_admission
    _, records = captured()
    previous = records['result_admission.json']['execution_storyboard']
    story = compile_story(records['planning_authoring.json'], previous['production_plan'])
    return records, apply_result_admission(story, records['result_admission.json']['units'])


def test_real_request_reproduces_exactly_and_new_formal_compilation_separates_execution_from_review():
    from pipeline.causality import compile_action_contract
    records, story = recompiled()
    old = records['result_admission.json']['execution_storyboard']
    actual = records['submission_requests/shot_001_attempt_001.json']['prompt']
    assert compile_normal_provider_prompt(old['shots'][0], old, has_observed_start=False) == actual
    assert len(actual) == 9267 and 'event_order' in actual
    prompt = compile_normal_provider_prompt(story['shots'][0], story, has_observed_start=False)
    assert 'event_order' not in prompt and 'after_ends' not in prompt
    for before, after in zip(old['shots'], story['shots'], strict=True):
        assert before['action_beats'] == after['action_beats']
        previous = compile_action_contract(before).review_projection
        current = deepcopy(compile_action_contract(after).review_projection)
        assert current['completion_contract'].pop('evidence_policy') == 'grounded-requirements-v2'
        assert current['completion_contract'].pop('execution_semantics') == 'intent-bounded-v1'
        if 'performance_policy' in current:
            assert current['performance_policy'].pop('time_basis') == 'sample-bounds-v1'
        assert previous == current  # 新协议改变证据表达，不改变原验收目标。
    review = next(v for k, v in records.items() if k.startswith('semantic_reviews/'))
    assert review['review']['accepted'] is False


def test_real_action_has_its_own_spatial_dependencies_and_unchanged_timeline():
    from pipeline.generation_instructions import shooting_projection
    _, story = recompiled()
    shot = story['shots'][0]
    snapshot = deepcopy(story)
    projection = shooting_projection(shot, story)
    action = projection['actions'][2]
    assert action['actor'] == 'LinAyi' and action['target'] == 'warm_water_glass'
    assert set(action['spatial_anchors']) >= {'LinAyi', 'warm_water_glass', 'empty_dining_chair', 'wooden_dining_table'}
    chair = action['spatial_anchors']['empty_dining_chair']
    assert chair['placement'] == shot['camera']['screen_positions']['empty_dining_chair']
    assert 'interaction_geometry/effect_region' in chair['sources']
    assert action['start_seconds'] == 3.5 and action['end_seconds'] == 6
    assert projection['actions'][1]['temporal_relation'] == 'overlap_previous'
    assert action['action'] == shot['action_beats'][2]['action']
    assert story == snapshot
    preview = compile_generation_preview(story, model='doubao-seedance-2.0-fast')
    assert preview['shots'][0]['execution_projection'] == projection
    prompt = preview['shots'][0]['profiles']['normal']['prompt']
    section = prompt.split('Beat 3:', 1)[1]
    assert chair['placement'] in section
    assert 'effect_motion=static' not in prompt


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_general_scenes_keep_all_actions_local_space_timing_and_request_preview(topic):
    from test_creation_schedule_flow import scenario_story
    from pipeline.generation_instructions import shooting_projection
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    before = deepcopy(story)
    for shot in story['shots']:
        projection = shooting_projection(shot, story)
        prompt = compile_normal_provider_prompt(shot, story, has_observed_start=False)
        for beat, action in zip(shot['action_beats'], projection['actions'], strict=True):
            for key in ('actor', 'target', 'action', 'start_seconds', 'end_seconds'):
                assert action[key] == beat[key]
            assert beat['action'] in prompt
            if beat.get('interaction_geometry'):
                assert beat['interaction_geometry']['effect_region'] in prompt
        assert 'event_order' not in prompt and 'after_ends' not in prompt
    assert story == before


def test_spatial_resolution_uses_exact_registered_ids_without_inventing_visibility_or_targets():
    from pipeline.generation_instructions import shooting_projection
    shot = {
        'shot_id': 1, 'duration': 8, 'characters': ['A'],
        'production_slot': {'authoring_protocol': 'scene-units-v1'},
        'key_props': ['Panel', 'Panel_Rail', 'Lamp', 'Unused'],
        'required_visible_entities': ['A'],
        'camera': {'screen_positions': {'A': 'left', 'Panel_Rail': 'back right', 'Unused': 'outside frame'}},
        'action_beats': [{'actor': 'A', 'target': 'Panel', 'action': 'Seat the panel in its rail.',
            'interaction_geometry': {'source': 'A的手', 'effect_region': 'Panel_Rail靠墙的一端；unknown_socket'},
            'start_seconds': 0, 'end_seconds': 8}],
    }
    before = deepcopy(shot)
    action = shooting_projection(shot, {'characters': [{'name': 'A'}]})['actions'][0]
    assert set(action['spatial_anchors']) == {'A', 'Panel', 'Panel_Rail'}
    assert action['spatial_anchors']['Panel']['sources'] == ['target']
    assert action['spatial_anchors']['A']['sources'] == ['actor', 'interaction_geometry/source']
    assert action['geometry']['effect_region'] == before['action_beats'][0]['interaction_geometry']['effect_region']
    assert shot == before  # 引用依赖不能升级为 required_visible_entities 或改写主体。
    shot['action_beats'][0].update(actor='Lamp', target='')
    shot['action_beats'][0]['interaction_geometry'] = {}
    independent = shooting_projection(shot, {})['actions'][0]
    assert independent['target'] == '' and set(independent['spatial_anchors']) == {'Lamp'}


@pytest.mark.parametrize('field', ['action', 'destination', 'camera', 'goal', 'playback', 'style'])
def test_final_submission_gate_rejects_lost_shooting_content(field):
    import config
    from pipeline.readiness import provider_submission_issues
    _, story = recompiled()
    shot = story['shots'][0]
    prompt = compile_normal_provider_prompt(shot, story, has_observed_start=False)
    values = {
        'action': shot['action_beats'][2]['action'],
        'destination': shot['action_beats'][2]['interaction_geometry']['effect_region'],
        'camera': 'positive lock - camera remains fixed throughout the shot',
        'goal': 'Required observable functions:',
        'playback': 'Present real-time motion, without slow motion',
        'style': 'Use only scene-native sound',
    }
    args = dict(prompt_profile='normal', model=config.SEEDANCE_MODEL, resolution=story['resolution'], duration=shot['duration'])
    assert provider_submission_issues(prompt, shot, story, **args) == []
    assert provider_submission_issues(prompt.replace(values[field], ''), shot, story, **args)


def test_explicit_final_outcome_and_result_only_spatial_anchor_reach_generation():
    from pipeline.generation_instructions import shooting_projection, shooting_instruction_parts
    _, story = recompiled()
    shot = deepcopy(story['shots'][-1])
    shot['production_slot']['required_outcome'] = 'The package has reached its registered receiver.'
    projection = shooting_projection(shot, story)
    assert projection['required_final_outcome'] == shot['production_slot']['required_outcome']
    assert projection['required_final_outcome'] in ' '.join(shooting_instruction_parts(projection))
    # 明确独立的合成空间输入：支撑物只在结果中出现，仍须与动作关联；不依赖场景词。
    minimal = {'shot_id': 1, 'duration': 8, 'key_props': ['Panel', 'Upper_Rail'],
        'camera': {'screen_positions': {'Upper_Rail': 'upper left'}},
        'action_beats': [{'actor': 'Operator', 'target': 'Panel', 'action': 'Install the component.',
            'outcomes': [{'entity': 'Panel', 'state': 'secured inside Upper_Rail'}],
            'start_seconds': 0, 'end_seconds': 8}]}
    action = shooting_projection(minimal, {})['actions'][0]
    assert action['spatial_anchors']['Upper_Rail'] == {
        'placement': 'upper left', 'sources': ['outcomes/0/state']}


def test_observed_start_reframes_without_replaying_planned_start_or_leaking_safe_details():
    from pipeline.provider_prompt import compile_policy_safe_prompt
    _, story = recompiled()
    shot = story['shots'][1]
    shot['observed_entry_path'] = 'UNIQUE_ENTRY_PATH: approach from the observed side of the table'
    shot['composition_change'] = 'large'
    normal = compile_normal_provider_prompt(shot, story, has_observed_start=True)
    safe = compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=True,
                                     image_role='reference_image', reference_count=1)
    assert 'observed physical and action state in Image 1' in normal
    assert 'unmistakably different coverage' in normal
    assert 'start exactly with' not in normal
    assert shot['observed_entry_path'] in normal and 'UNIQUE_ENTRY_PATH' not in safe


def test_actual_footage_and_original_observations_still_fail_after_generation_change(tmp_path):
    from types import SimpleNamespace
    from pipeline.semantic_review import SemanticTakeReviewer
    from test_full_production_flow import completion
    fixture, records = captured()
    _, story = recompiled()
    video = Path(__file__).parents[1] / fixture['source'] / 'shots/shot_001_rejected_1.mp4'
    if not video.exists():
        pytest.skip('local original video unavailable')
    assert hashlib.sha256(video.read_bytes()).hexdigest() == fixture['video_sha256']
    original = next(v for k, v in records.items() if k.startswith('semantic_reviews/'))['observation_response']
    calls = []
    def create(**request):
        import jsonschema
        jsonschema.validate(original, request['response_format']['json_schema']['schema'])
        calls.append(request)
        return completion(deepcopy(original))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-recorded-observation',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        result = reviewer.review(str(video), story['shots'][0], identity_crop_entities=['XiaoAn', 'LinAyi'])
        assert not result.accepted and not result.primary_action_completed
    assert len(calls) == 1  # 第二次只重放缓存，不重新观察或修补证据。

"""真实创作失败与通用动作词汇边界；服务替身不代表模型语义准确率。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from pipeline.planning_review import ensure_planning_semantics
from test_ordinary_scene_contract import offline


@pytest.mark.parametrize('actor,target,action,result', [
    ('attacker', 'defender', 'Looks at the returning colleague and smiles.', 'The observer visibly recognizes the colleague.'),
    ('attacker', '', 'Practices a sequence of footwork alone.', 'The performer finishes facing the doorway.'),
    ('Door', '', 'Rotates slowly on its own motor.', 'The panel faces the room.'),
])
def test_self_directed_performance_reaches_execution_without_fabricated_reaction(actor, target, action, result):
    import jsonschema
    from pipeline.action_authoring import authored_shot_response_schema, AuthoredShot, compile_action_units
    from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
    from pipeline.action_timing import action_timing_issues
    from test_authoring_schedule import raw_shot, slot
    current = {**slot(0), 'authoring_protocol': 'scene-units-v1'}
    unit = {'kind': 'performance', 'actor': actor, 'target': target,
            'action': action, 'result': result, 'weight': 1.}
    raw = raw_shot({'first': unit})
    before = deepcopy(raw)
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
        scene_entities=['Door', 'Dust'], bound_handoffs=True, allow_scene_subjects=True)
    jsonschema.validate({'shot': {k:v for k,v in raw.items() if k != 'shot_id'}},
                        scheduled_shot_schema(base, current))
    bound = bind_authored_schedule(raw, current)
    beats = compile_action_units(AuthoredShot.model_validate(bound), current)
    beat = beats[0]
    assert beat['action'] == action and beat['visible_result'] == result
    assert beat['activity'] == 'performance'
    assert not {'response_kind', 'interaction_geometry', 'effect_at_seconds', 'reaction_at_seconds'} & beat.keys()
    assert not action_timing_issues({'shot_id': 1, 'duration': 5, 'production_slot': current, 'action_beats': beats})
    assert raw == before


def test_controlled_object_change_is_not_forced_to_use_collision_response():
    import jsonschema
    from pipeline.action_authoring import AuthoredShot, compile_action_units
    from pipeline.authoring_schedule import bind_authored_schedule
    from pipeline.action_timing import action_timing_issues, sequence_events
    from pipeline.causality import event_execution_instruction
    from test_authoring_schedule import raw_shot, slot, interaction, schema_for
    current = {**slot(0), 'authoring_protocol': 'scene-units-v1'}
    unit = {**interaction(), 'kind': 'interaction', 'target': 'Door', 'response': 'state_change',
            'action': 'Turns the handle and opens the door.', 'result': 'The door is open.',
            'effect': {'mode': 'direct_contact', 'motion': 'sweep', 'source': 'hand',
                       'region': 'door handle', 'unaffected': 'rest of the room'},
            'resolution': {'states': {'Door': [{'state': 'open'}]}}}
    raw = raw_shot({'first': unit})
    jsonschema.validate({'shot': {k:v for k,v in raw.items() if k != 'shot_id'}}, schema_for(current))
    beats = compile_action_units(AuthoredShot.model_validate(bind_authored_schedule(raw,current)), current)
    shot = {'shot_id': 1, 'duration': 5, 'production_slot': current, 'action_beats': beats}
    assert not action_timing_issues(shot)
    event = sequence_events(shot)[0]
    assert event['response_kind'] == 'state_change'
    assert 'state_change' in event_execution_instruction(event)
    assert event['outcomes'][0]['entity'] == 'Door'


def test_real_nonphysical_action_misclassified_as_impact_stays_reproducible(tmp_path, monkeypatch):
    root = Path(__file__).parents[1] / 'output/20261104_133329_671965'
    if not root.is_dir():
        pytest.skip('captured local run unavailable')
    source = (root / 'storyboard.json').read_bytes()
    assert hashlib.sha256(source).hexdigest() == '9524613fdef44dd0ea91c74f5b246ce582f516184edf1cd4baa44d63b94fe049'
    story = json.loads(source)
    raw = json.loads((root / 'planning_shot_001_response.json').read_text())
    authored = json.loads(raw['content'])['shot']['action_plan']['following'][0]
    assert authored['kind'] == 'interaction' and authored['response'] == 'impact'
    assert story['shots'][0]['action_beats'][1]['response_kind'] == authored['response']
    shutil.copyfile(root / 'planning_semantic_review.json', tmp_path / 'planning_semantic_review.json')
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('must replay recorded verdict'))
    before = deepcopy(story)
    with pytest.raises(ValueError, match='Shot 1 动作 2'):
        ensure_planning_semantics(story, tmp_path, lambda *args: None)
    assert story == before and (root / 'storyboard.json').read_bytes() == source


@pytest.mark.parametrize('flexible', [False, True])
def test_state_change_meaning_survives_both_prompt_profiles_review_and_cache_identity(flexible):
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.action_vocabulary import STATE_CHANGE_EXECUTION
    from test_scene_target_flow import object_story
    from test_action_authoring import plan_for
    source = object_story('Panel', 'presses the control switch', 'indicator lit')
    plan = plan_for()
    for slot in plan['slots']:
        slot['choreography'] = 'flexible' if flexible else 'exact'
    for shot in source['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.pop('resolution')
                unit['outcomes'] = [{'entity': 'Panel', 'state': 'indicator lit'}]
    baseline = _compile_storyboard_contract(compile_authored_storyboard(source, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    for shot in source['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['response'] = 'state_change'
    story = _compile_storyboard_contract(compile_authored_storyboard(source, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    for shot, old in zip(story['shots'], baseline['shots']):
        contract = compile_action_contract(shot)
        assert bool(contract.review_projection.get('performance_policy')) == flexible
        assert contract.fingerprint() != compile_action_contract(old).fingerprint()
        assert STATE_CHANGE_EXECUTION in contract.event_review_instruction()
        for prompt in (compile_normal_provider_prompt(shot, story, has_observed_start=False),
                       compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                                  image_role=None, reference_count=0)):
            assert STATE_CHANGE_EXECUTION in prompt
            assert 'Panel' in prompt and 'indicator lit' in prompt


@pytest.mark.parametrize('extra', [
    {'response': 'impact'}, {'effect': {'mode': 'direct_contact'}},
    {'resolution': {'states': {'defender': [{'state': 'hurt'}]}}},
])
def test_performance_cannot_carry_hidden_external_effects(extra):
    from pydantic import ValidationError
    from pipeline.action_authoring import AuthoredShot
    from pipeline.authoring_schedule import bind_authored_schedule
    from test_authoring_schedule import raw_shot, slot
    current = {**slot(0), 'authoring_protocol': 'scene-units-v1'}
    unit = {'kind': 'performance', 'actor': 'attacker', 'target': 'defender',
            'action': 'Looks toward the colleague.', 'result': 'Smiles in recognition.',
            'weight': 1., **extra}
    with pytest.raises((ValidationError, ValueError)):
        AuthoredShot.model_validate(bind_authored_schedule(raw_shot({'first': unit}), current))

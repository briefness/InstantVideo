"""Shared event-contract regressions, independent of a particular story or weapon."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.models import validate_storyboard
from pipeline.readiness import storyboard_readiness_issues
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.visual_invariants import compile_visual_invariants


def sequence_story():
    story = json.loads((Path(__file__).resolve().parents[1] / 'examples/contact_previs/storyboard.json').read_text())
    shot = story['shots'][0]
    shot['duration'] = 5
    geometry = deepcopy(shot['interaction_geometry'])
    shot['action_beats'] = [
        {'phase': 'peak', 'actor': 'attacker', 'target': 'defender',
         'action': 'presses the guard aside', 'visible_result': 'defender moves back',
         'start_seconds': 0., 'end_seconds': 2.5, 'activity': 'interaction',
         'effect_at_seconds': 1., 'reaction_at_seconds': 1.2,
         'interaction_geometry': geometry},
        {'phase': 'peak', 'actor': 'defender', 'target': 'attacker',
         'action': 'returns the pressure', 'visible_result': 'attacker steps back',
         'start_seconds': 2.5, 'end_seconds': 4.5, 'activity': 'interaction',
         'effect_at_seconds': 3., 'reaction_at_seconds': 3.2,
         'interaction_geometry': {**geometry, 'actor': 'defender', 'target': 'attacker',
                                  'source': 'defender lead palm', 'effect_region': 'attacker guard',
                                  'reaction_scope': 'attacker only'}},
        {'phase': 'aftermath', 'actor': 'defender', 'target': 'attacker',
         'action': 'settles', 'visible_result': 'attacker remains back',
         'start_seconds': 4.5, 'end_seconds': 5., 'activity': 'hold'}]
    return story


def test_explicit_sequence_survives_schema_and_readiness_without_single_actor_repair():
    story = validate_storyboard(sequence_story())
    assert not storyboard_readiness_issues(story)


def test_normal_safe_and_review_preserve_each_event_owner_and_result():
    story = sequence_story()
    shot = story['shots'][0]
    contract = compile_action_contract(shot)
    events = contract.review_projection['action_events']
    assert [(e['actor'], e['target']) for e in events] == [('attacker', 'defender'), ('defender', 'attacker')]
    for prompt in [compile_normal_provider_prompt(shot, story, has_observed_start=False),
                   compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                              image_role=None, reference_count=0)]:
        assert 'event 1' in prompt and 'event 2' in prompt
        assert '1.2s' in prompt and '3.2s' in prompt
        assert 'do not introduce another major action' not in prompt
    assert [e['visible_result'] for e in events] == ['defender moves back', 'attacker steps back']


@pytest.mark.parametrize('mutate', [
    lambda s: s['action_beats'][1].pop('interaction_geometry'),
    lambda s: s['action_beats'][1].update(reaction_at_seconds=2.9),
    lambda s: s['action_beats'][1].update(effect_at_seconds=5.),
    lambda s: s['action_beats'][1]['interaction_geometry'].update(actor='attacker'),
])
def test_partial_or_incoherent_sequence_stops_before_submission(mutate):
    story = sequence_story()
    mutate(story['shots'][0])
    assert storyboard_readiness_issues(story)


def test_fixed_camera_does_not_freeze_subject_orientation():
    shot = sequence_story()['shots'][0]
    compiled = compile_visual_invariants(shot)
    assert compiled.review_projection['body_orientation_scope'].startswith('initial')
    assert not any('preserve these body orientations throughout' in p for p in compiled.prompt_parts)


def test_earlier_event_scope_cannot_hide_behind_valid_final_geometry():
    story = sequence_story()
    earlier = story['shots'][0]['action_beats'][0]['interaction_geometry']
    earlier.update(interaction_mode='directed_path', outcome_scope='all', effect_motion='static')
    assert any('sweep' in issue for issue in storyboard_readiness_issues(story))


def test_earlier_event_participant_must_be_visible():
    story = sequence_story()
    earlier = story['shots'][0]['action_beats'][0]
    earlier['target'] = 'receiver'
    earlier['interaction_geometry']['target'] = 'receiver'
    assert any('required_visible_entities' in issue for issue in storyboard_readiness_issues(story))


def test_one_long_contact_cannot_satisfy_multiple_event_requirement():
    from pipeline.action_timing import action_timing_issues
    shot = sequence_story()['shots'][0]
    shot['production_slot'] = {'min_interaction_events': 2, 'min_interaction_seconds': 3.5}
    shot['action_beats'].pop(1)
    shot['action_beats'][0]['end_seconds'] = 4.5
    assert any('数量1不足计划2' in issue for issue in action_timing_issues(shot))


def test_event_effect_scope_survives_next_shot_but_not_a_different_scene():
    from pipeline.causality import causal_storyboard_issues
    story = sequence_story()
    first = story['shots'][0]
    first['scene_id'] = 'workshop_a'
    after = {'shot_id': 2, 'scene_id': 'workshop_a', 'interaction_geometry': {
        **first['action_beats'][1]['interaction_geometry'], 'interaction_mode': 'none',
        'effect_phase': 'aftermath', 'effect_motion': 'none'}}
    story['shots'].append(after)
    assert not causal_storyboard_issues(story)
    after['scene_id'] = 'workshop_b'
    assert any('无原因扩大' in issue for issue in causal_storyboard_issues(story))


def test_v4_freezes_event_budget_while_v3_keeps_legacy_semantics():
    from pipeline.production_plan import build_production_plan, production_plan_issues, apply_production_plan
    from test_director_intent import draft, intent_reply
    request = '密集交互'
    intent = {**intent_reply(request), 'source_request': request}
    old = build_production_plan('action', 15, intent=intent, contract_version='production-plan-v3-intent-timing')
    new = build_production_plan('action', 15, intent=intent)
    assert all(slot.get('min_interaction_events', 0) == 0 for slot in old['slots'])
    assert all(slot['min_interaction_events'] == 2 for slot in new['slots'])
    board = draft()
    apply_production_plan(board, old)
    assert not production_plan_issues(board)
    apply_production_plan(board, new)
    assert not production_plan_issues(board)
    board['production_plan']['slots'][0]['min_interaction_events'] = 1
    assert production_plan_issues(board)

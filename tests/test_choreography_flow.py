"""Fluid action is an executable relation, not extra strikes or genre keywords."""
from copy import deepcopy
import json

import pytest

from pipeline.action_timing import action_timing_issues
from pipeline.causality import compile_action_contract
from pipeline.models import validate_storyboard
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.readiness import storyboard_readiness_issues
from pipeline.generator import VideoGenerator
from test_action_sequence import sequence_story
from test_sequence_review import offline_frames, event_response, reviewer_for  # noqa: F401


def flowing_story():
    story = sequence_story()
    first, second, _ = story['shots'][0]['action_beats']
    first.update(end_seconds=2.8, response_kind='evasion',
                 action='sweeps through the original guard line',
                 visible_result='defender slips beneath the sweep without being hit')
    first['interaction_geometry'].update(interaction_mode='directed_path',
        line_of_action_visible=True, effect_region='the vacated upper guard line')
    second.update(start_seconds=1.5, effect_at_seconds=2., reaction_at_seconds=2.2,
                  flow_from_previous='The defender turns the same slipping step into a rising counter, without resetting.')
    return story


def test_linked_overlapping_actions_survive_formal_schema_and_readiness():
    story = validate_storyboard(flowing_story())
    assert not storyboard_readiness_issues(story)


def test_flow_and_avoided_contact_survive_both_provider_profiles_and_fingerprint():
    story = flowing_story()
    shot = story['shots'][0]
    contract = compile_action_contract(shot)
    assert len(contract.review_projection['action_events']) == 2
    for prompt in (compile_normal_provider_prompt(shot, story, has_observed_start=False),
                   compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                               image_role=None, reference_count=0)):
        assert 'evasion' in prompt and 'without resetting' in prompt
        assert '1.5-4.5s' in prompt
    changed = deepcopy(shot)
    changed['action_beats'][1]['flow_from_previous'] = 'Continues the same retreating step into a turn.'
    assert VideoGenerator._compiled_contract_fingerprint(shot) != VideoGenerator._compiled_contract_fingerprint(changed)


def test_overlap_cannot_double_count_interaction_seconds():
    shot = flowing_story()['shots'][0]
    shot['production_slot'] = {'min_interaction_seconds': 4.8}
    assert any('有效交互时长4.5秒' in issue for issue in action_timing_issues(shot))


@pytest.mark.parametrize('change', [
    lambda b: b[1].pop('flow_from_previous'),
    lambda b: b[1].update(flow_from_previous='   '),
    lambda b: b[1].update(start_seconds=-.1),
    lambda b: b[1].update(end_seconds=2.7),
    lambda b: b[1].update(effect_at_seconds=.5),
    lambda b: b[0].update(flow_from_previous='unknown earlier event'),
    lambda b: b[0]['interaction_geometry'].update(interaction_mode='direct_contact'),
])
def test_unexplained_or_contradictory_overlap_fails_closed(change):
    story = flowing_story()
    change(story['shots'][0]['action_beats'])
    assert storyboard_readiness_issues(story)


@pytest.mark.parametrize('flow_valid', [True, False])
def test_review_requires_observed_flow_not_just_two_completed_events(tmp_path, flow_valid):
    response = event_response()
    for entry in response['action_event_evidence']:
        entry['flow_continuity_valid'] = flow_valid
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    review = reviewer.review(str(video), flowing_story()['shots'][0])
    assert review.accepted is flow_valid
    assert review.action_sequence_valid is flow_valid
    prompt = sdk.calls[0]['messages'][0]['content']
    assert 'evasion' in prompt and 'flow_continuity_valid' in prompt
    assert len(sdk.calls) == 1


def test_cached_success_cannot_hide_broken_flow(tmp_path):
    response = event_response()
    for entry in response['action_event_evidence']:
        entry['flow_continuity_valid'] = True
    shot = flowing_story()['shots'][0]
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert reviewer.review(str(video), shot).accepted
    cache = next((tmp_path / 'semantic_reviews').glob('*.json'))
    saved = json.loads(cache.read_text())
    saved['action_event_evidence'][1]['flow_continuity_valid'] = False
    cache.write_text(json.dumps(saved))
    assert not reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('response', ['evasion', 'deflection'])
def test_non_impact_result_is_explicit_and_does_not_add_strikes(response):
    story = flowing_story()
    shot = story['shots'][0]
    shot['action_beats'][0]['response_kind'] = response
    contract = compile_action_contract(shot)
    assert len(contract.review_projection['action_events']) == 2
    assert 'response_kind=' + response in ' '.join(contract.prompt_parts)
    assert not storyboard_readiness_issues(validate_storyboard(story))


@pytest.mark.parametrize('invalid', [None, 'true', 1])
def test_missing_or_non_boolean_flow_evidence_is_not_accepted(invalid):
    contract = compile_action_contract(flowing_story()['shots'][0])
    evidence = event_response()['action_event_evidence']
    for item in evidence:
        item['flow_continuity_valid'] = invalid
    assert contract.event_evidence_shape_issues(evidence)


def test_flow_links_cannot_smuggle_new_fields_or_nonphysical_events():
    from pydantic import ValidationError
    story = flowing_story()
    story['shots'][0]['action_beats'][1]['response_kind'] = 'magic_guess'
    with pytest.raises(ValidationError):
        validate_storyboard(story)
    story = flowing_story()
    story['shots'][0]['action_beats'][-1]['flow_from_previous'] = 'new strike in the hold'
    assert storyboard_readiness_issues(story)


def test_continuation_preserves_flow_and_response_kind(tmp_path):
    from pipeline.continuation import prepare_continuation
    from test_continuation import Planner
    shot = flowing_story()['shots'][0]
    shot['continuity_from_previous'] = 'seamless'
    tail = tmp_path / 'tail.jpg'
    tail.write_bytes(b'offline accepted tail')
    previous = {'scene_id': shot.get('scene_id', ''), 'observed_end_state': {'action_phase': 'approaching'}}
    changed = prepare_continuation(shot, previous, str(tail), tmp_path, client=Planner())
    assert changed['action_beats'] == shot['action_beats']


def test_formal_planner_preserves_fluid_exchange_without_extra_events(monkeypatch):
    from authoring_cases import authoring_from_legacy
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import draft, intent_reply, Responses
    request = '15秒武侠交锋，招式密集但要借势连贯，不能机械地一人一下'
    board = draft()
    for shot in board['shots']:
        first, second = shot['action_beats'][1:3]
        first.update(end_seconds=3., response_kind='evasion',
                     visible_result='defender slips beneath the attack without being hit')
        first['interaction_geometry'].update(interaction_mode='directed_path', line_of_action_visible=True)
        second.update(start_seconds=2.5, flow_from_previous='Continues the same slipping turn into the counter without resetting.')
    client = Responses(intent_reply(request), authoring_from_legacy(board))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard(request, target_duration=15)
    assert len(client.calls) == 5
    assert sum(shot['duration'] for shot in result['shots']) == 15
    for shot in result['shots']:
        events = compile_action_contract(shot).review_projection['action_events']
        assert len(events) == 2 and events[0]['response_kind'] == 'evasion'
        assert events[1]['flow_from_previous'].endswith('without resetting.')
    assert request in client.calls[1]['messages'][1]['content']


def test_continuous_attack_does_not_require_alternating_actors():
    story = flowing_story()
    first, second, _ = story['shots'][0]['action_beats']
    second.update(actor='attacker', target='defender', action='turns the same sweep into a descending press',
                  visible_result='defender yields lower', flow_from_previous='The attacker redirects the same sweep without resetting.')
    second['interaction_geometry'] = {**first['interaction_geometry'], 'interaction_mode': 'direct_contact'}
    assert not storyboard_readiness_issues(validate_storyboard(story))
    events = compile_action_contract(story['shots'][0]).review_projection['action_events']
    assert [event['actor'] for event in events] == ['attacker', 'attacker']


def test_flow_is_shared_with_noncombat_transfer_and_samples_its_boundary(tmp_path):
    from test_sequence_review import process_shot
    shot = process_shot('indirect_effect')
    first, second, _ = shot['action_beats']
    first['end_seconds'] = 2.8
    second.update(start_seconds=1.5, flow_from_previous='The receiver continues the same rotation into the handoff.')
    assert not action_timing_issues(shot)
    response = event_response()
    for entry in response['action_event_evidence']:
        entry['flow_continuity_valid'] = True
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert reviewer.review(str(video), shot).accepted
    labels = [item.get('text', '') for item in sdk.calls[0]['messages'][1]['content']]
    assert any('1.400s' in label for label in labels)
    assert any('1.600s' in label for label in labels)

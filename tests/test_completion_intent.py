"""Real captures and explicit counterfactual annotations; no fresh vision judgments."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.result_admission import apply_result_admission
from test_functional_results import format_for, annotated_wire, decode
from test_sequence_review import offline_frames


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_incidental_action_failure.json').read_text())
    sources = {}
    for name, item in record['sources'].items():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
        sources[name] = json.loads(item['text'])
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    original = sources['storyboard.json']
    plan = original['production_plan']
    story = _compile_storyboard_contract(compile_authored_storyboard(sources['planning_authoring.json'], plan),
        original['aspect_ratio'], original['resolution'], original['style'], production_plan=plan)
    return record, sources, story


def annotated_intents(story, units):
    """Strict legacy-test interpretation, not semantic extraction or production fallback."""
    units = deepcopy(units)
    for unit in units:
        i = next(i for i, s in enumerate(story['shots']) if s['shot_id'] == unit['shot_id'])
        beat = story['shots'][i]['action_beats'][unit['beat_index']]
        unit['completion_intent'] = {'kind': 'required', 'goal': beat['action'], 'basis': 'story',
            'reference': f'/shots/{i}/narrative_beat/state_change',
            'reason': 'Strict synthetic fixture: preserve every authored action as an obligation.'}
    return units


def admitted_capture():
    record, sources, story = captured()
    units = annotated_intents(story, sources['result_admission.json']['units'])
    first = [u for u in units if u['shot_id'] == story['shots'][0]['shot_id']]
    for unit, goal in zip(first, ['小安带行李回到餐厅门口，犹豫而未进入。',
                                '林阿姨注意到回家的小安。', '林阿姨将温水杯放在桌边空椅旁。']):
        unit['completion_intent'].update(goal=goal, basis='user', reference='/request/segments/1',
            reason='Offline annotation of the explicit original story requirement, not a new model response.')
    first[3]['completion_intent'] = {'kind': 'supporting', 'supports': 1,
        'reason': 'The authored look toward the cup is one realization of the still-hesitant arrival; no later event needs this gaze.'}
    return record, sources, apply_result_admission(story, units)


def test_captured_failure_is_action_authority_not_missing_core_event():
    _, sources, _ = captured()
    raw = next(v for k, v in sources.items() if '/responses/' in k)
    response = json.loads(raw['attempts'][0]['choices'][0]['content'])
    assert len(raw['attempts']) == 1
    assert [k for k, v in response['observations'].items() if v['status'] == 'violated'] == ['beat:4/completion']
    original = sources['result_admission.json']['execution_storyboard']['shots'][0]
    assert compile_action_contract(original).review_projection['completion_contract']['criteria']['beat:4/completion']['action'] == original['action_beats'][3]['action']


def test_admission_separates_function_from_realization_without_rewriting_actions():
    _, sources, story = admitted_capture()
    shot = story['shots'][0]
    assert shot['completion_policy'] == 'admitted-intent-v1'
    criteria = compile_action_contract(shot).review_projection['completion_contract']['criteria']
    assert criteria['beat:4/completion']['kind'] == 'realization'
    assert criteria['beat:4/completion']['supports'] == 'beat:1/completion'
    assert criteria['beat:1/completion']['action'] == '小安带行李回到餐厅门口，犹豫而未进入。'
    for new, old in zip(shot['action_beats'], sources['storyboard.json']['shots'][0]['action_beats']):
        assert new['action'] == old['action'] and new['visible_result'] == old['visible_result']


def test_real_observations_counterfactual_does_not_change_original_verdict():
    _, sources, story = admitted_capture()
    raw = next(v for k, v in sources.items() if '/responses/' in k)
    response = json.loads(raw['attempts'][0]['choices'][0]['content'])
    # Original statuses/text are retained; only the explicitly annotated admission differs.
    verdict = decode(story['shots'][0], response)
    assert verdict['primary_action_completed'] is True
    assert verdict['observed_end_state']['pose_and_gaze'] == response['observations']['beat:4/completion']['observation']
    original = sources['result_admission.json']['execution_storyboard']['shots'][0]
    assert decode(original, response)['primary_action_completed'] is False


@pytest.mark.parametrize('key', ['beat:1/completion', 'beat:2/completion', 'beat:3/completion', 'beat:3/outcomes/1'])
def test_missing_real_requirement_still_blocks(key):
    _, _, story = admitted_capture()
    shot = story['shots'][0]
    wire = annotated_wire(format_for(shot)['json_schema']['schema'])
    wire['observations'][key].update(status='violated', observation='Required event/state never occurs.')
    assert decode(shot, wire)['accepted'] is False


@pytest.mark.parametrize('supports', [0, 4, 5, True])
def test_support_cannot_erase_all_work_or_reference_itself(supports):
    _, sources, story = captured()
    units = annotated_intents(story, sources['result_admission.json']['units'])
    units[3]['completion_intent'] = {'kind': 'supporting', 'supports': supports, 'reason': 'invalid link'}
    with pytest.raises(ValueError):
        apply_result_admission(story, units)


@pytest.mark.parametrize('goal,detail', [
    ('The courier reaches the receiver with the parcel.', 'pauses with one hand on the hip'),
    ('The dancer completes a continuous turn.', 'finishes looking toward the ceiling'),
    ('The witness silently recognizes the suspect.', 'touches the collar with two fingers'),
    ('The lamp reveals its translucent material.', 'casts a tiny crescent on the tabletop'),
])
@pytest.mark.parametrize('explicit_detail', [False, True])
def test_same_detail_is_replaceable_or_required_by_source_not_keywords(goal, detail, explicit_detail):
    from test_ordinary_scene_contract import scene_case
    from test_staging_direction import compile_story
    from test_action_result_flow import annotated_verdict
    from pipeline.production_plan import build_production_plan
    from pipeline.models import DirectorIntentSpec
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.semantic_review import _review_contract

    _, reply, identity, _, shots = scene_case('family')
    # 此用例保留 v9 的原请求引用协议；新目标目录的对应边界见 test_goal_authority。
    reply.pop('completion_goals', None)
    request = goal + (' Also required: ' + detail if explicit_detail else '')
    reply.update(source_request=request, intent_evidence=request, duration_evidence=request, shot_count_evidence=request)
    intent = DirectorIntentSpec.model_validate(reply).model_dump()
    # These are test service inputs, never modifications to the captured production plan.
    intent['authoring_protocol'] = 'scene-units-v1'
    plan = build_production_plan(intent['content_focus'], 24, intent=intent)
    for shot in shots:
        shot['result_admission'] = 'review-required-v1'
        shot['action_plan'][0].update(action=goal, result=goal)
        shot['action_plan'][1].update(action=detail, result=detail)
        shot['story_change'] = goal
        shot['story_state_after'] = detail
    story = compile_story({**identity, 'shots': shots}, plan)
    units = annotated_intents(story, annotated_verdict(story)['units'])
    for unit in units:
        unit['completion_intent'].update(goal=goal, basis='user', reference='/request/segments/0')
        if unit['beat_index'] == 1:
            unit['completion_intent'] = ({**unit['completion_intent'], 'goal': detail} if explicit_detail else
                {'kind': 'supporting', 'supports': 1, 'reason': 'This is a replaceable realization of the first function.'})
    admitted = apply_result_admission(story, units)
    for shot in admitted['shots']:
        wire = annotated_wire(format_for(shot)['json_schema']['schema'])
        wire['observations']['beat:2/completion'].update(status='violated', observation='The small performance choice differs.')
        assert decode(shot, wire)['accepted'] is (not explicit_detail)
        wire['observations']['beat:1/completion'].update(status='violated', observation='The required event is only attempted.')
        assert decode(shot, wire)['accepted'] is False
        normal = compile_normal_provider_prompt(shot, admitted, has_observed_start=False)
        safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
        assert goal in normal and goal in safe and detail in normal
        review_text = json.dumps(_review_contract(shot), ensure_ascii=False)
        assert (detail in review_text) is explicit_detail


@pytest.mark.parametrize('fault', ['missing', 'forged_reference', 'unknown_basis', 'empty_goal', 'all_support', 'cycle'])
def test_incomplete_or_unbound_admission_cannot_be_frozen(fault):
    _, sources, story = captured()
    units = annotated_intents(story, sources['result_admission.json']['units'])
    if fault == 'missing':
        units[0].pop('completion_intent')
    elif fault == 'forged_reference':
        units[0]['completion_intent']['reference'] = '/shots/0/action_beats/0/action'
    elif fault == 'unknown_basis':
        units[0]['completion_intent']['basis'] = 'looks_important'
    elif fault == 'empty_goal':
        units[0]['completion_intent']['goal'] = ' '
    else:
        for unit in units[:4 if fault == 'all_support' else 2]:
            unit['completion_intent'] = {'kind': 'supporting', 'supports': 2 if unit['beat_index'] == 0 else 1, 'reason': 'invalid graph'}
    with pytest.raises(ValueError):
        apply_result_admission(story, units)


def test_formal_single_review_owns_intents_and_reuses_exact_record(tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from pipeline.run_state import RunWorkspace
    from test_director_intent import Responses
    from planning_protocol_cases import annotated_outcome
    _, sources, story = captured()
    _, _, annotated = admitted_capture()
    units = deepcopy(sources['result_admission.json']['units'])
    for unit in units:
        unit['completion_intent'] = annotated['shots'][unit['shot_id'] - 1]['action_beats'][unit['beat_index']]['completion_intent']
    sdk = Responses(annotated_outcome({'units': units, 'continuity_valid': True, 'requested_outcome_valid': True,
                     'reason': 'Offline annotated planning service; not a fresh semantic judgment.'}, story))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: sdk)
    save = lambda name, value: RunWorkspace._atomic_write_json(tmp_path / name, value)
    frozen = ensure_planning_semantics(story, tmp_path, save)
    assert ensure_planning_semantics(story, tmp_path, save) == frozen
    assert len(sdk.calls) == 1
    assert frozen['shots'][0]['action_beats'][3]['completion_intent']['kind'] == 'supporting'
    import jsonschema
    wire = json.loads((tmp_path / 'planning_semantic_review.json').read_text())['verdict']
    schema = sdk.calls[0]['response_format']['json_schema']['schema']
    jsonschema.validate(wire, schema)
    bad = deepcopy(wire)
    bad['units']['1:0'].pop('completion_intent')
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    assert all(b['action'] == old['action'] for s, source in zip(frozen['shots'], story['shots'])
               for b, old in zip(s['action_beats'], source['action_beats']))


def test_original_media_replay_uses_same_observation_for_handoff_without_new_calls(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer
    from types import SimpleNamespace
    record, sources, story = admitted_capture()
    video = Path(__file__).resolve().parents[1] / record['video']['path']
    if not video.is_file():
        pytest.skip('local original media unavailable; exact JSON replay remains covered')
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']
    raw = next(v for k, v in sources.items() if '/responses/' in k)['attempts'][0]['choices'][0]['content']
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(kwargs['messages'][-1]['content'], str):
            from functional_observation_cases import annotated_support_response
            from test_full_production_flow import completion
            return completion(annotated_support_response(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=raw, refusal=None))])
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-captured-counterfactual',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = reviewer.review(str(video), story['shots'][0])
    assert result.accepted
    assert reviewer.review(str(video), story['shots'][0]) == result
    assert len(calls) == 2
    assert '没有将视线移向' in result.observed_end_state['pose_and_gaze']
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']


def test_planned_pose_does_not_reenter_through_blocking_or_continuation(tmp_path):
    from pipeline.semantic_review import _review_contract
    from pipeline.visual_invariants import compile_visual_invariants
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    from test_director_intent import Responses
    _, _, story = admitted_capture()
    shot = story['shots'][0]
    contract = _review_contract(shot)
    assert 'pose_and_gaze' not in contract['start_state']
    assert all(set(value) == {'action_target'} for value in contract['blocking'].values())
    assert 'body_orientations' not in compile_visual_invariants(shot).review_projection
    _, previous, tail = scenario(tmp_path)
    previous['scene_id'] = shot['scene_id']
    shot['continuity_from_previous'] = 'seamless'
    before = deepcopy(shot)
    sdk = Responses({'compatible': True, 'entry_action': 'Connect the observed entry without repeating a completed event.',
                     'entry_visible_result': 'compatible', 'reason': 'offline annotation'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    context = json.loads(sdk.calls[0]['messages'][-1]['content'])
    planned = context['planned_shot']
    assert 'prompt_en' not in planned and 'blocking' not in planned
    assert 'pose_and_gaze' not in planned['end_state']
    assert planned['action_beats'][3]['action'] == planned['action_beats'][0]['action']
    assert result['action_beats'] == before['action_beats']
    assert result['start_state']['action_phase'] == previous['observed_end_state']['action_phase']
    assert shot == before


def test_captured_v8_planning_cache_keeps_original_authority_without_call(tmp_path, monkeypatch):
    import config
    from pipeline.planning_review import ensure_planning_semantics
    from pipeline.run_state import RunWorkspace
    _, sources, _ = captured()
    story = sources['storyboard.json']
    original_review = sources['planning_semantic_review.json']
    assert original_review['protocol_version'] == 'planning-semantics-v8-evidence-sources'
    monkeypatch.setattr(config, 'LLM_MODEL', 'doubao-seed-2.1-lite')
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: pytest.fail('historical cache must not be re-reviewed'))
    save = lambda name, value: RunWorkspace._atomic_write_json(tmp_path / name, value)
    save('planning_semantic_review.json', original_review)
    result = ensure_planning_semantics(story, tmp_path, save)
    assert result == sources['result_admission.json']['execution_storyboard']
    assert all(not s.get('completion_policy') for s in result['shots'])
    assert json.loads((tmp_path / 'planning_semantic_review.json').read_text()) == original_review


def test_new_independent_admission_of_old_pending_draft_produces_complete_protocol():
    from pipeline.models import ShotSpec
    _, sources, story = captured()
    for shot in story['shots']:
        shot.pop('result_policy', None)
    units = annotated_intents(story, sources['result_admission.json']['units'])
    result = apply_result_admission(story, units)
    for shot in result['shots']:
        assert ShotSpec.model_validate(shot).result_policy == 'functional-v1'
        assert compile_action_contract(shot).review_projection['completion_contract']['intent_policy'] == 'admitted-intent-v1'
    assert all(not s.get('result_policy') for s in story['shots'])


def optional_interaction_story():
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_action_result_flow import annotated_verdict
    _, _, raw, plan = scenario_story('货物交接')
    # A separately annotated low-density scene: one transfer, then an optional hand flourish.
    for slot in plan['slots']:
        slot.update(min_interaction_events=1, min_interaction_seconds=0, authoring_protocol='scene-units-v1')
    for source in raw['shots']:
        source['result_admission'] = 'review-required-v1'
    story = compile_story(raw, plan)
    units = annotated_intents(story, annotated_verdict(story)['units'])
    units[2]['completion_intent'] = {'kind': 'supporting', 'supports': 2,
        'reason': 'Synthetic fixture: the transfer is complete; the second hand flourish has no independent required function.'}
    for decision in units[2]['result_admission']['decisions']:
        decision.update(basis='direction', evidence='The flourish and its finishing posture are optional in this fixture.')
    return story, units


def test_supporting_physical_gesture_does_not_gain_a_second_causal_obligation():
    from pipeline.event_assessment import assess_event_evidence
    from pipeline.provider_prompt import compile_policy_safe_prompt
    story, units = optional_interaction_story()
    admitted = apply_result_admission(story, units)
    shot = admitted['shots'][0]
    contract = compile_action_contract(shot)
    assert [e['event_id'] for e in contract.review_projection['action_events']] == [2]
    wire = annotated_wire(format_for(shot)['json_schema']['schema'])
    wire['observations']['beat:3/completion'].update(status='violated', observation='The flourish is absent.')
    decoded = decode(shot, wire)
    assert decoded['accepted']
    decoded['action_event_evidence'][0].update(observed_start_seconds=0.4, observed_effect_seconds=0.8,
        observed_reaction_seconds=1., observed_end_seconds=2., active_intervals=[{'start_seconds': 0.4, 'end_seconds': 2.}])
    times = [i / 5 for i in range(26)]
    assert assess_event_evidence(contract, decoded['action_event_evidence'], timestamps=times)['status'] == 'passed'
    decoded['action_event_evidence'][0]['effect_visible'] = False
    assert assess_event_evidence(contract, decoded['action_event_evidence'], timestamps=times)['status'] == 'rejected'
    prompt = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'event 3:' in prompt and 'scope=' in prompt  # Generation retains the optional physical staging.


def test_supporting_labels_cannot_remove_existing_interaction_allocation():
    story, units = optional_interaction_story()
    story['shots'][0]['production_slot']['min_interaction_events'] = 2
    with pytest.raises(ValueError, match='交互.*分配|分配.*交互'):
        apply_result_admission(story, units)


def test_scene_with_only_supporting_interactions_does_not_fall_back_to_legacy_cause_review(tmp_path, offline_frames):
    from pipeline.semantic_review import SemanticTakeReviewer
    from types import SimpleNamespace
    import subprocess
    story, units = optional_interaction_story()
    story['shots'][0]['production_slot'].update(min_interaction_events=0, min_interaction_seconds=0)
    for unit in units[1:3]:
        unit['completion_intent'] = {'kind': 'supporting', 'supports': 1,
            'reason': 'Synthetic scene: arrival is required; the two hand gestures are replaceable performance.'}
        for decision in unit['result_admission']['decisions']:
            decision.update(basis='direction', evidence='Synthetic optional physical flourish.')
    shot = apply_result_admission(story, units)['shots'][0]
    assert compile_action_contract(shot).review_projection['action_events'] == []
    # Use existing offline media sampling adapter; exercise the actual reviewer schema boundary.
    video = tmp_path / 'offline.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=64x48:r=24:d=6.25',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        schema = kwargs['response_format']['json_schema']['schema']['properties']
        if 'decisions' in schema:
            from functional_observation_cases import annotated_support_response
            from test_full_production_flow import completion
            return completion(annotated_support_response(kwargs))
        assert 'causal_evidence' not in schema and 'action_event_evidence' not in schema
        wire = annotated_wire(kwargs['response_format']['json_schema']['schema'])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(wire), refusal=None))])
    reviewer = SemanticTakeReviewer(tmp_path / 'review', model='offline-annotation',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert reviewer.review(str(video), shot).accepted
    assert len(calls) == 2

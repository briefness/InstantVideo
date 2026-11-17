"""Motion speed is not event occupancy; offline service annotations are explicit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import format_authoring_slots
from pipeline.causality import compile_action_contract
from pipeline.director_intent import resolve_director_intent
from pipeline.production_plan import build_production_plan
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from test_director_intent import Responses
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline motion timing attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline motion timing attempted network'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_slow_motion_delivery.json').read_text())
    assert hashlib.sha256(record['source']['text'].encode()).hexdigest() == record['source']['sha256']
    return record, json.loads(record['source']['text'])['execution_storyboard']


def prompts(shot, story):
    following = shot['shot_id'] > 1
    return [compile_normal_provider_prompt(shot, story, has_observed_start=following),
            compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=following,
                                      image_role='first_frame' if following else None,
                                      reference_count=int(following))]


def test_real_delivery_remains_byte_equivalent_in_existing_contract_and_profiles():
    record, story = captured()
    for shot, baseline in zip(story['shots'], record['baseline']):
        assert compile_action_contract(shot).fingerprint() == baseline['contract']
        assert [hashlib.sha256(p.encode()).hexdigest() for p in prompts(shot, story)] == baseline['prompts']
    # This is the observed design gap, not an assertion that 4.5 s equals rapid action.
    last = story['shots'][-1]
    assert len([b for b in last['action_beats'] if b['activity'] == 'interaction']) == 1
    assert last['action_beats'][0]['end_seconds'] == 4.5


@pytest.mark.parametrize('brief,tempo,playback', [
    ('双人舞要快速连续旋转，按正常时间播放，不要慢镜头。', 'fast', 'real_time'),
    ('缓慢做精密装配，但摄影保持正常时间，不要慢放。', 'deliberate', 'real_time'),
    ('用慢镜头展示运动员快速的起跳发力。', 'fast', 'slow_motion'),
])
def test_timing_has_independent_source_through_intent_plan_author_and_both_requests(brief, tempo, playback):
    direction = {'performer_tempo': tempo, 'playback': playback, 'evidence': brief}
    # Independent service annotation: this does not assess remote language understanding.
    client = Responses({'content_focus': 'action', 'pacing': 'dense', 'intent_evidence': brief,
                        'choreography': 'flexible', 'motion_timing': direction})
    intent = resolve_director_intent(client, brief)
    plan = build_production_plan('action', 15, intent=intent)
    assert all(s['motion_timing'] == direction for s in plan['slots'])
    assert all(s['motion_timing'] == direction for s in json.loads(format_authoring_slots(plan)))
    _, story = captured()
    shot = deepcopy(story['shots'][1])
    shot['production_slot'] = plan['slots'][1]
    original_beats = deepcopy(shot['action_beats'])
    contract = compile_action_contract(shot)
    assert contract.review_projection['motion_timing'] == direction
    for prompt in prompts(shot, story):
        assert f'performer_tempo={tempo}' in prompt and f'playback={playback}' in prompt
        assert 'plot outcomes' in prompt
        assert 'invent extra events' not in prompt
        assert 'do not introduce events beyond the ordered contract' not in prompt
    assert shot['action_beats'] == original_beats


def test_schema_represents_speed_separately_from_density_and_grounded_multisentence_evidence():
    import jsonschema
    from pipeline.director_intent import director_intent_response_schema
    brief = '动作快速连续。用慢镜头呈现。'
    record, story = captured()
    legacy = story['production_plan']['director_intent']
    raw = {k: deepcopy(v) for k, v in legacy.items() if k not in {'source_request', 'choreography'}}
    raw.update(intent_evidence='动作快速连续。', ending_requirement='', ending_evidence='',
               completion_goals=[{'goal': '动作快速连续。', 'basis': 'user', 'evidence': '动作快速连续。'}],
               creative_direction=None,
               requested_duration_seconds=None, duration_evidence=None, duration_mode='approximate',
               requested_shot_count=None, shot_count_evidence=None,
               motion_timing={'performer_tempo': 'fast', 'playback': 'slow_motion', 'evidence': brief})
    selection = {'references': [f'request:0:{len(brief)}']}
    raw['intent_evidence'] = deepcopy(selection)
    raw['completion_goals'][0]['evidence'] = deepcopy(selection)
    raw['motion_timing']['evidence'] = deepcopy(selection)
    jsonschema.validate(raw, director_intent_response_schema(brief))
    raw['motion_timing'] = None
    jsonschema.validate(raw, director_intent_response_schema(brief))
    raw['motion_timing'] = {'performer_tempo': 'extreme', 'playback': 'real_time', 'evidence': brief}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, director_intent_response_schema(brief))


@pytest.mark.parametrize('mutation', [
    {'performer_tempo': 'cinematic'}, {'playback': '2x'}, {'evidence': 'not in source'},
    {'evidence': ''}, {'evidence': ' '}, {'performer_tempo': True}, {'playback': None},
    {'performer_tempo': 'unspecified', 'playback': 'unspecified'}, {'unexpected': 'ignore contract'},
])
def test_invalid_timing_stops_at_intent_without_retry(mutation):
    brief = '正常速度展示连续作业。'
    timing = {'performer_tempo': 'natural', 'playback': 'real_time', 'evidence': brief, **mutation}
    client = Responses({'content_focus': 'action', 'pacing': 'dense', 'intent_evidence': brief,
                        'motion_timing': timing})
    with pytest.raises(ValueError, match='无法可靠解析'):
        resolve_director_intent(client, brief)
    assert len(client.calls) == 1


def test_missing_direction_never_infers_speed_from_dense_cinematic_or_model_name():
    from pipeline.models import DirectorIntentSpec, ProductionSlotSpec
    _, story = captured()
    original = story['production_plan']['director_intent']
    assert DirectorIntentSpec.model_validate(original).model_dump() == original
    for slot in story['production_plan']['slots']:
        assert 'motion_timing' not in ProductionSlotSpec.model_validate(slot).model_dump()
    no_direction = resolve_director_intent(Responses({
        'content_focus': 'action', 'pacing': 'dense', 'intent_evidence': 'Fast电影感持续表演',
        'motion_timing': None}), 'Fast电影感持续表演')
    assert 'motion_timing' not in no_direction
    assert all('motion_timing' not in s for s in build_production_plan('action', 15, intent=no_direction)['slots'])


def test_product_movement_tempo_does_not_force_dense_interaction_or_change_duration():
    brief = '快速旋转展示产品，镜头保持稳定，正常播放。'
    direction = {'performer_tempo': 'fast', 'playback': 'real_time', 'evidence': brief}
    intent = {'source_request': brief, 'content_focus': 'product', 'pacing': 'balanced',
              'intent_evidence': brief, 'choreography': 'flexible'}
    old = build_production_plan('product', 15, intent=intent)
    new = build_production_plan('product', 15, intent={**intent, 'motion_timing': direction})
    assert [s['duration'] for s in old['slots']] == [s['duration'] for s in new['slots']]
    assert [{k: v for k, v in s.items() if k != 'motion_timing'} for s in new['slots']] == old['slots']
    shot = {'shot_id': 1, 'production_slot': new['slots'][0], 'interaction_geometry': {'effect_phase': 'none'}}
    contract = compile_action_contract(shot)
    assert contract.review_projection['motion_timing'] == direction
    assert 'performer_tempo=fast' in contract.review_instruction
    for prompt in prompts(shot, {}):
        assert 'performer_tempo=fast' in prompt
        assert 'connected footwork' not in prompt  # no invented human interaction for product motion


def test_exact_and_reference_paths_keep_event_constraints_and_receive_tempo():
    from test_action_sequence import sequence_story
    story = sequence_story()
    shot = story['shots'][0]
    before = deepcopy(shot['action_beats'])
    brief = '按参考动作顺序准确演示，保持自然动作速度。'
    shot.setdefault('production_slot', {}).update(choreography='exact', creative_brief=brief,
        motion_timing={'performer_tempo': 'natural', 'playback': 'real_time', 'evidence': brief})
    contract = compile_action_contract(shot)
    assert not contract.review_projection.get('performance_policy')
    assert 'performer_tempo=natural' in contract.event_review_instruction()
    for prompt in prompts(shot, story):
        assert 'performer_tempo=natural' in prompt and '1.2s' in prompt
        assert 'connected footwork' not in prompt
    assert shot['action_beats'] == before


def test_speed_changes_contract_identity_without_changing_actions_or_time_allocation():
    from pipeline.film_rhythm import planned_interaction_seconds
    from pipeline.production_plan import production_plan_issues
    _, story = captured()
    original = deepcopy(story)
    direction = {'performer_tempo': 'fast', 'playback': 'real_time',
                 'evidence': story['production_plan']['director_intent']['intent_evidence']}
    changed_intent = {**story['production_plan']['director_intent'], 'motion_timing': direction}
    plan = build_production_plan('action', 15, intent=changed_intent)
    story['production_plan'] = plan
    for shot, slot, prior in zip(story['shots'], plan['slots'], original['shots']):
        shot['production_slot'] = slot
        assert compile_action_contract(shot).fingerprint() != compile_action_contract(prior).fingerprint()
        assert shot['action_beats'] == prior['action_beats']
    assert planned_interaction_seconds(story['shots']) == planned_interaction_seconds(original['shots'])
    assert not production_plan_issues(story)
    story['shots'][1]['production_slot'] = deepcopy(story['shots'][1]['production_slot'])
    story['shots'][1]['production_slot']['motion_timing']['performer_tempo'] = 'deliberate'
    assert production_plan_issues(story)


@pytest.mark.asyncio
@pytest.mark.parametrize('topic,tempo,playback', [
    ('双人舞', 'fast', 'real_time'), ('货物交接', 'deliberate', 'real_time'),
    ('绳索救援', 'fast', 'slow_motion'),
])
async def test_official_generation_export_and_resume_keep_timing(
        scheduled_services, topic, tempo, playback):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.run_state import RunWorkspace
    from test_creation_schedule_flow import scenario_story
    from tools.ffmpeg_ops import get_video_info
    _, services = scheduled_services(topic)
    brief, reply, source, _ = scenario_story(topic)
    sentence = f'表演速度要求 {tempo}，播放处理 {playback}。'
    brief += sentence
    direction = {'performer_tempo': tempo, 'playback': playback, 'evidence': sentence}
    from pipeline.intent_evidence import evidence_catalog
    catalog = evidence_catalog(brief)
    key = next(k for k,v in catalog.items() if v['text'] == sentence)
    reply['motion_timing'] = {**direction, 'evidence': {'references': [key]}}
    ending_key = next(k for k,v in catalog.items() if v['text'] == reply['ending_evidence'])
    ending_parts = [next(iter(catalog)), ending_key]
    reply['ending_evidence'] = {'references': ending_parts}
    reply['completion_goals'][0]['evidence'] = {'references': ending_parts}
    direction['evidence'] = {'version': 'source-spans-v1', 'segments': [catalog[key]]}
    services.author = Responses(reply, source)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3,
                             take_policy='single_pass', delivery_mode='reviewed')
    output = await pipeline.run(brief)
    assert Path(output).name == 'final.mp4'
    assert services.planning[0]['outcome_review']['requirement']['segments'] == [catalog[k] for k in ending_parts]
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    for call in services.author.calls[2:]:
        context = json.loads(call['messages'][-1]['content'])
        assert context['slot']['motion_timing'] == direction
        assert '不是肢体动作的数量上限' in call['messages'][0]['content']
    assert len(services.submissions) == len(services.reviews) == 3
    for call in services.submissions:
        assert call['model'] == config.SEEDANCE_MODEL
        assert f'performer_tempo={tempo}' in call['prompt'] and f'playback={playback}' in call['prompt']
    frozen = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    assert all(s['production_slot']['motion_timing'] == direction for s in frozen['shots'])
    assert [[b['action'] for b in s['action_beats']] for s in frozen['shots']] == [
        [unit['action'] for unit in s['action_plan']] for s in source['shots']]
    state = RunWorkspace.resume(pipeline.workspace).manifest
    assert state.status.value == 'succeeded' and len(state.paid_take_budget.reservations) == 3
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360
    counts = len(services.author.calls), len(services.planning), len(services.submissions)
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.planning), len(services.submissions))


@pytest.mark.asyncio
async def test_existing_review_and_continuation_receive_same_timing_without_new_stage(scheduled_services):
    from pipeline.orchestrator import VideoPipeline
    from test_creation_schedule_flow import scenario_story
    _, services = scheduled_services('双人舞')
    brief, reply, source, _ = scenario_story('双人舞')
    evidence = '快速衔接旋转，以正常时间播放。'
    direction = {'performer_tempo': 'fast', 'playback': 'real_time', 'evidence': evidence}
    reply['motion_timing'] = direction
    services.author = Responses(reply, source)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief + evidence)
    assert Path(output).name == 'final.mp4'
    assert len(services.submissions) == len(services.reviews) == 3 and len(services.entries) == 2
    assert all(review['motion_timing'] == direction for review in services.reviews)
    assert all(review['performance_policy']['motion_timing'] == direction for review in services.reviews)
    assert len(services.author.calls) == 5 and len(services.planning) == 1


def test_motion_timing_input_cannot_be_forged_downstream():
    _, story = captured()
    shot = story['shots'][0]
    shot['production_slot']['motion_timing'] = {
        'performer_tempo': 'fast', 'playback': 'real_time', 'evidence': 'fabricated requirement'}
    with pytest.raises(ValueError, match='冻结需求原文'):
        compile_action_contract(shot)

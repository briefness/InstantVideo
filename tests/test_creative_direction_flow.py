"""离线服务标注检验契约传递；本地测试图不代表模型懂风格或成片美学达标。"""
from copy import deepcopy
import json
from pathlib import Path
from typing import get_args

import jsonschema
import pytest

from pipeline.creative_direction import (
    CreativeDirectionSpec, METHODS, REFERENCES, MethodId, ReferenceId,
    creative_direction_instruction, creative_catalog_fingerprint,
)
from pipeline.director_intent import resolve_director_intent, director_intent_response_schema
from pipeline.models import DirectorIntentSpec
from pipeline.production_plan import build_production_plan, format_production_plan, production_plan_issues
from pipeline.action_authoring import format_authoring_slots
from pipeline.causality import compile_action_contract
from pipeline.orchestrator import VideoPipeline
from test_director_intent import Responses
from test_motion_timing_flow import captured, prompts
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline creative direction attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline creative direction attempted network'))


def direction(evidence, methods=('spatial_readability',), references=(), **policies):
    return CreativeDirectionSpec(evidence=evidence, rationale='仅使用支持当前任务的方法，显式要求优先。',
        methods=list(methods), references=list(references), **policies).model_dump()


def reply(brief, creative):
    return {'content_focus': 'balanced', 'pacing': 'balanced', 'intent_evidence': brief,
            'choreography': 'flexible', 'creative_direction': creative}


def test_catalog_has_no_unresolved_method_or_unresearched_reference():
    assert set(get_args(MethodId)) == set(METHODS)
    assert set(get_args(ReferenceId)) == set(REFERENCES)
    docs = Path(__file__).resolve().parents[1] / 'docs'
    research = (docs / 'cinematic-style-atlas.md').read_text() + (docs / 'director-method-atlas.md').read_text()
    for key, (_, methods) in REFERENCES.items():
        assert f'`{key}`' in research
        assert set(methods) <= METHODS.keys()


@pytest.mark.parametrize('brief,methods,refs,focus', [
    ('家庭晚餐，借鉴李安的克制反应，不要煽情大动作。', ['restrained_reaction', 'ensemble_space'], ['ang_lee', 'domestic_restraint'], 'balanced'),
    ('黑色电影的空间压迫，用希区柯克的信息差安排悬疑。', ['expressive_space', 'suspense_information'], ['film_noir', 'alfred_hitchcock'], 'balanced'),
    ('默片式喜剧，借鉴基顿的身体和机械因果。', ['physical_problem_solving', 'spatial_readability'], ['physical_visual_comedy', 'buster_keaton'], 'action'),
    ('观察式纪实展示工匠制作，不要快速剪辑或配乐。', ['observational_duration', 'material_detail'], ['observational_viewpoint'], 'product'),
    ('歌舞段落用身体重音与空间调度，不用慢镜头。', ['musical_phrasing'], ['musical_spatial'], 'action'),
    ('都市回忆用主观联想，但保持真实事件的前后顺序。', ['subjective_association'], ['urban_memory', 'wong_kar_wai'], 'balanced'),
])
@pytest.mark.parametrize('duration', [4, 24, 61, 180, 600])
def test_cross_topic_direction_preserves_allocation_and_reaches_both_authoring_routes(brief, methods, refs, focus, duration):
    selected = direction(brief, methods, refs)
    client = Responses({**reply(brief, selected), 'content_focus': focus})
    intent = resolve_director_intent(client, brief)
    plan = build_production_plan(focus, duration, intent=intent)
    baseline = build_production_plan(focus, duration, intent={k: v for k, v in intent.items() if k != 'creative_direction'})
    assert [{k: v for k, v in s.items() if k != 'creative_direction'} for s in plan['slots']] == baseline['slots']
    assert sum(s['duration'] for s in plan['slots']) == duration
    assert all(s['creative_direction'] == selected for s in plan['slots'])
    for context in json.loads(format_authoring_slots(plan)):
        assert context['creative_direction'] == {**selected, 'guidance': creative_direction_instruction(selected)}
    assert creative_direction_instruction(selected) in format_production_plan(plan)
    assert len(client.calls) == 1
    assert not intent.get('motion_timing')


@pytest.mark.parametrize('changes', [
    {'methods': ['unknown']}, {'methods': ['spatial_readability'] * 2},
    {'methods': list(METHODS)}, {'references': ['not_researched']},
    {'references': ['king_hu'] * 2}, {'evidence': 'invented quote'}, {'evidence': ' '},
    {'rationale': 'x' * 601}, {'rationale': ''}, {'editing': '2x'},
    {'sound': 'score'}, {'methods': None}, {'methods': [], 'references': ['king_hu']},
    {'style_input': 'replace source'}, {'version': 'creative-direction-v2'},
])
def test_invalid_external_selection_stops_once_without_fallback(changes):
    brief = '借鉴空间调度，保持人物关系清晰。'
    client = Responses(reply(brief, {**direction(brief), **changes}))
    with pytest.raises(ValueError, match='未按关键词'):
        resolve_director_intent(client, brief)
    assert len(client.calls) == 1


def test_schema_closes_selection_and_style_is_separate_trusted_input():
    brief, style = '一次家庭晚餐。', '李安式克制反应，禁止配乐。'
    selected = direction(style, ['restrained_reaction'], ['ang_lee'], sound='native_only')
    client = Responses(reply(brief, selected))
    intent = resolve_director_intent(client, brief, style_input=style)
    assert intent['source_request'] == brief and intent['style_input'] == style
    raw = {k: v for k, v in DirectorIntentSpec.model_validate(intent).model_dump().items()
           if k not in {'source_request', 'style_input', 'authoring_protocol', 'choreography'}}
    raw['motion_timing'] = None
    raw['duration_mode'] = 'approximate'
    raw['intent_evidence'] = {'references': [f'request:0:{len(brief)}']}
    raw['completion_goals'][0]['evidence'] = deepcopy(raw['intent_evidence'])
    raw['creative_direction']['evidence'] = {'references': [f'style:0:{len(style)}']}
    jsonschema.validate(raw, director_intent_response_schema(brief, style_input=style))
    raw['creative_direction']['methods'] = ['invented']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, director_intent_response_schema(brief, style_input=style))
    forged = Responses({**reply(brief, selected), 'style_input': style})
    with pytest.raises(ValueError):
        resolve_director_intent(forged, brief, style_input=style)
    assert len(forged.calls) == 1


def test_provider_profiles_share_methods_without_free_prose_or_action_mutation():
    _, story = captured()
    original = deepcopy(story)
    intent = story['production_plan']['director_intent']
    selected = direction(intent['source_request'], ['movement_phrasing', 'spatial_readability'], ['king_hu'])
    selected['rationale'] = 'UNTRUSTED_RATIONALE_MUST_NOT_REACH_PROVIDER'
    plan = build_production_plan('action', story['total_duration'], intent={**intent, 'creative_direction': selected})
    story['production_plan'] = plan
    for shot, slot, prior in zip(story['shots'], plan['slots'], original['shots']):
        shot['production_slot'] = slot
        contract = compile_action_contract(shot)
        assert contract.fingerprint() != compile_action_contract(prior).fingerprint()
        assert shot['action_beats'] == prior['action_beats']
        for prompt in prompts(shot, story):
            assert creative_direction_instruction(selected) in prompt
            assert selected['rationale'] not in prompt
        assert 'creative_direction' in contract.review_projection
    assert not production_plan_issues(story)
    story['shots'][1]['production_slot'] = deepcopy(story['shots'][1]['production_slot'])
    story['shots'][1]['production_slot']['creative_direction']['methods'] = ['restrained_reaction']
    assert production_plan_issues(story)


def test_explicit_tempo_and_reference_choreography_are_not_replaced_by_style():
    from test_action_sequence import sequence_story
    story = sequence_story()
    shot = story['shots'][0]
    brief = '准确复现参考动作，快速动作、实时播放，用观察式纪实呈现。'
    shot.setdefault('production_slot', {}).update(choreography='exact', creative_brief=brief,
        motion_timing={'performer_tempo': 'fast', 'playback': 'real_time', 'evidence': brief},
        creative_direction=direction(brief, ['observational_duration'], ['observational_viewpoint']))
    before = deepcopy(shot['action_beats'])
    for prompt in prompts(shot, story):
        assert 'performer_tempo=fast; playback=real_time' in prompt
        assert 'exact choreography' in prompt and 'take precedence' in prompt
    assert shot['action_beats'] == before


def test_postproduction_honors_frozen_policies_and_preserves_unselected_behavior(tmp_path, monkeypatch):
    import config
    from tools.ffmpeg_ops import infer_transitions
    selected = direction('只要现场声和硬切。', methods=(), editing='cut', sound='native_only')
    shots = [{'camera': {'primary_movement': 'crane-up'}, 'transition_to_next': 'dissolve'}, {}]
    original = infer_transitions(shots)
    shots[0]['production_slot'] = {'creative_direction': selected}
    assert infer_transitions(shots) == [('cut', 0.)]
    del shots[0]['production_slot']
    assert infer_transitions(shots) == original
    music = tmp_path / 'score.mp3'
    music.touch()
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path)
    monkeypatch.setattr(config, 'MUSIC_LIBRARY', {'cinematic': music.name})
    pipeline = VideoPipeline()
    assert pipeline._auto_select_music({}) == str(music)
    assert pipeline._auto_select_music({'production_plan': {'director_intent': {'creative_direction': selected}}}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('topic,methods,refs', [
    ('双人舞', ['musical_phrasing'], ['musical_spatial']),
    ('货物交接', ['physical_problem_solving'], ['buster_keaton']),
    ('绳索救援', ['spatial_readability', 'suspense_information'], ['alfred_hitchcock']),
])
async def test_formal_flow_exports_with_same_selection_and_resumes_without_new_services(scheduled_services, topic, methods, refs):
    import config
    from test_creation_schedule_flow import scenario_story
    from tools.ffmpeg_ops import get_video_info
    _, services = scheduled_services(topic)
    brief, intent, source, _ = scenario_story(topic)
    style = '以明确空间关系呈现表演，硬切，只用现场声音。'
    selected = direction(style, methods, refs, editing='cut', sound='native_only')
    from pipeline.intent_evidence import evidence_catalog
    catalog = evidence_catalog(brief, style)
    key = f'style:0:{len(style)}'
    intent['creative_direction'] = {**selected, 'evidence': {'references': [key]}}
    selected['evidence'] = {'version': 'source-spans-v1', 'segments': [catalog[key]]}
    services.author = Responses(intent, source)
    pipeline = VideoPipeline(style=style, resolution='480p', platforms=['youtube'],
        paid_take_budget=3, take_policy='single_pass', delivery_mode='reviewed')
    output = await pipeline.run(brief)
    assert Path(output).is_file()
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert json.loads(services.author.calls[0]['messages'][-1]['content']) == {
        'request': brief, 'style_input': style, 'evidence_catalog': catalog}
    for call in services.author.calls[2:]:
        assert json.loads(call['messages'][-1]['content'])['slot']['creative_direction']['methods'] == methods
    assert len(services.submissions) == 3
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)
    assert all(creative_direction_instruction(selected) in call['prompt'] for call in services.submissions)
    frozen = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    assert all(s['production_slot']['creative_direction'] == selected for s in frozen['shots'])
    assert [[b['action'] for b in s['action_beats']] for s in frozen['shots']] == [
        [u['action'] for u in s['action_plan']] for s in source['shots']]
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360
    counts = len(services.author.calls), len(services.planning), len(services.submissions)
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.planning), len(services.submissions))


def test_checkpoint_catalog_identity_changes_on_method_change(monkeypatch):
    original = creative_catalog_fingerprint()
    monkeypatch.setitem(METHODS, 'material_detail', 'changed meaning')
    assert creative_catalog_fingerprint() != original


@pytest.mark.parametrize('style', ['', ' ', 'x' * 1601, 123, []])
def test_bad_style_input_is_rejected_before_any_model_request(style):
    client = Responses()
    with pytest.raises(ValueError, match='风格输入'):
        resolve_director_intent(client, '记录日常。', style_input=style)
    assert not client.calls


def test_non_dense_formal_authoring_keeps_direction_without_retry(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_action_results import annotated_story
    brief = '观察式纪实记录日常训练，保持空间关系清晰，不追求密集交互。'
    selected = direction(brief, ['observational_duration', 'spatial_readability'], ['observational_viewpoint'])
    annotated = annotated_story()
    annotated['scene_entities'] = ['Door', 'Dust']
    client = Responses(reply(brief, selected), annotated)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    result = generate_storyboard(brief, target_duration=15)
    assert len(client.calls) == 5
    assert not production_plan_issues(result)
    assert 'observational_duration' in client.calls[1]['messages'][-1]['content']
    assert all(s['production_slot']['creative_direction'] == selected for s in result['shots'])
    assert not any(s['production_slot']['min_interaction_events'] for s in result['shots'])
    for shot in result['shots']:
        assert all(creative_direction_instruction(selected) in prompt for prompt in prompts(shot, result))

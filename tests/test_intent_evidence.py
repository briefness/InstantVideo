"""真实意图失败与跨题材证据传输；禁止联网，不人工改写目标。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.director_intent import resolve_director_intent, director_intent_response_schema
from test_director_intent import Responses, intent_reply
from test_goal_allocation import offline


def captured():
    data = json.loads((Path(__file__).parent / 'fixtures/real_multi_source_intent_failure.json').read_text())
    for record in data['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    raw = json.loads(json.loads(data['files']['planning_intent_response.json']['text'])['content'])
    return data, raw


def test_real_unmodified_response_gets_lossless_source_locations_once():
    from pipeline.completion_intent import source_goals
    data, raw = captured()
    snapshot = deepcopy(raw)
    client = Responses(raw)
    result = resolve_director_intent(client, data['request'], style_input=data['style_input'])
    assert len(client.calls) == 1 and raw == snapshot
    goal = result['completion_goals'][3]
    assert goal['goal'] == raw['completion_goals'][3]['goal']
    spans = goal['evidence']['segments']
    assert len(spans) == 2 and [s['start'] for s in spans] == [32, 226]
    assert ''.join(s['text'] for s in spans) == raw['completion_goals'][3]['evidence']
    assert all(data['request'][s['start']:s['end']] == s['text'] for s in spans)
    assert len(source_goals({'production_plan': {'director_intent': result}})) == 4


@pytest.mark.parametrize('brief', [
    '母亲打开门。镜头保持安静。女儿走进家门。',
    '机器人举起零件。背景是工厂。机器人将零件装入底座。',
    'The runner starts. Show the crowd. The runner finishes.',
    '展示陶瓷花瓶。不要人物。最后展示瓶口纹理。',
])
def test_catalog_selection_is_a_source_address_not_a_generated_quote(brief):
    from pipeline.intent_evidence import evidence_catalog
    from pipeline.models import DirectorIntentSpec
    catalog = evidence_catalog(brief)
    parts = [key for key, value in catalog.items() if value['text'] != brief]
    raw = intent_reply(brief, focus='balanced', pacing='balanced')
    raw['completion_goals'][0]['evidence'] = {'references': [parts[0], parts[-1]]}
    client = Responses(raw)
    result = resolve_director_intent(client, brief)
    evidence = result['completion_goals'][0]['evidence']
    assert len(evidence['segments']) == 2
    assert DirectorIntentSpec.model_validate(result).model_dump() == result
    sent = json.loads(client.calls[0]['messages'][-1]['content'])
    assert sent['evidence_catalog'] == catalog
    schema = director_intent_response_schema(brief)
    assert schema['$defs']['CompletionGoal']['properties']['evidence']['properties']['references']['items']['enum'] == list(catalog)


@pytest.mark.parametrize('kind', ['unknown', 'duplicate', 'reverse', 'style', 'empty', 'extra'])
def test_invalid_selection_stops_once_with_field_diagnostic(kind):
    from pipeline.intent_evidence import evidence_catalog
    request = '开始走路。不要奔跑。最后停下。'
    catalog = evidence_catalog(request, '冷色')
    ids = [key for key, value in catalog.items() if value['source'] == 'request' and value['text'] != request]
    selection = {'references': [ids[0], ids[-1]]}
    if kind == 'unknown': selection['references'] = ['request:999:1000']
    elif kind == 'duplicate': selection['references'] = [ids[0], ids[0]]
    elif kind == 'reverse': selection['references'] = [ids[-1], ids[0]]
    elif kind == 'style': selection['references'] = [next(k for k,v in catalog.items() if v['source'] == 'style')]
    elif kind == 'empty': selection['references'] = []
    else: selection['text'] = 'forged'
    raw = intent_reply(request)
    raw['completion_goals'][0]['evidence'] = selection
    errors = {}
    client = Responses(raw)
    with pytest.raises(ValueError):
        resolve_director_intent(client, request, style_input='冷色', save_diagnostic=errors.__setitem__)
    assert len(client.calls) == 1
    assert errors['planning_intent_error.json']['fields'][0]['loc'] == ['completion_goals', 0, 'evidence']


def test_all_intent_dimensions_keep_locations_through_plan_camera_and_ending():
    from pipeline.intent_evidence import evidence_catalog
    from pipeline.production_plan import build_production_plan
    from pipeline.motion_timing import motion_timing_projection
    from pipeline.creative_direction import creative_direction_instruction
    from pipeline.requested_outcome import requested_ending, outcome_review_context
    from pipeline.models import DirectorIntentSpec
    brief = '24秒，四段。动作要快。正常速度播放。救援队将伤者带出。不要放下伤者。最后送到安全站。'
    style = '冷色，硬切，不配乐。'
    catalog = evidence_catalog(brief, style)
    ids = [key for key, s in catalog.items() if s['source'] == 'request' and s['text'] != brief]
    select = lambda *keys: {'references': list(keys)}
    raw = {**intent_reply(brief), 'intent_evidence': select(ids[1], ids[2]),
        'requested_duration_seconds': 24, 'duration_evidence': select(ids[0]),
        'requested_shot_count': 4, 'shot_count_evidence': select(ids[0]),
        'ending_requirement': '伤者被带出并送到安全站', 'ending_evidence': select(ids[3], ids[5]),
        'motion_timing': {'performer_tempo': 'fast', 'playback': 'real_time', 'evidence': select(ids[1], ids[2])},
        'creative_direction': {'methods': ['spatial_readability'], 'references': [], 'rationale': '保持空间清晰，按用户规定硬切。',
            'editing': 'cut', 'sound': 'native_only', 'evidence': select(ids[3], next(k for k,s in catalog.items() if s['source']=='style'))}}
    intent = resolve_director_intent(Responses(raw), brief, style_input=style)
    assert DirectorIntentSpec.model_validate(json.loads(json.dumps(intent))).model_dump() == intent
    plan = build_production_plan('action', 24, intent=intent)
    for slot in plan['slots']:
        timing = motion_timing_projection({'production_slot': slot})
        assert [s['text'] for s in timing['evidence']['segments']] == ['动作要快。', '正常速度播放。']
        assert 'direct cuts' in creative_direction_instruction(slot['creative_direction'])
    ending = requested_ending(plan)
    assert ending == intent['ending_evidence']
    assert [s['start'] for s in ending['segments']] == [brief.index('救援队'), brief.index('最后')]
    context = outcome_review_context({'production_plan': plan, 'shots': [{'action_beats': [], 'end_state': {}}]})
    assert context['requirement'] == ending and context['sources'] == {}
    assert '不要放下伤者。' in intent['source_request']  # 选段不删除完整上下文或否定项。


@pytest.mark.parametrize('mutation', ['hash', 'start', 'end', 'text', 'source', 'negative_removed'])
def test_saved_locations_cannot_outlive_changed_source_or_text(mutation):
    from pipeline.intent_evidence import evidence_catalog
    from pipeline.models import DirectorIntentSpec
    from pipeline.completion_intent import source_goals
    brief = '先移动。不要奔跑。最后停下。'
    ids = list(evidence_catalog(brief))
    raw = intent_reply(brief)
    raw['completion_goals'][0]['evidence'] = {'references': ids[:2]}
    intent = resolve_director_intent(Responses(raw), brief)
    span = intent['completion_goals'][0]['evidence']['segments'][0]
    if mutation == 'hash': span['document_sha256'] = '0' * 64
    elif mutation == 'start': span['start'] = True
    elif mutation == 'end': span['end'] = 999
    elif mutation == 'text': span['text'] = '先奔跑。'
    elif mutation == 'source': span['source'] = 'style'
    else: intent['source_request'] = brief.replace('不要', '')
    with pytest.raises(ValueError): DirectorIntentSpec.model_validate(intent)
    with pytest.raises(ValueError): source_goals({'production_plan': {'director_intent': intent}})


@pytest.mark.parametrize('quote', ['先走。伪造结果。', '最后停下。先走。', '先走。停下。'])
def test_legacy_compound_quote_never_fuzzy_matches_or_reorders(quote):
    raw = intent_reply('先走。不要跑。最后停下。')
    raw['completion_goals'][0]['evidence'] = quote
    with pytest.raises(ValueError):
        resolve_director_intent(Responses(raw), '先走。不要跑。最后停下。')


def test_ambiguous_legacy_sentences_require_explicit_addresses():
    from pipeline.intent_evidence import evidence_catalog
    brief = '出发。不要跑。出发。抵达。'
    raw = intent_reply(brief)
    raw['completion_goals'][0]['evidence'] = '出发。不要跑。抵达。'
    with pytest.raises(ValueError): resolve_director_intent(Responses(raw), brief)
    ids = list(evidence_catalog(brief))
    raw['completion_goals'][0]['evidence'] = {'references': [ids[0], ids[1], ids[3]]}
    intent = resolve_director_intent(Responses(raw), brief)
    assert intent['completion_goals'][0]['evidence']['segments'][0]['start'] == 0


def test_unicode_coordinates_and_document_limits_are_exact():
    from pipeline.intent_evidence import evidence_catalog, evidence_grounded
    brief = '🎬开始。\n不奔跑！ Cafe\u0301 finishes. 停下。'
    catalog = evidence_catalog(brief)
    for span in catalog.values():
        assert brief[span['start']:span['end']] == span['text']
    whole = next(v for v in catalog.values() if v['text'] == brief)
    evidence = {'version': 'source-spans-v1', 'segments': [whole]}
    assert evidence_grounded(evidence, brief)
    assert not evidence_grounded(evidence, brief.replace('e\u0301', 'é'))
    assert not evidence_grounded('a', None)
    assert not evidence_grounded(evidence, brief, 1)
    assert len(next(iter(evidence_catalog('字' * 16000).values()))['text']) == 16000
    for invalid in ('', ' ', '字' * 16001, None, 12):
        with pytest.raises(ValueError): evidence_catalog(invalid)
    with pytest.raises(ValueError): evidence_catalog('需求', 'a' * 1601)


def test_reference_motion_also_uses_source_locations():
    from pipeline.intent_evidence import evidence_catalog
    brief = '严格按提供的骨架轨迹生成。角色完成跳跃。'
    first = next(iter(evidence_catalog(brief)))
    raw = {**intent_reply(brief), 'motion_control': 'reference_motion',
           'motion_control_evidence': {'references': [first]}}
    result = resolve_director_intent(Responses(raw), brief)
    assert result['choreography'] == 'exact'
    assert result['motion_control_evidence']['segments'][0]['text'] == '严格按提供的骨架轨迹生成。'
    assert 'authoring_protocol' not in result


@pytest.mark.parametrize('brief', ['先走。不要跑。最后停下。', 'A. Keep context. B.'])
def test_transport_schema_does_not_replace_local_cardinality_checks(brief):
    import jsonschema
    from pipeline.intent_evidence import evidence_catalog, selection_schema
    from pipeline.llm_options import ark_output_schema
    catalog = evidence_catalog(brief)
    first = next(iter(catalog))
    for ids in ([], [first, first]):
        # Ark 投影会移除数量/唯一性关键字，本地仍须独立拒绝。
        jsonschema.validate({'references': ids}, ark_output_schema(selection_schema(catalog)))
        raw = intent_reply(brief)
        raw['completion_goals'][0]['evidence'] = {'references': ids}
        with pytest.raises(ValueError): resolve_director_intent(Responses(raw), brief)

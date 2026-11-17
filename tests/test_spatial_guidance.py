"""可视空间约束的离线协议测试；坐标替身不代表模型定位准确率。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_goal_allocation import offline
from test_full_production_flow import completion

ROOT = Path(__file__).parents[1] / 'output/20261110_190901_163328'


def captured():
    story = json.loads((ROOT / 'result_admission.json').read_text())['execution_storyboard']
    record = json.loads((ROOT / 'continuations/shot_002.json').read_text())
    shot = deepcopy(story['shots'][1])
    shot['spatial_control'] = 'grounded-diagram-v1'
    previous = {**story['shots'][0], 'observed_end_state': record['input']['observed_state'],
                '_accepted_video_path': str(ROOT / 'shots/shot_001.mp4')}
    return story, shot, previous, record


def annotation(shot):
    return [{'beat_id': 1, 'entity': 'clear_glass_cup_with_warm_water',
             'target_entity': 'dark_wood_home_dining_table', 'relation': 'on',
             'source_box': [.76, .49, .80, .56],
             'target_region': [[.67, .53], [.73, .53], [.73, .55], [.67, .55]],
             'requirement_quote': '把杯子从自己身侧桌面移到空木椅旁的桌面位置'}]


def test_real_request_gets_a_distinct_visual_control_without_replacing_original(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.generator import VideoGenerator
    from pipeline.provider_materials import material_plan
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    story, shot, previous, record = captured()
    calls = []
    def create(**request):
        calls.append(request)
        value = deepcopy(record['plan'])
        if 'spatial_guidance' in request['response_format']['json_schema']['schema']['properties']:
            value['spatial_guidance'] = annotation(shot)
        return completion(value)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tail = str(ROOT / 'shots/shot_001_lastframe.jpg')
    before = deepcopy(shot)
    result = prepare_continuation(shot, previous, tail, tmp_path, client=client)
    assert result.get('spatial_guidance'), 'actual request needs a non-text control channel'
    generator = object.__new__(VideoGenerator)
    generator.output_dir, generator.character_refs = tmp_path, {}
    refs, role = generator._build_image_refs(result, tail, previous)
    plan = material_plan(refs, role)
    assert [x['type'] for x in plan['content']] == ['video_url', 'image_url']
    assert refs[0] == tail and Path(refs[-1]).suffix == '.png'
    from PIL import Image
    with Image.open(refs[-1]) as image:
        assert image.size == (960, 540)
        assert len(image.getcolors(1_000_000)) > 2
    for prompt in (compile_normal_provider_prompt(result, story, has_observed_start=True),
                   compile_policy_safe_prompt(result, storyboard=story, image_role=role,
                                              reference_count=len(refs), has_state_reference=True)):
        assert 'SPATIAL GUIDE' in prompt
    assert shot == before and result['action_beats'] == before['action_beats']
    assert result['camera'] == before['camera']
    assert prepare_continuation(shot, previous, tail, tmp_path, client=client, allow_create=False) == result
    assert len(calls) == 1


@pytest.mark.parametrize('topic', ['parcel', 'panel', 'rescue_line'])
def test_generic_geometry_is_bound_to_authority_not_object_keywords(topic, tmp_path):
    from pipeline.spatial_guidance import guidance_context, bind_guidance, materialize_guidance
    _, shot, _, _ = captured()
    shot['action_beats'] = [{'actor': 'Worker', 'target': topic, 'action': f'places {topic} on Receiver',
        'start_seconds': 0, 'end_seconds': 3, 'activity': 'interaction', 'outcomes': []}]
    shot['camera']['screen_positions'] = {'Worker': 'left', topic: 'middle', 'Receiver': 'right'}
    proposal = [{'beat_id': 1, 'entity': topic, 'target_entity': 'Receiver', 'relation': 'on',
                 'source_box': [.1, .2, .2, .3],
                 'target_region': [[.6, .5], [.9, .5], [.9, .6], [.6, .6]],
                 'requirement_quote': f'places {topic} on Receiver'}]
    context = guidance_context(shot)
    bound = bind_guidance(proposal, context, 'a' * 64)
    shot['spatial_guidance'] = bound
    path = materialize_guidance(shot, tmp_path)
    assert path.is_file()
    changed = deepcopy(shot)
    changed['action_beats'][0]['action'] = 'different work'
    with pytest.raises(ValueError):
        materialize_guidance(changed, tmp_path)


@pytest.mark.parametrize('fault', ['outside', 'degenerate', 'unknown', 'quote', 'self', 'nan', 'duplicate'])
def test_bad_grounding_cannot_become_valid_visual_control(fault):
    from pipeline.spatial_guidance import guidance_context, bind_guidance
    _, shot, _, _ = captured()
    value = annotation(shot)
    if fault == 'outside':
        value[0]['source_box'][0] = -1
    elif fault == 'degenerate':
        value[0]['target_region'] = [[.5, .5]] * 4
    elif fault == 'unknown':
        value[0]['target_entity'] = 'invented'
    elif fault == 'quote':
        value[0]['requirement_quote'] = 'invented requirement'
    elif fault == 'self':
        value[0]['target_entity'] = value[0]['entity']
    elif fault == 'nan':
        value[0]['source_box'][0] = float('nan')
    else:
        value *= 2
    with pytest.raises(ValueError):
        bind_guidance(value, guidance_context(shot), 'a' * 64)


@pytest.mark.parametrize('field', ['entity', 'target_entity', 'relation'])
def test_external_wrong_types_are_rejected(field):
    from pipeline.spatial_guidance import guidance_context, bind_guidance
    _, shot, _, _ = captured()
    value = annotation(shot)
    value[0][field] = []
    with pytest.raises(ValueError):
        bind_guidance(value, guidance_context(shot), 'a' * 64)


def test_reference_capacity_source_and_cache_fail_closed(tmp_path):
    import hashlib
    from pipeline.spatial_guidance import guidance_context, bind_guidance, append_guidance
    _, shot, _, _ = captured()
    tail = ROOT / 'shots/shot_001_lastframe.jpg'
    shot['spatial_guidance'] = bind_guidance(annotation(shot), guidance_context(shot),
                                             hashlib.sha256(tail.read_bytes()).hexdigest())
    for refs, role, source in [([f'https://example.invalid/{i}.png' for i in range(9)], 'reference_image', str(tail)),
                               ([str(tail)], 'first_frame', str(tail)),
                               ([str(tail)], 'reference_image', None)]:
        original = list(refs)
        with pytest.raises(ValueError):
            append_guidance(shot, refs, role, source, tmp_path)
        assert refs == original
    refs, _ = append_guidance(shot, [str(tail)], 'reference_image', str(tail), tmp_path)
    Path(refs[-1]).write_bytes(b'corrupt cache')
    with pytest.raises(ValueError, match='缓存已变化'):
        append_guidance(shot, [str(tail)], 'reference_image', str(tail), tmp_path)
    assert Path(refs[-1]).read_bytes() == b'corrupt cache'


def test_capacity_uses_physical_materials_not_logical_locators(tmp_path):
    import hashlib
    from pipeline.spatial_guidance import guidance_context, bind_guidance, append_guidance
    from pipeline.provider_materials import material_plan
    _, shot, _, _ = captured()
    tail = ROOT / 'shots/shot_001_lastframe.jpg'
    shot['spatial_guidance'] = bind_guidance(annotation(shot), guidance_context(shot),
                                             hashlib.sha256(tail.read_bytes()).hexdigest())
    refs = [str(tail)] * 9
    combined, role = append_guidance(shot, refs, 'reference_image', str(tail), tmp_path)
    assert combined[:-1] == refs
    assert [item['type'] for item in material_plan(combined, role)['content']] == ['video_url', 'image_url']


def test_empty_grounding_keeps_references_and_scope_roles_are_distinct(tmp_path):
    from pipeline.spatial_guidance import guidance_context, bind_guidance, append_guidance, reference_clauses
    _, shot, _, _ = captured()
    shot['spatial_guidance'] = bind_guidance([], guidance_context(shot), 'a' * 64)
    refs = ['original-reference']
    assert append_guidance(shot, refs, 'first_frame', None, tmp_path) == (refs, 'first_frame')
    clauses = reference_clauses(3, 'reference_image', True)
    assert 'Image 3 is the SPATIAL GUIDE' in clauses[0]
    assert 'Image 1 supplies the accepted prior physical state' in clauses[1]
    assert 'Image 2 supplies identity and appearance only' in clauses[2]
    assert len(clauses) == 3

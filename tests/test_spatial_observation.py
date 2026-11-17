"""实际失败响应不能越过观察/指令边界；几何替身不代表识别准确率。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_goal_allocation import offline

ROOT = Path(__file__).parents[1] / 'output/20261111_131400_535263'
VERSION = 'observed-layout-v1'


def capture():
    story = json.loads((ROOT / 'result_admission.json').read_text())['execution_storyboard']
    shot = deepcopy(story['shots'][1])
    shot['spatial_control'] = VERSION
    record = json.loads((ROOT / 'continuations/shot_002.json').read_text())
    return story, shot, record


def test_real_failed_proposal_cannot_be_promoted_to_a_new_action():
    from pipeline.spatial_guidance import guidance_context, bind_guidance
    _, shot, record = capture()
    context = guidance_context(shot)
    with pytest.raises(ValueError):
        bind_guidance(record['plan']['spatial_guidance'], context, record['input']['tail_sha256'])


def test_grounder_only_localizes_fixed_entities_without_writing_goals():
    from pipeline.spatial_guidance import guidance_context, guidance_schema, bind_guidance
    _, shot, record = capture()
    context = guidance_context(shot)
    schema = guidance_schema(context)
    assert schema['type'] == 'object'
    assert schema['additionalProperties'] is False
    assert {'warm_water_glass', 'dining_table', 'empty_dining_chair'} <= set(context['entities'])
    assert 'An' not in context['entities'], 'unallocated gaze/hold cannot create spatial work'
    proposal = dict.fromkeys(context['entities'])
    proposal['dining_table'] = [[.63,.54],[.99,.54],[.99,.64],[.63,.64]]
    bound = bind_guidance(proposal, context, record['input']['tail_sha256'])
    assert 'warm_water_glass' in bound['unlocated']
    assert bound['items'] == [{'entity': 'dining_table', 'region': proposal['dining_table']}]
    assert not any(k in item for item in bound['items'] for k in ('relation', 'target_entity', 'requirement_quote'))
    bad = deepcopy(proposal)
    bad['warm_water_glass'] = {'substitute': 'Aunt Lin'}
    with pytest.raises(ValueError):
        bind_guidance(bad, context, record['input']['tail_sha256'])


@pytest.mark.parametrize('subject,support', [('parcel','shelf'), ('panel','fixture'), ('patient','stretcher')])
def test_grounding_semantics_survive_renaming_without_keyword_rules(subject, support):
    from pipeline.spatial_guidance import guidance_context, bind_guidance, guidance_instruction
    _, shot, _ = capture()
    shot['action_beats'] = [{'actor':'Worker','target':subject,'action':'authored work',
        'activity':'interaction','completion_intent':{'kind':'allocated','requirements':[{'reference':'goal'}]},
        'interaction_geometry':{'effect_region':support}}]
    shot['camera']['screen_positions'] = {'Worker':'left',subject:'center',support:'right','Bystander':'back'}
    context = guidance_context(shot)
    assert set(context['entities']) == {subject, support}
    proposal = {subject:None,support:[[.5,.5],[.9,.5],[.9,.9],[.5,.9]]}
    shot['spatial_guidance'] = bind_guidance(proposal, context, 'a'*64)
    text = guidance_instruction(shot)
    assert 'observed' in text and support in text
    assert 'intended target' not in text
    assert shot['action_beats'][0]['action'] == 'authored work'


def test_optional_map_cannot_override_a_selected_pixel_camera_input(tmp_path):
    from pipeline.spatial_guidance import guidance_context, bind_guidance, guidance_instruction, append_guidance
    _, shot, _ = capture()
    ctx = guidance_context(shot)
    proposal = dict.fromkeys(ctx['entities'])
    proposal['dining_table'] = [[.6,.5],[.9,.5],[.9,.6],[.6,.6]]
    shot['spatial_guidance'] = bind_guidance(proposal, ctx, 'a'*64)
    shot['reference_view'] = {'execution_capability': {'mode':'first_frame_pixels'}}
    refs = ['selected-pixel-view']
    assert append_guidance(shot, refs, 'first_frame', 'source', tmp_path) == (refs, 'first_frame')
    assert guidance_instruction(shot) == ''
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault', ['missing','extra','nan','bool','crossed','outside'])
def test_observation_boundary_rejects_invalid_external_data(fault):
    from pipeline.spatial_guidance import guidance_context, bind_guidance
    _, shot, _ = capture()
    ctx = guidance_context(shot)
    proposal = dict.fromkeys(ctx['entities'])
    points = [[.5,.5],[.9,.5],[.9,.9],[.5,.9]]
    proposal['dining_table'] = points
    if fault == 'missing': proposal.pop('dining_table')
    elif fault == 'extra': proposal['invented'] = None
    elif fault == 'nan': points[0][0] = float('nan')
    elif fault == 'bool': points[0][0] = True
    elif fault == 'crossed': points[1], points[2] = points[2], points[1]
    else: points[0][0] = -1
    with pytest.raises(ValueError):
        bind_guidance(proposal, ctx, 'a'*64)


def test_actual_source_runs_new_observation_path_and_resumes_without_another_call(tmp_path):
    from types import SimpleNamespace
    from test_full_production_flow import completion
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from pipeline.generator import VideoGenerator
    from pipeline.provider_materials import material_plan, material_prompt
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.spatial_guidance import reference_kind
    story, shot, record = capture()
    previous = {**story['shots'][0], 'observed_end_state':record['input']['observed_state'],
                '_accepted_video_path':str(ROOT/'shots/shot_001.mp4')}
    calls = []
    def create(**request):
        calls.append(request)
        ctx = json.loads(request['messages'][-1]['content'])
        result = deepcopy(record['plan'])
        # 离线观察替身，不把真实失败响应中的关系改成成功的生产记录。
        result['spatial_guidance'] = dict.fromkeys(ctx['spatial_context']['entities'])
        result['spatial_guidance']['dining_table'] = [[.63,.54],[.99,.54],[.99,.64],[.63,.64]]
        return completion(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tail = str(ROOT/'shots/shot_001_lastframe.jpg')
    before = deepcopy(shot)
    result = prepare_continuation(shot, previous, tail, tmp_path, client=client)
    assert result['spatial_guidance']['unlocated'] == ['empty_dining_chair', 'warm_water_glass']
    generator = object.__new__(VideoGenerator)
    generator.output_dir, generator.character_refs = tmp_path, {}
    refs, role = generator._build_image_refs(result, tail, previous)
    materials = material_plan(refs, role)
    assert [m['type'] for m in materials['content']] == ['video_url','image_url']
    normal = generator._inject_reference_scope(
        compile_normal_provider_prompt(result,story,has_observed_start=True), role,
        reference_count=len(refs), has_state_reference=True, spatial_reference=reference_kind(result))
    safe = compile_policy_safe_prompt(result,storyboard=story,has_state_reference=True,
                                     image_role=role,reference_count=len(refs))
    for text in (normal,safe):
        lowered = material_prompt(text, materials)
        assert 'observed object locations only' in lowered
        assert 'geometric destination guidance' not in lowered
        assert '1A=Aunt Lin' not in lowered
        assert '图片1 is the SPATIAL GUIDE' in lowered
    assert shot == before and result['action_beats'] == before['action_beats']
    assert result['camera'] == before['camera']
    assert prepare_continuation(shot,previous,tail,tmp_path,client=client,allow_create=False) == result
    changed = deepcopy(shot)
    changed['camera']['composition'] += ' changed'
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(changed,previous,tail,tmp_path,client=client,allow_create=False)
    assert len(calls) == 1


def test_real_legacy_artifact_bytes_are_not_reinterpreted():
    from pipeline.spatial_guidance import _legacy_image_bytes
    record = json.loads((ROOT/'shots/shot_002_rejected_1_generation.json').read_text())
    bound = record['spatial_guidance']
    expected = _legacy_image_bytes(bound)
    assert any(path.read_bytes() == expected for path in (ROOT/'spatial_guidance').glob('*.png'))

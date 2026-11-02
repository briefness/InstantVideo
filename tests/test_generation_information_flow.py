"""Information conservation at the real authoring/provider/continuation seams.

Service responses are explicit offline annotations, not model quality evidence.
"""
from copy import deepcopy
import json

import pytest

from pipeline.continuation import prepare_continuation
from pipeline.provider_prompt import compile_normal_provider_prompt
from pipeline.result_admission import apply_result_admission
from pipeline.storyboard import generate_storyboard
from test_action_results import annotated_story
from test_director_intent import Responses, intent_reply
from test_action_result_flow import annotated_verdict
from test_continuation import scenario, Planner
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('generation-chain replay attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


@pytest.mark.parametrize('visual', [
    '青瓷展台的釉面细裂纹，暖色窗光穿过磨砂玻璃形成层叠投影',
    '仓库红砖墙、磨损的黄铜栏杆，侧逆光照出货物包装的纤维质感',
    '装配车间的拉丝铝台面和蓝灰隔板，柔光下金属边缘清晰',
])
def test_authored_visual_direction_reaches_normal_provider(monkeypatch, visual):
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    for shot in raw['shots']:
        shot['visual_description'] = visual
    source = deepcopy(raw)
    brief = '15秒三段连续动作展示'
    client = Responses({**intent_reply(brief), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    compiled = generate_storyboard(brief, target_duration=15)
    assert raw == source and len(client.calls) == 5
    for shot in compiled['shots']:
        prompt = compile_normal_provider_prompt(shot, compiled, has_observed_start=False)
        assert visual in prompt


@pytest.mark.parametrize('entry_kind', ['transition', 'interaction'])
def test_reviewed_results_have_same_authority_for_every_entry(tmp_path, entry_kind):
    from test_action_authoring import plan_for
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    raw = annotated_story()
    plan = plan_for()
    for source in raw['shots']:
        for unit in source['action_plan']:
            if unit['kind'] == 'interaction':
                unit['resolution']['actor_state'] = 'OPTIONAL_POSE: elbow held at an exact angle'
    for slot in plan['slots']:
        slot['choreography'] = 'flexible'
    story = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
                                        '16:9', '720p', 'cinematic', production_plan=plan)
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    verdict = annotated_verdict(story)
    for unit in verdict['units']:
        for decision in unit['result_admission']['decisions']:
            if decision['source'] == 'outcomes/1':
                decision.update(basis='direction', evidence='offline label: replaceable actor posture')
    shot = apply_result_admission(story, verdict['units'])['shots'][1]
    if entry_kind == 'interaction':
        shot['action_beats'].pop(0)
    _, previous, tail = scenario(tmp_path)
    shot.update(scene_id='roof', continuity_from_previous='seamless')
    original = deepcopy(shot)
    response = {'compatible': True, 'entry_action': 'continues toward the receiver',
                'reason': 'offline annotation of compatible entry'}
    if entry_kind == 'transition':
        response['entry_visible_result'] = 'within reach of the receiver'
    client = Responses(response)
    prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    context = json.loads(client.calls[0]['messages'][-1]['content'])
    for beat in context['planned_shot']['action_beats']:
        if beat.get('activity') == 'interaction':
            assert len(beat['outcomes']) == 2  # target state + required Door state
            assert all('attacker' != state['entity'] for state in beat['outcomes'])
            assert 'result_directions' not in beat
    assert shot == original


def test_real_authoring_replay_keeps_visual_action_and_camera_without_source_rewrite():
    from pathlib import Path
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.causality import compile_action_contract
    root = Path(__file__).parent / 'fixtures'
    raw = json.loads((root / 'real_generation_authoring.json').read_text())['value']
    old = json.loads((root / 'real_result_admission_failure.json').read_text())['sources']['storyboard.json']['value']
    before = deepcopy(raw)
    plan = old['production_plan']
    new = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
                                      old['aspect_ratio'], old['resolution'], old['style'], production_plan=plan)
    assert raw == before
    for authored, legacy, rebuilt in zip(raw['shots'], old['shots'], new['shots']):
        prompt = compile_normal_provider_prompt(rebuilt, new, has_observed_start=False)
        assert authored['visual_description'] in prompt
        assert authored['visual_description'] not in compile_normal_provider_prompt(legacy, old, has_observed_start=False)
        assert rebuilt['camera']['primary_movement'] == authored['camera']['primary_movement']
        assert [b['action'] for b in rebuilt['action_beats']] == [u['action'] for u in authored['action_plan']]
        assert compile_action_contract(legacy).fingerprint() == compile_action_contract(deepcopy(legacy)).fingerprint()


def test_visual_direction_is_part_of_existing_provenance_not_action_admission(tmp_path):
    from pipeline.generator import VideoGenerator
    from pipeline.causality import compile_action_contract
    from test_action_results import scene_shot
    generator = object.__new__(VideoGenerator)
    generator.character_refs = {}
    generator.output_dir = tmp_path
    shot = scene_shot()
    original = generator._generation_provenance(shot, [], None)
    shot['visual_description'] = 'amber glass beside a textured blue wall'
    first = generator._generation_provenance(shot, [], None)
    accepted = generator._acceptance_context(shot, [], None, policy='technical_only')
    action = compile_action_contract(shot).fingerprint()
    shot['visual_description'] = 'matte grey metal beside a red wall'
    second = generator._generation_provenance(shot, [], None)
    assert 'visual_direction_fingerprint' not in original
    assert first['visual_direction_fingerprint'] != second['visual_direction_fingerprint']
    assert accepted != generator._acceptance_context(shot, [], None, policy='technical_only')
    assert compile_action_contract(shot).fingerprint() == action


def test_legacy_reviewed_transition_cache_is_reused_byte_for_byte(tmp_path):
    import hashlib
    import config
    from pipeline.continuation import VERSION, ContinuationUnavailableError
    from test_result_admission import candidate, annotated
    story = candidate()
    shot = apply_result_admission(story, annotated(story)['units'])['shots'][0]
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    _, previous, tail = scenario(tmp_path)
    # Exact legacy v3 input: transition entry did not project admitted results.
    context = {'version': VERSION, 'model': config.LLM_MODEL,
        'tail_sha256': hashlib.sha256(tail.read_bytes()).hexdigest(),
        'observed_state': previous['observed_end_state'], 'planned_shot': deepcopy(shot),
        'entry_editable': True}
    plan = {'compatible': True, 'entry_action': 'previously saved entry',
            'entry_visible_result': 'previously saved approach state', 'reason': 'saved annotation'}
    path = tmp_path / 'continuations/shot_002.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'version': VERSION, 'input': context, 'plan': plan,
        'input_fingerprint': hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}))
    saved = path.read_bytes()
    client = Responses()
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False)
    assert result['action_beats'][0]['action'] == 'previously saved entry'
    assert path.read_bytes() == saved and not client.calls
    changed = deepcopy(shot)
    changed['action_beats'][0]['action'] = 'a different input'
    with pytest.raises(ContinuationUnavailableError, match='不匹配'):
        prepare_continuation(changed, previous, str(tail), tmp_path, client=client, allow_create=False)
    assert path.read_bytes() == saved and not client.calls


@pytest.mark.parametrize('invalid', ['', 'x' * 1601, 42, ['wrong type']])
def test_visual_direction_has_the_same_bound_at_persistence_as_authoring(invalid):
    from pydantic import ValidationError
    from pipeline.models import ShotSpec
    from test_action_results import scene_shot
    with pytest.raises(ValidationError):
        ShotSpec.model_validate({**scene_shot(), 'visual_description': invalid})


def test_legacy_visual_fields_do_not_reactivate_and_safe_profile_stays_bounded():
    from pipeline.provider_prompt import compile_policy_safe_prompt
    from pipeline.models import ShotSpec
    from test_action_results import scene_shot
    shot = scene_shot()
    shot['prompt_en'] = 'LEGACY_UNTRUSTED_ACTION'
    shot['scene_description'] = 'LEGACY_UNTRUSTED_SCENE_ACTION'
    assert 'visual_description' not in ShotSpec.model_validate(shot).model_dump(exclude_none=True)
    before = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    assert 'LEGACY_UNTRUSTED' not in before
    shot['visual_description'] = 'NEW_AUTHORED_MATERIAL_DETAIL'
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'NEW_AUTHORED_MATERIAL_DETAIL' not in safe
    assert 'LEGACY_UNTRUSTED' not in safe


@pytest.mark.asyncio
@pytest.mark.parametrize('brief,prop,action,result,visual', [
    ('搬运交接', 'Parcel', 'passes the parcel to the receiver', 'holds the parcel securely', 'warm light on red brick and woven cardboard'),
    ('信件传递', 'Letter', 'slides the envelope into the listener hands', 'has received the envelope', 'amber windows reflected in a polished oak table'),
    ('车间装配', 'Panel', 'guides the panel into the receiver clamp', 'holds the fitted panel', 'brushed aluminium with blue-grey partitions'),
])
async def test_formal_fast_chain_reaches_real_sdk_payload_and_export(
        production_services, monkeypatch, brief, prop, action, result, visual):
    from pathlib import Path
    from types import SimpleNamespace
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.run_state import RunWorkspace
    from tools.seedance_api import SeedanceAPI
    _, services = production_services
    raw = annotated_story()
    raw['scene_entities'] = [prop, 'Dust']
    raw['story_arc'] = {key: brief + ': ' + result for key in raw['story_arc']}
    for character in raw['characters']:
        character['description'] = 'work clothes, empty hands at entry, no weapons'
    for index, shot in enumerate(raw['shots']):
        shot.update(visual_description=visual, key_props=[prop, 'Dust'],
                    scene_description='A workshop with a shared table',
                    story_change=f'{prop} transfer progresses through stage {index + 1}',
                    story_state_after=f'stage {index + 1} of the transfer is established')
        for boundary in ('start_state', 'end_state'):
            shot[boundary] = {**shot[boundary], 'location': 'workshop', 'subject': 'two workers',
                'action_phase': 'continuous transfer', 'prop_state': f'{prop} in hand',
                'pose_and_gaze': 'hands extended toward the shared work, eyes on the object',
                'open_motion': f'hands move with the {prop} across the table'}
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.update(action=action, result=result, weight=3.)
                unit['resolution'].update(target_state=result,
                    actor_state='OPTIONAL_POSTURE: elbow at a precise angle', scene_changes=[
                    {'entity': prop, 'state': result, 'importance': 'required'},
                    {'entity': 'Dust', 'state': 'briefly airborne', 'importance': 'incidental'}])
                unit['effect'].update(source='hands', region='shared workspace', unaffected='background remains unchanged')
            else:
                unit.update(action='continues the movement toward the shared workspace' if unit['kind'] == 'transition'
                            else 'maintains the established transfer', result='the ongoing transfer remains visible')
        # An uninterrupted middle shot exercises an interaction entry as well as transition entry.
        if index == 1:
            shot['action_plan'].pop(0)
    brief = '15秒，三段连续的' + brief
    services.author = Responses({**intent_reply(brief), 'choreography': 'flexible',
        'requested_duration_seconds': 15, 'duration_evidence': '15秒',
        'requested_shot_count': 3, 'shot_count_evidence': '三段'}, raw)
    base_create = services.create
    def annotated_create(**kwargs):
        response = base_create(**kwargs)
        content = kwargs['messages'][-1]['content']
        value = json.loads(response.choices[0].message.content)
        if isinstance(content, list):
            value['observed_end_state'] = {'location': 'workshop', 'subject': 'two workers',
                'action_phase': f'transfer stage {len(services.submissions)} established',
                'prop_state': f'{prop}: {result}', 'open_motion': f'{prop} continues toward the receiver'}
        elif 'observed_state' in json.loads(content):
            value['entry_action'] = f'continues the observed {prop} movement through the shared workspace'
            if 'entry_visible_result' in value:
                value['entry_visible_result'] = f'{prop} remains within reach of the receiver'
        if isinstance(content, str) and '"storyboard"' in content:
            for unit in value['units'].values():
                for source, decision in (unit.get('result_admission') or {}).get('decisions', {}).items():
                    if source == 'outcomes/1':
                        decision.update(basis='direction', reference=None,
                                        reason='offline annotation: optional posture')
        response.choices[0].message.content = json.dumps(value)
        return response
    monkeypatch.setattr(services, 'create', annotated_create)
    def create_task(**kwargs):
        services.submissions.append(kwargs)
        return SimpleNamespace(id=f'offline-{len(services.submissions)}')
    def get_task(*, task_id):
        return SimpleNamespace(status='succeeded', content=SimpleNamespace(video_url=task_id))
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(
        create=create_task, get=get_task)))
    api.download_video = services.download_video
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: api)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).is_file()
    assert len(services.author.calls) == 5 and len(services.submissions) == 3
    assert len(services.entries) == 2
    for index, request in enumerate(services.submissions):
        assert request['model'] == config.SEEDANCE_MODEL
        assert request['resolution'] == '720p' and request['duration'] == 5
        text = request['content'][0]['text']
        assert visual in text and action in text and result in text
        references = request['content'][1:]
        assert all(item['type'] != 'video_url' for item in references)
        if index:
            assert references[0]['role'] == 'first_frame'
            assert references[0]['image_url']['url'].startswith('data:image/')
        else:
            assert not references
    for context in services.entries:
        assert context['version'] == 'observed-entry-v7-path-guidance'
        assert context['observed_state']['prop_state'] == f'{prop}: {result}'
        assert context['observed_state']['open_motion'] == f'{prop} continues toward the receiver'
        for beat in context['planned_shot']['action_beats']:
            assert all('OPTIONAL_POSTURE' not in item['state'] for item in beat.get('outcomes', []))
    assert [c['entry_result_bound'] for c in services.entries] == [True, False]
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 3
    counts = (len(services.author.calls), len(services.submissions), len(services.entries))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.submissions), len(services.entries))

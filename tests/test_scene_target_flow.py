"""Scene-object targets are visible participants, not character identities.

Real authoring responses are replayed unchanged. Semantic labels in integration
tests are synthetic and do not approve the recorded film or its prose.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units, compile_authored_storyboard
from pipeline.causality import compile_action_contract
from pipeline.participants import visible_character_names
from pipeline.provider_prompt import compile_normal_provider_prompt
from pipeline.production_plan import build_production_plan
from pipeline.readiness import ensure_storyboard_ready
from pipeline.storyboard import generate_storyboard, _compile_storyboard_contract
from test_action_authoring import plan_for
from test_action_results import annotated_story
from test_director_intent import Responses, intent_reply
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('scene target replay attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def recorded():
    sources = json.loads((Path(__file__).parent / 'fixtures/real_scene_target_failure.json').read_text())['sources']
    intent = json.loads(sources['planning_intent_response.json']['value']['content'])
    story = deepcopy(sources['planning_identity.json']['value'])
    story['shots'] = [{'shot_id': i, **deepcopy(sources[f'planning_shot_{i:03d}.json']['value']['shot'])}
                      for i in range(1, 4)]
    request = sources['planning_compiled.json']['value']['production_plan']['director_intent']['source_request']
    return request, intent, story, sources


def object_story(target, action, state):
    raw = annotated_story()
    raw['scene_entities'] = [target, 'Dust', 'Unused_Lamp']
    for shot in raw['shots']:
        shot['key_props'] = [target, 'Dust', 'Unused_Lamp']
        shot['camera']['screen_positions'][target] = 'lower center beside the working hand'
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.update(target=target, action=action, result=state, response='impact')
                unit['effect'].update(mode='direct_contact', source='the working hand',
                                      region=f'the surface of {target}')
                unit['resolution'] = {'target_state': state, 'actor_state': None,
                    'scene_changes': [{'entity': 'Dust', 'state': 'briefly airborne', 'importance': 'incidental'}]}
    return raw


def test_exact_real_object_targets_survive_without_hiding_invalid_staging():
    request, intent, raw, sources = recorded()
    before = deepcopy(raw)
    # Historical authoring is compiler input, not a new identity response.
    plan = build_production_plan('action', 15, intent={**intent, 'source_request': request})
    result = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        ensure_storyboard_ready(result)
    assert raw == before
    tile = 'Loose_Roof_Tile_Fragments'
    shot = result['shots'][1]
    assert tile in shot['required_visible_entities'] and tile not in shot['characters']
    assert visible_character_names(shot, [c['name'] for c in result['characters']]) == shot['characters']
    assert sources['planning_check.json']['value']['critical'] is True
    for source, compiled in zip(raw['shots'], result['shots']):
        assert [u['action'] for u in source['action_plan']] == [b['action'] for b in compiled['action_beats']]
        assert [u['result'] for u in source['action_plan']] == [b['visible_result'] for b in compiled['action_beats']]
    # Compiler correctness does not prove this contradictory semantic label true.
    assert shot['action_beats'][2]['response_kind'] == 'evasion'
    assert '无法做出格挡或闪避' in shot['action_beats'][2]['outcomes'][0]['state']


@pytest.mark.parametrize('target,action,state', [
    ('Door', 'pushes the door open with a continuous forward step', 'open with the passage clear'),
    ('Control_Panel', 'presses the control switch with one hand', 'indicator lit and switch depressed'),
    ('Parcel', 'turns the parcel on the table to show the label', 'label facing the viewer'),
    ('Violin_String', 'draws the bow along the string', 'string visibly vibrating'),
])
def test_object_roles_survive_compilation_prompt_and_review(target, action, state):
    raw = object_story(target, action, state)
    before = deepcopy(raw)
    plan = build_production_plan('action', 15, intent={**intent_reply('continuous action'),
        'source_request': 'continuous action', 'choreography': 'flexible'})
    compiled = compile_authored_storyboard(raw, plan)
    for shot in compiled['shots']:
        assert target in shot['required_visible_entities']
        assert target not in shot['characters']
        assert 'Dust' not in shot['required_visible_entities']
        assert 'Unused_Lamp' not in shot['required_visible_entities']
        for beat in shot['action_beats']:
            if beat['activity'] == 'interaction':
                assert beat['interaction_geometry']['target_screen_position'] == 'lower center beside the working hand'
    result = _compile_storyboard_contract(compiled, '16:9', '720p', 'cinematic', production_plan=plan)
    ensure_storyboard_ready(result)
    for shot in result['shots']:
        prompt = compile_normal_provider_prompt(shot, result, has_observed_start=False)
        assert target in prompt and state in prompt
        contract = compile_action_contract(shot)
        assert all(e['target'] == target for e in contract.review_projection['action_events'])
        assert target not in visible_character_names(shot, [c['name'] for c in result['characters']])
    assert raw == before


@pytest.mark.parametrize('explicit', [False, True])
def test_geometry_uses_authored_position_without_inventing_one(explicit):
    raw = object_story('Panel', 'presses the panel', 'pressed')['shots'][0]
    if explicit:
        raw['blocking']['Panel'] = {'frame_position': 'right foreground on the workbench'}
    else:
        raw['camera']['screen_positions'].pop('Panel')
    parsed = AuthoredShot.model_validate(raw)
    beats = compile_action_units(parsed, plan_for()['slots'][0])
    for b in beats:
        if b['activity'] == 'interaction':
            assert b['interaction_geometry']['target_screen_position'] == ('right foreground on the workbench' if explicit else '')


def test_later_object_target_is_not_replaced_by_last_event_pair():
    raw = object_story('Panel', 'presses the panel', 'pressed')
    shot = raw['shots'][0]
    events = [u for u in shot['action_plan'] if u['kind'] == 'interaction']
    events[-1]['target'] = 'defender'
    compiled = compile_authored_storyboard(raw, plan_for())['shots'][0]
    assert compiled['interaction_geometry']['target'] == 'defender'
    assert 'Panel' in compiled['required_visible_entities']
    assert len(compiled['required_visible_entities']) == len(set(compiled['required_visible_entities']))


def test_unknown_object_target_remains_rejected():
    raw = object_story('Panel', 'presses the panel', 'pressed')
    raw['shots'][0]['action_plan'][1]['target'] = 'Unregistered'
    with pytest.raises(ValueError, match='未注册'):
        compile_authored_storyboard(raw, plan_for())


@pytest.mark.asyncio
async def test_real_authoring_stops_at_first_invalid_boundary_without_retry(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_action_result_flow import annotated_verdict
    from planning_protocol_cases import annotated_outcome
    request, intent, raw, _ = recorded()
    plan = build_production_plan('action', 15, intent={**intent, 'source_request': request})
    historical = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    for shot in historical['shots']:
        shot['result_admission'] = 'review-required-v1'
    calls = []
    class Reviewer:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            story = json.loads(kwargs['messages'][-1]['content'])['storyboard']
            verdict = annotated_verdict(story)
            # Explicit offline negative label, not a new model judgment.
            unit = next(u for u in verdict['units'] if (u['shot_id'], u['beat_index']) == (2, 2))
            unit.update(classification_matches_description=False,
                        reason='offline annotation: evasion contradicts inability to evade')
            calls.append(deepcopy(story))
            return Responses(annotated_outcome(verdict, story)).create(**kwargs)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: deepcopy(historical))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: Reviewer())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('paid stage reached'))
    pipeline = VideoPipeline(paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await pipeline.run(request)
    assert not calls
    assert pipeline.run_workspace.manifest.paid_take_budget.reservations == []
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert not calls
    # 隔离验证第二个已知缺陷：原始语义矛盾也不能通过评审。
    from pipeline.planning_review import ensure_planning_semantics
    with pytest.raises(ValueError, match='evasion contradicts'):
        ensure_planning_semantics(historical, tmp_path / 'semantic-seam', lambda *args: None)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('target,action,state', [
    ('Door', 'pushes the door open', 'open with the passage clear'),
    ('Panel', 'presses the panel switch', 'switch depressed and indicator lit'),
])
@pytest.mark.parametrize('response_kind', ['impact', 'state_change'])
async def test_formal_object_flow_sdk_review_continuation_export_and_resume(
        production_services, monkeypatch, target, action, state, response_kind):
    from types import SimpleNamespace
    import config
    from pipeline.orchestrator import VideoPipeline
    from tools.seedance_api import SeedanceAPI
    from test_full_production_flow import original_task_response
    import subprocess
    _, services = production_services
    raw = object_story(target, action, state)
    brief = '15秒，三段连续展示人物操作物体的过程'
    for shot in raw['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
                unit['response'] = response_kind
    services.author = Responses({**intent_reply(brief), 'choreography': 'flexible',
        'requested_duration_seconds': 15, 'duration_evidence': '15秒',
        'requested_shot_count': 3, 'shot_count_evidence': '三段'}, raw)
    base_create = services.create
    def annotated_create(**kwargs):
        response = base_create(**kwargs)
        content = kwargs['messages'][-1]['content']
        value = json.loads(response.choices[0].message.content)
        if isinstance(content, list):
            # Two character identities plus one visible object, independently
            # labelled; the object does not acquire a third face crop.
            value['required_entities_visible'] = [True, True, True]
            if 'terminal_observation' in value:
                value['terminal_observation']['prop_state'] = f'{target}: {state}'
            elif 'observations' in value:
                key = 'beat:2/completion'
                value['observations'][key]['observation'] = f'{target}: {state}'
                value['end_state_sources']['prop_state'] = [key]
            else:
                value['observed_end_state']['prop_state'] = f'{target}: {state}'
        elif 'observed_state' in json.loads(content):
            value['entry_action'] = f'continues the movement beside {target}'
            if 'entry_visible_result' in value:
                value['entry_visible_result'] = f'{target} remains within reach'
        response.choices[0].message.content = json.dumps(value)
        return response
    monkeypatch.setattr(services, 'create', annotated_create)
    def create_task(**kwargs):
        services.submissions.append(kwargs)
        return SimpleNamespace(id=f'offline-{len(services.submissions)}')
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(
        create=create_task, get=lambda **kw: original_task_response(kw['task_id']))))
    api.download_video = services.download_video
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: api)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).is_file()
    duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=nw=1:nk=1', output], text=True))
    assert 14.9 <= duration <= 15.2
    assert len(services.submissions) == 3 and len(services.reviews) == 3 and len(services.entries) == 2
    for request in services.submissions:
        assert request['model'] == config.SEEDANCE_MODEL
        text = request['content'][0]['text']
        assert target in text and action in text and state in text
        if response_kind == 'state_change':
            from pipeline.action_vocabulary import STATE_CHANGE_EXECUTION
            assert STATE_CHANGE_EXECUTION in text
    for contract in services.reviews:
        assert all(e['target'] == target for e in contract['action_events'])
        assert all(e['response_kind'] == response_kind for e in contract['action_events'])
    for entry in services.entries:
        assert target in entry['planned_shot']['required_visible_entities']
        assert target not in entry['planned_shot']['characters']
        assert entry['observed_state']['prop_state'] == f'{target}: {state}'
    counts = (len(services.author.calls), len(services.planning), len(services.submissions), len(services.reviews))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.planning), len(services.submissions), len(services.reviews))

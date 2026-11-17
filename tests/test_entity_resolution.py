"""身份解析不得覆盖已登记对象；真实原稿逐字重放，禁止网络。"""
from copy import deepcopy
import hashlib
from itertools import permutations
import json
from pathlib import Path

import pytest

from pipeline.participants import (EntityRegistry, canonical_participant_id, canonical_entity_id,
    canonical_target_id, shot_entity_registry, normalize_shot_participants)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline test attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline test attempted network'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_exact_entity_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    return record, {n: json.loads(s['text']) for n, s in record['sources'].items()}


def test_real_creation_survives_final_compilation_without_target_or_result_changes():
    from pipeline.production_plan import build_production_plan
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.models import DirectorIntentSpec
    from pipeline.provider_prompt import compile_generation_preview
    record, values = captured()
    raw = values['planning_authoring.json']
    before = deepcopy(raw)
    intent = DirectorIntentSpec.model_validate({
        **json.loads(values['planning_intent_response.json']['content']),
        'source_request': record['options']['request'], 'style_input': record['options']['style'],
        'authoring_protocol': 'scene-units-v1'}).model_dump()
    plan = build_production_plan(intent['content_focus'], 24, intent=intent)
    compiled = compile_authored_storyboard(raw, plan)
    preserved = deepcopy(compiled)
    result = _compile_storyboard_contract(compiled, '16:9', '480p', record['options']['style'], production_plan=plan)
    for source, shot in zip(preserved['shots'], result['shots']):
        for a, b in zip(source['action_beats'], shot['action_beats']):
            for field in ('actor', 'target', 'action', 'visible_result', 'outcomes', 'scene_changes'):
                assert a.get(field) == b.get(field), (shot['shot_id'], field)
    preview = compile_generation_preview(result, model='doubao-seedance-2.0-fast')
    # 编译成功不能代替尚未执行的独立审查；该阻止提交的提示必须保留。
    assert all(p['preflight_issues'] == [f"Shot {s['shot_id']}: 结果候选尚未完成必要性审查，不能提交视频任务"]
               for s in preview['shots'] for p in s['profiles'].values())
    assert sum(s['duration'] for s in result['shots']) == 24 and raw == before


@pytest.mark.parametrize('character,entity', [('AN', 'transparent_warm_water_glass'),
    ('Li', 'linen_shade'), ('A', 'crate'), ('机', '机器面板')])
@pytest.mark.parametrize('namespace', ['props', 'themes'])
def test_exact_registered_target_is_stable_under_colliding_names(character, entity, namespace):
    for names in permutations([character, 'Operator']):
        registry = EntityRegistry(names, (entity,) if namespace == 'props' else (),
                                  (entity,) if namespace == 'themes' else ())
        assert canonical_target_id(entity, registry, exclude=['Operator']) == entity
        assert canonical_target_id(canonical_target_id(entity, registry), registry) == entity


@pytest.mark.parametrize('value,characters,props', [
    ('red crate panel', ('red',), ('red_crate',)),
    ('ab', ('AB',), ('A_B',)),
])
def test_ambiguity_across_kinds_is_not_resolved_by_discarding_object_candidates(value, characters, props):
    assert canonical_target_id(value, EntityRegistry(characters, props, ())) == value


def test_iterable_catalogs_and_exclusions_are_consumed_once():
    assert canonical_entity_id('blue fighter', iter(['blue_guard'])) == 'blue_guard'
    assert canonical_entity_id('blue guard', ['blue_guard', 'red_guard'], exclude=iter(['blue_guard'])) == 'blue guard'
    assert canonical_entity_id('blue_guard', ['blue_guard', 'red_guard'], exclude=['blue_guard']) == 'blue_guard'


@pytest.mark.parametrize('names', [('Panel', 'PanelSwitch'), ('Cup', 'Cup_Lid'), ('Gate', 'Gate_Handle')])
def test_structured_registration_does_not_merge_distinct_object_ids(names):
    for order in permutations(names):
        shot = {'production_slot': {'authoring_protocol': 'scene-units-v1'},
                'characters': ['Operator'], 'key_props': list(order), 'continuity_props': list(names)}
        assert set(shot_entity_registry(shot, ['Operator']).props) == set(names)
        normalize_shot_participants(shot, ['Operator'])
        assert shot['key_props'] == list(order) and shot['continuity_props'] == list(names)


@pytest.mark.parametrize('value,candidates,expected', [
    ('Defender on the right', ['Attacker', 'Defender'], 'Defender'),
    ('blue fighter', ['Blue_Guard', 'Red_Guard'], 'Blue_Guard'),
    ('robot-arm', ['Robot_Arm'], 'Robot_Arm'),
    ('unknown', ['A', 'B'], 'unknown'),
    ('...', ['A'], '...'),
    ('', ['A'], ''), ('none', ['A'], 'none'),
])
def test_legacy_aliases_and_unresolved_labels_keep_their_contract(value, candidates, expected):
    assert canonical_entity_id(value, candidates) == expected


def test_legacy_single_candidate_fallback_is_explicit():
    assert canonical_participant_id('unknown', ['Actor']) == 'unknown'
    assert canonical_participant_id('unknown', ['Actor'], fallback_to_single=True) == 'Actor'


@pytest.mark.asyncio
async def test_exact_captured_responses_reach_formal_preview_with_separate_annotated_review(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses
    from test_action_result_flow import annotated_verdict
    from planning_protocol_cases import annotated_outcome
    record, values = captured()
    author = Responses(json.loads(values['planning_intent_response.json']['content']),
                       values['planning_identity.json'],
                       *[values[f'planning_shot_{i:03d}.json'] for i in range(1, 5)])
    reviews = []
    def review(**kwargs):
        story = json.loads(kwargs['messages'][-1]['content'])['storyboard']
        reviews.append(story)
        # 原运行未到审查阶段；该响应是独立协议标注，不是补造真实审查记录。
        return Responses(annotated_outcome(annotated_verdict(story), story)).create(**kwargs)
    reviewer = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=review)))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: reviewer)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **_: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, planning_repair=False, paid_take_budget=0,
                             take_policy='single_pass', style=record['options']['style'])
    result = await pipeline.run(record['options']['request'])
    preview = json.loads((pipeline.workspace / 'generation_preview.json').read_text())
    assert all(not p['preflight_issues'] for s in preview['shots'] for p in s['profiles'].values())
    for i in range(1, 5):
        assert json.loads((pipeline.workspace / f'planning_shot_{i:03d}.json').read_text()) == values[f'planning_shot_{i:03d}.json']
    frozen = json.loads(Path(result).read_text())
    for source, shot in zip(values['planning_authoring.json']['shots'], frozen['shots']):
        assert [(u['actor'], u['target'], u['action']) for u in source['action_plan']] == [
            (b['actor'], b['target'], b['action']) for b in shot['action_beats']]
    assert len(author.calls) == 6 and len(reviews) == 1
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    assert await resumed.run() == result
    assert len(author.calls) == 6 and len(reviews) == 1
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations

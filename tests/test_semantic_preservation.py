"""原稿到执行的语义保持；不改失败记录，不访问模型或视频服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import compile_authored_storyboard
from pipeline.storyboard import _compile_storyboard_contract, _apply_coverage_defaults


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    r = json.loads((Path(__file__).parent / 'fixtures/real_semantic_lowering_failure.json').read_text())
    assert hashlib.sha256(r['source_text'].encode()).hexdigest() == r['source_sha256']
    return r, json.loads(r['source_text'])


def compile_record():
    record, raw = captured()
    lowered = compile_authored_storyboard(raw, record['production_plan'])
    before = deepcopy(lowered)
    final = _compile_storyboard_contract(lowered, **record['format'], production_plan=record['production_plan'])
    return record, before, final


def test_original_record_keeps_every_action_and_geometry_through_full_compilation():
    record, before, final = compile_record()
    for source, output in zip(before['shots'], final['shots'], strict=True):
        # 只允许序列化补齐缺省字段；所有已经明确的动作内容必须逐字段相同。
        for original, beat in zip(source['action_beats'], output['action_beats'], strict=True):
            assert all(beat[k] == v for k,v in original.items())
        assert all(output['interaction_geometry'][k] == v for k,v in source['interaction_geometry'].items())
    for shot_id in (2,3):
        beat = final['shots'][shot_id-1]['action_beats'][-1]
        assert beat['actor'] == 'rain_smudged_window_glass' and beat['target'] == ''
        assert record['failed_beats'][str(shot_id)]['actor'] != beat['actor']


def test_defaults_are_idempotent_and_never_inherit_another_actions_participants():
    _, _, final = compile_record()
    before = deepcopy(final['shots'])
    _apply_coverage_defaults(final['shots'])
    _apply_coverage_defaults(final['shots'])
    assert final['shots'] == before


def assert_round_trip(story, tmp_path):
    from pipeline.models import RunOptions
    from pipeline.run_state import RunWorkspace
    from pipeline.generator import VideoGenerator
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_generation_preview
    import config
    before = deepcopy(story)
    preview = compile_generation_preview(story, model=config.SEEDANCE_MODEL)
    fingerprints = [compile_action_contract(s).fingerprint() for s in story['shots']]
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='offline semantic round trip', paid_take_budget=0))
    workspace.save_storyboard(story)
    restored = RunWorkspace.resume(workspace.path).load_storyboard(include_observations=False)
    VideoGenerator._normalize_continuity(restored['shots'])
    assert restored == before == story
    assert [compile_action_contract(s).fingerprint() for s in restored['shots']] == fingerprints
    assert compile_generation_preview(restored, model=config.SEEDANCE_MODEL) == preview
    assert not workspace.manifest.paid_take_budget.reservations
    return preview


def test_original_record_survives_save_resume_generation_entry_and_both_prompt_profiles(tmp_path):
    record, raw = captured()
    original = deepcopy(raw)
    _, _, final = compile_record()
    preview = assert_round_trip(final, tmp_path)
    assert raw == original
    from pipeline.readiness import storyboard_readiness_issues
    assert not storyboard_readiness_issues(final)
    for row in preview['shots']:
        for profile in row['profiles'].values():
            assert profile['preflight_issues'] == [f"Shot {row['shot_id']}: 结果候选尚未完成必要性审查，不能提交视频任务"]
    for shot_id in (2, 3):
        for profile in preview['shots'][shot_id - 1]['profiles'].values():
            # 审查前的候选稿不等于审查通过；这里只验证请求来源仍是正确的环境主体。
            assert 'rain smudged window glass' in profile['prompt'].replace('_', ' ')


@pytest.mark.parametrize('subject,target,action,result', [
    ('Vent_Fan', '', 'The ceiling fan keeps rotating.', 'Air circulates above the workbench.'),
    ('Stage_Light', '', 'The spotlight tracks across the floor.', 'The stage edge remains illuminated.'),
    ('Conveyor', 'Parcel', 'The belt carries the parcel forward.', 'The parcel approaches the scanner.'),
    ('attacker_Lamp', '', 'The lamp flickers beside the doorway.', 'Light alternates on the wall.'),
    ('attacker', 'defender', 'Looks toward the colleague.', 'Recognition becomes visible.'),
    ('defender', '', 'Practices footwork alone.', 'The performer faces the doorway.'),
])
def test_mixed_independent_subjects_preserved_across_full_pipeline(tmp_path, subject, target, action, result):
    from test_scene_target_flow import object_story
    from test_action_authoring import plan_for
    raw = object_story('Panel', 'presses the control switch', 'indicator lit')
    plan = plan_for()
    for slot in plan['slots']:
        slot['authoring_protocol'] = 'scene-units-v1'
        slot['choreography'] = 'flexible'
    for shot in raw['shots'][:-1]:
        shot['key_props'] += [s for s in (subject, target) if s and s not in ('attacker', 'defender')]
        shot['action_plan'][-1] = {'id': shot['action_plan'][-1]['id'],
            'kind': 'performance', 'actor': subject, 'target': target,
            'action': action, 'result': result, 'weight': .1}
    original = deepcopy(raw)
    lowered = compile_authored_storyboard(raw, plan)
    before = deepcopy(lowered)
    compiled = _compile_storyboard_contract(lowered, '16:9', '480p', 'cinematic', production_plan=plan)
    assert lowered == before and raw == original
    for shot in compiled['shots'][:-1]:
        beat = shot['action_beats'][-1]
        assert (beat['actor'], beat['target'], beat['action'], beat['visible_result']) == (subject, target, action, result)
    assert_round_trip(compiled, tmp_path)


@pytest.mark.parametrize('field,value', [('actor', 'unregistered'), ('target', 'unregistered')])
def test_invalid_authored_participant_is_rejected_without_inheritance(field, value):
    record, raw = captured()
    raw['shots'][1]['action_plan'][-1][field] = value
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='未注册的动作参与者'):
        compile_authored_storyboard(raw, record['production_plan'])
    assert raw == before


@pytest.mark.parametrize('field,value', [('actor', 'AuntLin'), ('target', 'plain_white_ceramic_water_cup'),
                                       ('action', 'invented action'), ('visible_result', 'invented result')])
def test_future_lowering_mutations_fail_at_compilation_boundary(monkeypatch, field, value):
    import pipeline.storyboard as module
    record, raw = captured()
    lowered = compile_authored_storyboard(raw, record['production_plan'])
    before = deepcopy(lowered)
    defaults = module._apply_defaults
    def corrupt(story, *args):
        defaults(story, *args)
        story['shots'][1]['action_beats'][-1][field] = value
    monkeypatch.setattr(module, '_apply_defaults', corrupt)
    with pytest.raises(ValueError, match=r'改写已确定的动作语义: shots\[1\].action_beats'):
        module._compile_storyboard_contract(lowered, **record['format'], production_plan=record['production_plan'])
    assert lowered == before


def test_original_legacy_record_is_not_migrated_or_given_new_review_identity():
    from pipeline.causality import compile_action_contract
    record, _ = captured()
    # 同一份已失败的历史数据继续按旧合同解释，不能因升级被偷偷改判。
    original = record['compiled_text'].encode()
    assert hashlib.sha256(original).hexdigest() == record['compiled_sha256']
    legacy = json.loads(original)
    before = deepcopy(legacy)
    fingerprints = [compile_action_contract(s).fingerprint() for s in legacy['shots']]
    _apply_coverage_defaults(legacy['shots'])
    assert legacy == before
    assert [compile_action_contract(s).fingerprint() for s in legacy['shots']] == fingerprints


def test_explicit_event_geometry_reaches_child_review_without_legacy_rewrite():
    from pipeline.causality import compile_action_contract
    from pipeline.action_timing import sequence_events, event_shot
    _, _, story = compile_record()
    for shot in story['shots']:
        for event in sequence_events(shot):
            child = {**event_shot(event), 'action_semantics': shot['action_semantics']}
            contract = compile_action_contract(child)
            assert contract.effect_motion == event['interaction_geometry']['effect_motion']
            assert contract.review_projection['interaction_geometry'] == event['interaction_geometry']
            assert contract.review_projection['action_semantics'] == 'authored-v1'
            legacy = compile_action_contract(event_shot(event))
            assert legacy.fingerprint() != contract.fingerprint()

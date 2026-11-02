"""A continuation model must not echo or edit program-owned result states."""
from copy import deepcopy
import json

import pytest

from pipeline.action_results import required_outcomes
from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
from pipeline.performance import outcome_text
from test_action_results import scene_shot
from test_continuation import scenario, Planner
from test_director_intent import Responses


def test_multiple_long_results_are_preserved_without_model_echo_and_resume(tmp_path):
    _, previous, tail = scenario(tmp_path)
    shot = scene_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    shot['action_beats'][0]['outcomes'][0]['state'] = 'ready ' * 200
    shot['action_beats'][0]['scene_changes'][0]['state'] = 'open ' * 240
    original = deepcopy(shot)
    required = outcome_text(required_outcomes(shot['action_beats'][0]))
    assert len(required) > 1600
    client = Responses({'compatible': True, 'entry_action': 'continues the existing movement', 'reason': 'entry remains compatible'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert result['action_beats'][0]['visible_result'] == required
    assert result['action_beats'][0]['outcomes'] == shot['action_beats'][0]['outcomes']
    assert result['action_beats'][0]['scene_changes'] == shot['action_beats'][0]['scene_changes']
    assert shot == original
    schema = client.calls[0]['response_format']['json_schema']['schema']
    assert set(schema['required']) == {'compatible', 'entry_action', 'reason'}
    assert 'entry_visible_result' not in schema['properties']
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False) == result
    assert len(client.calls) == 1
    cached = json.loads((tmp_path / 'continuations/shot_002.json').read_text())
    assert cached['version'] == 'observed-entry-v5-bound-results'
    assert 'entry_visible_result' not in cached['plan']


def test_result_override_is_rejected_not_silently_discarded(tmp_path):
    _, previous, tail = scenario(tmp_path)
    shot = scene_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    with pytest.raises(ContinuationUnavailableError, match='结构无效'):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=Planner(entry_visible_result='Door remains closed'))
    assert not (tmp_path / 'continuations/shot_002.json').exists()


def test_reviewed_entry_path_cannot_overwrite_action_and_participates_in_provenance(tmp_path):
    from pipeline.generator import VideoGenerator
    from pipeline.staging_direction import staging_projection
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    _, previous, tail = scenario(tmp_path)
    shot = scene_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless',
                result_admission='reviewed-results-v1', generation_direction='authored-staging-v1')
    original = deepcopy(shot)
    path = 'continues the existing movement through a shorter approach'
    client = Responses({'compatible': True, 'entry_action': path, 'reason': 'offline compatible path'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert shot == original
    assert result['action_beats'] == shot['action_beats']
    assert result['observed_entry_path'] == path
    prompt = compile_normal_provider_prompt(result, {}, has_observed_start=True)
    assert path in prompt and shot['action_beats'][0]['action'] in prompt
    safe = compile_policy_safe_prompt(result, has_state_reference=True,
                                     image_role='first_frame', reference_count=1)
    assert path not in safe
    assert 'observed_entry_path' in staging_projection(result, profile='policy_safe')['omitted_fields']
    g = object.__new__(VideoGenerator)
    g.output_dir, g.character_refs = tmp_path, {}
    changed = deepcopy(result)
    changed['observed_entry_path'] = 'approaches around the outside of the existing contact'
    assert g._generation_provenance(result, [], None) != g._generation_provenance(changed, [], None)
    assert g._acceptance_context(result, [], None) != g._acceptance_context(changed, [], None)
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False) == result
    assert len(client.calls) == 1


def test_legacy_reviewed_entry_cache_keeps_original_request_semantics(tmp_path):
    import hashlib
    from pipeline.continuation import LEGACY_REVIEWED_VERSION
    _, previous, tail = scenario(tmp_path)
    shot = scene_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless',
                result_admission='reviewed-results-v1', generation_direction='authored-staging-v1')
    client = Responses({'compatible': True, 'entry_action': 'legacy adapted interaction',
                        'reason': 'offline cached annotation'})
    prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    path = tmp_path / 'continuations/shot_002.json'
    cached = json.loads(path.read_text())
    cached['version'] = cached['input']['version'] = LEGACY_REVIEWED_VERSION
    cached['input_fingerprint'] = hashlib.sha256(json.dumps(
        cached['input'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cached['execution_shot'].pop('observed_entry_path')
    cached['execution_shot']['action_beats'][0]['action'] = cached['plan']['entry_action']
    path.write_text(json.dumps(cached))
    saved = path.read_bytes()
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False)
    assert result['action_beats'][0]['action'] == 'legacy adapted interaction'
    assert 'observed_entry_path' not in result
    assert len(client.calls) == 1 and path.read_bytes() == saved


@pytest.mark.parametrize('invalid', ['', ' \t', 42, None, 'x' * 1601])
def test_entry_path_is_validated_at_generation_projection(invalid):
    from pipeline.staging_direction import staging_projection
    shot = scene_shot()
    shot.update(generation_direction='authored-staging-v1', observed_entry_path=invalid)
    with pytest.raises(ValueError, match='续拍路径'):
        staging_projection(shot)

import json
from copy import deepcopy

import pytest

from pipeline.scene_binding import apply_scene_bindings, resolve_scene_bindings
from pipeline.storyboard import _should_use_previous_tail_reference
from test_planning_semantics import compiled
from test_director_intent import Responses


def locations():
    story = compiled()
    for i, shot in enumerate(story['shots']):
        shot['scene_id'] = f'roof_shot_{i + 1}'
    return story


def review(anchors):
    return {'bindings': [{'shot_id': i, 'anchor_shot_id': anchor, 'certain': True,
                         'reason': 'fixture: shared roof or an explicit location change'}
                        for i, anchor in enumerate(anchors, 1)]}


def test_binding_restores_tail_reference_without_rewriting_actions_or_first_shot():
    original = locations()
    before = deepcopy(original)
    bound = apply_scene_bindings(original, review([1, 1, 1]))
    assert original == before and bound['shots'][0] == original['shots'][0]
    for old, new in zip(original['shots'], bound['shots']):
        assert {k: v for k, v in old.items() if k != 'scene_id'} == {k: v for k, v in new.items() if k != 'scene_id'}
    assert _should_use_previous_tail_reference(bound['shots'][1], bound['shots'][0])
    assert not _should_use_previous_tail_reference(original['shots'][1], original['shots'][0])


def test_different_locations_and_return_to_earlier_location_remain_distinct():
    bound = apply_scene_bindings(locations(), review([1, 2, 1]))
    assert [s['scene_id'] for s in bound['shots']] == ['roof_shot_1', 'roof_shot_2', 'roof_shot_1']
    assert not _should_use_previous_tail_reference(bound['shots'][1], bound['shots'][0])


@pytest.mark.parametrize('anchors', [[2, 2, 2], [1, 1, 2], [1, 2], [1, 2, 99]])
def test_invalid_binding_does_not_mutate_input(anchors):
    story = locations()
    before = deepcopy(story)
    with pytest.raises(ValueError):
        apply_scene_bindings(story, review(anchors))
    assert story == before


def test_cached_binding_preserves_recovery_and_stops_stale_or_locked_rebinding(tmp_path, monkeypatch):
    story = locations()
    client = Responses(review([1, 1, 1]))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    save = lambda name, value: (tmp_path / name).write_text(json.dumps(value))
    # Existing shot 1 keeps its exact context, so a pending task can still be polled.
    bound = resolve_scene_bindings(story, tmp_path, save, locked_shot_ids={1})
    assert resolve_scene_bindings(story, tmp_path, save, locked_shot_ids={1, 2}) == bound
    assert len(client.calls) == 1
    changed = deepcopy(story)
    changed['shots'][1]['scene_description'] += ' changed'
    with pytest.raises(ValueError, match='不一致'):
        resolve_scene_bindings(changed, tmp_path, save)
    (tmp_path / 'scene_binding_review.json').unlink()
    # Lost application metadata cannot turn an existing response into a fresh call.
    with pytest.raises(ValueError, match='不能隐式重试'):
        resolve_scene_bindings(story, tmp_path, save, locked_shot_ids={2})
    assert not (tmp_path / 'scene_binding_review.json').exists()
    assert len(client.calls) == 1


def test_uncertain_location_stops(tmp_path, monkeypatch):
    raw = review([1, 1, 1])
    raw['bindings'][1]['certain'] = False
    client = Responses(raw)
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    save = lambda name, value: (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match='不确定'):
        resolve_scene_bindings(locations(), tmp_path, save)
    assert not (tmp_path / 'execution_storyboard.json').exists()
    assert json.loads((tmp_path / 'scene_binding_review.json').read_text())['verdict'] == raw

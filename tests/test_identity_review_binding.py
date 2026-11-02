import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pipeline.generator import VideoGenerator
from pipeline.semantic_review import SemanticReviewUnavailableError
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames
from test_visual_assets import assets
from pipeline.visual_assets import prepare_identity_assets


def test_review_labels_each_actual_identity_image_and_invalidates_rebinding(tmp_path):
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    refs = [tmp_path / 'a.jpg', tmp_path / 'b.jpg']
    for i, path in enumerate(refs):
        path.write_bytes(f'reference {i}'.encode())
    args = dict(identity_reference_paths=[str(p) for p in refs],
                identity_reference_entities=['defender', 'attacker'])
    assert reviewer.review(str(video), outcome_shot(), **args).accepted
    content = sdk.calls[0]['messages'][1]['content']
    labels = [item['text'] for item in content if item['type'] == 'text']
    assert 'Canonical identity reference 1 for character "defender":' in labels
    assert 'Canonical identity reference 2 for character "attacker":' in labels
    reviewer.review(str(video), outcome_shot(), **args)
    assert len(sdk.calls) == 1
    args['identity_reference_entities'].reverse()
    reviewer.review(str(video), outcome_shot(), **args)
    assert len(sdk.calls) == 2


@pytest.mark.parametrize('names', [[], ['attacker', 'defender'], [''], [True]])
def test_bad_reference_bindings_stop_before_review(tmp_path, names):
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    ref = tmp_path / 'ref.jpg'
    ref.write_bytes(b'identity')
    with pytest.raises(SemanticReviewUnavailableError, match='绑定'):
        reviewer.review(str(video), outcome_shot(), identity_reference_paths=[str(ref)],
                        identity_reference_entities=names)
    assert not sdk.calls


@pytest.mark.asyncio
async def test_state_reference_generation_and_identity_review_have_distinct_provenance(tmp_path):
    story, _ = assets(tmp_path)
    refs = prepare_identity_assets(tmp_path, story)
    generator = VideoGenerator(str(tmp_path))
    generator.character_refs.update(refs)
    previous = deepcopy(story['shots'][0])
    current = {**previous, 'shot_id': 2, 'continuity_from_previous': 'seamless'}
    tail = str(tmp_path / 'tail.jpg')
    inputs, role = generator._build_image_refs(current, tail, prev_shot=previous)
    assert inputs == [tail] and role == 'first_frame'
    generation = generator._generation_provenance(current, inputs, role)
    assert generation['identity_bindings'] == []
    first = generator._generation_provenance(previous, [refs['defender'], refs['attacker']], 'reference_image')
    assert first['identity_bindings'] == [
        {'image_index': 1, 'character': 'defender'}, {'image_index': 2, 'character': 'attacker'}]
    calls = []
    def review(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(accepted=True)
    generator.semantic_reviewer = SimpleNamespace(review=review)
    await generator._review_take('offline.mp4', current, previous_frame_path=tail,
                                 previous_shot=previous, storyboard=story)
    assert calls[0]['identity_reference_entities'] == current['characters']
    assert calls[0]['identity_reference_paths'] == [refs[name] for name in current['characters']]


@pytest.mark.asyncio
async def test_identity_review_does_not_silently_truncate_cast_to_two(tmp_path):
    generator = VideoGenerator(str(tmp_path))
    names = ['Alice', 'Bo', 'Chen']
    generator.character_refs = {name: f'{name}.jpg' for name in names}
    calls = []
    def review(*args, **kwargs):
        calls.append(kwargs)
    generator.semantic_reviewer = SimpleNamespace(review=review)
    await generator._review_take('offline.mp4', {'shot_id': 1, 'characters': names},
        storyboard={'characters': [{'name': name} for name in names]})
    assert calls[0]['identity_reference_entities'] == names
    assert len(calls[0]['identity_reference_paths']) == 3

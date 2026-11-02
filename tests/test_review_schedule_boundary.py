"""Observation requests must not supply the timestamps they ask vision to measure."""
from copy import deepcopy
import json

from pipeline.semantic_review import _review_contract
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


def test_outcome_observation_is_independent_of_suggested_schedule(tmp_path):
    shot = outcome_shot()
    original = deepcopy(shot)
    expected = _review_contract(shot)
    shifted = deepcopy(shot)
    for beat in shifted['action_beats']:
        for key in ('effect_at_seconds', 'reaction_at_seconds'):
            if key in beat:
                beat[key] += .01
    assert _review_contract(shifted) == expected
    assert shot == original
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    assert reviewer.review(str(video), shot).accepted
    prompt = sdk.calls[0]['messages'][1]['content'][0]['text']
    for key in ('effect_at_seconds', 'reaction_at_seconds', '"action_timing"'):
        assert key not in prompt
    assert 'min_interaction_seconds' in prompt and 'result_hold_seconds' in prompt
    assert any(item.get('text', '').startswith('Current ') for item in sdk.calls[0]['messages'][1]['content'])


def test_exact_choreography_retains_schedule_authority():
    shot = outcome_shot()
    shot['production_slot']['choreography'] = 'exact'
    contract = _review_contract(shot)
    assert 'action_timing' in contract
    assert 'effect_at_seconds' in json.dumps(contract)

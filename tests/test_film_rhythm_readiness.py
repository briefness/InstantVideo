import pytest

from pipeline.film_rhythm import planning_rhythm_context
from pipeline.production_plan import build_production_plan
from tools.prepare_authored_run import prepare_authored_run
from test_authored_run import source


def test_imported_flexible_script_without_outcome_evidence_contract_stops_before_workspace(tmp_path):
    authored = source()
    authored['intent']['choreography'] = 'flexible'
    for shot in authored['storyboard']['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
    with pytest.raises(ValueError, match='独立结果'):
        prepare_authored_run(authored, tmp_path, paid_take_budget=3)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('focus,pacing', [('product', 'balanced'), ('balanced', 'balanced'),
                                        ('action', 'atmospheric')])
def test_non_dense_intents_do_not_acquire_a_film_interaction_quota(focus, pacing):
    intent = dict(source_request='offline brief', intent_evidence='offline brief',
                  content_focus=focus, pacing=pacing, choreography='flexible')
    plan = build_production_plan(focus, 30, intent=intent)
    assert plan.get('min_interaction_seconds') is None
    assert planning_rhythm_context(plan, []) is None


def test_impossible_remaining_budget_stops_before_next_authoring_call():
    plan = {'min_interaction_seconds': 11, 'slots': [{'duration': 5}, {'duration': 5, 'result_hold_seconds': .5}]}
    with pytest.raises(ValueError, match='剩余镜头'):
        planning_rhythm_context(plan, [])

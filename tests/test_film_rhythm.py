import pytest

from pipeline.production_plan import build_production_plan
from pipeline.run_state import RunWorkspace, PaidTakeBudgetExhaustedError
from pipeline.models import RunOptions


@pytest.mark.parametrize('focus,brief,duration', [
    ('action', '人物连续追逐，开头入场，结尾解救', 15),
    ('action', '机械臂连续装配，开头定位，结尾展示成品', 30),
    ('action', '连续展示协作实验，开头准备，结尾观察结果', 60),
])
def test_flexible_density_belongs_to_film_not_every_shot(focus, brief, duration):
    intent = dict(source_request=brief, intent_evidence=brief, content_focus=focus,
                  pacing='dense', choreography='flexible')
    plan = build_production_plan(focus, duration, intent=intent)
    assert plan['version'] == 'production-plan-v8-request-shape'
    assert plan['min_interaction_seconds'] == pytest.approx(plan['planned_duration'] * .7)
    assert all(s['min_interaction_seconds'] == 0 for s in plan['slots'])
    old = build_production_plan(focus, duration, intent=intent, contract_version='production-plan-v6-performance')
    assert 'min_interaction_seconds' not in old
    assert sum(s['min_interaction_seconds'] for s in old['slots']) == pytest.approx(plan['min_interaction_seconds'])


def workspace(tmp_path, *, cap=3, policy='completion_first'):
    w = RunWorkspace.create(tmp_path, RunOptions(request='offline', paid_take_budget=cap, take_policy=policy))
    w.save_storyboard({'title': 'offline', 'mood': 'cinematic', 'shots': [
        {'shot_id': i, 'duration': 5, 'scene_description': 'workshop', 'prompt_en': 'test'} for i in (1, 2, 3)]})
    return w


def submit(w, shot):
    token = w.reserve_paid_take(shot)
    w.confirm_paid_take_submission(token, 'offline-' + token)
    w.reconcile_paid_take(token, 'offline-' + token)
    return token


def test_retry_cannot_consume_unsubmitted_shots_first_opportunities(tmp_path):
    w = workspace(tmp_path)
    submit(w, 1)
    with pytest.raises(PaidTakeBudgetExhaustedError, match='后续'):
        w.reserve_paid_take(1)
    assert len(w.manifest.paid_take_budget.reservations) == 1
    submit(w, 2)
    submit(w, 3)


def test_single_pass_survives_resume_even_with_extra_total_budget(tmp_path):
    w = workspace(tmp_path, cap=10, policy='single_pass')
    submit(w, 1)
    w = RunWorkspace.resume(w.path)
    with pytest.raises(PaidTakeBudgetExhaustedError, match='一次'):
        w.reserve_paid_take(1)

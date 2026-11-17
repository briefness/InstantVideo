"""One durable submission ledger across entrypoints, failures and concurrent resumes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from pipeline.models import RunOptions
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace, PaidTakeBudgetExhaustedError
from test_film_rhythm import workspace, submit


def test_insufficient_budget_stops_before_first_submission(tmp_path):
    w = workspace(tmp_path, cap=2)
    with pytest.raises(PaidTakeBudgetExhaustedError, match='首次机会'):
        w.reserve_paid_take(1)
    assert not RunWorkspace.resume(w.path).manifest.paid_take_budget.reservations


@pytest.mark.parametrize('policy', ['completion_first', 'single_pass'])
def test_released_unsubmitted_reservation_does_not_consume_opportunity(tmp_path, policy):
    w = workspace(tmp_path, policy=policy)
    token = w.reserve_paid_take(1)
    w.release_unsubmitted_paid_take(token)
    w = RunWorkspace.resume(w.path)
    if policy == 'single_pass':
        # 金额预留可释放，单次提交许可不可由释放操作重新授权。
        with pytest.raises(PaidTakeBudgetExhaustedError):
            submit(w, 1)
    else:
        submit(w, 1)
    submit(w, 2)
    submit(w, 3)
    assert sum(r.status != 'released' for r in w.manifest.paid_take_budget.reservations) == (
        2 if policy == 'single_pass' else 3)


@pytest.mark.parametrize('policy', ['completion_first', 'single_pass'])
def test_unconfirmed_submission_still_consumes_opportunity_after_crash(tmp_path, policy):
    w = workspace(tmp_path, policy=policy)
    w.reserve_paid_take(1)
    restored = VideoPipeline.from_workspace(w.path)
    assert restored.take_policy == policy
    with pytest.raises(PaidTakeBudgetExhaustedError):
        restored.run_workspace.reserve_paid_take(1)
    assert len(restored.run_workspace.manifest.paid_take_budget.reservations) == 1


def test_concurrent_retries_can_only_spend_surplus_not_other_shots(tmp_path):
    w = workspace(tmp_path, cap=4)
    submit(w, 1)
    barrier = Barrier(4)
    workers = [RunWorkspace.resume(w.path) for _ in range(4)]
    def reserve(worker):
        barrier.wait(timeout=5)
        try:
            return worker.reserve_paid_take(1)
        except PaidTakeBudgetExhaustedError:
            return 'blocked'
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, workers))
    assert results.count('blocked') == 3
    w = RunWorkspace.resume(w.path)
    submit(w, 2)
    submit(w, 3)
    assert len(w.manifest.paid_take_budget.reservations) == 4


def test_old_options_keep_old_policy_and_new_entrypoints_freeze_new_policy(tmp_path):
    from web_server import _parse_create_payload
    from test_authored_run import source
    from tools.prepare_authored_run import prepare_authored_run
    assert RunOptions.model_validate({'request': 'old workspace'}).take_policy == 'legacy'
    assert VideoPipeline().take_policy == 'completion_first'
    assert _parse_create_payload({'request': 'new web run'}).take_policy == 'completion_first'
    assert _parse_create_payload({'request': 'new web run', 'take_policy': 'single_pass'}).take_policy == 'single_pass'
    w = prepare_authored_run(source(), tmp_path)
    assert w.options.take_policy == 'completion_first'
    assert VideoPipeline.from_workspace(w.path).take_policy == 'completion_first'
    old = workspace(tmp_path, policy='legacy')
    submit(old, 1)
    submit(old, 1)
    submit(old, 1)


@pytest.mark.parametrize('policy', ['legacy', None, True, {}, 'unlimited'])
def test_web_cannot_select_unknown_or_legacy_submission_policy(policy):
    from web_server import _parse_create_payload
    with pytest.raises(ValueError, match='生成策略'):
        _parse_create_payload({'request': 'offline', 'take_policy': policy})


@pytest.mark.parametrize('policy', ['completion_first', 'single_pass'])
def test_unplanned_shot_cannot_reserve_budget(tmp_path, policy):
    w = workspace(tmp_path, policy=policy)
    with pytest.raises(ValueError, match='计划'):
        w.reserve_paid_take(99)
    assert not w.manifest.paid_take_budget.reservations

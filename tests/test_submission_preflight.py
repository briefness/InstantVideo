"""A stop policy must stop scheduling, not just reject a late reservation."""
from dataclasses import asdict
from pathlib import Path

import pytest

from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace, PaidTakeBudgetExhaustedError
from test_action_sequence import sequence_story
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames
from test_film_rhythm import workspace, submit


def test_preflight_is_read_only_and_final_reservation_rechecks(tmp_path):
    w = workspace(tmp_path, cap=3, policy='single_pass')
    before = (w.path / 'run_manifest.json').read_bytes()
    w.check_paid_take(1)
    assert (w.path / 'run_manifest.json').read_bytes() == before
    submit(RunWorkspace.resume(w.path), 1)
    for operation in (w.check_paid_take, w.reserve_paid_take):
        with pytest.raises(PaidTakeBudgetExhaustedError):
            operation(1)
    assert len(RunWorkspace.resume(w.path).manifest.paid_take_budget.reservations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('policy,cap', [('single_pass', 10), ('completion_first', 3)])
async def test_rejected_take_stops_before_retake_preparation(tmp_path, monkeypatch, capsys, policy, cap):
    story = sequence_story()
    story['shots'] = [{**outcome_shot(), 'shot_id': i} for i in (1, 2, 3)]
    for shot in story['shots']:
        shot['production_slot'].update(shot_id=shot['shot_id'], duration=5, allowed_effect_phases=['active'],
            requires_visible_result=True, coverage_roles=['interaction'], framing_family='wide',
            reference_policy='independent')
    w = RunWorkspace.create(tmp_path, RunOptions(request='offline', paid_take_budget=cap, take_policy=policy))
    w.save_storyboard(story)
    def progress(result):
        fields = asdict(result)
        fields.pop('character_ref_path')
        w.record_shot(**fields)
    generator = VideoGenerator(str(w.path), on_progress=progress,
        check_paid_take=w.check_paid_take,
        reserve_paid_take=w.reserve_paid_take, confirm_paid_take_submission=w.confirm_paid_take_submission,
        reconcile_paid_take=w.reconcile_paid_take, release_unsubmitted_paid_take=w.release_unsubmitted_paid_take)
    class API:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            assert self.calls == 1
            kwargs['on_submitted']('offline-original')
            return {'status': 'succeeded', 'provider_task_id': 'offline-original', 'video_url': 'offline'}
        async def download_video(self, url, destination):
            Path(destination).write_bytes(b'offline media')
    generator.api = API()
    response = outcome_response()
    response['action_event_evidence'][0]['effect_visible'] = False
    _, generator.semantic_reviewer, sdk = reviewer_for(w.path, response)
    monkeypatch.setattr('pipeline.generator.check_video_quality',
        lambda _: {'pass': True, 'quality_score': 100, 'issues': []})
    preparations = []
    original = generator._build_image_refs
    def refs(*args, **kwargs):
        preparations.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(generator, '_build_image_refs', refs)
    result = await generator._generate_single_shot(story['shots'][0], None, None, story)
    assert result.status == 'failed' and result.semantic_accepted is False
    assert result.provider_task_id == 'offline-original' and Path(result.local_path).is_file()
    assert generator.api.calls == len(sdk.calls) == 1
    assert '执行唯一一次定向重拍' not in capsys.readouterr().out
    assert any('语义验收不通过' in error for error in result.errors)
    assert len(RunWorkspace.resume(w.path).manifest.paid_take_budget.reservations) == 1
    source_model = result.model_used
    # Resume may inspect the preserved evidence but must not prepare a new prompt/reference set.
    preparations.clear()
    result = await generator._generate_single_shot(story['shots'][0], None, None, story)
    assert result.status == 'failed' and generator.api.calls == 1
    assert result.model_used == source_model and result.provider_task_id == 'offline-original'
    assert 'reused_local_take' in result.recovery_actions
    assert RunWorkspace.resume(w.path).manifest.shots['1'].take_history[0].model_used == source_model
    assert len(preparations) <= 1  # Initial provenance check only, no generation-loop preparation.

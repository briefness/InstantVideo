"""受理后的局部处理失败只能恢复同一任务，不能消耗第二次提交。"""
from dataclasses import asdict
from pathlib import Path
import socket

import pytest

from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from test_run_state import storyboard


class SubmittedVideo:
    supports_last_frame = True

    def __init__(self):
        self.creates = 0
        self.polls = []
        self.downloads = 0

    async def generate(self, **kwargs):
        self.creates += 1
        kwargs['on_submitted']('offline-authoritative-task')
        return self.result()

    def result(self):
        return {'status': 'succeeded', 'provider_task_id': 'offline-authoritative-task',
                'video_url': 'offline-footage'}

    async def poll_task(self, task_id, **kwargs):
        self.polls.append(task_id)
        return self.result()

    async def download_video(self, source, destination):
        self.downloads += 1
        Path(destination).write_bytes(b'immutable offline footage')


def generator_for(run, api, monkeypatch):
    def progress(result):
        fields = asdict(result)
        fields.pop('character_ref_path')
        run.record_shot(**fields)

    generator = VideoGenerator(str(run.path), on_progress=progress,
        reserve_paid_take=run.reserve_paid_take, check_paid_take=run.check_paid_take,
        confirm_paid_take_submission=run.confirm_paid_take_submission,
        reconcile_paid_take=run.reconcile_paid_take,
        release_unsubmitted_paid_take=run.release_unsubmitted_paid_take,
        resume_tasks=run.resumable_pending_tasks(), review_content=False)
    generator.api = api
    tail = run.path / 'shots/shot_001_lastframe.jpg'
    tail.write_bytes(b'offline tail')
    monkeypatch.setattr(generator, '_extract_local_tail_frame', lambda *_: str(tail))
    return generator


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['single_pass', 'completion_first'])
@pytest.mark.parametrize('failure', ['capture_output', 'acceptance_context', 'timeout', 'sensitive_error'])
async def test_post_submission_failure_preserves_task_and_resumes_materialization(
        tmp_path, monkeypatch, policy, failure):
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr('pipeline.generator.check_video_quality',
                        lambda _: {'pass': True, 'quality_score': 100, 'issues': []})
    # 有充足预算，确保无第二次提交来自生命周期边界，而不是碰巧被额度拦住。
    run = RunWorkspace.create(tmp_path, RunOptions(
        request='offline scene', paid_take_budget=10, take_policy=policy))
    story = run.save_storyboard(storyboard())
    api = SubmittedVideo()
    generator = generator_for(run, api, monkeypatch)

    def interrupted(*_, **__):
        if failure == 'sensitive_error':
            raise OSError('download https://example.invalid/video?signature=PRIVATE_TEST_SIGNATURE Authorization: PRIVATE_TEST_KEY')
        raise TimeoutError('local processing interrupted') if failure == 'timeout' else OSError('local disk full')

    with monkeypatch.context() as failed:
        if failure == 'acceptance_context':
            failed.setattr(generator, '_write_acceptance_context', interrupted)
        else:
            failed.setattr('pipeline.generator.capture_output', interrupted)
        result = await generator._generate_single_shot(story['shots'][0], None, None, story)

    assert result.status == 'running'
    assert not any('PRIVATE_TEST_' in error or 'https://' in error for error in result.errors)
    assert api.creates == 1 and not api.polls
    assert Path(result.local_path).is_file()
    run = RunWorkspace.resume(run.path)
    assert 'PRIVATE_TEST_' not in run.manifest_path.read_text()
    run.ensure_recoverable_submissions()
    pending = run.resumable_pending_tasks()[1]
    assert pending['task_id'] == 'offline-authoritative-task'
    assert pending['resume_from'] == 'provider_result'
    assert len(run.manifest.paid_take_budget.reservations) == 1
    assert run.manifest.paid_take_budget.reservations[0].status == 'reconciled'

    resumed = generator_for(run, api, monkeypatch)
    result = await resumed._generate_single_shot(story['shots'][0], None, None, story)
    assert result.status == 'success'
    assert result.provider_task_id == 'offline-authoritative-task'
    assert api.creates == 1 and api.polls == ['offline-authoritative-task']
    assert api.downloads == 2  # Existing but unfinished bytes cannot bypass source materialization.
    assert len(RunWorkspace.resume(run.path).manifest.paid_take_budget.reservations) == 1

from dataclasses import asdict
from pathlib import Path

import pytest

from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from test_action_sequence import sequence_story
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


@pytest.mark.asyncio
@pytest.mark.parametrize('policy,cap', [('completion_first', 3), ('single_pass', 10)])
async def test_semantic_retake_cannot_bypass_frozen_submission_strategy(tmp_path, monkeypatch, policy, cap):
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
        reserve_paid_take=w.reserve_paid_take, confirm_paid_take_submission=w.confirm_paid_take_submission,
        reconcile_paid_take=w.reconcile_paid_take, release_unsubmitted_paid_take=w.release_unsubmitted_paid_take)
    class API:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            assert self.calls == 1, 'a blocked retake reached the provider'
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
    async def no_wait(*args):
        pass
    monkeypatch.setattr('pipeline.generator.asyncio.sleep', no_wait)
    result = await generator._generate_single_shot(story['shots'][0], None, None, story)
    assert result.status == 'failed'
    assert generator.api.calls == 1 and len(sdk.calls) == 1
    restored = RunWorkspace.resume(w.path)
    assert len(restored.manifest.paid_take_budget.reservations) == 1
    assert restored.manifest.shots['1'].provider_task_id == 'offline-original'
    assert restored.reserve_paid_take(2)
    assert restored.reserve_paid_take(3)

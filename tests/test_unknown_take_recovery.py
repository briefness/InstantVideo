from dataclasses import asdict
from pathlib import Path

import pytest

from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SemanticReviewUnavailableError
from test_action_sequence import sequence_story
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


@pytest.mark.asyncio
async def test_unknown_identity_preserves_original_submission_even_when_retries_are_allowed(tmp_path, monkeypatch):
    story = sequence_story()
    shot = outcome_shot()
    shot['production_slot'].update(shot_id=1, duration=5, allowed_effect_phases=['active'],
        requires_visible_result=True, coverage_roles=['interaction'], framing_family='wide',
        reference_policy='independent')
    story['shots'] = [shot]
    w = RunWorkspace.create(tmp_path, RunOptions(request='offline', paid_take_budget=10))
    w.save_storyboard(story)
    def progress(result):
        fields = asdict(result)
        fields.pop('character_ref_path')
        w.record_shot(**fields)
    generator = VideoGenerator(str(w.path), on_progress=progress,
        reserve_paid_take=w.reserve_paid_take, confirm_paid_take_submission=w.confirm_paid_take_submission,
        reconcile_paid_take=w.reconcile_paid_take)
    class API:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            assert self.calls == 1, 'unknown identity triggered a new video'
            kwargs['on_submitted']('offline-original')
            return {'status': 'succeeded', 'provider_task_id': 'offline-original', 'video_url': 'offline'}
        async def download_video(self, url, destination):
            Path(destination).write_bytes(b'offline media')
    generator.api = API()
    response = outcome_response()
    response['identity_continuity_valid'] = None
    _, generator.semantic_reviewer, sdk = reviewer_for(w.path, response)
    monkeypatch.setattr('pipeline.generator.check_video_quality',
        lambda _: {'pass': True, 'quality_score': 100, 'issues': []})
    with pytest.raises(SemanticReviewUnavailableError, match='观察不足'):
        await generator._generate_single_shot(shot, None, None, story)
    assert generator.api.calls == len(sdk.calls) == 1
    restored = RunWorkspace.resume(w.path)
    assert len(restored.manifest.paid_take_budget.reservations) == 1
    assert restored.manifest.shots['1'].provider_task_id == 'offline-original'
    assert (w.path / 'shots/shot_001.mp4').is_file()
    assert not list((w.path / 'shots').glob('*rejected*'))


@pytest.mark.parametrize('field', ['identity_continuity_valid', 'boundary_continuity_valid'])
def test_missing_live_judgment_is_not_the_legacy_default(tmp_path, field):
    response = outcome_response()
    response.pop(field)
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    kwargs = {}
    if field == 'boundary_continuity_valid':
        tail = tmp_path / 'previous.jpg'
        tail.write_bytes(b'accepted prior frame')
        kwargs['previous_frame_path'] = str(tail)
        response.update(environment_continuity_valid=True, action_handoff_valid=True,
                        screen_direction_valid=True, prop_continuity_valid=True)
    with pytest.raises(SemanticReviewUnavailableError, match='无效结构'):
        reviewer.review(str(video), outcome_shot(), **kwargs)
    assert field in sdk.calls[0]['response_format']['json_schema']['schema']['required']
    assert len(sdk.calls) == 2

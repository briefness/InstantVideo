"""Formal orchestration, synthetic SDK observations and real local FFmpeg export."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from pipeline.causality import compile_action_contract
from pipeline.generator import ShotResult
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SemanticReviewUnavailableError, SemanticTakeReviewer
from tools import ffmpeg_ops
from authoring_cases import authoring_from_legacy
from test_director_intent import Responses, intent_reply, draft
from test_director_export import clip
from test_performance_outcomes import outcome_response
from test_planning_semantics import verdict_for
from test_sequence_review import LocalCompletions, offline_frames


@pytest.mark.asyncio
@pytest.mark.parametrize('assessment', ['passed', 'failed', 'uncertain'])
async def test_formal_flow_exports_only_when_observed_film_rhythm_passes(tmp_path, monkeypatch, assessment):
    brief = '15秒连贯的训练交流，三段，交互密集且自然'
    authored = authoring_from_legacy(draft())
    authored['scene_entities'] = []
    authored['locations'] = [{'id': 'training_roof', 'description': 'training roof'}]
    for shot in authored['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.pop('carry', None)
                unit['link_previous'] = False  # Explicit synthetic new-protocol response.
                unit.update(weight=3., resolution={'target_state': unit['result'], 'actor_state': None, 'scene_changes': []})
    author = Responses({**intent_reply(brief), 'choreography': 'flexible'}, authored)
    class PlanningReview:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            story = json.loads(kwargs['messages'][-1]['content'])['storyboard']
            verdict = verdict_for(story)
            for unit in verdict['units']:
                beat = story['shots'][unit['shot_id'] - 1]['action_beats'][unit['beat_index']]
                if beat.get('outcomes'):
                    unit.update(outcome_contract_valid=True, evidence_fields=['outcomes'])
            return Responses(verdict).create(**kwargs)
    media = []
    class Generated:
        def __init__(self, directory, *, on_progress, **kwargs):
            self.directory, self.progress = Path(directory), on_progress
        async def generate_all(self, story):
            results = []
            for shot in story['shots']:
                i = shot['shot_id']
                video = self.directory / f'raw_{i}.mp4'
                # Different source durations make independently generated media bytes.
                clip(video, 5.1 + i / 24)
                media.append(video)
                response = outcome_response()
                contract = compile_action_contract(shot)
                for event, expected in zip(response['action_event_evidence'], contract.review_projection['action_events']):
                    event['event_id'] = expected['event_id']
                if assessment == 'failed':
                    for event in response['action_event_evidence']:
                        start = event['observed_start_seconds']
                        event['active_intervals'] = [{'start_seconds': start, 'end_seconds': start + .5}]
                elif assessment == 'uncertain':
                    response['action_event_evidence'][-1]['active_intervals'][-1]['end_seconds'] = 3.8
                sdk = LocalCompletions(response)
                reviewer = SemanticTakeReviewer(self.directory,
                    client=SimpleNamespace(chat=SimpleNamespace(completions=sdk)), model='offline-vision')
                assert reviewer.review(str(video), shot).accepted
                result = ShotResult(shot_id=i, status='success', local_path=str(video), semantic_accepted=True,
                    accepted_contract_fingerprint=contract.fingerprint())
                self.progress(result)
                results.append(result)
            return results
    async def no_voice(*args):
        return []
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no_luts')
    monkeypatch.setitem(config.SEEDANCE_OUTPUT_DIMENSIONS['480p'], '16:9', '320:180')
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: PlanningReview())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', Generated)
    monkeypatch.setattr('pipeline.orchestrator.synthesize_voiceover', no_voice)
    monkeypatch.setattr(VideoPipeline, '_auto_select_music', lambda *args: None)
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.export_for_platform',
                        lambda source, platform, dest: shutil.copyfile(source, dest))
    validate = ffmpeg_ops.validate_publish_ready
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.validate_publish_ready',
                        lambda *args, platform=None, **kwargs: validate(*args, **kwargs))
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=0,
                             take_policy='single_pass')
    if assessment == 'passed':
        output = await pipeline.run(brief)
        streams = ffmpeg_ops.get_video_info(output)['streams']
        video = next(s for s in streams if s['codec_type'] == 'video')
        assert int(video['nb_frames']) == 360
        assert float(video['duration']) == pytest.approx(15, abs=1 / 24)
        assert pipeline._transitions == [('cut', 0.), ('cut', 0.)]
        subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
        assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    else:
        error = RuntimeError if assessment == 'failed' else SemanticReviewUnavailableError
        with pytest.raises(error, match='不自动重拍'):
            await pipeline.run(brief)
        assert not (pipeline.workspace / 'concat.mp4').exists()
        assert not (pipeline.workspace / 'normalized').exists()
    assert len(media) == 3 and all(p.is_file() for p in media)
    assert len(author.calls) == 5
    saved = json.loads((pipeline.workspace / 'film_rhythm_review.json').read_text())
    assert saved['status'] == assessment
    workspace = RunWorkspace.resume(pipeline.workspace)
    assert workspace.options.take_policy == 'single_pass'
    assert not workspace.manifest.paid_take_budget.reservations

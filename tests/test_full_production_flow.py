"""Offline production traversal, replacing only external service boundaries.

Synthetic observations are protocol fixtures, not evidence of cinematic quality.
The planner, generator, reviewer, ledger, frame extraction and export stay real.
"""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import httpx
import pytest
import requests

import config
from pipeline.orchestrator import VideoPipeline
from pipeline.generator import RemoteTaskPendingError
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SemanticReviewUnavailableError
from test_action_result_flow import annotated_verdict
from test_action_results import annotated_story
from test_director_intent import Responses, intent_reply
from test_performance_outcomes import outcome_response
from tools import ffmpeg_ops


def completion(value):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=json.dumps(value), refusal=None))])


class ProductionServices:
    """Fixed remote responses; never generates or approves local pipeline state."""
    def __init__(self, sources, author, *, unknown_second=False):
        self.sources, self.author = sources, author
        self.unknown_second = unknown_second
        self.submissions, self.reviews, self.entries, self.planning = [], [], [], []
        self.chat = SimpleNamespace(completions=self)
        self.supports_last_frame = True

    def create(self, **kwargs):
        user = kwargs['messages'][-1]['content']
        if isinstance(user, list):
            text = user[0]['text']
            contract = json.JSONDecoder().raw_decode(text.split('Shot contract:\n', 1)[1])[0]
            self.reviews.append(contract)
            response = outcome_response()
            for event, expected in zip(response['action_event_evidence'], contract['action_events']):
                event['event_id'] = expected['event_id']
            response.update(
                identity_crop_boxes=[None, None], composition_change_valid=True,
                observed_end_state={'location': 'training roof', 'subject': 'attacker and defender',
                    'action_phase': 'guard yielded; both remain upright', 'prop_state': 'Door open; empty hands'},
                environment_continuity_valid=True, action_handoff_valid=True,
                screen_direction_valid=True, prop_continuity_valid=True,
                boundary_state_evidence={key: True for key in (
                    'prior_state_preserved', 'state_progress_not_reversed',
                    'open_motion_handoff_valid', 'persistent_entities_preserved',
                    'scene_identity_preserved')})
            if self.unknown_second and len(self.submissions) == 2:
                response['identity_continuity_valid'] = None
            return completion(response)
        context = json.loads(user)
        if 'storyboard' in context:
            self.planning.append(context)
            from planning_protocol_cases import wire_verdict
            return completion(wire_verdict(annotated_verdict(context['storyboard']), kwargs))
        if 'observed_state' in context:
            self.entries.append(context)
            result = {'compatible': True, 'entry_action': 'continues from the observed guard',
                      'reason': 'offline compatibility annotation'}
            if not context.get('entry_result_bound', context['version'] == 'observed-entry-v5-bound-results'):
                result['entry_visible_result'] = 'palms remain in range'
            return completion(result)
        return self.author.create(**kwargs)

    async def generate(self, **kwargs):
        self.submissions.append(kwargs)
        task = f'offline-{len(self.submissions)}'
        kwargs['on_submitted'](task)
        return {'status': 'succeeded', 'provider_task_id': task, 'video_url': task}

    async def download_video(self, url, destination):
        shutil.copyfile(self.sources[int(url.split('-')[-1]) - 1], destination)
        return destination


@pytest.fixture
def production_services(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline production test attempted network access')
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    brief = '15秒，三段连续的双人训练交流，动作密集自然'
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    for shot in raw['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
    author = Responses({**intent_reply(brief), 'choreography': 'flexible',
        'requested_duration_seconds': 15, 'duration_evidence': '15秒',
        'requested_shot_count': 3, 'shot_count_evidence': '三段'}, raw)
    sources = []
    for index in range(3):
        path = tmp_path / f'source-{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'testsrc2=s=1280x720:r=24:d={5.125 + index / 24}', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=6', '-vf', f'hue=h={index * 90}',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-shortest', str(path)], check=True)
        sources.append(path)
    services = ProductionServices(sources, author)
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **kwargs: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    # Optional presentation assets are absent; do not replace their implementations.
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    return brief, services


@pytest.mark.asyncio
async def test_brief_to_three_clips_to_final_and_completed_resume(tmp_path, production_services):
    brief, services = production_services
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert len(services.author.calls) == 5
    assert len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 3
    assert len(services.entries) == 2
    assert [call['image_role'] for call in services.submissions] == [None, 'first_frame', 'first_frame']
    for index, call in enumerate(services.submissions[1:], 1):
        assert Path(call['image_urls'][0]).is_file()
        assert services.entries[index - 1]['observed_state'] == {
            'location': 'training roof', 'subject': 'attacker and defender',
            'action_phase': 'guard yielded; both remain upright', 'prop_state': 'Door open; empty hands'}
    assert services.submissions[0]['model'] == config.SEEDANCE_MODEL
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert len(manifest.paid_take_budget.reservations) == 3
    assert manifest.status.value == 'succeeded'
    assert json.loads((pipeline.workspace / 'film_rhythm_review.json').read_text())['status'] == 'passed'
    video = next(s for s in ffmpeg_ops.get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360
    assert float(video['duration']) == pytest.approx(15, abs=1 / 24)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    counts = (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))


@pytest.mark.asyncio
async def test_uncertain_second_clip_resumes_same_media_without_resubmission(production_services):
    brief, services = production_services
    services.unknown_second = True
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(SemanticReviewUnavailableError, match='观察不足'):
        await pipeline.run(brief)
    assert len(services.submissions) == 2
    assert not (pipeline.workspace / 'final.mp4').exists()
    second = pipeline.workspace / 'shots/shot_002.mp4'
    before = second.read_bytes()
    # Reproduce the lifecycle boundary: observations alter the restored view,
    # even though no authored field or saved storyboard has changed.
    frozen = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    restored = RunWorkspace.resume(pipeline.workspace).load_storyboard()
    assert restored != frozen
    for shot in restored['shots']:
        shot.pop('observed_end_state', None)
    assert restored == frozen
    services.unknown_second = False
    output = await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert Path(output).is_file() and second.read_bytes() == before
    assert len(services.submissions) == 3
    assert len(services.author.calls) == 5
    assert len(services.entries) == 2
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 3


@pytest.mark.asyncio
async def test_pending_second_clip_polls_original_task_then_finishes(production_services, monkeypatch):
    brief, services = production_services
    generate = services.generate
    polls = []
    async def pending(**kwargs):
        result = await generate(**kwargs)
        if len(services.submissions) == 2:
            return {'status': 'failed', 'error_type': 'poll_timeout',
                    'provider_task_id': result['provider_task_id'], 'error': 'offline timeout'}
        return result
    async def poll(task_id, **kwargs):
        polls.append(task_id)
        return {'status': 'succeeded', 'provider_task_id': task_id, 'video_url': task_id}
    monkeypatch.setattr(services, 'generate', pending)
    monkeypatch.setattr(services, 'poll_task', poll, raising=False)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(RemoteTaskPendingError):
        await pipeline.run(brief)
    assert len(services.submissions) == 2 and not polls
    output = await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert Path(output).is_file()
    assert polls == ['offline-2'] and len(services.submissions) == 3
    assert len(services.author.calls) == 5 and len(services.entries) == 2
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('change_plan', [False, True])
async def test_export_failure_reuses_takes_but_never_ignores_changed_plan(
        production_services, monkeypatch, change_plan):
    brief, services = production_services
    export = ffmpeg_ops.export_for_platform
    def disk_failure(*args, **kwargs):
        raise OSError('offline simulated disk failure')
    monkeypatch.setattr(ffmpeg_ops, 'export_for_platform', disk_failure)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(OSError, match='disk failure'):
        await pipeline.run(brief)
    assert len(services.submissions) == 3
    frozen_path = pipeline.workspace / 'storyboard.json'
    original = frozen_path.read_bytes()
    if change_plan:
        changed = json.loads(original)
        changed['title'] = 'A different authored plan'
        frozen_path.write_text(json.dumps(changed))
    monkeypatch.setattr(ffmpeg_ops, 'export_for_platform', export)
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    if change_plan:
        with pytest.raises(ValueError, match='当前稿件或模型不一致'):
            await resumed.run()
    else:
        output = await resumed.run()
        assert Path(output).is_file()
        assert frozen_path.read_bytes() == original
        assert resumed.run_workspace.manifest.status.value == 'succeeded'
    assert len(services.submissions) == len(services.reviews) == 3
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert len(services.entries) == 2

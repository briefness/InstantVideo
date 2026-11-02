"""Variable event counts through the real pipeline, with explicit service fixtures.

Test-pattern videos and annotated observations validate transport and lifecycle,
not model accuracy or cinematic quality. Only external service boundaries change.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import httpx
import pytest
import requests

import config
from pipeline.orchestrator import VideoPipeline
from pipeline.performance import OBSERVED_TIMES
from pipeline.run_state import RunWorkspace
from test_action_result_flow import annotated_verdict
from test_creation_schedule_flow import SCENARIOS, scenario_story
from test_director_intent import Responses
from test_full_production_flow import ProductionServices, completion
from test_performance_outcomes import outcome_response
from planning_protocol_cases import annotated_outcome, wire_verdict
from tools.ffmpeg_ops import get_video_info


# Independent observation annotations, not timestamps copied from compiled plans.
OBSERVATIONS = [
    [(2, (0., .6, 1.1, 2.8)), (3, (2.4, 2.9, 3.3, 4.3))],
    [(1, (0., .4, .7, 1.9)), (2, (1.4, 1.8, 2.1, 3.2)), (3, (2.7, 3.2, 3.6, 4.3))],
    [(1, (0., 2., 3.5, 4.3))],
]


class ScheduledServices(ProductionServices):
    def __init__(self, sources, author, story, *, rejected_shot=None):
        super().__init__(sources, author)
        self.story = story
        self.rejected_shot = rejected_shot

    def create(self, **kwargs):
        user = kwargs['messages'][-1]['content']
        if isinstance(user, list):
            contract = json.JSONDecoder().raw_decode(user[0]['text'].split('Shot contract:\n', 1)[1])[0]
            index = len(self.submissions) - 1
            observed = OBSERVATIONS[index]
            assert [e['event_id'] for e in contract['action_events']] == [e[0] for e in observed]
            assert all(e['actor'] == 'Lead' and e['target'] == 'Partner' for e in contract['action_events'])
            self.reviews.append(contract)
            response = outcome_response()
            template = response['action_event_evidence'][0]
            response['action_event_evidence'] = []
            for event_id, times in observed:
                response['action_event_evidence'].append({
                    **deepcopy(template), 'event_id': event_id, **dict(zip(OBSERVED_TIMES, times)),
                    'flow_continuity_valid': True,
                    'active_intervals': [{'start_seconds': times[0], 'end_seconds': times[-1]}]})
            if index + 1 == self.rejected_shot:
                response['action_event_evidence'][-1]['result_visible'] = False
            response.update(
                identity_crop_boxes=[None, None], composition_change_valid=True,
                observed_end_state={**{k: self.story['shots'][index]['end_state'][k]
                    for k in ('location', 'subject', 'action_phase')}, 'prop_state': ''},
                environment_continuity_valid=True, action_handoff_valid=True,
                screen_direction_valid=True, prop_continuity_valid=True,
                boundary_state_evidence={key: True for key in (
                    'prior_state_preserved', 'state_progress_not_reversed',
                    'open_motion_handoff_valid', 'persistent_entities_preserved', 'scene_identity_preserved')})
            return completion(response)
        context = json.loads(user)
        if 'storyboard' in context:
            self.planning.append(context)
            story = context['storyboard']
            return completion(wire_verdict(annotated_outcome(annotated_verdict(story), story), kwargs))
        if 'observed_state' in context:
            self.entries.append(context)
            return completion({'compatible': True,
                'entry_action': 'continues the cooperative movement from the observed shared position',
                'reason': 'explicit offline compatibility annotation'})
        return self.author.create(**kwargs)


@pytest.fixture
def scheduled_services(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('offline scheduled production attempted external HTTP')
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    sources = []
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    for index in range(3):
        source = tmp_path / f'source-{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'testsrc2=s={dimensions}:r=24:d={5.125 + index / 24}', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=6', '-vf', f'hue=h={index * 90}',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-shortest', str(source)], check=True)
        sources.append(source)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')

    def prepare(topic, *, rejected_shot=None):
        brief, intent, story, _ = scenario_story(topic)
        services = ScheduledServices(sources, Responses(intent, story), story, rejected_shot=rejected_shot)
        for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
            monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
        monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
        return brief, services
    return prepare


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', SCENARIOS)
async def test_variable_work_reaches_final_export_and_resumes_without_services(scheduled_services, topic):
    brief, services = scheduled_services(topic)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert [len(r['action_events']) for r in services.reviews] == [2, 3, 1]
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert len(services.submissions) == 3 and len(services.entries) == 2
    assert all(s['model'] == config.SEEDANCE_MODEL for s in services.submissions)
    assert [s['image_role'] for s in services.submissions] == [None, 'first_frame', 'first_frame']
    for index, entry in enumerate(services.entries):
        assert entry['observed_state'] == {k: services.story['shots'][index]['end_state'][k]
            for k in ('location', 'subject', 'action_phase')}
    frozen = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    assert [s['production_slot']['min_interaction_events'] for s in frozen['shots']] == [2, 3, 1]
    assert frozen['shots'][-1]['production_slot']['result_hold_seconds'] == .5
    assert services.planning[0]['outcome_review']['requirement'] in brief
    rhythm = json.loads((pipeline.workspace / 'film_rhythm_review.json').read_text())
    assert rhythm['status'] == 'passed' and rhythm['interaction_seconds'][0] >= 10.5
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360 and float(video['duration']) == pytest.approx(15, abs=1 / 24)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    state = RunWorkspace.resume(pipeline.workspace).manifest
    assert state.status.value == 'succeeded' and len(state.paid_take_budget.reservations) == 3
    counts = [len(values) for values in (services.author.calls, services.planning, services.submissions,
                                        services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(values) for values in (services.author.calls, services.planning, services.submissions,
                                               services.reviews, services.entries)]


@pytest.mark.asyncio
@pytest.mark.parametrize('rejected_shot', [2, 3])
async def test_missing_middle_or_final_result_stops_without_extra_submissions(scheduled_services, rejected_shot):
    brief, services = scheduled_services('绳索救援', rejected_shot=rejected_shot)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(RuntimeError, match=f'Shot {rejected_shot} 生成失败'):
        await pipeline.run(brief)
    assert len(services.submissions) == len(services.reviews) == rejected_shot
    assert not (pipeline.workspace / 'final.mp4').exists()
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert len(manifest.paid_take_budget.reservations) == rejected_shot
    assert manifest.shots[str(rejected_shot)].canonical_take_id is None
    assert all(manifest.shots[str(i)].canonical_take_id for i in range(1, rejected_shot))
    counts = [len(values) for values in (services.author.calls, services.planning, services.submissions,
                                        services.reviews, services.entries)]
    with pytest.raises(RuntimeError):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert counts == [len(values) for values in (services.author.calls, services.planning, services.submissions,
                                               services.reviews, services.entries)]
    assert not (pipeline.workspace / 'final.mp4').exists()

"""原始创作响应内容保真重放到正式四段交付；审查标注和测试图均明确为离线替身。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_full_production_flow import ProductionServices, completion
from test_director_intent import Responses
from test_action_result_flow import annotated_verdict
from test_functional_results import annotated_wire
from planning_protocol_cases import annotated_outcome
from test_goal_allocation import annotation, captured, offline
from tools.ffmpeg_ops import get_video_info


class GoalServices(ProductionServices):
    def create(self, **kwargs):
        content = kwargs['messages'][-1]['content']
        if isinstance(content, list):
            contract = json.JSONDecoder().raw_decode(content[0]['text'].split('Shot contract:\n')[1])[0]
            self.reviews.append(contract)
            schema = kwargs['response_format']['json_schema']['schema']
            legacy = deepcopy(schema)
            legacy['properties'].pop('terminal_observation')
            legacy['properties']['end_state_sources'] = {'type': 'object', 'properties': {}}
            response = annotated_wire(legacy)
            response.pop('end_state_sources')
            count = schema['properties']['terminal_observation']['properties']['sample_index']['const']
            # 声明的替身标注只证明状态流和文件交付，不是对测试图的视觉判断。
            state = self.planning[0]['storyboard']['shots'][len(self.submissions) - 1]['end_state']
            response['terminal_observation'] = {'sample_index': count, 'status': 'observed',
                **{k: state.get(k, '') for k in ('action_phase', 'pose_and_gaze', 'prop_state', 'open_motion')}}
            response['end_state_context'] = {k: state.get(k, '') for k in response['end_state_context']}
            response['identity_crop_boxes'] = [None, None]
            return completion(response)
        context = json.loads(content)
        if 'storyboard' in context:
            self.planning.append(context)
            verdict = annotated_verdict(context['storyboard'])
            for unit in verdict['units']:
                unit.pop('result_admission', None)
            verdict['goal_allocation'] = annotation(context['storyboard'])
            return Responses(annotated_outcome(verdict, context['storyboard'])).create(**kwargs)
        return super().create(**kwargs)


@pytest.mark.asyncio
async def test_raw_creation_to_four_clips_export_and_resume_without_replanning(tmp_path, monkeypatch):
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_goal_creation_responses.json').read_text())
    values = []
    for record in fixture['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
        values.append(json.loads(json.loads(record['text'])['content']))
    original = deepcopy(values)
    story, _ = captured()
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    # 正向交付用例需有四段不同动态，复用同一片段会正确触发重演拒绝。
    sources = []
    for index in range(4):
        source = tmp_path / f'test-pattern-{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'testsrc2=s={dimensions}:r=24:d={6.125+index*7}', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=6.125', '-vf', f'trim=start={index*7},setpts=PTS-STARTPTS',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
            '-shortest', str(source)], check=True)
        sources.append(source)
    services = GoalServices(sources, Responses(*values))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    intent = story['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    output = await pipeline.run(intent['source_request'])
    assert values == original
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 4
    assert all(c['model'] == config.SEEDANCE_MODEL for c in services.submissions)
    assert [c['completion_contract']['intent_policy'] for c in services.reviews] == ['goal-allocation-v1'] * 4
    third = services.reviews[2]['completion_contract']['criteria']
    assert len([c for c in third.values() if c['kind'] == 'completion']) == 1
    submitted = services.planning[0]['storyboard']
    assert [[b['action'] for b in s['action_beats']] for s in submitted['shots']] == [
        [b['action'] for b in s['action_beats']] for s in story['shots']]
    for call, shot in zip(services.submissions, submitted['shots']):
        assert all(b['action'] in call['prompt'] for b in shot['action_beats'])
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert float(video['duration']) == pytest.approx(24, abs=1 / 24)
    assert int(video['nb_frames']) == 576
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded' and len(manifest.paid_take_budget.reservations) == 4
    counts = [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]

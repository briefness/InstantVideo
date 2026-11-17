"""真实原始创作和审查不改写；外部生成/视觉观察明确使用离线替身。"""
from copy import deepcopy
import json
import subprocess

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_cross_shot_goal import capture
from test_goal_allocation import offline
from test_goal_allocation_flow import GoalServices
from test_full_production_flow import completion
from test_director_intent import Responses
from tools.ffmpeg_ops import get_video_info


class CapturedGoalServices(GoalServices):
    def __init__(self, *args, verdict, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdict = verdict

    def create(self, **kwargs):
        content = kwargs['messages'][-1]['content']
        if isinstance(content, str) and 'storyboard' in json.loads(content):
            import jsonschema
            jsonschema.validate(self.verdict, kwargs['response_format']['json_schema']['schema'])
            self.planning.append(json.loads(content))
            return completion(deepcopy(self.verdict))
        return super().create(**kwargs)


@pytest.mark.asyncio
async def test_real_creation_and_review_replay_to_four_clips_export_and_resume(tmp_path, monkeypatch):
    fixture, story, review = capture()
    values = [json.loads(json.loads(r['text'])['content']) for n, r in fixture['files'].items()
              if n.endswith('_response.json')]
    original = deepcopy((values, review))
    source = tmp_path / 'test-pattern.mp4'
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc2=s={dimensions}:r=24:d=6.125', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6.125',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
    services = CapturedGoalServices([source] * 4, Responses(*values), verdict=review['verdict'])
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
    assert (values, review) == original
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 4
    assert all(c['model'] == config.SEEDANCE_MODEL for c in services.submissions)
    for i in (0, 1):
        assert any(c.get('execution_scope', {}).get('shot_id') == i + 1
                   for c in services.reviews[i]['completion_contract']['criteria'].values())
        prompt = services.submissions[i]['prompt']
        criteria = services.reviews[i]['completion_contract']['criteria'].values()
        for criterion in criteria:
            scope = criterion.get('execution_scope')
            if scope:
                for unit in scope['units']:
                    assert 'action' not in unit
                    beat = story['shots'][unit['shot_id'] - 1]['action_beats'][unit['beat_index']]
                    assert (unit['actor'], unit['target']) == (beat['actor'], beat.get('target', ''))
                    assert beat['action'] in prompt
                assert 'Joint goal context: ' + criterion['goal_context']['action'] in prompt
                assert "Execute only this shot's contribution in beats" in prompt
    submitted = services.planning[0]['storyboard']
    assert [[b['action'] for b in s['action_beats']] for s in submitted['shots']] == [
        [b['action'] for b in s['action_beats']] for s in story['shots']]
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert float(video['duration']) == pytest.approx(24, abs=1 / 24)
    assert int(video['nb_frames']) == 576
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded' and len(manifest.paid_take_budget.reservations) == 4
    counts = [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]

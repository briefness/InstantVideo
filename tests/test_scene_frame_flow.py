"""最新真实创作贯穿正式流程；原片负例与明确服务替身的交付正例分开。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_optional_view_flow import OptionalViewServices
from test_scene_frame_execution import captured
from test_goal_allocation import offline
from test_director_intent import Responses
from test_full_production_flow import completion
from tools.ffmpeg_ops import get_video_info


class SceneFrameServices(OptionalViewServices):
    def __init__(self, *args, failed_observation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_observation = failed_observation

    def create(self, **kwargs):
        response = super().create(**kwargs)
        if (self.failed_observation is not None and len(self.reviews) == 2
                and isinstance(kwargs['messages'][-1]['content'], list)):
            import jsonschema
            candidate = deepcopy(self.failed_observation)
            # 新增边界问题使用显式离线服务替身；原记录中的事实和肯定标签不改写。
            # 测试要求即使替身全部肯定，实际原片的几何反证仍能阻止交付。
            synthetic = json.loads(response.choices[0].message.content)
            candidate['observations'].update({k: v for k, v in synthetic['observations'].items()
                                              if k not in candidate['observations']})
            jsonschema.validate(candidate, kwargs['response_format']['json_schema']['schema'])
            return completion(candidate)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize('actual_failure', [True, False])
async def test_original_authoring_reaches_formal_delivery_or_stops_on_original_footage(tmp_path, monkeypatch, actual_failure):
    fixture, records = captured()
    root = Path(__file__).parents[1] / fixture['source']
    first, second = root / 'shots/shot_001.mp4', root / 'shots/shot_002_rejected_1.mp4'
    if not first.exists() or not second.exists():
        pytest.skip('local original videos unavailable')
    reviews = {p.name: records['semantic_reviews/' + hashlib.sha256(p.read_bytes()).hexdigest() + '.json']
               for p in (first, second)}
    first_review = reviews[first.name]['observation_response']
    second_review = reviews[second.name]['observation_response']
    raw = [json.loads(records[name]['content']) for name in ['planning_intent_response.json',
        'planning_identity_response.json', *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5))]]
    snapshot = deepcopy((raw, first_review, second_review))
    pattern = tmp_path / 'explicit-synthetic-service.mp4'
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc2=s={dimensions}:r=24:d=6.125', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6.125',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(pattern)], check=True)
    services = SceneFrameServices([first, second if actual_failure else pattern, pattern, pattern],
        Responses(*raw), verdict=records['planning_semantic_review.json']['verdict'],
        first_review=first_review, entry_response=records['continuations/shot_002.json']['plan'],
        failed_observation=second_review if actual_failure else None)
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    intent = records['storyboard.json']['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    if actual_failure:
        with pytest.raises(RuntimeError, match='Shot 2'):
            await pipeline.run(intent['source_request'])
        assert len(services.submissions) == 2 and not (Path(pipeline.workspace) / 'final.mp4').exists()
        rejected = RunWorkspace.resume(pipeline.workspace).manifest.shots['2']
        assert rejected.status.value == 'failed'
        camera_cache = next(v for p in (Path(pipeline.workspace) / 'semantic_reviews').glob('*.json')
                            if not p.name.endswith('.evidence.json')
                            and (v := json.loads(p.read_text())).get('composition_measurement'))
        assert camera_cache['observation_response']['composition_change_valid'] is True
        assert camera_cache['review']['composition_change_valid'] is False
    else:
        output = await pipeline.run(intent['source_request'])
        assert len(services.submissions) == 4
        assert float(get_video_info(output)['format']['duration']) == pytest.approx(24, abs=.1)
        counts = [len(x) for x in (services.author.calls, services.planning, services.submissions, services.reviews)]
        assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
        assert counts == [len(x) for x in (services.author.calls, services.planning, services.submissions, services.reviews)]
    assert (raw, first_review, second_review) == snapshot
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert all(s['generation_execution'] == 'scene-instructions-v2' for s in services.planning[0]['storyboard']['shots'])
    assert all('visible props:' not in call['prompt'] for call in services.submissions)
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)

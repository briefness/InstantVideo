"""最新真实创作/审查原样重放，取景与视频服务替身只证明正式执行和恢复。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_cross_shot_goal_flow import CapturedGoalServices
from test_director_intent import Responses
from test_full_production_flow import completion
from test_goal_allocation import offline
from test_view_execution import capture
from tools.ffmpeg_ops import get_video_info


class ViewServices(CapturedGoalServices):
    def create(self, **kwargs):
        response = super().create(**kwargs)
        content = kwargs['messages'][-1]['content']
        context = json.loads(content) if isinstance(content, str) else {}
        if context.get('camera_view'):
            value = json.loads(response.choices[0].message.content)
            # 明确的合成服务标注；测试图不含人物，不能用它声称视觉定位正确。
            value['view_window'] = {'left': .1, 'top': .1, 'scale': .75,
                'regions': {e: [.2, .2, .7, .7] for e in context['camera_view']['region_entities']}}
            if context['camera_view'].get('presentation_policy') == 'local-reframe-v1':
                value['view_window'] = {'regions': value['view_window']['regions']}
            return completion(value)
        return response


@pytest.mark.asyncio
async def test_latest_raw_creation_to_four_segments_with_executed_view_and_no_duplicate_resume(tmp_path, monkeypatch):
    _, captured = capture()
    story, review = captured['storyboard.json'], captured['planning_semantic_review.json']
    names = ['planning_intent_response.json', 'planning_identity_response.json',
             *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5))]
    values = [json.loads(captured[n]['content']) for n in names]
    original = deepcopy((values, review, story))
    source = tmp_path / 'offline-pattern.mp4'
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc2=s={dimensions}:r=24:d=6.125', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6.125',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
    services = ViewServices([source] * 4, Responses(*values), verdict=review['verdict'])
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
    assert (values, review, story) == original
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 4
    # 动态连续性也检查停留镜；不能因不需要改写入场动作而跳过原片观察。
    assert len(services.entries) == 3
    assert all(len(e['temporal_evidence']['sources'][0]['frames']) == 9 for e in services.entries)
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)
    # 参与者以未经修改的作者响应为准；其余字段和时序逐项保留。
    units = [[value['shot']['action_plan']['first'], *value['shot']['action_plan']['following']]
             for value in values[2:]]
    expected_beats = [[{**beat, 'temporal_relation': 'sequential',
                       'actor': unit['actor'], 'target': unit.get('target', '')}
                      for beat, unit in zip(shot['action_beats'], authored, strict=True)]
                     for shot, authored in zip(story['shots'], units, strict=True)]
    assert [s['action_beats'] for s in services.planning[0]['storyboard']['shots']] == expected_beats
    run = Path(pipeline.workspace)
    provenance = json.loads((run / 'shots/shot_002_generation.json').read_text())
    assert provenance['image_role'] == 'first_frame' and provenance['reference_view']
    assert provenance['reference_source_fingerprints'] != provenance['reference_fingerprints']
    submitted_refs = services.submissions[1]['image_urls']
    assert len(submitted_refs) == 1 and Path(submitted_refs[0]).parent.name == 'reference_views'
    assert Path(submitted_refs[0]).with_suffix('.json').is_file()
    assert services.reviews[1]['boundary_context']['reference_view_version'] == 'observed-camera-view-v1'
    manifest = RunWorkspace.resume(run).manifest
    assert manifest.status.value == 'succeeded' and len(manifest.paid_take_budget.reservations) == 4
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert float(video['duration']) == pytest.approx(story['total_duration'], abs=1/24)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    counts = [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(run).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]

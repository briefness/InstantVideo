"""013300原始创作、审查、第一段观察和失败续拍原样重放；后续视频使用明确的离线替身。"""
from copy import deepcopy
import hashlib
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
from test_optional_view import captured
from tools.ffmpeg_ops import get_video_info


class OptionalViewServices(CapturedGoalServices):
    def __init__(self, *args, first_review, entry_response, camera_failed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_review = first_review
        self.entry_response = entry_response
        self.camera_failed = camera_failed

    def create(self, **kwargs):
        response = super().create(**kwargs)
        content = kwargs['messages'][-1]['content']
        if isinstance(content, list):
            if len(self.reviews) == 1:
                # 当前schema对这份真实第一段记录无字段迁移，无新的观察或改判。
                import jsonschema
                jsonschema.validate(self.first_review, kwargs['response_format']['json_schema']['schema'])
                return completion(deepcopy(self.first_review))
            if len(self.reviews) == 2 and self.camera_failed:
                value = json.loads(response.choices[0].message.content)
                value['composition_change_valid'] = False
                return completion(value)
        elif json.loads(content).get('camera_view') and len(self.entries) == 1:
            entry = deepcopy(self.entry_response)
            synthetic = json.loads(response.choices[0].message.content)
            if 'observed_motion' in synthetic:
                entry['observed_motion'] = synthetic['observed_motion']
            if 'spatial_guidance' in synthetic:
                entry['spatial_guidance'] = synthetic['spatial_guidance']
            return completion(entry)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize('camera_failed', [False, True])
async def test_real_failure_reply_reaches_formal_export_or_real_quality_stop_without_retry(tmp_path, monkeypatch, camera_failed):
    fixture, records, _, observed, entry = captured()
    original_media = Path(__file__).parents[1] / fixture['source'] / 'shots/shot_001.mp4'
    if not original_media.is_file():
        pytest.skip('original local media unavailable; isolated raw response replay remains covered')
    assert hashlib.sha256(original_media.read_bytes()).hexdigest() == observed['video_hash']
    story, review = records['storyboard.json'], records['planning_semantic_review.json']
    names = ['planning_intent_response.json', 'planning_identity_response.json',
             *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5))]
    values = [json.loads(records[n]['content']) for n in names]
    original = deepcopy((values, review, entry, observed))
    pattern = tmp_path / 'synthetic-service-video.mp4'
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc2=s={dimensions}:r=24:d=6.125', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6.125',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(pattern)], check=True)
    services = OptionalViewServices([original_media, pattern, pattern, pattern], Responses(*values),
        verdict=review['verdict'], first_review=observed['observation_response'], entry_response=entry,
        camera_failed=camera_failed)
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    intent = story['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    if camera_failed:
        with pytest.raises(RuntimeError, match='Shot 2'):
            await pipeline.run(intent['source_request'])
    else:
        output = await pipeline.run(intent['source_request'])
    assert (values, review, entry, observed) == original
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == (2 if camera_failed else 4)
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)
    # 逐项对照未经修改的作者响应；历史编译稿中的空目标回填不能作为正确基准。
    units = [[value['shot']['action_plan']['first'], *value['shot']['action_plan']['following']]
             for value in values[2:]]
    expected_beats = [[{**beat, 'temporal_relation': 'sequential',
                       'actor': unit['actor'], 'target': unit.get('target', '')}
                      for beat, unit in zip(shot['action_beats'], authored, strict=True)]
                     for shot, authored in zip(story['shots'], units, strict=True)]
    assert [s['action_beats'] for s in services.planning[0]['storyboard']['shots']] == expected_beats
    run = Path(pipeline.workspace)
    continuation = json.loads((run / 'continuations/shot_002.json').read_text())
    assert {k: v for k, v in continuation['plan'].items() if k not in {'observed_motion', 'spatial_guidance'}} == entry
    assert continuation['plan']['spatial_guidance'] == dict.fromkeys(continuation['input']['spatial_context']['entities'])
    assert continuation['plan']['observed_motion']['sample_indices'] == [1, 9]
    assert continuation['view_decision']['mode'] == 'native_reference'
    # 原响应原样重放到新接口：新接口不再接受模型指定的裁切窗口。
    assert continuation['view_decision']['issues'] == [
        {'field': field, 'code': 'extra_forbidden'} for field in ('left', 'top', 'scale')]
    provenance_path = run / 'shots' / ('shot_002_rejected_1_generation.json' if camera_failed else 'shot_002_generation.json')
    provenance = json.loads(provenance_path.read_text())
    assert provenance['reference_view_decision'] == continuation['view_decision']
    assert provenance['image_role'] == 'reference_image' and not provenance.get('reference_view')
    assert all('first_frame' != services.submissions[1].get(k) for k in ('image_role', 'role'))
    assert len(list((run / 'continuations/responses/shot_002').glob('*.json'))) == 1
    manifest = RunWorkspace.resume(run).manifest
    assert len(manifest.paid_take_budget.reservations) == (2 if camera_failed else 4)
    if camera_failed:
        assert not (run / 'final.mp4').exists()
    else:
        assert manifest.status.value == 'succeeded'
        video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
        assert float(video['duration']) == pytest.approx(story['total_duration'], abs=1/24)
        subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
        counts = [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
        assert await VideoPipeline.from_workspace(run).run() == output
        assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert hashlib.sha256(original_media.read_bytes()).hexdigest() == observed['video_hash']

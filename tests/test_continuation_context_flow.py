"""本次真实创作原样重放；外部视频和后续观察是显式离线服务替身。"""
from copy import deepcopy
import hashlib
import json
import subprocess

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from pipeline.continuation_context import expand_context, request_context
from test_continuation_context import ROOT
from test_cross_shot_goal_flow import CapturedGoalServices
from test_director_intent import Responses
from test_goal_allocation import offline
from tools.ffmpeg_ops import get_video_info


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['reviewed', 'normal'])
async def test_original_planning_to_export_or_strict_quality_stop(tmp_path, monkeypatch, mode):
    names = ['planning_intent_response.json', 'planning_identity_response.json',
             *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5)),
             'planning_semantic_review.json', 'storyboard.json']
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    records = {n: json.loads((ROOT / n).read_text()) for n in names}
    values = [json.loads(records[n]['content']) for n in names[:6]]
    snapshot = deepcopy(values)
    sources = [ROOT / 'shots/shot_001.mp4']
    for index in range(3):
        path = tmp_path / f'explicit-offline-{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
            f'testsrc2=s=864x496:r=24:d={6.125+index*7}', '-f', 'lavfi', '-i',
            'sine=frequency=440:duration=6.125', '-vf', f'trim=start={index*7},setpts=PTS-STARTPTS',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
            '-shortest', str(path)], check=True)
        sources.append(path)
    services = CapturedGoalServices(sources, Responses(*values),
        verdict=records['planning_semantic_review.json']['verdict'])
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    intent = records['storyboard.json']['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        delivery_mode=mode, paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    if mode == 'reviewed':
        # 此离线观察替身的短因果区间在现有采样下不确定；严格模式必须停止。
        # 不改变时间标注或生产审查来让成功路径通过。
        from pipeline.semantic_review import SemanticReviewUnavailableError
        with pytest.raises(SemanticReviewUnavailableError, match='时间观察语义不确定'):
            await pipeline.run(intent['source_request'])
        assert len(services.submissions) == 1 and not services.entries
        assert not (pipeline.workspace / 'final.mp4').exists()
        assert values == snapshot
        assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in hashes.items())
        return
    output = await pipeline.run(intent['source_request'])
    assert values == snapshot
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == 4 and len(services.entries) == 3
    admitted = json.loads((pipeline.workspace / 'result_admission.json').read_text())['execution_storyboard']
    for index, entry in enumerate(services.entries, 2):
        path = pipeline.workspace / f'continuations/shot_{index:03d}.json'
        saved = json.loads(path.read_text())
        assert expand_context(entry) == saved['input']
        assert request_context(saved['input']) == entry
        assert saved['execution_shot']['action_beats'] == admitted['shots'][index-1]['action_beats']
        assert saved['execution_shot']['camera'] == admitted['shots'][index-1]['camera']
    for request, shot in zip(services.submissions, admitted['shots'], strict=True):
        assert all(b['action'] in request['prompt'] for b in shot['action_beats'])
        assert request['model'] == config.SEEDANCE_MODEL
    assert float(get_video_info(output)['format']['duration']) == pytest.approx(24, abs=.1)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded'
    assert len(manifest.paid_take_budget.reservations) == 4
    assert manifest.options.delivery_mode == mode
    counts = [len(v) for v in (services.author.calls, services.planning, services.entries,
                               services.submissions, services.reviews)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.entries,
                                      services.submissions, services.reviews)]
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in hashes.items())

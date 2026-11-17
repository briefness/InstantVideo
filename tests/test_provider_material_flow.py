"""最新原始规划和首片观察重放，真实 SDK 请求构造，云端视频明确用测试替身。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest
import config
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from tools.seedance_api import SeedanceAPI
from tools.ffmpeg_ops import get_video_info
from test_scene_frame_flow import SceneFrameServices
from test_director_intent import Responses
from test_goal_allocation import offline


class OriginalMaterialServices(SceneFrameServices):
    requires_original_materials = True

    def __init__(self, *args, reject_second=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdk_requests = []
        self.reject_second = reject_second
        self.created_at = int(time.time())

    async def generate(self, **kwargs):
        self.submissions.append(kwargs)
        def submit(**request):
            self.sdk_requests.append(request)
            if self.reject_second and len(self.sdk_requests) == 2:
                class Rejected(Exception):
                    status_code = 400
                    code = 'InputImageSensitiveContentDetected.PrivacyInformation'
                    message = "The request failed because the input image 'content[1]' may contain real person."
                raise Rejected()
            return SimpleNamespace(id=f'offline-{len(self.sdk_requests)}')
        def get(*, task_id):
            number = int(task_id.split('-')[-1])
            return SimpleNamespace(status='succeeded', model='doubao-seedance-2.0-fast',
                created_at=self.created_at, content=SimpleNamespace(
                    video_url=f'https://example.com/original-{number}.mp4',
                    last_frame_url=f'https://example.com/official-{number}.png', audio_url=None))
        api = object.__new__(SeedanceAPI)
        api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(
            tasks=SimpleNamespace(create=submit, get=get)))
        return await api.generate(**kwargs)

    async def download_video(self, url, destination):
        index = int(url.rsplit('-', 1)[-1].split('.')[0]) - 1
        shutil.copyfile(self.sources[index], destination)
        return destination


@pytest.mark.asyncio
@pytest.mark.parametrize('rejected', [False, True])
async def test_original_failure_material_lifecycle_to_export_or_single_rejection(tmp_path, monkeypatch, rejected):
    root = Path(__file__).parents[1] / 'output/20261104_030303_322135'
    if not root.is_dir():
        pytest.skip('original local run unavailable')
    first = root / 'shots/shot_001.mp4'
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert digest == '8b10ffd80f539f75a1fd352d6d1ded13c8bf2f642d70223824a3fb38b8f0923d'
    records = {name: json.loads((root / name).read_text()) for name in [
        'storyboard.json', 'planning_intent_response.json', 'planning_identity_response.json',
        *(f'planning_shot_{i:03d}_response.json' for i in range(1,5)),
        'planning_semantic_review.json', 'continuations/shot_002.json',
        f'semantic_reviews/{digest}.json']}
    responses = [json.loads(records[name]['content']) for name in ['planning_intent_response.json',
        'planning_identity_response.json', *(f'planning_shot_{i:03d}_response.json' for i in range(1,5))]]
    snapshot = deepcopy(records)
    sources = [first]
    for index in range(3):
        synthetic = tmp_path / f'explicit-offline-service-{index}.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'testsrc2=s=864x496:r=24:d={6.125+index*7}',
            '-f','lavfi','-i','sine=frequency=440:duration=6.125',
            '-vf',f'trim=start={index*7},setpts=PTS-STARTPTS', '-c:v','libx264','-preset','ultrafast',
            '-pix_fmt','yuv420p','-c:a','aac','-shortest',str(synthetic)], check=True)
        sources.append(synthetic)
    services = OriginalMaterialServices(sources, Responses(*responses),
        verdict=records['planning_semantic_review.json']['verdict'],
        first_review=records[f'semantic_reviews/{digest}.json']['observation_response'],
        entry_response=records['continuations/shot_002.json']['plan'], reject_second=rejected)
    for module in ('storyboard','planning_review','continuation','semantic_review','scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'explicit-offline-service-key')
    intent = records['storyboard.json']['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    if rejected:
        with pytest.raises(RuntimeError, match='Shot 2'):
            await pipeline.run(intent['source_request'])
        assert len(services.sdk_requests) == 2
        assert not (pipeline.workspace / 'final.mp4').exists()
    else:
        output = await pipeline.run(intent['source_request'])
        assert len(services.sdk_requests) == 4
        assert float(get_video_info(output)['format']['duration']) == pytest.approx(24, abs=.1)
        counts = [len(v) for v in (services.author.calls, services.sdk_requests, services.reviews)]
        assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
        assert counts == [len(v) for v in (services.author.calls, services.sdk_requests, services.reviews)]
    second = services.sdk_requests[1]
    assert second['model'] == 'doubao-seedance-2.0-fast'
    assert second['content'][1:] == [{'type':'video_url', 'video_url':
        {'url':'https://example.com/original-1.mp4'}, 'role':'reference_video'}]
    assert 'data:image' not in json.dumps(second)
    assert 'XiaoAn' in second['content'][0]['text'] and 'LinAyi' in second['content'][0]['text']
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert records == snapshot
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == (2 if rejected else 4)

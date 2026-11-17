"""正式编译到交付/恢复；云端与视觉判断为显式替身，本地媒体链路真实执行。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest
import config
from pipeline.orchestrator import VideoPipeline
from pipeline.semantic_review import SemanticTakeReviewer
from pipeline.run_state import RunWorkspace
from test_cross_shot_goal_flow import CapturedGoalServices
from test_director_intent import Responses
from test_full_production_flow import completion
from test_goal_allocation import offline
from test_presentation_media import make_video
from test_provider_material_flow import OriginalMaterialServices
from test_view_execution import capture
from tools import ffmpeg_ops


class PresentationServices(CapturedGoalServices):
    requires_original_materials = True
    generate = OriginalMaterialServices.generate
    download_video = OriginalMaterialServices.download_video

    def __init__(self, *args, reject_crop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdk_requests = []
        self.created_at = int(time.time())
        self.reject_second = False
        self.reject_crop = reject_crop

    def create(self, **kwargs):
        response = super().create(**kwargs)
        content = kwargs['messages'][-1]['content']
        value = json.loads(response.choices[0].message.content)
        if isinstance(content, str):
            context = json.loads(content)
            if context.get('camera_view'):
                value['view_window'] = {'regions': {e: [.2,.2,.7,.7]
                    for e in context['camera_view']['region_entities']}}
        elif self.reject_crop and len(self.reviews) == 2:
            value['composition_change_valid'] = False
        return completion(value)


@pytest.mark.asyncio
@pytest.mark.parametrize('reject_crop', [False, True])
async def test_one_presentation_for_review_history_export_and_no_paid_retry(tmp_path, monkeypatch, reject_crop):
    _, records = capture()
    story, planning = records['storyboard.json'], records['planning_semantic_review.json']
    names = ['planning_intent_response.json', 'planning_identity_response.json',
             *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5))]
    responses = [json.loads(records[n]['content']) for n in names]
    untouched = deepcopy(responses)
    source, _ = make_video(tmp_path, duration=6.125)
    # 第四段的供应商替身已改变构图；呈现层须保留它，不能再次套同一裁切。
    reframed = tmp_path / 'provider-reframed.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-vf',
        'crop=600:344:50:50,scale=864:496', '-c:v', 'libx264', '-c:a', 'copy', str(reframed)], check=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    services = PresentationServices([source, source, source, reframed], Responses(*responses),
        verdict=planning['verdict'], reject_crop=reject_crop)
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'explicit-offline-service-key')
    reviewed, normalized = [], []
    review = SemanticTakeReviewer.review
    normalize = ffmpeg_ops.normalize_video

    def observe_review(self, video, *args, **kwargs):
        reviewed.append(str(video))
        return review(self, video, *args, **kwargs)

    def observe_normalize(video, *args, **kwargs):
        normalized.append(str(video))
        return normalize(video, *args, **kwargs)

    monkeypatch.setattr(SemanticTakeReviewer, 'review', observe_review)
    monkeypatch.setattr(ffmpeg_ops, 'normalize_video', observe_normalize)
    intent = story['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    if reject_crop:
        with pytest.raises(RuntimeError):
            await pipeline.run(intent['source_request'])
        assert len(services.sdk_requests) == 2
        assert not normalized and not (pipeline.workspace / 'final.mp4').exists()
    else:
        result = await pipeline.run(intent['source_request'])
        assert len(services.sdk_requests) == 4
        assert normalized == reviewed
        assert ffmpeg_ops.get_video_duration(result) == pytest.approx(24., abs=.1)
        # 第三镜的动态证据确实来自第二镜呈现，而非它未裁切的原件。
        assert services.entries[1]['temporal_evidence']['sources'][0]['video_sha256'] == hashlib.sha256(
            Path(reviewed[1]).read_bytes()).hexdigest()
        counts = (len(services.sdk_requests), len(reviewed), len(services.entries), len(normalized))
        assert await VideoPipeline.from_workspace(pipeline.workspace).run() == result
        assert counts == (len(services.sdk_requests), len(reviewed), len(services.entries), len(normalized))
        from web_server import _run_summary
        summary = _run_summary(pipeline.workspace)
        assert '/presentations/' in summary['shots'][1]['video_url']
        Path(reviewed[1]).write_bytes(b'corrupt local presentation')
        broken = _run_summary(pipeline.workspace)
        assert broken['shots'][1]['video_url'] is None
        assert '暂不可预览' in broken['shots'][1]['errors'][-1]
    assert Path(reviewed[1]).parent.name == 'presentations'
    assert hashlib.sha256(Path(reviewed[1]).read_bytes()).hexdigest() != source_hash
    assert responses == untouched and hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert all(r['model'] == config.SEEDANCE_MODEL for r in services.sdk_requests)
    assert any(i['type'] == 'video_url' for i in services.sdk_requests[1]['content'])
    assert 'presentations/' not in json.dumps(services.sdk_requests)
    budget = RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget
    assert len(budget.reservations) == (2 if reject_crop else 4)

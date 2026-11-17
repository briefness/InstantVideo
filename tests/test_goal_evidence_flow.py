"""本轮原始创作/审查逐字回放；云端视频和视觉判断为明确的离线替身。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import config
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_goal_evidence_capacity import capture
from test_goal_allocation import offline
from test_director_intent import Responses
from test_presentation_flow import PresentationServices
from test_presentation_media import make_video
from tools.ffmpeg_ops import get_video_info


class TimedServices(PresentationServices):
    def create(self, **kwargs):
        from test_full_production_flow import completion
        response = super().create(**kwargs)
        if not isinstance(kwargs['messages'][-1]['content'], list):
            return response
        value = json.loads(response.choices[0].message.content)
        # 测试图没有真实动作；这些时间是明确的服务替身标注，仅验证媒体与合同的数据流。
        shot = self.planning[0]['storyboard']['shots'][len(self.submissions) - 1]
        for event in value.get('action_event_evidence', []):
            beat = shot['action_beats'][event['event_id'] - 1]
            for destination, source in [('observed_start_seconds', 'start_seconds'),
                    ('observed_effect_seconds', 'effect_at_seconds'),
                    ('observed_reaction_seconds', 'reaction_at_seconds'), ('observed_end_seconds', 'end_seconds')]:
                event[destination] = beat[source]
            event['active_intervals'] = [{'start_seconds': beat['start_seconds'], 'end_seconds': beat['end_seconds']}]
        return completion(value)


@pytest.mark.asyncio
async def test_real_fifteen_unit_creation_to_full_export_and_resume(tmp_path, monkeypatch):
    story, review = capture()
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_goal_evidence_creation.json').read_text())
    responses = []
    for record in fixture['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
        responses.append(json.loads(json.loads(record['text'])['content']))
    original = deepcopy((responses, review))
    source, _ = make_video(tmp_path, duration=6.125)
    sources = [source]
    # 独立素材具有不同构图和字节来源；不能复制同一原片充当多个已接受镜头。
    for index, crop in enumerate(['600:344:50:50', '740:424:100:30', '560:320:220:100'], 1):
        reframed = tmp_path / f'reframed-test-pattern-{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-vf',
            f'crop={crop},scale=864:496', '-c:v', 'libx264', '-c:a', 'copy', str(reframed)], check=True)
        sources.append(reframed)
    services = TimedServices(sources, Responses(*responses), verdict=review['verdict'])
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'offline-service-key')
    intent = story['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    result = await pipeline.run(intent['source_request'])
    assert (responses, review) == original
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.sdk_requests) == len(services.reviews) == 4
    assert all(r['model'] == config.SEEDANCE_MODEL for r in services.sdk_requests)
    for index, contract in enumerate(services.reviews):
        criteria = contract['completion_contract']['criteria']
        contribution = next(c for c in criteria.values()
            if c.get('source', {}).get('reference') == '/intent/completion_goals/3')
        assert contribution['kind'] == 'contribution'
        assert len(contribution['execution_scope']['units']) == [5, 4, 4, 2][index]
        assert contribution['goal_context']['participating_shots'] == [1, 2, 3, 4]
    video = next(s for s in get_video_info(result)['streams'] if s['codec_type'] == 'video')
    assert float(video['duration']) == pytest.approx(24, abs=1 / 24)
    assert int(video['nb_frames']) == 576
    subprocess.run(['ffmpeg', '-v', 'error', '-i', result, '-f', 'null', '-'], check=True)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded'
    assert len(manifest.paid_take_budget.reservations) == 4  # 仅替身账本，真实提交为零。
    counts = (len(services.author.calls), len(services.planning), len(services.sdk_requests), len(services.reviews))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == result
    assert counts == (len(services.author.calls), len(services.planning), len(services.sdk_requests), len(services.reviews))

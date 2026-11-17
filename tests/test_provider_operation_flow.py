"""最新原始创作走正式入口，服务替身只重放历史产物与否定证据。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import config

from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_director_intent import Responses
from test_goal_allocation import offline
from test_full_production_flow import completion
from test_presentation_flow import PresentationServices


class ReplayServices(PresentationServices):
    def create(self, **kwargs):
        content = kwargs['messages'][-1]['content']
        if isinstance(content, list):
            contract = json.JSONDecoder().raw_decode(content[0]['text'].split('Shot contract:\n', 1)[1])[0]
            self.reviews.append(contract)
            return completion(deepcopy(self.observations[len(self.reviews) - 1]))
        context = json.loads(content)
        if 'observed_state' in context:
            self.entries.append(context)
            return completion(deepcopy(self.entry))
        return super().create(**kwargs)


@pytest.mark.asyncio
async def test_formal_real_replay_keeps_failed_movie_failed_without_retry(tmp_path, monkeypatch):
    root = Path(__file__).parents[1] / 'output/20261112_184640_228865'
    if not root.is_dir():
        pytest.skip('本机真实运行不可用')
    names = ['storyboard.json', 'planning_semantic_review.json',
        'planning_intent_response.json', 'planning_identity_response.json',
        *(f'planning_shot_{i:03d}_response.json' for i in range(1, 5)), 'continuations/shot_002.json']
    records = {n: json.loads((root / n).read_text()) for n in names}
    hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}
    sources = [root / f'shots/{n}.mp4' for n in ('shot_001', 'shot_002_rejected_1')]
    reviews = [json.loads((root / f'semantic_reviews/{hashlib.sha256(s.read_bytes()).hexdigest()}.json').read_text())
               for s in sources]
    responses = [json.loads(records[n]['content']) for n in names[2:-1]]
    unchanged = deepcopy(responses)
    services = ReplayServices(sources, Responses(*responses), verdict=records['planning_semantic_review.json']['verdict'])
    services.observations = [r['observation_response'] for r in reviews]
    services.entry = records['continuations/shot_002.json']['plan']
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'offline-replay-service-key')
    intent = records['storyboard.json']['production_plan']['director_intent']
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
        paid_take_budget=4, take_policy='single_pass', style=intent['style_input'])
    with pytest.raises(RuntimeError, match='Shot 2'):
        await pipeline.run(intent['source_request'])
    assert len(services.sdk_requests) == 2
    second = services.sdk_requests[1]
    assert second['model'] == 'doubao-seedance-2.0-fast'
    assert second['content'][0]['text'].startswith('参考素材生成下一镜的新视频')
    assert 'tracking' in second['content'][0]['text'] and 'medium shot' in second['content'][0]['text']
    assert any(c['type'] == 'video_url' for c in second['content'])
    evidence = json.loads((pipeline.workspace / 'submission_requests/shot_002_attempt_001.json').read_text())
    assert evidence['provenance']['provider_materials']['operation']['camera_boundary'] == 'cut_to_current_shot'
    assert not (pipeline.workspace / 'final.mp4').exists()
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 2
    assert responses == unchanged
    assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == digest for n, digest in hashes.items())

"""普通场景走正式交付；测试图与独立服务标注只证明链路，不证明画面质量。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import config
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from planning_protocol_cases import wire_verdict
from test_director_intent import Responses
from test_full_production_flow import ProductionServices, completion
from test_ordinary_scene_contract import scene_case, offline
from test_planning_semantics import verdict_for
from test_sequence_review import event_response
from tools.ffmpeg_ops import get_video_info


class SceneServices(ProductionServices):
    def __init__(self, sources, author, shots, character_count):
        super().__init__(sources, author)
        self.shots = shots
        self.character_count = character_count

    def create(self, **kwargs):
        user = kwargs['messages'][-1]['content']
        if isinstance(user, list):
            contract = json.JSONDecoder().raw_decode(user[0]['text'].split('Shot contract:\n', 1)[1])[0]
            assert not contract.get('action_events')
            self.reviews.append(contract)
            shot = self.shots[len(self.submissions) - 1]
            response = event_response()
            response.pop('action_event_evidence')
            response.update(creative_brief_valid=True, composition_change_valid=True,
                identity_crop_boxes=[None] * self.character_count,
                observed_end_state={**{k: shot['end_state'][k] for k in ('location', 'subject', 'action_phase')},
                                    'prop_state': ''},
                environment_continuity_valid=True, action_handoff_valid=True,
                screen_direction_valid=True, prop_continuity_valid=True,
                boundary_state_evidence={k: True for k in (
                    'prior_state_preserved', 'state_progress_not_reversed',
                    'open_motion_handoff_valid', 'persistent_entities_preserved', 'scene_identity_preserved')})
            from functional_observation_cases import annotated_observation_response
            return completion(annotated_observation_response(response, kwargs))
        context = json.loads(user)
        if context.get('protocol') in {'observation-support-v4-evidence-context', 'observation-support-v5-visual-evidence'}:
            return super().create(**kwargs)
        if 'storyboard' in context:
            self.planning.append(context)
            verdict = verdict_for(context['storyboard'])
            for unit in verdict['units']:
                unit.pop('result_admission', None)
            return completion(wire_verdict(verdict, kwargs))
        if 'observed_state' in context:
            return super().create(**kwargs)
        return self.author.create(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
@pytest.mark.parametrize('active_performance', [False, True])
async def test_scene_creation_reaches_final_media_and_reuses_completed_delivery(tmp_path, monkeypatch, topic, active_performance):
    brief, reply, identity, wires, shots = scene_case(topic)
    if active_performance:
        # 独立合成输入，不重写历史记录；主动行为从正式作者响应入口进入。
        for wire, shot in zip(wires, shots):
            wire['shot']['action_plan']['first']['kind'] = 'performance'
            shot['action_plan'][0]['kind'] = 'performance'
    original = deepcopy(wires)
    source = tmp_path / 'test-pattern.mp4'
    dimensions = config.SEEDANCE_OUTPUT_DIMENSIONS['480p']['16:9'].replace(':', 'x')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc2=s={dimensions}:r=24:d=6.125', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6.125',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
    services = SceneServices([source] * 4, Responses(reply, identity, *wires), shots, len(identity['characters']))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: services)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no-luts')
    monkeypatch.setattr(config, 'MUSIC_DIR', tmp_path / 'no-music')
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], planning_repair=False,
                             paid_take_budget=4, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert len(services.author.calls) == 6 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 4
    assert len(services.entries) == 3  # 展示/停留镜同样需要时序交接，不改写原动作。
    assert all(len(c['temporal_evidence']['sources'][0]['frames']) == 9 for c in services.entries)
    assert all(c['model'] == config.SEEDANCE_MODEL for c in services.submissions)
    assert [c['image_role'] for c in services.submissions] == [None, 'reference_image', 'reference_image', 'reference_image']
    assert all(Path(c['image_urls'][0]).is_file() for c in services.submissions[1:])
    for call, shot in zip(services.submissions, shots):
        assert all(unit['action'] in call['prompt'] for unit in shot['action_plan'])
        assert all(unit['result'] in call['prompt'] for unit in shot['action_plan'])
    if active_performance:
        assert all(s['action_beats'][0]['activity'] == 'performance'
                   for s in services.planning[0]['storyboard']['shots'])
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert float(video['duration']) == pytest.approx(24, abs=1 / 24)
    assert int(video['nb_frames']) == 576
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded' and len(manifest.paid_take_budget.reservations) == 4
    counts = [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions, services.reviews, services.entries)]
    assert wires == original

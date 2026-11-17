"""Whole-film delivery is distinct from content approval; no real service calls."""
import json
import hashlib
from pathlib import Path

import pytest

import config
from pipeline.generator import RemoteTaskPendingError
from pipeline.models import RunOptions
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_scheduled_production_flow import scheduled_services
from tools import ffmpeg_ops
from web_server import _parse_create_payload, _run_summary


def preview_pipeline(**kwargs):
    return VideoPipeline(resolution='480p', platforms=['youtube'],
                         paid_take_budget=3, take_policy='single_pass',
                         delivery_mode='preview', **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_complete_preview_keeps_actual_tail_and_never_claims_content_approval(
        scheduled_services, monkeypatch, topic):
    brief, services = scheduled_services(topic)
    original_enabled = config.SEMANTIC_REVIEW_ENABLED
    def unexpected_review(*args, **kwargs):
        pytest.fail('preview delivery must not instantiate the visual reviewer')
    monkeypatch.setattr('pipeline.generator.SemanticTakeReviewer', unexpected_review)
    pipeline = preview_pipeline()
    output = await pipeline.run(brief)
    assert Path(output).name == 'preview.mp4'
    assert not (pipeline.workspace / 'final.mp4').exists()
    assert len(services.submissions) == 3
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert not services.reviews and not services.entries
    assert all(call['model'] == config.SEEDANCE_MODEL for call in services.submissions)
    for shot_id, submission in enumerate(services.submissions[1:], 1):
        assert submission['image_role'] == 'reference_image'
        assert Path(submission['image_urls'][0]) == pipeline.workspace / f'shots/shot_{shot_id:03d}_lastframe.jpg'
        assert Path(submission['image_urls'][0]).is_file()
        provenance = json.loads((pipeline.workspace / f'shots/shot_{shot_id + 1:03d}_generation.json').read_text())
        assert provenance['reference_fingerprints'][0] == hashlib.sha256(
            Path(submission['image_urls'][0]).read_bytes()).hexdigest()
        assert provenance['acceptance_context']['policy'] == 'technical_only'
    assert not (pipeline.workspace / 'continuations').exists()
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert manifest.status.value == 'succeeded'
    assert len(manifest.paid_take_budget.reservations) == 3
    assert all(s.semantic_accepted is None and not s.observed_end_state
               and s.acceptance_policy == 'technical_only' for s in manifest.shots.values())
    report = json.loads((pipeline.workspace / 'delivery_report.json').read_text())
    assert report['content_acceptance'] == 'not_reviewed'
    assert report['technical_validation'] == 'passed'
    assert not (pipeline.workspace / 'film_rhythm_review.json').exists()
    summary = _run_summary(pipeline.workspace)
    assert summary['options']['delivery_mode'] == 'preview'
    assert summary['assets']['final_url'].endswith('/preview.mp4')
    assert summary['assets']['exports']['youtube']
    info = ffmpeg_ops.validate_publish_ready(output, expected_duration=15, duration_tolerance=.15)
    assert info['duration'] == pytest.approx(15, abs=1/24)
    assert config.SEMANTIC_REVIEW_ENABLED == original_enabled
    counts = len(services.submissions), len(services.author.calls), len(services.planning)
    restored = VideoPipeline.from_workspace(pipeline.workspace)
    assert restored.delivery_mode == 'preview'
    assert await restored.run() == output
    assert counts == (len(services.submissions), len(services.author.calls), len(services.planning))


@pytest.mark.asyncio
@pytest.mark.parametrize('review_content', [None, True])
async def test_only_explicit_preview_can_use_unreviewed_reference_handoff(tmp_path, monkeypatch, review_content):
    from pipeline.generator import VideoGenerator
    from pipeline.continuation import ContinuationUnavailableError
    from test_temporal_visual_handoff import captured, ROOT
    story, shot, previous, _ = captured()
    previous.pop('observed_end_state')
    monkeypatch.setattr(config, 'SEMANTIC_REVIEW_ENABLED', False)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: object())
    monkeypatch.setattr('pipeline.generator.SemanticTakeReviewer', lambda *_, **__: object())
    monkeypatch.setattr('pipeline.continuation.OpenAI', lambda **_: pytest.fail('missing evidence cannot invoke a model'))
    generator = VideoGenerator(str(tmp_path), review_content=review_content)
    with pytest.raises(ContinuationUnavailableError, match='缺少上一镜观察状态'):
        await generator._generate_single_shot(
            shot, str(ROOT / 'shots/shot_001_lastframe.jpg'), previous, story)


@pytest.mark.asyncio
async def test_preview_export_failure_reuses_all_takes(scheduled_services, monkeypatch):
    brief, services = scheduled_services('货物交接')
    original_export = ffmpeg_ops.export_for_platform
    def disk_failure(*args, **kwargs):
        raise OSError('offline disk full')
    monkeypatch.setattr(ffmpeg_ops, 'export_for_platform', disk_failure)
    pipeline = preview_pipeline()
    with pytest.raises(OSError, match='disk full'):
        await pipeline.run(brief)
    assert RunWorkspace.resume(pipeline.workspace).manifest.status.value == 'failed'
    assert not (pipeline.workspace / 'delivery_report.json').exists()
    originals = {p.name: p.read_bytes() for p in (pipeline.workspace / 'shots').glob('shot_*.mp4')}
    monkeypatch.setattr(ffmpeg_ops, 'export_for_platform', original_export)
    output = await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert Path(output).is_file() and len(services.submissions) == 3
    assert originals == {p.name: p.read_bytes() for p in (pipeline.workspace / 'shots').glob('shot_*.mp4')}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['pending', 'invalid_media'])
async def test_preview_does_not_hide_provider_or_media_failures(scheduled_services, monkeypatch, failure):
    brief, services = scheduled_services('绳索救援')
    generate, download = services.generate, services.download_video
    async def pending(**kwargs):
        result = await generate(**kwargs)
        if len(services.submissions) == 2:
            return {'status': 'pending', 'error_type': 'poll_timeout',
                    'provider_task_id': result['provider_task_id']}
        return result
    async def invalid(url, destination):
        if len(services.submissions) == 2:
            Path(destination).write_bytes(b'not a decodable video')
            return destination
        return await download(url, destination)
    monkeypatch.setattr(services, 'generate', pending if failure == 'pending' else generate)
    monkeypatch.setattr(services, 'download_video', invalid if failure == 'invalid_media' else download)
    pipeline = preview_pipeline()
    with pytest.raises(RemoteTaskPendingError if failure == 'pending' else RuntimeError):
        await pipeline.run(brief)
    assert len(services.submissions) == 2
    assert not (pipeline.workspace / 'preview.mp4').exists()
    assert not (pipeline.workspace / 'delivery_report.json').exists()
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 2


@pytest.mark.asyncio
async def test_preview_budget_insufficient_before_any_paid_submission(scheduled_services):
    brief, services = scheduled_services('双人舞')
    pipeline = preview_pipeline()
    pipeline.paid_take_budget = 2
    with pytest.raises(RuntimeError, match='Shot 1 生成失败'):
        await pipeline.run(brief)
    assert not services.submissions
    manifest = RunWorkspace.resume(pipeline.workspace).manifest
    assert not manifest.paid_take_budget.reservations
    assert '首次机会' in manifest.shots['1'].errors[-1]


def test_delivery_mode_defaults_are_compatible_and_invalid_modes_rejected(tmp_path):
    assert RunOptions(request='old').delivery_mode == 'reviewed'
    assert VideoPipeline().delivery_mode == 'reviewed'
    assert _parse_create_payload({'request': 'new'}).delivery_mode == 'reviewed'
    for value in ('preview', 'reviewed'):
        opts = _parse_create_payload({'request': 'new', 'delivery_mode': value})
        workspace = RunWorkspace.create(tmp_path, opts)
        assert VideoPipeline.from_workspace(workspace.path).delivery_mode == value
    for value in ('anything', None, True, {}, []):
        with pytest.raises(ValueError):
            _parse_create_payload({'request': 'new', 'delivery_mode': value})


def test_legacy_planning_identity_restores_without_rewriting_checkpoint(tmp_path):
    import hashlib
    from pipeline.planning_checkpoint import PlanningCheckpoint, FILENAME, VERSION
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='historical planning'))
    historical = workspace.options.model_dump(mode='json', exclude={'delivery_mode'})
    assert 'delivery_mode' not in historical
    context = digest({'version': VERSION, 'options': historical,
                      'model': config.LLM_MODEL, 'endpoint': config.ARK_BASE_URL})
    record = {'version': VERSION, 'context': context, 'steps': []}
    record['digest'] = digest(record)
    path = workspace.path / FILENAME
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    assert PlanningCheckpoint(RunWorkspace.resume(workspace.path), resume=True).context == context
    assert path.read_bytes() == original
    assert RunOptions.model_validate(historical).model_dump(mode='json') == historical
    assert RunOptions(request='preview', delivery_mode='preview').model_dump()['delivery_mode'] == 'preview'
    assert 'delivery_mode' not in RunOptions(request='explicit', delivery_mode='reviewed').model_dump()


def test_cli_freezes_mode_and_rejects_resume_override(monkeypatch, tmp_path, capsys):
    import main
    calls = []
    async def run(self, request):
        calls.append((self.delivery_mode, self.take_policy, self.paid_take_budget))
        return 'preview.mp4'
    monkeypatch.setattr(VideoPipeline, 'run', run)
    monkeypatch.setattr('sys.argv', ['main.py', 'offline', '--delivery-mode', 'preview',
                                   '--take-policy', 'single_pass', '--paid-take-budget', '3'])
    main.main()
    assert calls == [('preview', 'single_pass', 3)]
    assert '待验收' in capsys.readouterr().out
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='old'))
    original = workspace.manifest_path.read_bytes()
    monkeypatch.setattr('sys.argv', ['main.py', '--resume', str(workspace.path), '--delivery-mode', 'preview'])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert workspace.manifest_path.read_bytes() == original


@pytest.mark.asyncio
async def test_web_entry_passes_persisted_delivery_mode_to_pipeline(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from test_web_server import FakePipeline
    from web_server import create_app
    constructed = []
    def factory(**kwargs):
        constructed.append(kwargs['delivery_mode'])
        assert kwargs['run_workspace'].options.delivery_mode == kwargs['delivery_mode']
        return FakePipeline(**kwargs)
    async with TestClient(TestServer(create_app(output_root=tmp_path, pipeline_factory=factory))) as client:
        response = await client.post('/api/runs', json={'request': 'offline', 'delivery_mode': 'preview'})
        assert response.status == 202
        detail = await response.json()
        assert detail['options']['delivery_mode'] == 'preview'
        assert constructed == ['preview']


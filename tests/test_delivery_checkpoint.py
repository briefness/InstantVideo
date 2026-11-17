"""交付记录边界测试；媒体探测替身不代表真实画面通过验收。"""
import json
import os
from pathlib import Path

import pytest

import config
from pipeline import delivery
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from test_run_state import storyboard


@pytest.fixture
def ready(tmp_path, monkeypatch):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='5秒包裹交接', platforms=['youtube']))
    workspace.save_storyboard(storyboard())
    final = workspace.path / 'final.mp4'
    final.write_bytes(b'unit fixture master, not actual media')
    calls = []

    def export(source, platform, output):
        calls.append(platform)
        Path(output).write_bytes(b'unit fixture exported media')

    monkeypatch.setattr(delivery.ffmpeg_ops, 'export_for_platform', export)
    monkeypatch.setattr(delivery.ffmpeg_ops, 'validate_publish_ready',
                        lambda path, **kwargs: {'path': path, 'duration': 5.0})
    report = {'version': 'delivery-v1', 'mode': 'reviewed', 'output': 'final.mp4',
              'technical_validation': 'passed', 'content_acceptance': 'not_reviewed', 'shots': []}
    with workspace.execution():
        delivery.prepare_delivery(workspace, str(final), expected_duration=5, report=report)
    return workspace, calls


def test_delivery_is_idempotent_and_repairs_only_derived_outputs(ready):
    workspace, calls = ready
    with workspace.execution():
        result = delivery.resume_delivery(workspace)
        assert workspace.completed_output() == result
        assert delivery.resume_delivery(workspace) == result
        assert calls == ['youtube']
        (workspace.path / 'exports/youtube.mp4').write_bytes(b'corrupted export')
        assert workspace.completed_output() is None
        assert delivery.resume_delivery(workspace) == result
    assert calls == ['youtube', 'youtube']
    report = json.loads((workspace.path / 'delivery_report.json').read_text())
    assert report['content_acceptance'] == 'not_reviewed'
    assert report['delivery_fingerprint'] == workspace.manifest.delivery_fingerprint


@pytest.mark.parametrize('damage', ['invalid_record', 'missing_master',
                                  'model', 'options', 'symlink_master', 'fifo_master', 'fifo_record'])
def test_invalid_checkpoint_never_falls_back_to_production_or_export(ready, tmp_path, monkeypatch, damage):
    workspace, calls = ready
    record = workspace.path / delivery.FILENAME
    master = workspace.path / 'final.mp4'
    if damage == 'invalid_record':
        record.write_text('{"base":{},"exports":{},"digest":"forged"}')
    elif damage == 'missing_master':
        master.unlink()
    elif damage == 'model':
        monkeypatch.setattr(config, 'SEEDANCE_MODEL', 'changed-model')
    elif damage == 'options':
        workspace.manifest.options.style = 'different'
    elif damage == 'symlink_master':
        outside = tmp_path / 'outside.mp4'
        outside.write_bytes(master.read_bytes())
        master.unlink()
        master.symlink_to(outside)
    else:
        path = master if damage == 'fifo_master' else record
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(ValueError):
        delivery.resume_delivery(workspace)
    assert not calls


def test_missing_export_progress_recovers_from_authoritative_manifest_base(ready):
    """新记录缺进度文件不是母版证据缺失，只重做可派生的平台导出。"""
    workspace, calls = ready
    record = workspace.path / delivery.FILENAME
    record.unlink()
    master = workspace.path / 'final.mp4'
    original = master.read_bytes()
    assert workspace.manifest.delivery_base is not None
    assert delivery.resume_delivery(workspace) == str(master)
    assert calls == ['youtube'] and master.read_bytes() == original
    assert record.exists()


def test_legacy_fingerprint_without_record_cannot_reconstruct_master_evidence(ready):
    """历史仅有指纹时缺失原记录仍拒绝，不能推断或补造验收依据。"""
    workspace, calls = ready
    workspace.manifest.delivery_base = None
    (workspace.path / delivery.FILENAME).unlink()
    with pytest.raises(ValueError, match='缺失或损坏'):
        delivery.resume_delivery(workspace)
    assert not calls


def test_unknown_orphan_record_does_not_allow_reproduction(ready):
    workspace, calls = ready
    workspace.manifest.delivery_fingerprint = None
    with pytest.raises(ValueError, match='交付记录'):
        delivery.resume_delivery(workspace)
    assert not calls


def test_export_failure_keeps_prior_destination_and_master(ready, monkeypatch):
    workspace, calls = ready
    before = (workspace.path / 'final.mp4').read_bytes()
    destination = workspace.path / 'exports/youtube.mp4'
    destination.parent.mkdir()
    destination.write_bytes(b'old partial output')

    def fail(source, platform, path):
        Path(path).write_bytes(b'partial new output')
        raise OSError('disk full')

    monkeypatch.setattr(delivery.ffmpeg_ops, 'export_for_platform', fail)
    with pytest.raises(OSError, match='disk full'):
        delivery.resume_delivery(workspace)
    assert destination.read_bytes() == b'old partial output'
    assert (workspace.path / 'final.mp4').read_bytes() == before
    assert not list(destination.parent.glob('.*.mp4'))
    assert not json.loads((workspace.path / delivery.FILENAME).read_text())['exports']


def test_failed_export_validation_is_not_committed(ready, monkeypatch):
    workspace, calls = ready

    def invalid(*args, **kwargs):
        raise RuntimeError('invalid export media')

    monkeypatch.setattr(delivery.ffmpeg_ops, 'validate_publish_ready', invalid)
    with pytest.raises(RuntimeError, match='invalid export'):
        delivery.resume_delivery(workspace)
    assert calls == ['youtube']
    assert not (workspace.path / 'exports/youtube.mp4').exists()
    assert not json.loads((workspace.path / delivery.FILENAME).read_text())['exports']


def test_delivery_uses_existing_execution_lock(ready):
    workspace, _ = ready
    second = RunWorkspace.resume(workspace.path)
    with workspace.execution():
        with pytest.raises(RuntimeError):
            with second.execution():
                pytest.fail('concurrent owner entered delivery')

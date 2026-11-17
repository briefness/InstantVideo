"""母版身份是单次原子提交；导出进度损坏不能授予新的生产权限。"""
import json
import shutil

import pytest

from pipeline import delivery
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from test_run_state import storyboard
from test_full_production_flow import production_services


def local_delivery(tmp_path, monkeypatch, mode='reviewed'):
    run = RunWorkspace.create(tmp_path, RunOptions(
        request='offline delivery', delivery_mode=mode, platforms=['youtube']))
    run.save_storyboard(storyboard())
    master = run.path / ('preview.mp4' if mode == 'preview' else 'final.mp4')
    master.write_bytes(b'offline immutable accepted master')
    monkeypatch.setattr(delivery.ffmpeg_ops, 'validate_publish_ready',
                        lambda path, **_: {'path': str(path), 'duration': 5.})
    monkeypatch.setattr(delivery.ffmpeg_ops, 'export_for_platform',
                        lambda source, platform, target: shutil.copyfile(source, target))
    return run, master


@pytest.mark.parametrize('mode', ['preview', 'reviewed'])
def test_checkpoint_write_failure_recovers_only_from_committed_base(tmp_path, monkeypatch, mode):
    run, master = local_delivery(tmp_path, monkeypatch, mode)
    with monkeypatch.context() as failed:
        failed.setattr(delivery, '_write', lambda *_: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError, match='disk full'):
            delivery.prepare_delivery(run, str(master), expected_duration=5, report={'mode': mode})
    resumed = RunWorkspace.resume(run.path)
    assert resumed.manifest.delivery_base is not None
    assert not (run.path / delivery.FILENAME).exists()
    assert delivery.verified_output(resumed) is None
    assert delivery.resume_delivery(resumed) == str(master)
    assert delivery.verified_output(RunWorkspace.resume(run.path)) == str(master)
    assert not resumed.manifest.paid_take_budget.reservations


def test_manifest_commit_failure_never_publishes_an_unbound_checkpoint(tmp_path, monkeypatch):
    run, master = local_delivery(tmp_path, monkeypatch)
    with monkeypatch.context() as failed:
        failed.setattr(run, '_atomic_write_json',
                       lambda *_: (_ for _ in ()).throw(OSError('manifest disk failure')))
        with pytest.raises(OSError, match='manifest disk failure'):
            delivery.prepare_delivery(run, str(master), expected_duration=5, report={})
    resumed = RunWorkspace.resume(run.path)
    assert resumed.manifest.delivery_base is None
    assert resumed.manifest.delivery_fingerprint is None
    assert not (run.path / delivery.FILENAME).exists()
    assert delivery.resume_delivery(resumed) is None
    delivery.prepare_delivery(resumed, str(master), expected_duration=5, report={})
    assert delivery.resume_delivery(resumed) == str(master)


def test_failure_after_manifest_publication_keeps_recoverable_authority(tmp_path, monkeypatch):
    run, master = local_delivery(tmp_path, monkeypatch)
    atomic_write = run._atomic_write_json

    def fail_after_publication(path, payload):
        atomic_write(path, payload)
        raise OSError('publication acknowledgement unavailable')

    with monkeypatch.context() as failed:
        failed.setattr(run, '_atomic_write_json', fail_after_publication)
        with pytest.raises(OSError, match='acknowledgement unavailable'):
            delivery.prepare_delivery(run, str(master), expected_duration=5, report={})
    resumed = RunWorkspace.resume(run.path)
    assert resumed.manifest.delivery_base is not None
    assert not (run.path / delivery.FILENAME).exists()
    assert delivery.resume_delivery(resumed) == str(master)


@pytest.mark.parametrize('damage', ['checkpoint', 'authoritative_base', 'master', 'unbound_checkpoint'])
def test_delivery_recovery_rejects_changed_or_unbound_evidence(tmp_path, monkeypatch, damage):
    run, master = local_delivery(tmp_path, monkeypatch)
    delivery.prepare_delivery(run, str(master), expected_duration=5, report={'accepted': True})
    checkpoint = run.path / delivery.FILENAME
    if damage == 'checkpoint':
        record = json.loads(checkpoint.read_text())
        record['base']['report']['accepted'] = False
        delivery._write(run, record)  # Even a self-consistent checkpoint cannot change the authority.
    elif damage == 'authoritative_base':
        run.manifest.delivery_base['report']['accepted'] = False
        run._save_manifest()
    elif damage == 'master':
        master.write_bytes(b'other footage')
    else:
        run.manifest.delivery_base = None
        run.manifest.delivery_fingerprint = None
        run._save_manifest()
    with pytest.raises(ValueError):
        delivery.resume_delivery(RunWorkspace.resume(run.path))
    assert not (run.path / 'exports/youtube.mp4').exists()


def test_legacy_bound_checkpoint_remains_readable_but_missing_one_cannot_be_recreated(tmp_path, monkeypatch):
    run, master = local_delivery(tmp_path, monkeypatch)
    delivery.prepare_delivery(run, str(master), expected_duration=5, report={})
    run.manifest.delivery_base = None
    run._save_manifest()
    assert delivery.resume_delivery(RunWorkspace.resume(run.path)) == str(master)
    (run.path / delivery.FILENAME).unlink()
    with pytest.raises(ValueError, match='缺失或损坏'):
        delivery.resume_delivery(RunWorkspace.resume(run.path))


def test_delivery_base_binding_checks_identity_and_is_immutable(tmp_path, monkeypatch):
    run, master = local_delivery(tmp_path, monkeypatch)
    delivery.prepare_delivery(run, str(master), expected_duration=5, report={})
    original = run.manifest.delivery_base
    changed = {**original, 'output': 'other.mp4'}
    with pytest.raises(ValueError):
        run.bind_delivery(run.manifest.delivery_fingerprint, base=changed)
    with pytest.raises(ValueError):
        run.bind_delivery(delivery._digest(changed), base=changed)
    assert RunWorkspace.resume(run.path).manifest.delivery_base == original


@pytest.mark.asyncio
async def test_formal_resume_after_checkpoint_commit_failure_skips_all_production(production_services, monkeypatch):
    from pipeline.orchestrator import VideoPipeline
    brief, services = production_services
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    write = delivery._write
    with monkeypatch.context() as failed:
        failed.setattr(delivery, '_write', lambda *_: (_ for _ in ()).throw(OSError('checkpoint unavailable')))
        with pytest.raises(OSError, match='checkpoint unavailable'):
            await pipeline.run(brief)
    master = pipeline.workspace / 'final.mp4'
    original = master.read_bytes()
    counts = (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))
    monkeypatch.setattr('pipeline.orchestrator.synthesize_voiceover',
                        lambda *_: pytest.fail('delivery recovery must not restart voice generation'))
    assert delivery._write is write
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == str(master)
    assert master.read_bytes() == original
    assert counts == (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))


@pytest.mark.asyncio
async def test_formal_resume_after_manifest_failure_reassembles_accepted_takes(production_services, monkeypatch):
    from pipeline.orchestrator import VideoPipeline
    brief, services = production_services
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    with monkeypatch.context() as failed:
        failed.setattr(RunWorkspace, 'bind_delivery',
                       lambda *_, **__: (_ for _ in ()).throw(OSError('manifest commit unavailable')))
        with pytest.raises(OSError, match='manifest commit unavailable'):
            await pipeline.run(brief)
    counts = (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))
    assert not (pipeline.workspace / delivery.FILENAME).exists()
    assert RunWorkspace.resume(pipeline.workspace).manifest.delivery_base is None
    # 未绑定 final 不能直接获得信任；正式流程从已验收原片重新组装。
    master = pipeline.workspace / 'final.mp4'
    master.write_bytes(b'uncommitted master cannot grant acceptance')
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == str(master)
    assert master.read_bytes() != b'uncommitted master cannot grant acceptance'
    assert counts == (len(services.author.calls), len(services.submissions), len(services.reviews), len(services.entries))
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 3

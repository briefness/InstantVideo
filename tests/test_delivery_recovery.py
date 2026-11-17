"""交付恢复必须只做本地导出，不再次调用已完成的生产阶段。"""
from pathlib import Path

import pytest

from pipeline.orchestrator import VideoPipeline
from test_full_production_flow import production_services
from tools import ffmpeg_ops


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['export_failure', 'missing_export', 'changed_master'])
async def test_delivery_recovery_does_not_reenter_production(production_services, monkeypatch, damage):
    brief, services = production_services
    voice_calls = []

    async def voiceover(*args):
        voice_calls.append(args)
        return []

    monkeypatch.setattr('pipeline.orchestrator.synthesize_voiceover', voiceover)
    export = ffmpeg_ops.export_for_platform
    exports = []

    def exporting(*args, **kwargs):
        exports.append(args[1])
        if damage == 'export_failure' and len(exports) == 2:
            raise OSError('offline disk failure')
        return export(*args, **kwargs)

    monkeypatch.setattr(ffmpeg_ops, 'export_for_platform', exporting)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube', 'tiktok'],
                             paid_take_budget=3, take_policy='single_pass')
    if damage == 'export_failure':
        with pytest.raises(OSError, match='disk failure'):
            await pipeline.run(brief)
    else:
        await pipeline.run(brief)
    counts = (len(services.author.calls), len(services.submissions), len(services.reviews),
              len(services.entries), len(voice_calls))
    final = pipeline.workspace / 'final.mp4'
    original = final.read_bytes()
    youtube = (pipeline.workspace / 'exports/youtube.mp4').read_bytes()
    if damage == 'missing_export':
        (pipeline.workspace / 'exports/tiktok.mp4').unlink()
    if damage == 'changed_master':
        final.write_bytes(b'changed after acceptance')
        with pytest.raises(ValueError, match='成片.*变化|成片.*损坏'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    else:
        assert await VideoPipeline.from_workspace(pipeline.workspace).run() == str(final)
        assert final.read_bytes() == original
        assert (pipeline.workspace / 'exports/tiktok.mp4').is_file()
        assert (pipeline.workspace / 'exports/youtube.mp4').read_bytes() == youtube
        assert exports == ['youtube', 'tiktok', 'tiktok']
    assert counts == (len(services.author.calls), len(services.submissions), len(services.reviews),
                      len(services.entries), len(voice_calls))

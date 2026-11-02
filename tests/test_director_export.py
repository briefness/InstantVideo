"""Real FFmpeg export from the official entry, with external generation replaced."""
import json
import subprocess
from pathlib import Path

import pytest

import config
from pipeline.generator import ShotResult
from pipeline.orchestrator import VideoPipeline
from tools import ffmpeg_ops


def clip(path, duration=5.12):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=s=320x180:r=24:d={duration}', '-f', 'lavfi', '-i',
                    f'sine=frequency=440:duration={duration}', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(path)], check=True)


@pytest.mark.asyncio
async def test_official_export_keeps_fifteen_seconds_and_no_extra_title(tmp_path, monkeypatch):
    raw = tmp_path / 'raw.mp4'
    clip(raw)
    calls = []
    def draft(**kwargs):
        calls.append(kwargs)
        return {'title': 'Duel', 'content_focus': 'action', 'mood': 'cinematic',
                'resolution': '480p', 'aspect_ratio': '16:9', 'shots': [
                    {'shot_id': i, 'duration': 5, 'prompt_en': 'An action shot',
                     'scene_description': 'A rooftop duel', 'transition_to_next': 'crossfade'}
                    for i in range(1, 4)]}
    class Generated:
        def __init__(self, *_args, on_progress=None, **_kwargs):
            self.progress = on_progress
        async def generate_all(self, storyboard):
            results = []
            for shot in storyboard['shots']:
                result = ShotResult(shot_id=shot['shot_id'], status='success', local_path=str(raw))
                self.progress(result)
                results.append(result)
            return results
    async def no_voice(*_args):
        return []
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path / 'runs')
    monkeypatch.setattr(config, 'LUTS_DIR', tmp_path / 'no_luts')
    monkeypatch.setitem(config.SEEDANCE_OUTPUT_DIMENSIONS['480p'], '16:9', '320:180')
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', draft)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', Generated)
    monkeypatch.setattr('pipeline.orchestrator.synthesize_voiceover', no_voice)
    monkeypatch.setattr(VideoPipeline, '_auto_select_music', lambda *_args: None)
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.export_for_platform',
                        lambda source, _platform, dest: __import__('shutil').copyfile(source, dest))
    validate = ffmpeg_ops.validate_publish_ready
    monkeypatch.setattr('pipeline.orchestrator.ffmpeg_ops.validate_publish_ready',
                        lambda *args, platform=None, **kwargs: validate(*args, **kwargs))
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=0)
    output = await pipeline.run('15秒武打短片，三个镜头，最后决胜')
    assert calls[0]['target_duration'] == 15
    info = ffmpeg_ops.get_video_info(output)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360
    assert float(video['duration']) == pytest.approx(15, abs=1 / 24)
    assert float(info['format']['duration']) == pytest.approx(15, abs=.15)
    assert ffmpeg_ops.has_audio_track(output)
    assert pipeline._transitions == [('cut', 0.0), ('cut', 0.0)]
    assert not (pipeline.workspace / 'title.mp4').exists()
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert len(calls) == 1
    subprocess.run(['ffmpeg', '-v', 'error', '-i', output, '-f', 'null', '-'], check=True)


def test_short_source_is_not_frozen_to_fill_planned_action(tmp_path):
    raw = tmp_path / 'short.mp4'
    clip(raw, duration=1)
    with pytest.raises(ValueError, match='短于计划'):
        ffmpeg_ops.normalize_video(str(raw), str(tmp_path / 'out.mp4'), duration=5)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), True, '5'])
def test_invalid_edit_duration_stops_before_media_access(tmp_path, duration):
    with pytest.raises(ValueError, match='有限正数'):
        ffmpeg_ops.normalize_video('missing.mp4', str(tmp_path / 'out.mp4'), duration=duration)

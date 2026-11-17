"""口播副作用按段持久化；全部引擎调用由离线替身提供。"""
import json
from pathlib import Path
import shutil
import socket
import subprocess

import pytest

import config
from pipeline.run_state import RunWorkspace
from tools import tts
from tools.voiceover_state import VoiceoverStateError
from test_full_production_flow import production_services


class OfflineVoice(tts.MacOSSayEngine):
    def __init__(self, source):
        super().__init__(voice='offline-voice')
        self.source = source
        self.calls = []
        self.fail = False

    async def synthesize(self, text, output_path):
        self.calls.append(text)
        if self.fail:
            raise TimeoutError('offline unconfirmed request')
        shutil.copyfile(self.source, output_path)
        return tts._get_audio_duration(output_path)


@pytest.fixture
def voice(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    source = tmp_path / 'offline_source.m4a'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=0.5', '-c:a', 'aac', str(source)], check=True)
    engine = OfflineVoice(source)
    monkeypatch.setattr(config, 'TTS_ENGINE', 'macos')
    monkeypatch.setattr(config, 'TTS_VOICE', engine.voice)
    monkeypatch.setattr(tts, 'get_tts_engine', lambda *_: engine)
    return engine


def narrated():
    return {'shots': [{'shot_id': 1, 'subtitle_text': '离线口播'},
                      {'shot_id': 2, 'subtitle_text': ''}]}


def test_public_volcano_identity_tracks_request_parameters_without_credentials(monkeypatch):
    monkeypatch.setattr(config, 'VOLCANO_TTS_API_KEY', 'offline-secret-value')
    engine = tts.VolcanoTTSEngine(tts.VoiceProfile('离线音色', 'offline-speaker', ()))
    profile = tts._voiceover_profile(engine)
    assert profile['request'] == {'speaker': 'offline-speaker',
                                  'audio_params': {'format': 'mp3', 'sample_rate': 48000}}
    assert 'offline-secret-value' not in json.dumps(profile)
    engine.api_key = 'rotated-offline-secret'
    assert tts._voiceover_profile(engine) == profile
    monkeypatch.setattr(config, 'VOLCANO_TTS_ENDPOINT', 'wss://offline.invalid/different-endpoint')
    assert tts._voiceover_profile(engine) != profile


@pytest.mark.asyncio
async def test_same_request_reuses_bound_audio_without_synthesis(tmp_path, voice):
    first = await tts.synthesize_voiceover(narrated(), str(tmp_path))
    before = Path(first[0].audio_path).read_bytes()
    second = await tts.synthesize_voiceover(narrated(), str(tmp_path))
    assert first == second and len(voice.calls) == 1
    assert Path(second[0].audio_path).read_bytes() == before
    record = json.loads((tmp_path / 'voiceover/vo_shot_001.json').read_text())
    assert record['request']['profile']['voice'] == 'offline-voice'
    assert record['status'] == 'succeeded' and record['duration'] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['text', 'voice', 'missing', 'corrupt', 'receipt', 'duration', 'symlink'])
async def test_changed_or_damaged_voiceover_never_triggers_new_synthesis(tmp_path, voice, monkeypatch, change):
    story = narrated()
    first = await tts.synthesize_voiceover(story, str(tmp_path))
    audio = Path(first[0].audio_path)
    receipt = audio.with_suffix('.json')
    if change == 'text':
        story['shots'][0]['subtitle_text'] = '不同要求'
    elif change == 'voice':
        monkeypatch.setattr(config, 'TTS_VOICE', 'other-voice')
    elif change == 'missing':
        audio.unlink()
    elif change == 'corrupt':
        audio.write_bytes(b'not audio')
    elif change == 'symlink':
        audio.unlink()
        audio.symlink_to(voice.source)
    else:
        record = json.loads(receipt.read_text())
        if change == 'duration':
            record['duration'] = float('nan')
        else:
            record['unexpected'] = True
        receipt.write_text(json.dumps(record))
    with pytest.raises(VoiceoverStateError):
        await tts.synthesize_voiceover(story, str(tmp_path))
    assert len(voice.calls) == 1


@pytest.mark.asyncio
async def test_unbound_existing_audio_is_not_accepted_or_overwritten(tmp_path, voice):
    audio = tmp_path / 'voiceover/vo_shot_001.m4a'
    audio.parent.mkdir()
    shutil.copyfile(voice.source, audio)
    before = audio.read_bytes()
    with pytest.raises(VoiceoverStateError, match='没有请求绑定'):
        await tts.synthesize_voiceover(narrated(), str(tmp_path))
    assert not voice.calls and audio.read_bytes() == before


@pytest.mark.asyncio
async def test_cache_restore_does_not_initialize_unavailable_engine(
        tmp_path, voice, monkeypatch):
    await tts.synthesize_voiceover(narrated(), str(tmp_path))
    monkeypatch.setattr(tts, 'get_tts_engine',
                        lambda *_: (_ for _ in ()).throw(ValueError('unavailable engine')))
    result = await tts.synthesize_voiceover(narrated(), str(tmp_path))
    assert len(result) == 1 and result[0].duration > 0
    assert len(voice.calls) == 1


@pytest.mark.asyncio
async def test_volcano_cache_replays_without_credentials_but_rejects_changed_public_parameters(
        tmp_path, voice, monkeypatch):
    story = narrated()
    monkeypatch.setattr(config, 'TTS_ENGINE', 'volcano')
    monkeypatch.setattr(config, 'VOLCANO_TTS_API_KEY', 'offline-only-key')
    engine = tts.VolcanoTTSEngine(tts.select_voice_for_storyboard(story))
    monkeypatch.setattr(engine, 'synthesize', voice.synthesize)
    monkeypatch.setattr(tts, 'get_tts_engine', lambda *_: engine)
    original = await tts.synthesize_voiceover(story, str(tmp_path))
    monkeypatch.setattr(config, 'VOLCANO_TTS_API_KEY', '')
    monkeypatch.setattr(tts, 'get_tts_engine', lambda *_: pytest.fail('cache must not initialize the engine'))
    assert await tts.synthesize_voiceover(story, str(tmp_path)) == original
    assert len(voice.calls) == 1
    monkeypatch.setattr(config, 'VOLCANO_TTS_ENDPOINT', 'wss://offline.invalid/changed')
    with pytest.raises(VoiceoverStateError, match='参数已变化'):
        await tts.synthesize_voiceover(story, str(tmp_path))
    assert len(voice.calls) == 1


@pytest.mark.asyncio
async def test_local_engine_initialization_failure_has_no_pending_request_and_can_resume(
        tmp_path, voice, monkeypatch):
    story = narrated()
    monkeypatch.setattr(config, 'TTS_ENGINE', 'volcano')
    monkeypatch.setattr(config, 'VOLCANO_TTS_API_KEY', '')

    async def synthesize_with_durable_intent(text, output_path):
        record = json.loads(Path(output_path).with_suffix('.json').read_text())
        assert record['status'] == 'requested'
        return await voice.synthesize(text, output_path)

    def initialize(storyboard):
        engine = tts.VolcanoTTSEngine(tts.select_voice_for_storyboard(storyboard))
        monkeypatch.setattr(engine, 'synthesize', synthesize_with_durable_intent)
        return engine

    monkeypatch.setattr(tts, 'get_tts_engine', initialize)
    with pytest.raises(VoiceoverStateError, match='尚未发出或记录'):
        await tts.synthesize_voiceover(story, str(tmp_path))
    assert not voice.calls
    assert not (tmp_path / 'voiceover/vo_shot_001.json').exists()
    assert not (tmp_path / 'voiceover/vo_shot_001.m4a').exists()
    monkeypatch.setattr(config, 'VOLCANO_TTS_API_KEY', 'offline-restored-key')
    result = await tts.synthesize_voiceover(story, str(tmp_path))
    assert len(result) == 1 and len(voice.calls) == 1


@pytest.mark.asyncio
async def test_unknown_request_and_success_commit_failure_never_retry(tmp_path, voice, monkeypatch):
    voice.fail = True
    with pytest.raises(VoiceoverStateError, match='后续处理未完成'):
        await tts.synthesize_voiceover(narrated(), str(tmp_path))
    voice.fail = False
    with pytest.raises(VoiceoverStateError, match='尚未确认'):
        await tts.synthesize_voiceover(narrated(), str(tmp_path))
    assert len(voice.calls) == 1
    second_root = tmp_path / 'second'
    second_root.mkdir()
    write = RunWorkspace._atomic_write_json

    def fail_success(path, record):
        if record.get('status') == 'succeeded':
            raise OSError('success acknowledgement failed')
        write(path, record)

    with monkeypatch.context() as failed:
        failed.setattr(RunWorkspace, '_atomic_write_json', staticmethod(fail_success))
        with pytest.raises(VoiceoverStateError, match='后续处理未完成'):
            await tts.synthesize_voiceover(narrated(), str(second_root))
    with pytest.raises(VoiceoverStateError, match='尚未确认'):
        await tts.synthesize_voiceover(narrated(), str(second_root))
    assert len(voice.calls) == 2


@pytest.mark.asyncio
async def test_request_commit_failure_never_calls_engine(tmp_path, voice, monkeypatch):
    monkeypatch.setattr(RunWorkspace, '_atomic_write_json',
                        lambda *_: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(VoiceoverStateError, match='未开始合成'):
        await tts.synthesize_voiceover(narrated(), str(tmp_path))
    assert not voice.calls


@pytest.mark.asyncio
async def test_partial_voiceover_preserves_completed_segments_and_unknown_request(tmp_path, voice, monkeypatch):
    story = {'shots': [{'shot_id': 1, 'subtitle_text': '第一段'},
                       {'shot_id': 2, 'subtitle_text': '第二段'}]}
    synthesize = voice.synthesize

    async def second_unknown(text, output_path):
        if voice.calls:
            voice.calls.append(text)
            raise TimeoutError('second request result unknown')
        return await synthesize(text, output_path)

    with monkeypatch.context() as failed:
        failed.setattr(voice, 'synthesize', second_unknown)
        with pytest.raises(VoiceoverStateError):
            await tts.synthesize_voiceover(story, str(tmp_path))
    first_audio = (tmp_path / 'voiceover/vo_shot_001.m4a').read_bytes()
    with pytest.raises(VoiceoverStateError, match='尚未确认'):
        await tts.synthesize_voiceover(story, str(tmp_path))
    assert voice.calls == ['第一段', '第二段']
    assert (tmp_path / 'voiceover/vo_shot_001.m4a').read_bytes() == first_audio


@pytest.mark.asyncio
async def test_empty_narration_does_not_initialize_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, 'get_tts_engine', lambda *_: pytest.fail('not needed'))
    assert await tts.synthesize_voiceover({'shots': []}, str(tmp_path)) == []


@pytest.mark.asyncio
async def test_formal_manifest_failure_reuses_audio_on_local_reassembly(
        production_services, voice, monkeypatch):
    from pipeline import orchestrator
    brief, services = production_services
    generate = orchestrator.generate_storyboard

    def authored_narration(**kwargs):
        # 明确口播是这个离线测试的冻结规划输入；恢复时不再修改稿件。
        story = generate(**kwargs)
        story['shots'][0]['subtitle_text'] = '离线正式流程口播'
        return story

    monkeypatch.setattr(orchestrator, 'generate_storyboard', authored_narration)
    pipeline = orchestrator.VideoPipeline(resolution='480p', platforms=['youtube'],
                                         paid_take_budget=3, take_policy='single_pass')
    with monkeypatch.context() as failed:
        failed.setattr(RunWorkspace, 'bind_delivery',
                       lambda *_, **__: (_ for _ in ()).throw(OSError('manifest unavailable')))
        with pytest.raises(OSError, match='manifest unavailable'):
            await pipeline.run(brief)
    assert len(voice.calls) == 1
    audio = pipeline.workspace / 'voiceover/vo_shot_001.m4a'
    before = audio.read_bytes()
    counts = (len(services.submissions), len(services.author.calls), len(services.reviews))
    output = await orchestrator.VideoPipeline.from_workspace(pipeline.workspace).run()
    assert Path(output).is_file() and len(voice.calls) == 1
    assert audio.read_bytes() == before
    assert counts == (len(services.submissions), len(services.author.calls), len(services.reviews))


@pytest.mark.asyncio
async def test_formal_unknown_voiceover_cannot_silently_deliver_or_retry(
        production_services, voice, monkeypatch):
    from pipeline import orchestrator
    brief, services = production_services
    generate = orchestrator.generate_storyboard

    def authored_narration(**kwargs):
        story = generate(**kwargs)
        story['shots'][0]['subtitle_text'] = '必须保留的口播'
        return story

    monkeypatch.setattr(orchestrator, 'generate_storyboard', authored_narration)
    voice.fail = True
    pipeline = orchestrator.VideoPipeline(resolution='480p', platforms=['youtube'],
                                         paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(VoiceoverStateError, match='后续处理未完成'):
        await pipeline.run(brief)
    assert not (pipeline.workspace / 'final.mp4').exists()
    counts = (len(services.submissions), len(services.author.calls), len(services.reviews))
    voice.fail = False
    with pytest.raises(VoiceoverStateError, match='尚未确认'):
        await orchestrator.VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(voice.calls) == 1 and not (pipeline.workspace / 'final.mp4').exists()
    assert counts == (len(services.submissions), len(services.author.calls), len(services.reviews))

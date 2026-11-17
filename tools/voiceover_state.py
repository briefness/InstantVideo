"""在工作区执行锁内保存逐段口播的请求和产物，恢复不重复外部合成。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat

from pipeline.run_state import RunWorkspace
from pipeline.structured_response import load_unique_json


VERSION = 'voiceover-segment-v1'
MAX_RECORD_BYTES = 1024 * 1024


class VoiceoverStateError(RuntimeError):
    """已有请求或产物不能被安全恢复，不能跳过后继续宣称完整交付。"""


def _artifact(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError('invalid audio artifact')
        return {'size': info.st_size, 'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}


def _duration(value: object) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError('invalid audio duration')
    return float(value)


def _read(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECORD_BYTES:
            raise ValueError('invalid voiceover receipt')
        return load_unique_json(handle.read(MAX_RECORD_BYTES + 1).decode())


async def synthesize_segment(engine_factory, text: str, audio: Path, *, profile: dict, probe) -> float:
    """请求前先落盘；只有绑定相同请求、字节及实测时长的音频才能复用。"""
    receipt = audio.with_suffix('.json')
    request = {'version': VERSION, 'text': text, 'profile': profile, 'output': audio.name}
    if len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode()) > MAX_RECORD_BYTES - 2048:
        raise VoiceoverStateError('口播请求超出记录容量；未开始合成')
    if audio.parent.is_symlink() or audio.is_symlink() or receipt.is_symlink():
        raise VoiceoverStateError('口播路径不安全；未开始或重新合成')
    if receipt.exists():
        try:
            record = _read(receipt)
            if record.get('request') != request:
                raise VoiceoverStateError('口播输入或引擎参数已变化；保留原请求，不自动重新合成')
            if record.get('status') == 'requested':
                if set(record) != {'request', 'status'}:
                    raise ValueError('invalid pending receipt')
                raise VoiceoverStateError('口播请求结果尚未确认；保留原音频和记录，不自动重新合成')
            if set(record) != {'request', 'status', 'artifact', 'duration'} or record['status'] != 'succeeded':
                raise ValueError('invalid completed receipt')
            duration = _duration(record['duration'])
            if _artifact(audio) != record['artifact'] or abs(_duration(probe(str(audio))) - duration) > .05:
                raise ValueError('changed audio artifact')
            return duration
        except VoiceoverStateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
            raise VoiceoverStateError('口播记录或音频缺失、损坏；保留原请求，不自动重新合成') from exc
    if audio.exists():
        raise VoiceoverStateError('现有口播音频没有请求绑定；未复用、覆盖或重新合成')
    try:
        # 内置引擎构造只做本地配置检查；失败尚未发出请求，不创建未知记录。
        engine = engine_factory()
    except Exception as exc:
        raise VoiceoverStateError('口播引擎初始化失败；尚未发出或记录合成请求，可修正配置后继续') from exc
    try:
        RunWorkspace._atomic_write_json(receipt, {'request': request, 'status': 'requested'})
    except OSError as exc:
        raise VoiceoverStateError('口播请求无法持久化；未开始合成') from exc
    try:
        reported = _duration(await engine.synthesize(text, str(audio)))
        duration = _duration(probe(str(audio)))
        if abs(reported - duration) > .05:
            raise ValueError('audio duration differs from engine result')
        artifact = _artifact(audio)
        with audio.open('rb') as handle:
            os.fsync(handle.fileno())
        RunWorkspace._atomic_write_json(receipt, {
            'request': request, 'status': 'succeeded', 'artifact': artifact, 'duration': duration})
        return duration
    except Exception as exc:
        raise VoiceoverStateError(
            f'口播请求已记录，后续处理未完成（{type(exc).__name__}）；恢复不会自动重新合成'
        ) from exc

"""从已验收母版完成本地交付；恢复不再进入生成、口播或混音。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

import config
from pipeline.structured_response import load_unique_json
from tools import ffmpeg_ops


FILENAME = 'delivery_checkpoint.json'
VERSION = 'delivery-checkpoint-v1'
MAX_BYTES = 4 * 1024 * 1024


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _path(workspace, relative: str) -> Path:
    path = workspace.path / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or any(p.is_symlink() for p in (path, *path.parents) if p != workspace.path.parent)
            or not path.resolve().is_relative_to(workspace.path.resolve())):
        raise ValueError('交付文件路径不安全；未重新生成')
    return path


def _artifact(path: Path) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError('交付文件为空或不是普通文件')
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'size': info.st_size, 'sha256': digest}


def _matches(path: Path, expected: dict) -> bool:
    try:
        return _artifact(path) == expected
    except (OSError, ValueError):
        return False


def _context(workspace) -> str:
    # 交付依赖冻结稿与运行配置，不依赖可过期的素材URL、密钥或外部服务在线状态。
    storyboard = workspace.load_storyboard(include_observations=False)
    return _digest({'options': workspace.options.model_dump(mode='json'),
                    'storyboard': storyboard, 'model': config.SEEDANCE_MODEL})


def _record_payload(record: dict) -> dict:
    payload = {k: v for k, v in record.items() if k != 'digest'}
    payload['digest'] = _digest(payload)
    if len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_BYTES:
        raise ValueError('交付记录超出容量限制')
    return payload


def _write(workspace, record: dict) -> None:
    payload = _record_payload(record)
    workspace._atomic_write_json(_path(workspace, FILENAME), payload)


def prepare_delivery(workspace, final_path: str, *, expected_duration: float, report: dict) -> None:
    """先冻结通过技术验收的母版，再开始任何可恢复的平台导出。"""
    checkpoint = _path(workspace, FILENAME)
    if (checkpoint.exists() or workspace.manifest.delivery_fingerprint
            or workspace.manifest.delivery_base is not None):
        raise ValueError('交付记录已存在；应恢复原母版，不得重建生产流程')
    name = 'preview.mp4' if workspace.options.delivery_mode == 'preview' else 'final.mp4'
    master = _path(workspace, name)
    if Path(final_path).resolve() != master.resolve():
        raise ValueError('交付母版路径与运行模式不一致')
    validation = ffmpeg_ops.validate_publish_ready(
        str(master), expected_duration=expected_duration, duration_tolerance=.15)
    base = {'version': VERSION, 'context': _context(workspace), 'output': name,
            'artifact': _artifact(master), 'validation': validation, 'report': report}
    record = _record_payload({'base': base, 'exports': {}})
    # 先原子保存完整权威依据，再写可重建的进度文件。只保存 hash 无法
    # 恢复双文件提交中断；反过来信任未绑定文件又会绕过母版身份检查。
    workspace.bind_delivery(_digest(base), base=base)
    _write(workspace, record)


def _read(workspace) -> dict:
    try:
        path = _path(workspace, FILENAME)
        authoritative = workspace.manifest.delivery_base
        if authoritative is not None and _digest(authoritative) != workspace.manifest.delivery_fingerprint:
            raise ValueError('invalid authoritative delivery identity')
        if authoritative is not None and not path.exists():
            # 只有已原子绑定的母版能重建导出进度；历史无绑定记录不能补信任。
            record = _record_payload({'base': authoritative, 'exports': {}})
        else:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                    raise ValueError('invalid delivery file')
                data = handle.read(MAX_BYTES + 1)
            record = load_unique_json(data.decode())
        base = record['base']
        if (set(record) != {'base', 'exports', 'digest'}
                or base['version'] != VERSION
                or _digest(base) != workspace.manifest.delivery_fingerprint
                or (authoritative is not None and base != authoritative)
                or record['digest'] != _digest({k: v for k, v in record.items() if k != 'digest'})
                or not isinstance(record['exports'], dict)
                or set(record['exports']) - set(workspace.options.platforms)):
            raise ValueError('invalid delivery identity')
        duration = base['validation']['duration']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid master duration')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError('交付记录缺失或损坏；保留现有素材，未重新生成') from exc
    if base['context'] != _context(workspace):
        raise ValueError('交付记录与当前稿件或模型不一致；未重新生成')
    expected_name = 'preview.mp4' if workspace.options.delivery_mode == 'preview' else 'final.mp4'
    if base['output'] != expected_name or not _matches(_path(workspace, expected_name), base['artifact']):
        raise ValueError('已验收成片缺失、损坏或内容变化；未重新生成或覆盖')
    return record


def _export_matches(workspace, platform: str, record: dict) -> bool:
    entry = record['exports'].get(platform)
    return bool(isinstance(entry, dict) and isinstance(entry.get('validation'), dict)
                and _matches(_path(workspace, f'exports/{platform}.mp4'), entry.get('artifact')))


def verified_output(workspace) -> str | None:
    """只读检查当前交付是否完整；历史工作区不补造母版证明。"""
    record = _read(workspace)
    if all(_export_matches(workspace, p, record) for p in workspace.options.platforms):
        return str(_path(workspace, record['base']['output']))
    return None


def resume_delivery(workspace) -> str | None:
    """由正式工作区执行锁保护；只能复用母版并补做本地导出。"""
    checkpoint = _path(workspace, FILENAME)
    if (not workspace.manifest.delivery_fingerprint
            and workspace.manifest.delivery_base is None and not checkpoint.exists()):
        return None
    record = _read(workspace)
    if not checkpoint.exists():
        _write(workspace, record)
    base = record['base']
    master = _path(workspace, base['output'])
    exports_dir = _path(workspace, 'exports')
    exports_dir.mkdir(exist_ok=True)
    for platform in workspace.options.platforms:
        if _export_matches(workspace, platform, record):
            continue
        destination = _path(workspace, f'exports/{platform}.mp4')
        fd, temporary = tempfile.mkstemp(prefix=f'.{platform}-', suffix='.mp4', dir=exports_dir)
        os.close(fd)
        try:
            ffmpeg_ops.export_for_platform(str(master), platform, temporary)
            validation = ffmpeg_ops.validate_publish_ready(
                temporary, platform=platform, expected_duration=base['validation']['duration'])
            artifact = _artifact(Path(temporary))
            with open(temporary, 'rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            directory = os.open(exports_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            validation['path'] = str(destination)
            record['exports'][platform] = {'artifact': artifact, 'validation': validation}
            _write(workspace, record)
        finally:
            Path(temporary).unlink(missing_ok=True)
    workspace._atomic_write_json(_path(workspace, 'publish_validation.json'), {
        'final': base['validation'],
        'exports': {p: record['exports'][p]['validation'] for p in workspace.options.platforms}})
    workspace._atomic_write_json(_path(workspace, 'delivery_report.json'), {
        **base['report'], 'artifact_integrity': 'sha256',
        'delivery_fingerprint': workspace.manifest.delivery_fingerprint})
    workspace.mark_succeeded(str(master))
    return str(master)

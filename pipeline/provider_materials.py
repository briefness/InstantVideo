"""供应商原始产物与本地观察分离；派生图只保存指向原素材的定位。

信任不是文件名或 generated=True。来源记录由当前视频凭据签名，绑定 API
返回的任务、模型、时间、地址和本地原片字节；不跨账号、不剪辑后冒充原片。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import config
from pipeline.readiness import GenerationReadinessError
from pipeline.run_state import RunWorkspace


VERSION = 'provider-original-materials-v1'


def _hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def _seal(value: dict) -> str:
    key = str(config.ARK_API_KEY_SEEDANCE or '').encode()
    if not key:
        raise GenerationReadinessError('缺少视频凭据，不能核对原始素材所属调用范围')
    return hmac.new(key, _canonical(value), hashlib.sha256).hexdigest()


def _write(path: Path, payload: dict) -> None:
    # 地址可能含短期签名；不写日志，记录只允许当前系统用户读取。
    RunWorkspace._atomic_write_json(path, {**payload, 'seal': _seal(payload)})
    path.chmod(0o600)


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128_000:
        raise GenerationReadinessError('素材来源记录缺失或大小无效；不能把本地加工图当作受信原图')
    try:
        payload = json.loads(path.read_text())
        signature = payload.pop('seal')
        if not isinstance(signature, str) or not hmac.compare_digest(signature, _seal(payload)):
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError):
        raise GenerationReadinessError('素材来源记录已变化或不属于当前视频凭据；未提交生成') from None
    return payload


def receipt_path(video: str | Path) -> Path:
    path = Path(video)
    return path.with_name(path.stem + '_provider_output.json')


def derivation_path(image: str | Path) -> Path:
    return Path(str(image) + '.material.json')


def move_original(source: Path, destination: Path) -> None:
    """候选片的来源与原片一起归档；中途失败最多暂停，不留下可误用的来源。"""
    receipt = receipt_path(source)
    target = receipt_path(destination)
    if destination.exists():
        raise GenerationReadinessError('原片归档目标已存在；未覆盖其他候选片')
    if receipt.exists():
        payload = _read(receipt)
        if payload['video_sha256'] != _hash(source):
            raise GenerationReadinessError('归档原片与供应商来源不一致')
        if target.exists() and _read(target) != payload:
            raise GenerationReadinessError('归档目标包含其他任务的来源')
        _write(target, payload)
    elif target.exists():
        raise GenerationReadinessError('归档目标来源与当前原片无法核对')
    source.replace(destination)
    receipt.unlink(missing_ok=True)


def _https(value: object) -> str:
    if not isinstance(value, str) or len(value) > 8192:
        raise GenerationReadinessError('供应商原始素材地址无效')
    try:
        parts = urlsplit(value)
        if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
                or parts.port not in (None, 443) or parts.fragment
                or any(ord(c) < 33 for c in value)):
            raise ValueError
    except ValueError:
        raise GenerationReadinessError('供应商原始素材地址无效') from None
    return value


def capture_output(video: str, result: dict) -> None:
    """下载成功后记录真实查询响应；旧服务/历史数据不凭空补来源。"""
    raw = result.get('provider_output')
    if raw is None:
        return
    if (not isinstance(raw, dict) or raw.get('task_id') != result.get('provider_task_id')
            or type(raw.get('created_at')) not in (int, float)
            or not math.isfinite(raw['created_at']) or raw['created_at'] <= 0
            or not isinstance(raw.get('model'), str) or not raw['model']):
        raise GenerationReadinessError('官方产物响应缺少任务、模型或创建时间；保留原片，不重新生成')
    from tools.ffmpeg_ops import get_video_info
    try:
        stream = next(s for s in get_video_info(video)['streams'] if s.get('codec_type') == 'video')
        duration = float(stream['duration'])
        digest = _hash(Path(video))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise GenerationReadinessError('已生成原片的媒体信息不可确认；保留任务，不重新提交') from exc
    payload = {'version': VERSION, 'task_id': raw['task_id'], 'model': raw['model'],
               'created_at': raw['created_at'], 'base_url': config.ARK_BASE_URL_SEEDANCE,
               'video_url': _https(raw.get('video_url')), 'video_sha256': digest,
               'last_frame_url': _https(raw['last_frame_url']) if raw.get('last_frame_url') else None,
               'duration': duration, 'width': stream['width'], 'height': stream['height'],
               'fps': stream.get('avg_frame_rate'), 'codec': stream.get('codec_name')}
    path = receipt_path(video)
    if path.exists() and _read(path) != payload:
        raise GenerationReadinessError('同一原片的供应商来源发生变化；未覆盖来源或重新生成')
    _write(path, payload)


def bind_derivation(image: str, video: str, *, kind: str, timestamp: float,
                    character: str | None = None, box=None,
                    selected_end_seconds: float | None = None) -> None:
    """裁剪是观察证据，生成端将携带其原视频及定位；不上传裁剪字节。"""
    if not receipt_path(video).exists():
        return  # 历史结果保持历史解释，不能推测它曾属于当前账号。
    source = _read(receipt_path(video))
    if (kind not in {'identity', 'official_tail', 'selected_state'}
            or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
            or not 0 <= timestamp <= source['duration']):
        raise GenerationReadinessError('派生观察的用途或时间点无效')
    if box is not None and (not isinstance(box, (tuple, list)) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        raise GenerationReadinessError('人物观察区域无效')
    if selected_end_seconds is not None and (kind == 'identity'
            or type(selected_end_seconds) not in (int, float) or not math.isfinite(selected_end_seconds)
            or not timestamp <= selected_end_seconds <= source['duration']):
        raise GenerationReadinessError('状态观察的选定结束位置无效')
    _write(derivation_path(image), {'version': VERSION, 'image_sha256': _hash(Path(image)),
        'video': str(Path(video).resolve()), 'video_sha256': source['video_sha256'],
        'kind': kind, 'timestamp': timestamp, 'character': character, 'box': box,
        **({'selected_end_seconds': selected_end_seconds} if selected_end_seconds is not None else {})})


def bind_view(image: str, source: str, window: dict) -> None:
    if not derivation_path(source).exists():
        return
    record = _read(derivation_path(source))
    if record['image_sha256'] != _hash(Path(source)):
        raise GenerationReadinessError('取景源图与来源记录不一致')
    if record.get('box'):
        # 来源已是本地呈现的区域，下一次裁切必须映回同一官方原件坐标。
        x, y, r, b = record['box']
        left, top, scale = (window[k] for k in ('left', 'top', 'scale'))
        record['box'] = [x + left * (r-x), y + top * (b-y),
                         x + (left+scale) * (r-x), y + (top+scale) * (b-y)]
        record.pop('view_window', None)
        record['image_sha256'] = _hash(Path(image))
    else:
        record.update(image_sha256=_hash(Path(image)), view_window=window)
    _write(derivation_path(image), record)


def _source(image: str) -> tuple[dict, dict] | None:
    if image.startswith(('https://', 'http://', 'asset://', 'data:')):
        return None
    path = derivation_path(image)
    if not path.exists():
        return None
    record = _read(path)
    source = _read(receipt_path(record['video']))
    try:
        matches = (record.get('version') == VERSION and source.get('version') == VERSION
            and record['image_sha256'] == _hash(Path(image))
            and record['video_sha256'] == source['video_sha256']
            and source['video_sha256'] == _hash(Path(record['video']))
            and source['base_url'] == config.ARK_BASE_URL_SEEDANCE)
    except (OSError, KeyError, TypeError):
        matches = False
    if not matches:
        raise GenerationReadinessError('原始素材、观察图或调用网关已变化；未提交生成')
    return record, source


def view_execution_capability(image: str) -> dict:
    """派生视图不能继承原件的信任或像素首帧能力；规划和提交共用此边界。"""
    source = _source(image)
    if source is None:
        return {'mode': 'first_frame_pixels'}
    record, original = source
    return {'mode': 'original_video_guidance',
            'source_task_id': original['task_id'], 'source_sha256': original['video_sha256'],
            'observation_sha256': record['image_sha256'], 'timestamp': record['timestamp']}


def material_plan(images: list[str], role: str | None, *, live: bool = False,
                  strict_generated: bool = False, execution: dict | None = None) -> dict | None:
    """编译单一素材清单，来源比较不受时间影响；仅新提交检查有效期。"""
    sources = [_source(path) for path in images]
    if strict_generated:
        for path, source in zip(images, sources):
            if path.startswith(('https://', 'http://', 'asset://', 'data:')):
                continue
            p = Path(path)
            if source is None and (p.parent.name in {'character_refs', 'reference_views'}
                    or (p.parent.name == 'shots' and p.name.endswith('_lastframe.jpg'))):
                raise GenerationReadinessError('历史派生素材缺少官方原始产物来源；需先恢复原任务来源，不能直接上传裁剪或抽帧图试错')
    if not any(sources):
        return None
    content, bindings, originals, images_count, videos_count = [], [], {}, 0, 0
    duration = 0.
    for index, (path, item) in enumerate(zip(images, sources), 1):
        if item is None:
            images_count += 1
            content.append({'type': 'image_url', 'image_url': {'url': path}, 'role': role or 'reference_image'})
            bindings.append({'logical_index': index, 'label': f'图片{images_count}',
                             'sha256': _hash(Path(path)) if Path(path).is_file() else hashlib.sha256(path.encode()).hexdigest()})
            continue
        record, source = item
        model = source['model'].replace('.', '-').lower()
        if not model.startswith(('doubao-seedance-2-0', 'doubao-seedance-2-5')):
            raise GenerationReadinessError('原素材模型不在官方人像产物信任范围内')
        if live:
            age = time.time() - source['created_at']
            # 30 天是产物信任期，原始 URL 只有 24 小时；不能混为一谈。
            if not 0 <= age < 23 * 3600:
                raise GenerationReadinessError('原始素材链接已接近或超过24小时有效期；需先转存未修改原件或确认原任务可用地址，不重新生成')
        original_tail = record['kind'] == 'official_tail' and not record.get('view_window')
        if original_tail and source['last_frame_url']:
            kind, url = 'image_url', source['last_frame_url']
        else:
            kind, url = 'video_url', source['video_url']
        if kind == 'video_url' and role == 'first_frame':
            raise GenerationReadinessError('剪辑区间或裁剪图没有对应官方首帧；视频参考不能冒充像素一致的 first_frame，未提交生成')
        key = (kind, url)
        if key not in originals:
            if kind == 'image_url':
                images_count += 1
                label = f'图片{images_count}'
                actual_role = role or 'reference_image'
            else:
                videos_count += 1
                label, actual_role = f'视频{videos_count}', 'reference_video'
                seconds = source['duration']
                pixels = source['width'] * source['height']
                from fractions import Fraction
                try:
                    fps = float(Fraction(source['fps']))
                except (ValueError, TypeError, ZeroDivisionError):
                    fps = 0.
                if (not 2 <= seconds <= 15 or not 407696 <= pixels <= 8295044
                        or not 300 <= min(source['width'], source['height'])
                        or max(source['width'], source['height']) > 6000
                        or not .4 <= source['width'] / source['height'] <= 2.5
                        or not 24 <= fps <= 60 or source['codec'] not in {'h264', 'hevc'}
                        or Path(record['video']).stat().st_size > 200_000_000):
                    raise GenerationReadinessError('原视频超出 Fast 参考视频时长或像素范围；不能剪辑或缩放受信原件后试错')
                duration += seconds
            originals[key] = label
            content.append({'type': kind, kind: {'url': url}, 'role': actual_role})
        label = originals[key]
        locator = '' if kind == 'image_url' else f'在{record["timestamp"]:.3f}秒处'
        if record.get('box'):
            locator += f'、归一化区域{json.dumps(record["box"], separators=(",", ":"))}'
        if record.get('character'):
            locator += f'内的人物{json.dumps(record["character"], ensure_ascii=False)}'
        if record.get('view_window'):
            locator += f'，按区域{json.dumps(record["view_window"], ensure_ascii=False)}建立新镜头取景'
        bindings.append({'logical_index': index, 'label': label + locator,
            'source_task_id': source['task_id'], 'source_sha256': source['video_sha256'],
            'observation_sha256': record['image_sha256'],
            'receipt_sha256': hashlib.sha256(_canonical(source)).hexdigest()})
        if execution is not None:
            bindings[-1].update(usage=record['kind'], media_label=label, media_type=kind,
                timestamp=record['timestamp'], source_duration=source['duration'], source_fps=source['fps'],
                selected_end_seconds=record.get('selected_end_seconds'))
    if images_count > 9 or videos_count > 3 or duration > 15:
        raise GenerationReadinessError('完整参考超出 Fast 的9图/3视频/视频合计15秒限制；未删除参考、裁剪原片或提交任务')
    if videos_count and any(c['role'] == 'first_frame' for c in content):
        raise GenerationReadinessError('官方首帧与视频参考不可混用；未改变连续性要求')
    plan = {'version': VERSION, 'content': content, 'bindings': bindings,
            'video_duration': duration, 'image_role': role if images_count else None}
    if execution is not None:
        from pipeline.provider_operation import compile_operation
        plan.update(execution=dict(execution), operation=compile_operation(bindings, execution))
    return plan


def plan_evidence(plan: dict | None) -> dict | None:
    if plan is None:
        return None
    return {'version': VERSION, 'bindings': plan['bindings'], 'image_role': plan['image_role'],
            'content_sha256': hashlib.sha256(_canonical(plan['content'])).hexdigest(),
            **({key: plan[key] for key in ('execution', 'operation')} if 'operation' in plan else {})}


def record_request(root: Path, shot_id: int, attempt: int, payload: dict) -> None:
    """创建前保存最终请求证据；400 无 task ID 也能回放，地址仅留指纹。"""
    path = root / 'submission_requests' / f'shot_{shot_id:03d}_attempt_{attempt:03d}.json'
    path.parent.mkdir(exist_ok=True)
    if path.exists() and _read(path) != payload:
        raise GenerationReadinessError('已有提交请求证据与当前请求不同；未覆盖或提交')
    _write(path, payload)


def material_prompt(prompt: str, plan: dict) -> str:
    labels = {b['logical_index']: b['label'] for b in plan['bindings']}
    # 只降低程序写入的编号引用，不改作者动作、目标或人物名称。
    def replace(match):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start or end - start > 9 or any(i not in labels for i in range(start, end + 1)):
            raise GenerationReadinessError('生成提示词的素材编号与实际素材不一致')
        return '、'.join(labels[i] for i in range(start, end + 1))
    prompt = re.sub(r'\bImages?\s+(\d+)(?:\s*[-–]\s*(\d+))?\b', replace, prompt, flags=re.I)
    if 'operation' in plan:
        from pipeline.provider_operation import operation_prefix
        return (operation_prefix(plan['operation']) + '[素材定位：' + '；'.join(labels.values())
                + '。人物身份与动作状态按上述来源绑定；新动作与摄影按当前分镜执行。] ' + prompt)
    return ('[原始素材引用：' + '；'.join(labels.values())
            + '。视频只提供所指人物身份或指定时间的场景状态；生成当前分镜的新动作与运镜，'
              '不要重演整段参考视频，不继承参考声音。] ' + prompt)

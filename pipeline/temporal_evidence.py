"""有界的原片时序证据；摘要不能代替像素，摄影与物理进展分别解释。"""
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

POLICY = 'temporal-window-v1'
VISUAL_SUPPORT = 'observation-support-v5-visual-evidence'
INSTRUCTION = (
    'Read the labelled previous_tail/previous_clip images in timestamp order, not as alternative identity references. '
    'They show recent action history, contact/support relationships and unfinished motion. '
    'A single endpoint cannot establish velocity or completion. Describe only visible motion, including '
    'whether it continues or has stopped; do not invent numerical speed or unseen transitions. '
    'Keep the current camera plan independent from the prior camera. A cut may change framing, '
    'not silently relocate objects, reset completed work or reverse an ongoing action. '
    'The next local action must be feasible from the actual history; planned placements are not observations.'
)
SUPPORT_INSTRUCTION = (
    '原始画面是主要证据，文字observation是待核对的陈述，不是已经证实的事实。'
    '按scope和timestamp分别查看上一镜末段、本镜开头和本镜过程，不能补写未出现的过渡。'
    '核对当前任务要求中的参与者、对象、承载/包含/接触关系与动作方向：靠近某物不等于放在其上，'
    '接触不等于释放、到达或完成。不要把世界中的物体与相机坐标混淆。'
    '用场景用途解释必要功能，但不得新增通用物品禁令、表演姿态、精确轨迹或原要求之外的终态。'
    '镜头切换可以省略无关表演，不能用想象的移位补足明确的物理衔接。'
    '图像直接反驳文字陈述时返回contradicted；遮挡、采样间隔或证据不足时返回unknown，不能猜通过。'
    '上一镜证据仅用于衔接，不能证明本镜完成了某个事件。仍只返回既定判定目录，不新增审查项。'
)


def enabled(shot: dict) -> bool:
    policy = shot.get('continuity_evidence')
    if policy not in (None, POLICY):
        raise ValueError('未知的时序证据协议')
    return policy == POLICY


def sample_clip(video_path: str, shot: dict, *, scope: str, timestamps=None) -> dict:
    from tools.ffmpeg_ops import get_video_info
    from tools.frame_extractor import extract_frame
    from pipeline.media_selection import selected_media_window
    path = Path(video_path)
    if scope not in {'previous_tail', 'previous_clip', 'current_clip'} or not path.is_file() or path.stat().st_size > 1_000_000_000:
        raise ValueError('时序证据缺少有效的有界原片')
    info = get_video_info(str(path))
    stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), {})
    duration = float(stream.get('duration', 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('时序证据无法确认视频时长')
    window = selected_media_window(shot, duration)
    end = window['last_sample_seconds'] if window else max(0., duration - .05)
    start = max(0., end - 2.) if scope == 'previous_tail' else 0.
    count = 9 if scope == 'previous_tail' else min(25, max(2, math.ceil(end * 4) + 1))
    times = sorted({round(start + (end - start) * i / (count - 1), 3) for i in range(count)})
    if timestamps is not None:
        times = list(timestamps)
        if (scope != 'current_clip' or not 1 <= len(times) <= 64 or times != sorted(set(times))
                or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= end + .001 for t in times)):
            raise ValueError('核对采样与原观察区间不一致')
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    source = {'scope': scope, 'video_sha256': digest, 'start_seconds': times[0],
              'end_seconds': times[-1], 'frames': []}
    content = []
    with tempfile.TemporaryDirectory(prefix='temporal-evidence-') as directory:
        for index, timestamp in enumerate(times, 1):
            frame = Path(directory) / f'{index}.jpg'
            extract_frame(str(path), str(frame), timestamp=timestamp)
            data = frame.read_bytes()
            row = {'index': index, 'timestamp': timestamp, 'sha256': hashlib.sha256(data).hexdigest()}
            source['frames'].append(row)
            content.extend([
                {'type': 'text', 'text': json.dumps({'scope': scope, 'video_sha256': digest, **row}, sort_keys=True)},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(),
                                                   'detail': 'high'}},
            ])
    return {'manifest': {'version': POLICY, 'sources': [source]}, 'content': content}


def combine(*bundles: dict | None) -> dict:
    present = [b for b in bundles if b is not None]
    result = {'manifest': {'version': POLICY, 'sources': [s for b in present for s in b['manifest']['sources']]},
              'content': [item for b in present for item in b['content']]}
    validate_bundle(result)
    return result


def validate_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or set(manifest) != {'version', 'sources'}
            or manifest['version'] != POLICY or not isinstance(manifest['sources'], list)
            or not 1 <= len(manifest['sources']) <= 2):
        raise ValueError('时序证据清单无效')
    scopes = set()
    for source in manifest['sources']:
        if (not isinstance(source, dict) or set(source) != {
                'scope', 'video_sha256', 'start_seconds', 'end_seconds', 'frames'}):
            raise ValueError('时序证据来源结构无效')
        scope = source['scope']
        if not isinstance(scope, str) or scope not in {'previous_tail', 'previous_clip', 'current_clip'} or scope in scopes:
            raise ValueError('时序证据来源重复或无效')
        scopes.add(scope)
        frames = source['frames']
        if not isinstance(frames, list) or not 1 <= len(frames) <= (9 if scope == 'previous_tail' else 64):
            raise ValueError('时序采样超出范围')
        for i, row in enumerate(frames, 1):
            if (not isinstance(row, dict) or set(row) != {'index', 'timestamp', 'sha256'}
                    or type(row['index']) is not int or row['index'] != i):
                raise ValueError('时序采样编号不连续')
        times = [f['timestamp'] for f in frames]
        if (any(type(t) not in (float, int) or not math.isfinite(t) or t < 0
                for t in [*times, source['start_seconds'], source['end_seconds']])
                or times != sorted(set(times)) or times[0] != source['start_seconds']
                or times[-1] != source['end_seconds']):
            raise ValueError('时序采样顺序或区间不一致')
        for digest in [source['video_sha256'], *(f['sha256'] for f in frames)]:
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('时序证据摘要无效')


def validate_bundle(bundle: dict) -> dict:
    """缓存指纹必须绑定实际送入核对器的像素、时间与来源。"""
    manifest, content = bundle['manifest'], bundle['content']
    validate_manifest(manifest)
    expected = []
    for source in manifest['sources']:
        scope = source['scope']
        for row in source['frames']:
            expected.append((json.dumps({'scope': scope, 'video_sha256': source['video_sha256'], **row}, sort_keys=True), row['sha256']))
    if len(content) != len(expected) * 2:
        raise ValueError('时序证据像素数量不一致')
    for i, (label, digest) in enumerate(expected):
        if content[2*i] != {'type': 'text', 'text': label}:
            raise ValueError('时序证据标签与清单不一致')
        url = content[2*i+1]['image_url']['url']
        if not url.startswith('data:image/jpeg;base64,'):
            raise ValueError('核对只能使用绑定的本地原片像素')
        data = base64.b64decode(url.split(',', 1)[1], validate=True)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('时序证据像素来源不一致')
    return deepcopy(manifest)


def validate_motion(value: dict, manifest: dict) -> dict:
    validate_manifest(manifest)
    if len(manifest['sources']) != 1 or manifest['sources'][0]['scope'] != 'previous_tail':
        raise ValueError('运动交接只能引用上一镜历史')
    frames = manifest['sources'][0]['frames']
    if (not isinstance(value, dict) or set(value) != {'summary', 'sample_indices'}
            or not isinstance(value['summary'], str) or not value['summary'].strip() or len(value['summary']) > 1200):
        raise ValueError('运动摘要缺少有效的观察来源')
    indices = value['sample_indices']
    if (not isinstance(indices, list) or len(indices) < 2 or any(type(i) is not int or not 1 <= i <= len(frames) for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('运动摘要必须引用有序的连续画面证据')
    return deepcopy(value)


def motion_instruction(shot: dict) -> str:
    motion = shot.get('motion_handoff')
    if motion is None:
        return ''
    if not enabled(shot) or set(motion) != {'summary', 'sample_indices', 'evidence'}:
        raise ValueError('运动交接协议不一致')
    validate_motion({k: motion[k] for k in ('summary', 'sample_indices')}, motion['evidence'])
    source = motion['evidence']['sources'][0]
    return (f"Observed incoming motion over the prior video interval {source['start_seconds']:g}-{source['end_seconds']:g}s: "
            + motion['summary'] + '. Continue this physical progress into the local work; do not replay the reference clip. '
            'The original reference video is motion/identity guidance, not an enforced camera pose.')

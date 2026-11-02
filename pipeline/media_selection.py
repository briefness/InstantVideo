"""新镜头的评审、身份中点与续拍端点共享既有剪辑前缀。"""
from __future__ import annotations

import math


def media_selection_policy(shot: dict | None) -> dict | None:
    version = (shot or {}).get('media_selection')
    if version is None:
        return None
    duration = shot.get('duration')
    if (version != 'planned-prefix-v1' or type(duration) not in (int, float)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError('无效的成片媒体区间')
    return {'version': version, 'start_seconds': 0., 'end_seconds': duration}


def selected_media_window(shot: dict | None, source_duration: float) -> dict | None:
    policy = media_selection_policy(shot)
    if policy is None:
        return None
    if (type(source_duration) not in (int, float) or not math.isfinite(source_duration)
            or source_duration <= 0):
        raise ValueError('原片时长必须是有限正数')
    # 短原片不延长。既有全片/导出检查仍负责判断可用长度，不新增时长门槛。
    duration = min(policy['end_seconds'], source_duration)
    return {**policy, 'planned_end_seconds': policy['end_seconds'], 'end_seconds': duration,
            'duration': duration, 'midpoint_seconds': duration / 2,
            # 沿用结果观察的末端采样间隔；尾帧不再另取原片的完整尾部。
            'last_sample_seconds': round(max(0., duration - .05), 3)}


def resolve_media_window(shot: dict | None, video_path: str) -> dict | None:
    if media_selection_policy(shot) is None:
        return None
    from tools.ffmpeg_ops import get_video_info
    streams = get_video_info(video_path).get('streams', [])
    video = next((stream for stream in streams if stream.get('codec_type') == 'video'), {})
    # 音轨填充和封装时长不能冒充可观察画面；与 normalize_video 的源范围一致。
    try:
        duration = float(video['duration'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('无法确认原片视频流的有效范围') from exc
    return selected_media_window(shot, duration)

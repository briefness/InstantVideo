"""Bounded output requirements; never silently shorten or add requested shots."""
from math import ceil

import config


def allocate_requested_shape(duration: int, intent: dict | None) -> list[int]:
    if type(duration) is not int or not config.MIN_SHOT_DURATION <= duration <= 600:
        raise ValueError('单次计划时长必须为 4–600 秒整数；不会自动截短需求')
    intent = intent or {}
    requested = intent.get('requested_duration_seconds')
    if requested is not None and requested != duration:
        raise ValueError('结构化时长与已确认的用户时长冲突')
    minimum = ceil(duration / config.MAX_SHOT_DURATION)
    maximum = duration // config.MIN_SHOT_DURATION
    count = intent.get('requested_shot_count')
    if count is None:
        count = max(minimum, min(10, ceil(duration / 6), maximum))
    if type(count) is not int or not minimum <= count <= maximum:
        raise ValueError(f'{duration} 秒无法分为 {count} 个镜头：每镜需 4–15 秒，可用段数为 {minimum}–{maximum}；未改写需求')
    base, remainder = divmod(duration, count)
    return [base + (index < remainder) for index in range(count)]

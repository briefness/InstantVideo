"""Bounded output requirements; never silently shorten or add requested shots."""
from math import ceil, floor

import config
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DurationLimitsSpec(BaseModel):
    """冻结规划所用的容量；恢复历史稿件时不按新配置重算拓扑。"""
    model_config = ConfigDict(extra='forbid', strict=True)

    min_duration: int = Field(ge=config.MIN_SHOT_DURATION, le=config.MAX_SHOT_DURATION)
    max_duration: int = Field(ge=config.MIN_SHOT_DURATION, le=config.MAX_SHOT_DURATION)

    @model_validator(mode='after')
    def ordered(self):
        if self.min_duration > self.max_duration:
            raise ValueError('单镜时长能力范围为空')
        return self


def duration_bounds(duration: int, intent: dict | None) -> tuple[int, int]:
    if type(duration) is not int or not config.MIN_SHOT_DURATION <= duration <= 600:
        raise ValueError('单次计划时长必须为 4–600 秒整数；不会自动截短需求')
    margin = max(1, floor(duration / 10)) if (intent or {}).get('duration_mode') == 'approximate' else 0
    return max(config.MIN_SHOT_DURATION, duration-margin), min(600, duration+margin)


def allocate_requested_shape(duration: int, intent: dict | None, allocation: list[int] | None = None,
                             *, duration_limits: dict | None = None) -> list[int]:
    if type(duration) is not int or not config.MIN_SHOT_DURATION <= duration <= 600:
        raise ValueError('单次计划时长必须为 4–600 秒整数；不会自动截短需求')
    intent = intent or {}
    requested = intent.get('requested_duration_seconds')
    if requested is not None and requested != duration:
        raise ValueError('结构化时长与已确认的用户时长冲突')
    lower, upper = duration_bounds(duration, intent)
    limits = DurationLimitsSpec.model_validate(duration_limits) if duration_limits is not None else None
    shot_min = limits.min_duration if limits else config.MIN_SHOT_DURATION
    shot_max = limits.max_duration if limits else config.MAX_SHOT_DURATION
    minimum = ceil(lower / shot_max)
    maximum = upper // shot_min
    if minimum > maximum:
        raise ValueError(f'{duration} 秒的允许范围与单镜 {shot_min}–{shot_max} 秒能力无交集；未改写需求')
    count = intent.get('requested_shot_count')
    if count is None:
        preferred = min(10, ceil(duration / 6))
        if limits:
            # 先保持用户目标可达，再考虑段数偏好。容差只用于内容分配或无精确解时。
            count = min(range(minimum, maximum + 1), key=lambda n: (
                abs(max(n * shot_min, min(duration, n * shot_max)) - duration),
                abs(n - preferred), n))
        else:
            # 历史计划没有冻结能力，保留原算法以验证和恢复旧稿件。
            count = max(minimum, min(preferred, maximum))
    if type(count) is not int or not minimum <= count <= maximum:
        raise ValueError(f'{duration} 秒无法分为 {count} 个镜头：每镜需 {shot_min}–{shot_max} 秒，可用段数为 {minimum}–{maximum}；未改写需求')
    if allocation is not None:
        if (intent.get('duration_mode') not in ('exact', 'approximate')
                or not isinstance(allocation, list) or len(allocation) != count
                or any(type(d) is not int or not shot_min <= d <= shot_max for d in allocation)
                or not lower <= sum(allocation) <= upper):
            raise ValueError('创作时长分配超出总时长、段数或单镜能力边界；未截短、补长或重试')
        return list(allocation)
    feasible = max(count * shot_min, min(duration, count * shot_max))
    base, remainder = divmod(feasible, count)
    return [base + (index < remainder) for index in range(count)]

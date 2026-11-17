"""审查传输容量与语义证据范围分离；选择始终绑定当前原稿目录。"""

# 既有单次审查的资源边界，不是全片动作数或单个目标的语义上限。
MAX_REVIEW_UNITS = 120
BATCH_REVIEW_UNITS = 24


def selection_schema(targets: dict, *, legacy_limit: int | None = None) -> dict:
    if not isinstance(targets, dict) or not 1 <= len(targets) <= MAX_REVIEW_UNITS:
        raise ValueError('分镜审查项目数量超出单批范围')
    return {'type': 'array', 'items': {'type': 'string', 'enum': list(targets)},
            'minItems': 1, 'maxItems': legacy_limit if legacy_limit is not None else len(targets),
            'uniqueItems': True}


def validate_selection(keys: object, targets: dict, *, label: str,
                       legacy_limit: int | None = None) -> list[str]:
    """不排序、截断或去重响应；诊断指出实际违反的约束。"""
    schema = selection_schema(targets, legacy_limit=legacy_limit)
    if not isinstance(keys, list) or not keys:
        raise ValueError(f'{label}：目标证据必须是非空动作引用列表')
    if any(not isinstance(k, str) for k in keys):
        raise ValueError(f'{label}：目标证据引用必须是字符串')
    if len(set(keys)) != len(keys):
        raise ValueError(f'{label}：目标证据重复引用同一动作')
    if any(k not in targets for k in keys):
        raise ValueError(f'{label}：目标证据引用不属于当前审查动作目录，不能跨批或虚构来源')
    if len(keys) > schema['maxItems']:
        raise ValueError(f'{label}：历史协议目标证据容量超限（{len(keys)}/{schema["maxItems"]}），'
                         '保留原记录，不能自动升级协议或重新审查')
    order = {key: i for i, key in enumerate(targets)}
    if keys != sorted(keys, key=order.__getitem__):
        raise ValueError(f'{label}：目标证据未按原稿时间线顺序引用')
    return keys

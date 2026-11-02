"""Keep explicit unknown observations separate from false judgments and missing fields."""
from copy import deepcopy


def project_boundary_observations(value, *, applicable: bool, fields: tuple[str, ...]):
    """无上一镜时跨镜判断不参与合取；true 是内部中性值，不是模型观察结论。"""
    if applicable or not isinstance(value, dict):
        return value
    result = deepcopy(value)
    for field in fields:
        result[field] = True
    return result


OBSERVABILITY_INSTRUCTION = (
    'Observation protocol: every boolean judgment supports true, false, or null. '
    'Use true only for visible supporting evidence, false for a visible violation, '
    'and null when occlusion, blur, absent references or sampling gaps prevent judgment. '
    'This applies to identity, camera, continuity, outcome, entity visibility and nested '
    'event/sample evidence. A clearly absent required entity is false; an unidentifiable '
    'entity is null. An inapplicable check may be true; never use it to hide a missing '
    'required observation. Do not guess from the prompt or planned state. Explain the '
    'unobservable dimension in failure_reason. Null overrides earlier binary wording; '
    'it pauses review and must not authorize a retake. Identity reference labels bind '
    'each reference to exactly its named character; do not exchange identities.'
)


def nullable_observation_schema(schema: dict) -> dict:
    result = deepcopy(schema)
    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get('type') == 'boolean':
            node['type'] = ['boolean', 'null']
        for child in node.get('properties', {}).values():
            visit(child)
        if isinstance(node.get('items'), dict):
            visit(node['items'])
    visit(result)
    return result


def unobservable_fields(value, schema: dict, path: str = '') -> list[str]:
    """Inspect only declared boolean observation positions, not optional crop boxes."""
    kind = schema.get('type')
    if kind == 'boolean' or kind == ['boolean', 'null']:
        return [path] if value is None else []
    gaps = []
    if isinstance(value, dict) and kind == 'object':
        for key, child in schema.get('properties', {}).items():
            if key in value:
                gaps.extend(unobservable_fields(value[key], child, f'{path}.{key}' if path else key))
    elif kind == 'array' and isinstance(value, (list, dict)):
        # Accepted caches store entity visibility as a mapping; live responses use arrays.
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            gaps.extend(unobservable_fields(item, schema.get('items', {}), f'{path}[{key}]'))
            if len(gaps) >= 32:
                break
    return gaps[:32]

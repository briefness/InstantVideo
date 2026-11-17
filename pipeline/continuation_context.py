"""续拍请求的无损来源目录；审查合同仍是权威，不让关联数放大原文。

类似数据库的实体/关联表：原稿动作按身份只存一次，范围保留有序身份列表。
仅改变传输表示，不摘要、不截断、不改目标，也不改变审查和执行的数据模型。
"""
from copy import deepcopy
import hashlib
import json

from pipeline.goal_allocation import GoalEvidenceUnit

VERSION = 'source-catalog-v1'
LIMIT = 64000
INSTRUCTION = (
    ' With request_encoding=source-catalog-v1, source_catalog.units stores unchanged authored '
    'evidence keyed by shot_id:beat_index (beat_index is zero-based); source_catalog.scopes '
    'stores ordered lists of those keys. A {scope_ref: key} at scope or execution_scope.units '
    'means exactly that ordered source list, not missing evidence. Resolve it through the '
    'catalog. References preserve scope membership and order; they add no actions or obligations. '
    'Other shots remain context, never actions to repeat in this shot.'
)


def serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def request_context(source: dict) -> dict:
    """旧缓存仍使用原表示；新请求通过完整往返验证防止悄悄丢失来源。"""
    if 'request_encoding' not in source:
        return deepcopy(source)
    if source['request_encoding'] != VERSION or 'source_catalog' in source:
        raise ValueError('续拍来源目录协议无效')
    units, scopes, scope_ids = {}, {}, {}

    def scope_ref(values):
        if not isinstance(values, list) or not values:
            raise ValueError('续拍来源范围缺失')
        keys = []
        for raw in values:
            unit = GoalEvidenceUnit.model_validate(raw).model_dump()
            if unit != raw:
                raise ValueError('续拍来源不能在传输时改写')
            key = f"{unit['shot_id']}:{unit['beat_index']}"
            if key in units and units[key] != unit:
                raise ValueError('同一动作来源对应不同原文，不能合并')
            units[key] = unit
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError('续拍来源范围包含重复动作')
        identity = tuple(keys)
        if identity not in scope_ids:
            name = f'scope:{len(scopes) + 1}'
            scope_ids[identity] = name
            scopes[name] = keys
        return {'scope_ref': scope_ids[identity]}

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        if 'scope_ref' in value:
            raise ValueError('原始续拍合同不能包含未解析来源引用')
        return {k: scope_ref(v) if (k == 'scope' and isinstance(v, list)) or (
                    k == 'units' and set(value) == {'shot_id', 'units'}) else visit(v)
                for k, v in value.items()}

    result = visit(source)
    result['source_catalog'] = {'units': units, 'scopes': scopes}
    if expand_context(result) != source:
        raise ValueError('续拍来源目录无法无损还原')
    return result


def expand_context(request: dict) -> dict:
    """审计用确定性还原；非法、悬空和冲突来源不能当成有效证据。"""
    if 'request_encoding' not in request:
        return deepcopy(request)
    if request['request_encoding'] != VERSION:
        raise ValueError('续拍来源目录协议无效')
    catalog = request.get('source_catalog')
    if not isinstance(catalog, dict) or set(catalog) != {'units', 'scopes'}:
        raise ValueError('续拍来源目录结构无效')
    units, scopes = catalog['units'], catalog['scopes']
    if not isinstance(units, dict) or not isinstance(scopes, dict):
        raise ValueError('续拍来源目录结构无效')
    for key, raw in units.items():
        unit = GoalEvidenceUnit.model_validate(raw).model_dump()
        if raw != unit or key != f"{unit['shot_id']}:{unit['beat_index']}":
            raise ValueError('续拍动作来源身份不匹配')
    for keys in scopes.values():
        if (not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys)
                or len(set(keys)) != len(keys) or any(k not in units for k in keys)):
            raise ValueError('续拍范围引用无效或缺失')
    used_scopes = set()

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        if 'scope_ref' in value:
            key = value['scope_ref']
            if set(value) != {'scope_ref'} or not isinstance(key, str) or key not in scopes:
                raise ValueError('续拍范围引用无效或缺失')
            used_scopes.add(key)
            return [deepcopy(units[k]) for k in scopes[key]]
        return {k: visit(v) for k, v in value.items()}

    result = visit({k: v for k, v in request.items() if k != 'source_catalog'})
    used_units = {k for s in used_scopes for k in scopes[s]}
    if used_scopes != set(scopes) or used_units != set(units):
        raise ValueError('续拍来源目录包含未绑定证据')
    return result


def context_diagnostics(source: dict, request: dict) -> dict:
    """只记录大小、数量和哈希；错误消息不泄露模型内容或素材链接。"""
    encoded = serialized(request)
    catalog = request.get('source_catalog') or {}
    return {'encoding': source.get('request_encoding', 'legacy-inline'),
            'source_chars': len(serialized(source)), 'request_chars': len(encoded),
            'limit_chars': LIMIT,
            'source_sha256': hashlib.sha256(serialized(source).encode()).hexdigest(),
            'request_sha256': hashlib.sha256(encoded.encode()).hexdigest(),
            'source_field_chars': {k: len(serialized(v)) for k, v in source.items()},
            'request_field_chars': {k: len(serialized(v)) for k, v in request.items()},
            'unique_units': len(catalog.get('units', {})),
            'unique_scopes': len(catalog.get('scopes', {}))}

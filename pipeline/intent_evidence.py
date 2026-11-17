"""意图引用的来源定位：传输选择地址，持久化保留逐段原文与坐标。

定位只证明文字来自哪里，不证明引用支持目标；后续语义审查保持原职责。
旧的连续引文仍严格验证。旧响应中的多句拼接仅在能逐字、唯一、有序地
映射为完整目录条目时无损解码，不猜测省略、改写或歧义引用。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source: Literal['request', 'style']
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=16000)


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: Literal['source-spans-v1']
    segments: list[EvidenceSpan] = Field(min_length=1, max_length=16000)


Evidence = Annotated[str, Field(max_length=16000)] | EvidenceSelection
RequiredEvidence = Annotated[str, Field(min_length=1, max_length=16000)] | EvidenceSelection


class EvidenceSourceError(ValueError):
    def __init__(self, location: list, reason: str):
        self.location, self.reason = location, reason
        super().__init__(f'原文证据定位失败（{reason}）：' + '/'.join(map(str, location)))


def _ranges(text: str):
    # 仅分配地址，不解释题材、否定或语义。英文小数不因小数点被拆开。
    for match in re.finditer(r'.+?(?:[。！？!?]|\.(?=\s)|\n+|$)', text, re.S):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - len(raw) + len(raw.rstrip())
        if start < end:
            yield start, end


def evidence_catalog(request: str, style_input: str | None = None) -> dict:
    if (not isinstance(request, str) or not request.strip() or len(request) > 16000
            or style_input is not None and (not isinstance(style_input, str)
                or not style_input.strip() or len(style_input) > 1600)):
        raise ValueError('原文目录需要有效且有界的需求与风格文档')
    catalog = {}
    for source, text in (('request', request), ('style', style_input)):
        if not text:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        for start, end in dict.fromkeys([*_ranges(text), (0, len(text))]):
            catalog[f'{source}:{start}:{end}'] = {
                'source': source, 'document_sha256': digest, 'start': start, 'end': end,
                'text': text[start:end]}
    return catalog


def has_evidence(value: Evidence | dict | None) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def evidence_grounded(value: Evidence | dict, request: str, style_input: str | None = None) -> bool:
    if not isinstance(request, str) or style_input is not None and not isinstance(style_input, str):
        return False
    sources = {'request': request, **({'style': style_input} if style_input else {})}
    if isinstance(value, str):
        return bool(value.strip()) and any(value in text for text in sources.values())
    try:
        selection = EvidenceSelection.model_validate(value)
        previous = (-1, -1)
        for span in selection.segments:
            text = sources.get(span.source)
            order = (0 if span.source == 'request' else 1, span.start)
            if (text is None or span.document_sha256 != hashlib.sha256(text.encode()).hexdigest()
                    or not 0 <= span.start < span.end <= len(text)
                    or text[span.start:span.end] != span.text or order < previous):
                return False
            # 下一段必须位于上一段之后，不能重复、重叠或调换证据顺序。
            previous = (order[0], span.end)
        return True
    except (ValueError, TypeError):
        return False


def selection_schema(catalog: dict, *, style: bool = False) -> dict:
    ids = [key for key, span in catalog.items() if style or span['source'] == 'request']
    return {'type': 'object', 'properties': {'references': {'type': 'array',
        'minItems': 1, 'maxItems': len(ids), 'uniqueItems': True,
        'items': {'type': 'string', 'enum': ids}}},
        'required': ['references'], 'additionalProperties': False}


def _decode(value, catalog: dict, request: str, style: str | None, location: list):
    if value is None or value == '':
        return value  # 有无值与对应意图是否成对仍由领域模型检查。
    if isinstance(value, str) and evidence_grounded(value, request, style):
        return value  # 保留历史连续引文解释，不迁移已保存计划。
    allowed = {key: span for key, span in catalog.items() if style or span['source'] == 'request'}
    if isinstance(value, str):
        # 兼容旧线协议；只接受完整句目录的唯一、逐字映射，不做模糊匹配。
        if len(value) > 16000:
            raise EvidenceSourceError(location, 'quote_too_long')
        pieces = [value[a:b] for a, b in _ranges(value)]
        if len(pieces) < 2 or ''.join(pieces) != value:
            raise EvidenceSourceError(location, 'unresolved_quote')
        ids = []
        for piece in pieces:
            matches = [key for key, span in allowed.items() if span['text'] == piece]
            if len(matches) != 1:
                raise EvidenceSourceError(location, 'ambiguous_or_missing_quote')
            ids.append(matches[0])
    elif isinstance(value, dict) and set(value) == {'references'}:
        ids = value['references']
    else:
        raise EvidenceSourceError(location, 'invalid_selection')
    if (not isinstance(ids, list) or not 1 <= len(ids) <= len(allowed)
            or any(not isinstance(key, str) or key not in allowed for key in ids)):
        raise EvidenceSourceError(location, 'unknown_or_forbidden_source')
    selection = {'version': 'source-spans-v1', 'segments': [deepcopy(allowed[key]) for key in ids]}
    if not evidence_grounded(selection, request, style):
        raise EvidenceSourceError(location, 'duplicate_overlapping_or_reordered_source')
    return selection


def decode_intent_evidence(value: dict, request: str, style_input: str | None = None) -> dict:
    result = deepcopy(value)
    catalog = evidence_catalog(request, style_input)
    fields = [(result, name, [name], None) for name in (
        'intent_evidence', 'ending_evidence', 'motion_control_evidence', 'duration_evidence', 'shot_count_evidence')]
    for name in ('motion_timing', 'creative_direction'):
        if isinstance(result.get(name), dict):
            fields.append((result[name], 'evidence', [name, 'evidence'],
                           style_input if name == 'creative_direction' else None))
    if isinstance(result.get('completion_goals'), list):
        fields.extend((goal, 'evidence', ['completion_goals', i, 'evidence'], None)
            for i, goal in enumerate(result['completion_goals']) if isinstance(goal, dict))
    for parent, key, location, style in fields:
        if key in parent:
            parent[key] = _decode(parent[key], catalog, request, style, location)
    return result

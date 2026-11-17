"""结构化输出的接收边界：保留原文，拒绝歧义，不修写模型内容。"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from pipeline.operation_failure import mark_failure_phase, add_secondary_failure


class StructuredResponseError(ValueError):
    """不可通过 JSON 语法修复消除的响应协议错误。"""


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise StructuredResponseError('模型响应含同一对象内的重复字段；未覆盖、修写或重新生成')
        result[key] = value
    return result


def load_unique_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_object)


def completion_response_record(response, *, model: str) -> dict:
    """只保存模型响应正文和结束状态，不序列化 SDK 请求、异常或认证信息。"""
    choices = getattr(response, 'choices', None)
    records = []
    for choice in choices if isinstance(choices, (list, tuple)) else []:
        message = getattr(choice, 'message', None)
        content = getattr(message, 'content', None)
        finish = getattr(choice, 'finish_reason', None)
        records.append({'content': content if isinstance(content, str) else None,
                        'finish_reason': finish if isinstance(finish, str) else None,
                        'refused': bool(getattr(message, 'refusal', None))})
    return {'model': model, 'choices': records}


@dataclass
class _Container:
    kind: str
    expect_key: bool = False
    keys: set[str] = field(default_factory=set)


class ObjectKeyGuard:
    """按 JSON 字符串/容器边界检查键；完整语法仍交给最终 JSON 解析器。"""

    def __init__(self):
        self.stack: list[_Container] = []
        self.in_string = False
        self.escaped = False
        self.key_token: list[str] | None = None

    def feed(self, content: str) -> None:
        for char in content:
            if self.in_string:
                if self.key_token is not None:
                    self.key_token.append(char)
                if self.escaped:
                    self.escaped = False
                elif char == '\\':
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                    if self.key_token is not None:
                        key = json.loads(''.join(self.key_token))
                        frame = self.stack[-1]
                        if key in frame.keys:
                            raise StructuredResponseError(
                                '模型响应含同一对象内的重复字段；已停止接收，未覆盖、修写或重新生成'
                            )
                        frame.keys.add(key)
                        frame.expect_key = False
                        self.key_token = None
                continue
            if char == '"':
                self.in_string = True
                is_key = self.stack and self.stack[-1].kind == '{' and self.stack[-1].expect_key
                self.key_token = ['"'] if is_key else None
            elif char in '{[':
                self.stack.append(_Container(char, expect_key=char == '{'))
            elif char in '}]':
                if not self.stack or self.stack[-1].kind != ('{' if char == '}' else '['):
                    raise StructuredResponseError('模型响应的 JSON 容器不匹配；已停止接收')
                self.stack.pop()
            elif char == ',' and self.stack and self.stack[-1].kind == '{':
                self.stack[-1].expect_key = True


def collect_structured_response(response, *, on_failure: Callable[[dict], None] | None = None):
    """消费单次 SDK 流并及时关闭；兼容调用方提供的已完成响应，不重发请求。"""
    parts: list[str] = []
    guard = ObjectKeyGuard()
    finish = None
    usage = None
    primary = None
    try:
        if hasattr(response, 'choices'):
            if not response.choices:
                raise StructuredResponseError('模型响应没有候选结果')
            message = response.choices[0].message
            parts.append(message.content or '')
            if getattr(message, 'refusal', None):
                raise StructuredResponseError('模型拒绝生成结构化内容；未重新生成')
            guard.feed(parts[0])
            return response
        for chunk in response:
            if getattr(chunk, 'usage', None) is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if getattr(choice, 'index', 0) != 0:
                    raise StructuredResponseError('模型返回了未请求的多个候选结果')
                content = getattr(choice.delta, 'content', None)
                if content:
                    parts.append(content)
                    guard.feed(content)
                if getattr(choice.delta, 'refusal', None):
                    raise StructuredResponseError('模型拒绝生成结构化内容；未重新生成')
                if choice.finish_reason is not None:
                    finish = choice.finish_reason
        if finish is None:
            raise StructuredResponseError('模型响应流缺少结束标记；未将不完整响应判为通过')
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=''.join(parts)), finish_reason=finish,
        )], usage=usage)
    except BaseException as error:
        primary = error
        mark_failure_phase(error, 'receive_response')
        if on_failure and isinstance(error, Exception):
            try:
                on_failure({'content': ''.join(parts), 'finish_reason': (
                    'protocol_error' if isinstance(error, ValueError) else 'transport_error'
                )})
            except Exception as secondary:
                add_secondary_failure(error, secondary, 'save_diagnostic')
        raise
    finally:
        close = getattr(response, 'close', None)
        if callable(close):
            try:
                close()
            except Exception as secondary:
                if primary is None:
                    mark_failure_phase(secondary, 'close_response')
                    raise
                add_secondary_failure(primary, secondary, 'close_response')

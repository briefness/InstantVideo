"""操作故障事实与重试权限分离；只保存白名单元数据。"""
from __future__ import annotations

from typing import Literal

import httpx
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, ConfigDict, Field

from tools.submission_evidence import creation_diagnostic, identifier


Phase = Literal['unknown', 'request', 'receive_response', 'save_diagnostic',
                'close_response', 'save_checkpoint', 'planning_step', 'save_run_failure', 'save_run_status']


class CauseEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    type: str = Field(pattern=r'^[A-Za-z0-9_.:-]{1,200}$')
    errno: int | None = None


class SecondaryFailure(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    phase: Phase
    type: str = Field(pattern=r'^[A-Za-z0-9_.:-]{1,200}$')


class FailureEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    version: Literal['operation-failure-v1'] = 'operation-failure-v1'
    phase: Phase = 'unknown'
    category: Literal['transport', 'http', 'validation', 'local_io', 'unknown']
    remote_outcome: Literal['unknown', 'response_received', 'not_applicable']
    chain: list[CauseEvidence] = Field(min_length=1, max_length=8)
    chain_truncated: bool = False
    http_status: int | None = Field(default=None, ge=100, le=599)
    request_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_.:-]{1,200}$')
    secondary: list[SecondaryFailure] = Field(default_factory=list, max_length=4)


def mark_failure_phase(exc: BaseException, phase: Phase) -> None:
    # 首个知道失败位置的边界负责定位；上层不能把接收失败改写成发送失败。
    if not hasattr(exc, '_operation_failure_phase'):
        exc._operation_failure_phase = phase


def add_secondary_failure(primary: BaseException, secondary: BaseException, phase: Phase) -> None:
    records = list(getattr(primary, '_operation_secondary_failures', ()))[:4]
    if len(records) < 4:
        records.append({'phase': phase, 'type': identifier(type(secondary).__name__) or 'Exception'})
    primary._operation_secondary_failures = records


def failure_evidence(exc: BaseException, *, phase: Phase = 'unknown') -> dict:
    chain, exceptions, seen = [], [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        exceptions.append(current)
        code = getattr(current, 'errno', None)
        chain.append({'type': identifier(type(current).__name__) or 'Exception',
                      'errno': code if type(code) is int else None})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    http_error = next((e for e in exceptions if isinstance(e, APIStatusError)), None)
    if http_error is not None:
        category, outcome = 'http', 'response_received'
    elif any(isinstance(e, (APIConnectionError, httpx.TransportError)) for e in exceptions):
        # 连不上、写断、读断均不能证明远端没有处理过请求，更不能授权重发。
        category, outcome = 'transport', 'unknown'
    elif isinstance(exc, ValueError):
        category, outcome = 'validation', 'not_applicable'
    elif isinstance(exc, OSError):
        category, outcome = 'local_io', 'not_applicable'
    else:
        category, outcome = 'unknown', 'unknown'
    metadata = creation_diagnostic(http_error, phase='request') if http_error is not None else {}
    return FailureEvidence(
        phase=getattr(exc, '_operation_failure_phase', phase), category=category,
        remote_outcome=outcome, chain=chain, chain_truncated=current is not None,
        http_status=metadata.get('http_status'), request_id=metadata.get('request_id'),
        secondary=list(getattr(exc, '_operation_secondary_failures', ()))[:4],
    ).model_dump(mode='json')


def failure_message(exc: Exception) -> str:
    evidence = failure_evidence(exc)
    if evidence['category'] == 'transport':
        return '外部服务连接或接收失败，远端处理结果未知；已停止，未自动重发。详见 run_failure.json。'
    if evidence['category'] == 'http':
        return f"外部服务返回 HTTP {evidence['http_status']}；已停止，未自动重发。详见 run_failure.json。"
    return str(exc)


def record_run_failure(workspace, exc: Exception) -> None:
    """诊断/状态落盘也可能失败，但不得用次生错误取代真实故障。"""
    try:
        path = workspace.path / 'run_failure.json'
        # 正式入口持有工作区执行锁。恢复被拒绝是后果，不能覆盖首次故障现场。
        if not path.exists() and not path.is_symlink():
            workspace._atomic_write_json(path, {
                'version': 'run-failure-v1', 'stage': workspace.manifest.stage,
                'failure': failure_evidence(exc),
            })
    except Exception as secondary:
        add_secondary_failure(exc, secondary, 'save_run_failure')
    try:
        workspace.mark_failed(failure_message(exc))
    except Exception as secondary:
        add_secondary_failure(exc, secondary, 'save_run_status')

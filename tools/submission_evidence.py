"""创建副作用的可追查证据；不记录异常正文、认证头或签名地址。"""
from datetime import datetime, timezone
import re
from pathlib import Path
import config


def identifier(value):
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', value)
            or value in {config.ARK_API_KEY, config.ARK_API_KEY_SEEDANCE}):
        return None
    return value


def creation_diagnostic(exc: Exception, *, phase: str) -> dict:
    status = getattr(exc, 'status_code', None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', {})
    sdk_request_id = identifier(getattr(exc, 'request_id', None))
    response_request_id = None
    if hasattr(headers, 'get'):
        response_request_id = identifier(headers.get('x-request-id')) or identifier(headers.get('x-tt-logid'))
    body = getattr(exc, 'body', None)
    body = body if isinstance(body, dict) else {}
    error = body.get('error', body)
    error = error if isinstance(error, dict) else {}
    body_request_id = (identifier(body.get('request_id')) or identifier(body.get('RequestId'))
                       or identifier(error.get('request_id')))
    message = error.get('message')
    if not body_request_id and isinstance(message, str) and len(message) <= 16384:
        match = re.search(r'\bRequest\s+id:\s*(\S+)', message, re.I)
        body_request_id = identifier(match.group(1)) if match else None
    # Ark SDK 异常属性来自客户端请求头，不能覆盖服务端审核追踪标识。
    # 分开保留来源；兼容旧 request_id 读取方，但不修改历史记录。
    request_id = response_request_id or body_request_id or sdk_request_id
    return {'version': 'submission-evidence-v1', 'phase': phase,
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'exception_type': identifier(type(exc).__name__),
            'cause_type': identifier(type(exc.__cause__).__name__) if exc.__cause__ else None,
            'http_status': status, 'request_id': request_id,
            'sdk_request_id': sdk_request_id, 'response_request_id': response_request_id,
            'body_request_id': body_request_id,
            'error_code': identifier(error.get('code'))}


def record_submission_result(root: Path, shot_id: int, attempt: int, result: dict) -> None:
    """先持久化白名单结果，再改变本地预算或镜头状态；不覆盖历史结果。"""
    from pipeline.run_state import RunWorkspace
    from pipeline.readiness import GenerationReadinessError
    from pipeline.models import SubmissionDiagnosticState
    from pydantic import ValidationError
    diagnostic = result.get('submission_diagnostic')
    try:
        if diagnostic is not None:
            diagnostic = SubmissionDiagnosticState.model_validate(diagnostic).model_dump(
                mode='json', exclude_none=True)
    except ValidationError as exc:
        raise GenerationReadinessError('提交诊断不符合持久化契约；保留预算，停止生成') from exc
    payload = {'version': 'submission-result-v1', 'shot_id': shot_id, 'attempt': attempt,
               'status': result.get('status'), 'error_type': result.get('error_type'),
               'provider_task_id': identifier(result.get('provider_task_id')),
               'submission_diagnostic': diagnostic}
    path = Path(root) / 'submission_results' / f'shot_{shot_id:03d}_attempt_{attempt:03d}.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise GenerationReadinessError('已有提交结果证据；未覆盖或重新生成')
        RunWorkspace._atomic_write_json(path, payload)
        path.chmod(0o600)
    except OSError as exc:
        raise GenerationReadinessError('提交结果证据无法保存；保留预算，停止生成') from exc

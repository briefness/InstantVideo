"""不确定提交保留诊断与预算；所有视频调用均为离线替身。"""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from tools.seedance_api import SeedanceAPI


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [None, 408, 500, 503])
async def test_unknown_create_keeps_safe_diagnostic(status):
    error = RuntimeError('Authorization Bearer SECRET https://private/?token=SECRET')
    error.status_code = status
    error.request_id = 'req-offline-123'
    api = object.__new__(SeedanceAPI)
    create = Mock(side_effect=error)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    result = await api.generate('test', model='doubao-seedance-2.0-fast', duration=4)
    assert create.call_count == 1
    assert result['error_type'] == 'submission_unknown'
    evidence = result['submission_diagnostic']
    assert evidence['exception_type'] == 'RuntimeError'
    assert evidence['request_id'] == 'req-offline-123'
    assert evidence['http_status'] == status
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [SimpleNamespace(), SimpleNamespace(id=None), SimpleNamespace(id=''), SimpleNamespace(id='https://secret')])
async def test_response_without_valid_identity_cannot_be_polled(response):
    api = object.__new__(SeedanceAPI)
    create, get = Mock(return_value=response), Mock(side_effect=AssertionError('must not poll'))
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create, get=get)))
    result = await api.generate('test', model='doubao-seedance-2.0-fast', duration=4)
    assert result['error_type'] == 'submission_unknown'
    assert result['submission_diagnostic']['phase'] == 'create_response'
    get.assert_not_called()
    assert create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
async def test_formal_flow_persists_unknown_and_resume_never_creates_again(tmp_path, monkeypatch, topic):
    import config
    from pathlib import Path
    from pipeline.orchestrator import VideoPipeline
    from pipeline.generator import RemoteTaskPendingError
    from test_scene_delivery_flow import SceneServices
    from test_ordinary_scene_contract import scene_case
    from test_director_intent import Responses
    brief, reply, identity, wires, shots = scene_case(topic)
    services = SceneServices([], Responses(reply, identity, *wires), shots, len(identity['characters']))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: services)
    error = RuntimeError('SECRET transport details')
    error.request_id = 'offline-' + topic
    api = object.__new__(SeedanceAPI)
    create = Mock(side_effect=error)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: api)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    pipeline = VideoPipeline(resolution='480p', planning_repair=False, paid_take_budget=4, take_policy='single_pass')
    with pytest.raises(RemoteTaskPendingError):
        await pipeline.run(brief)
    root = Path(pipeline.workspace)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['status'] == 'interrupted'
    assert manifest['shots']['1']['status'] == 'running'
    assert manifest['paid_take_budget']['reservations'][0]['status'] == 'reserved'
    diagnostic = manifest['shots']['1']['prompt_attempts'][0]['submission_diagnostic']
    assert diagnostic['request_id'] == 'offline-' + topic
    evidence = json.loads((root / 'submission_results/shot_001_attempt_001.json').read_text())
    assert evidence['submission_diagnostic'] == diagnostic
    assert 'SECRET' not in json.dumps(evidence)
    with pytest.raises(RuntimeError, match='无法确认|未确认|核对'):
        await VideoPipeline.from_workspace(str(root)).run()
    assert create.call_count == 1 and not (root / 'final.mp4').exists()


def test_diagnostic_reads_only_safe_fields(monkeypatch):
    import config
    from tools.submission_evidence import creation_diagnostic
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'SECRET')
    error = RuntimeError('SECRET')
    error.request_id = 'SECRET'
    error.response = SimpleNamespace(headers={'authorization': 'SECRET', 'x-request-id': 'req-safe'})
    error.body = {'error': {'code': 'InternalError', 'message': 'SECRET'}}
    result = creation_diagnostic(error, phase='create_request')
    assert result['request_id'] == 'req-safe' and result['error_code'] == 'InternalError'
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.asyncio
async def test_sdk_success_response_decode_failure_is_not_a_rejection():
    import httpx
    from volcenginesdkarkruntime._exceptions import ArkAPIResponseValidationError
    response = httpx.Response(200, request=httpx.Request('POST', 'https://offline.invalid/tasks'))
    error = ArkAPIResponseValidationError(response, {}, request_id='req-decode')
    api = object.__new__(SeedanceAPI)
    create = Mock(side_effect=error)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    result = await api.generate('offline', model='doubao-seedance-2.0-fast', duration=4)
    assert result['error_type'] == 'submission_unknown'
    assert result['submission_diagnostic']['http_status'] == 200
    assert result['submission_diagnostic']['request_id'] == 'req-decode'
    assert create.call_count == 1


def test_result_evidence_write_failure_is_terminal_and_never_overwrites(tmp_path, monkeypatch):
    from pipeline.readiness import GenerationReadinessError
    from pipeline.run_state import RunWorkspace
    from tools.submission_evidence import record_submission_result
    result = {'status': 'failed', 'error_type': 'submission_unknown'}
    with monkeypatch.context() as patch:
        patch.setattr(RunWorkspace, '_atomic_write_json', Mock(side_effect=OSError('disk full')))
        with pytest.raises(GenerationReadinessError, match='保留预算'):
            record_submission_result(tmp_path, 1, 1, result)
    record_submission_result(tmp_path, 1, 1, result)
    path = tmp_path / 'submission_results/shot_001_attempt_001.json'
    saved = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(GenerationReadinessError, match='未覆盖'):
        record_submission_result(tmp_path, 1, 1, {'status': 'succeeded'})
    assert path.read_bytes() == saved


@pytest.mark.asyncio
async def test_actual_unknown_run_resume_blocks_all_providers(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from pipeline.orchestrator import VideoPipeline
    source = Path(__file__).parents[1] / 'output/20261104_190747_846437'
    root = tmp_path / 'offline-copy'
    shutil.copytree(source, root)
    def forbidden(*_, **__):
        pytest.fail('unknown submission must not invoke any provider')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: client)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: SimpleNamespace(
        generate=forbidden, query_task=forbidden, download_video=forbidden))
    original = (source / 'run_manifest.json').read_bytes()
    with pytest.raises(RuntimeError, match='提交结果尚未确认'):
        await VideoPipeline.from_workspace(str(root)).run()
    manifest = json.loads((root / 'run_manifest.json').read_text())
    reservations = manifest['paid_take_budget']['reservations']
    assert len(reservations) == 1 and reservations[0]['status'] == 'reserved'
    assert not (root / 'final.mp4').exists()
    assert (source / 'run_manifest.json').read_bytes() == original


def test_rejected_request_releases_money_but_not_single_pass_permission(tmp_path):
    from test_film_rhythm import workspace
    from pipeline.run_state import PaidTakeBudgetExhaustedError, RunWorkspace
    run = workspace(tmp_path, cap=3, policy='single_pass')
    reservation = run.reserve_paid_take(1)
    run.release_unsubmitted_paid_take(reservation)
    resumed = RunWorkspace.resume(run.path)
    with pytest.raises(PaidTakeBudgetExhaustedError):
        resumed.reserve_paid_take(1)
    assert len(resumed.manifest.paid_take_budget.reservations) == 1
    assert resumed.manifest.paid_take_budget.reservations[0].status == 'released'


@pytest.mark.asyncio
async def test_accepted_task_checkpoint_failure_preserves_identity_without_poll_or_retry():
    from tools.seedance_api import SubmittedTaskCheckpointError
    api = object.__new__(SeedanceAPI)
    create = Mock(return_value=SimpleNamespace(id='offline-accepted'))
    get = Mock(side_effect=AssertionError('checkpoint must precede polling'))
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create, get=get)))
    checkpoint = Mock(side_effect=OSError('offline disk failure'))
    with pytest.raises(SubmittedTaskCheckpointError) as raised:
        await api.generate('offline', model='doubao-seedance-2.0-fast', duration=4,
                           on_submitted=checkpoint)
    assert raised.value.task_id == 'offline-accepted'
    checkpoint.assert_called_once_with('offline-accepted')
    get.assert_not_called()
    assert create.call_count == 1

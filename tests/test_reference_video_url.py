"""Offline network fakes exercise URL, DNS, TLS, hash and billing boundaries."""

import copy
import hashlib
import io
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

import config
from pipeline.generator import VideoGenerator
from pipeline.models import MotionReferenceSpec, RunOptions
from pipeline.motion_reference import ensure_motion_lineage, motion_reference_context
from pipeline.run_state import RunWorkspace
from tools import reference_video_url as remote
from tools.prepare_motion_run import main, prepare_motion_run
from tools.seedance_api import SeedanceAPI


MODEL = "doubao-seedance-2-0-mini-260615"
URL = "https://media.example.com/motion/immutable.mp4"
DATA = b"fake encoded video bytes"
DIGEST = hashlib.sha256(DATA).hexdigest()


@pytest.fixture(autouse=True)
def no_real_connections(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.setattr(socket, "getaddrinfo", Mock(side_effect=AssertionError("DNS forbidden")))


@pytest.fixture
def transport(monkeypatch):
    response = SimpleNamespace(status=200, read=io.BytesIO(DATA).read)
    response.read1 = lambda size: response.read(size)
    headers = {"Content-Length": str(len(DATA))}
    response.getheader = lambda key, default=None: headers.get(key, default)
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(remote, "_PinnedHTTPSConnection", factory)
    monkeypatch.setattr(socket, "getaddrinfo", Mock(return_value=[
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]))
    return response, headers, connection, factory


@pytest.mark.parametrize("url", [
    "http://example.com/x", "ftp://example.com/x", "file:///x", "data:video/mp4;base64,x",
    "https://localhost/x", "https://node.local/x", "https://node.internal/x",
    "https://127.0.0.1/x", "https://10.0.0.1/x", "https://169.254.169.254/x",
    "https://[::1]/x", "https://[fc00::1]/x", "https://[ff02::1]/x",
    "https://user:password@example.com/x", "https://example.com/x?signature=secret",
    "https://example.com/x#fragment", "https://example.com:8443/x", "https://example.com:bad/x",
    "https://example.com\\@127.0.0.1/x", "https://example.com/\nvideo", " https://example.com/x",
    "https://2130706433/x", "https://127.1/x", "https://%31%32%37.0.0.1/x",
    "https://example.com/" + "a" * 2048, "", None,
])
def test_unsafe_url_syntax_rejected_without_dns(url):
    with pytest.raises(ValueError):
        remote.validate_public_video_url(url)
    socket.getaddrinfo.assert_not_called()


def test_https_url_syntax_validation_is_offline():
    assert remote.validate_public_video_url(URL).hostname == "media.example.com"
    socket.getaddrinfo.assert_not_called()


def test_explicit_public_dns_pin_preserves_hostname_tls(transport, monkeypatch):
    _, _, _, factory = transport
    monkeypatch.setenv("MOTION_REFERENCE_DNS_PIN", "media.example.com=104.16.230.132")
    remote.verify_reference_video_url(URL, DIGEST)
    factory.assert_called_once_with("media.example.com", "104.16.230.132")
    socket.getaddrinfo.assert_not_called()


@pytest.mark.parametrize("pin", ["media.example.com=127.0.0.1", "media.example.com=198.18.0.1",
    "media.example.com=169.254.169.254", "media.example.com=::1", "media.example.com=not-an-ip",
    "media.example.com/path=104.16.230.132", "bad", "x=y=z", "a" * 385])
def test_dns_pin_does_not_bypass_public_address_checks(transport, monkeypatch, pin):
    monkeypatch.setenv("MOTION_REFERENCE_DNS_PIN", pin)
    with pytest.raises(ValueError):
        remote.verify_reference_video_url(URL, DIGEST)
    transport[3].assert_not_called()


def test_pin_does_not_apply_to_another_hostname(transport, monkeypatch):
    monkeypatch.setenv("MOTION_REFERENCE_DNS_PIN", "other.example.com=104.16.230.132")
    remote.verify_reference_video_url(URL, DIGEST)
    transport[3].assert_called_once_with("media.example.com", "93.184.216.34")


def test_exact_remote_bytes_checked_over_pinned_connection(transport):
    _, _, connection, factory = transport
    remote.verify_reference_video_url(URL, DIGEST)
    factory.assert_called_once_with("media.example.com", "93.184.216.34")
    connection.request.assert_called_once_with("GET", "/motion/immutable.mp4", headers={"Accept-Encoding": "identity"})
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [301, 302, 307, 401, 403, 404, 206, 500])
def test_non_200_responses_do_not_redirect_or_submit(transport, status):
    response, _, connection, _ = transport
    response.status = status
    with pytest.raises(ValueError, match="HTTP 200"):
        remote.verify_reference_video_url(URL, DIGEST)
    assert connection.request.call_count == 1
    connection.close.assert_called_once()


@pytest.mark.parametrize("addresses", [[], ["127.0.0.1"], ["169.254.169.254"],
    ["93.184.216.34", "10.1.2.3"], ["::ffff:127.0.0.1"], ["224.0.0.1"]])
def test_all_dns_results_must_be_public(transport, addresses):
    _, _, _, factory = transport
    socket.getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in addresses]
    with pytest.raises(ValueError, match="非公网"):
        remote.verify_reference_video_url(URL, DIGEST)
    factory.assert_not_called()


@pytest.mark.parametrize("fault", ["changed", "empty", "truncated", "oversize_header", "oversize_stream",
                                      "compressed", "timeout", "disconnect", "bad_length"])
def test_transfer_failure_is_bounded_and_closes_connection(transport, monkeypatch, fault):
    response, headers, connection, _ = transport
    if fault == "changed":
        response.read = io.BytesIO(b"x" * len(DATA)).read
    elif fault == "empty":
        headers.clear()
        response.read = io.BytesIO(b"").read
    elif fault == "truncated":
        response.read = io.BytesIO(DATA[:2]).read
    elif fault == "oversize_header":
        headers["Content-Length"] = str(remote.MAX_VIDEO_REFERENCE_BYTES + 1)
    elif fault == "oversize_stream":
        headers.clear()
        monkeypatch.setattr(remote, "MAX_VIDEO_REFERENCE_BYTES", 2)
    elif fault == "compressed":
        headers["Content-Encoding"] = "gzip"
    elif fault == "timeout":
        monkeypatch.setattr(remote.time, "monotonic", Mock(side_effect=[0, 31]))
    elif fault == "disconnect":
        response.read = Mock(side_effect=TimeoutError("secret signed URL"))
    else:
        headers["Content-Length"] = "invalid"
    with pytest.raises(ValueError) as error:
        remote.verify_reference_video_url(URL, DIGEST)
    assert "secret" not in str(error.value)
    connection.close.assert_called_once()


def test_tls_pins_checked_ip_preserves_hostname_verification(monkeypatch):
    sock, wrapped = Mock(), Mock()
    monkeypatch.setattr(socket, "create_connection", Mock(return_value=sock))
    connection = remote._PinnedHTTPSConnection("media.example.com", "93.184.216.34")
    context = Mock()
    context.wrap_socket.return_value = wrapped
    connection._context = context
    connection.connect()
    socket.create_connection.assert_called_once_with(("93.184.216.34", 443), timeout=10)
    context.wrap_socket.assert_called_once_with(sock, server_hostname="media.example.com")
    socket.getaddrinfo.assert_not_called()


def test_slow_body_is_stopped_between_single_reads(transport, monkeypatch):
    response, _, connection, _ = transport
    response.read1 = Mock(return_value=b"x")
    monkeypatch.setattr(remote.time, "monotonic", Mock(side_effect=[0, 1, 11, 21, 31]))
    with pytest.raises(ValueError, match="超时"):
        remote.verify_reference_video_url(URL, DIGEST)
    assert response.read1.call_count == 3
    connection.close.assert_called_once()


def spec(**kwargs):
    return MotionReferenceSpec(path="motion_refs/reference.mp4", sha256=DIGEST,
                               source_sha256="a" * 64, model=MODEL, **kwargs)


def test_schema_preserves_legacy_context_and_rejects_two_sources():
    legacy = motion_reference_context({"motion_reference": spec(asset_uri="asset://old").model_dump()})
    assert "video_url" not in legacy
    with pytest.raises(ValidationError, match="只能选择一个"):
        spec(asset_uri="asset://old", video_url=URL)
    assert spec(video_url=URL).video_url == URL


def fake_api():
    api = object.__new__(SeedanceAPI)
    create = Mock(return_value=SimpleNamespace(id="new-task"))
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create)))
    api._poll_ark = AsyncMock(return_value={"status": "pending", "provider_task_id": "new-task"})
    return api, create


@pytest.mark.asyncio
async def test_api_checks_bytes_before_task_and_poll_resume_does_not_fetch(transport, use_test_model):
    use_test_model(MODEL)
    _, _, connection, _ = transport
    api, create = fake_api()
    await api.generate("test", model=MODEL, reference_video=URL, reference_video_sha256=DIGEST)
    assert create.call_count == 1
    assert create.call_args.kwargs["content"][-1]["video_url"]["url"] == URL
    await api.generate("ignored", model=MODEL, task_id="old-task", reference_video="invalid")
    assert create.call_count == connection.request.call_count == 1
    assert api._poll_ark.call_args.args[0] == "old-task"


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [None, "bad", "0" * 64])
async def test_api_hash_failure_does_not_create_task(transport, digest, use_test_model):
    use_test_model(MODEL)
    api, create = fake_api()
    result = await api.generate("test", model=MODEL, reference_video=URL, reference_video_sha256=digest)
    assert result["error_type"] == "reference_validation"
    create.assert_not_called()
    api._poll_ark.assert_not_awaited()


def test_url_change_or_removal_invalidates_existing_lineage(tmp_path):
    shot = {"motion_reference": spec(video_url=URL).model_dump()}
    context = motion_reference_context(shot)
    (tmp_path / "take_generation.json").write_text(json.dumps({"motion_reference": context}))
    ensure_motion_lineage(shot, tmp_path / "take.mp4")
    shot["motion_reference"]["video_url"] = "https://media.example.com/different.mp4"
    with pytest.raises(ValueError, match="禁止复用"):
        ensure_motion_lineage(shot, tmp_path / "take.mp4")
    shot["motion_reference"] = None
    with pytest.raises(ValueError, match="禁止复用"):
        ensure_motion_lineage(shot, tmp_path / "take.mp4")


@pytest.mark.asyncio
async def test_remote_mismatch_releases_budget_and_never_resubmits(tmp_path, monkeypatch, transport, use_test_model):
    from pipeline import generator as module
    monkeypatch.setattr(SeedanceAPI, "_init_clients", lambda self: None)
    monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
    use_test_model(MODEL)
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="test", paid_take_budget=1))
    shot = {"shot_id": 1, "duration": 4, "prompt_en": "neutral scene",
            "motion_reference": spec(video_url=URL).model_dump()}
    # Local media validation has separate real-FFmpeg coverage; isolate billing here.
    monkeypatch.setattr(module, "prepare_motion_reference", lambda *args: motion_reference_context(shot))
    monkeypatch.setattr(module, "motion_prompt_suffix", lambda shot: " reference motion")
    generator = VideoGenerator(str(workspace.path), reserve_paid_take=workspace.reserve_paid_take,
                               release_unsubmitted_paid_take=workspace.release_unsubmitted_paid_take)
    api, create = fake_api()
    generator.api = api
    transport[0].read = io.BytesIO(b"x" * len(DATA)).read
    result = await generator._generate_single_shot(shot, None, None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]})
    assert result.status == "failed"
    create.assert_not_called()
    assert len(workspace.manifest.paid_take_budget.reservations) == 1
    assert workspace.manifest.paid_take_budget.reservations[0].status == "released"
    assert workspace.reserve_paid_take(1)  # Authorization remains available.


def test_cli_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        main(["--storyboard", "s.json", "--package", "pack", "--output-root", "out",
              "--asset-uri", "asset://id", "--video-url", URL])
    assert error.value.code == 2


@pytest.fixture(scope="module")
def local_package(tmp_path_factory):
    from pipeline.motion_plan import MotionPlan
    from tools.motion_previs import build_motion_previs
    root = Path(__file__).resolve().parents[1] / "examples/contact_previs"
    storyboard = json.loads((root / "storyboard.json").read_text())
    plan = MotionPlan.model_validate_json((root / "motion.json").read_text())
    package = build_motion_previs(storyboard, plan, tmp_path_factory.mktemp("url-previs") / "package")
    return storyboard, package


def test_prepare_url_workspace_is_offline_and_survives_round_trip(local_package, tmp_path):
    storyboard, package = local_package
    before = copy.deepcopy(storyboard)
    workspace = prepare_motion_run(storyboard, package, tmp_path, model=MODEL, video_url=URL)
    restored = RunWorkspace.resume(workspace.path)
    reference = restored.load_storyboard()["shots"][0]["motion_reference"]
    assert reference["video_url"] == URL
    assert reference.get("asset_uri") is None
    assert restored.options.paid_take_budget == 0
    assert hashlib.sha256((restored.path / reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    assert storyboard == before
    socket.getaddrinfo.assert_not_called()
    socket.create_connection.assert_not_called()


@pytest.mark.asyncio
async def test_zero_budget_url_run_never_fetches(local_package, tmp_path, monkeypatch):
    storyboard, package = local_package
    monkeypatch.setattr(SeedanceAPI, "_init_clients", lambda self: None)
    monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
    monkeypatch.setattr(config, "SEEDANCE_MODEL", MODEL)
    workspace = prepare_motion_run(storyboard, package, tmp_path, model=MODEL, video_url=URL)
    generator = VideoGenerator(str(workspace.path), reserve_paid_take=workspace.reserve_paid_take)
    generator.api.generate = AsyncMock(side_effect=AssertionError("budget must stop first"))
    saved = workspace.load_storyboard()
    result = await generator._generate_single_shot(saved["shots"][0], None, None, saved)
    assert result.status == "failed"
    generator.api.generate.assert_not_awaited()
    socket.getaddrinfo.assert_not_called()


@pytest.mark.asyncio
async def test_pending_url_task_uses_original_id_without_network(local_package, tmp_path, monkeypatch):
    from pipeline.causality import ACTION_CONTRACT_VERSION
    storyboard, package = local_package
    monkeypatch.setattr(SeedanceAPI, "_init_clients", lambda self: None)
    monkeypatch.setattr(config, "SEMANTIC_REVIEW_ENABLED", False)
    monkeypatch.setattr(config, "SEEDANCE_MODEL", MODEL)
    workspace = prepare_motion_run(storyboard, package, tmp_path, model=MODEL, video_url=URL)
    generator = VideoGenerator(str(workspace.path), reserve_paid_take=workspace.reserve_paid_take)
    saved = workspace.load_storyboard()
    shot = saved["shots"][0]
    descriptor = {"task_id": "original-task", "prompt_profile": "normal", "prompt_fingerprint": "a" * 64,
                  "compiled_contract_version": ACTION_CONTRACT_VERSION,
                  "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot)}
    generator._write_generation_provenance(1, generator._generation_provenance(
        shot, [], None, prompt_profile="normal", prompt_fingerprint="a" * 64,
    ), "original-task")
    generator.resume_tasks = {1: descriptor}
    generator.api.poll_task = AsyncMock(return_value={"status": "pending", "provider_task_id": "original-task"})
    generator.api.generate = AsyncMock(side_effect=AssertionError("must only poll"))
    result = await generator._generate_single_shot(shot, None, None, saved)
    assert result.provider_task_id == "original-task"
    generator.api.poll_task.assert_awaited_once_with("original-task", timeout=config.GENERATION_TIMEOUT)
    generator.api.generate.assert_not_awaited()
    socket.getaddrinfo.assert_not_called()

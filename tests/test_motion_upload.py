"""Conditional uploads are exercised only against in-memory transport doubles."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.motion_plan import MotionPlan
from tools import publish_motion_reference as upload
from tools.motion_previs import build_motion_previs

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/contact_previs"


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    storyboard = json.loads((EXAMPLE / "storyboard.json").read_text())
    plan = MotionPlan.model_validate_json((EXAMPLE / "motion.json").read_text())
    return build_motion_previs(storyboard, plan, tmp_path_factory.mktemp("upload") / "package")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("real network forbidden")
    monkeypatch.setattr(socket, "getaddrinfo", forbid)
    monkeypatch.setattr(socket, "create_connection", forbid)


def test_dry_run_never_uses_upload_authorization(package, monkeypatch, capsys):
    monkeypatch.setenv("MOTION_UPLOAD_PUT_URL", "secret-invalid-value")
    assert upload.main(["--package", str(package)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["remote_calls"] == 0 and result["public_url"] is None
    plan, payload = upload.prepare_upload(package, "https://media.example.com/motion")
    assert plan["public_url"].endswith(plan["sha256"] + ".mp4")
    assert len(payload) == plan["bytes"]


@pytest.mark.parametrize("url", ["http://media.example.com", "https://127.0.0.1", "https://media.example.com/?secret=x"])
def test_invalid_public_base_rejected_offline(package, url):
    with pytest.raises(ValueError):
        upload.prepare_upload(package, url)


@pytest.mark.parametrize("status", [200, 201, 204, 412])
def test_conditional_put_and_public_digest_verification(package, monkeypatch, status):
    plan, payload = upload.prepare_upload(package, "https://media.example.com")
    connection = Mock()
    connection.getresponse.return_value = SimpleNamespace(status=status)
    constructor = Mock(return_value=connection)
    verify = Mock()
    monkeypatch.setattr(upload, "_resolve_public_address", lambda host: "93.184.216.34")
    monkeypatch.setattr(upload, "_PinnedHTTPSConnection", constructor)
    monkeypatch.setattr(upload, "verify_reference_video_url", verify)
    signed = f"https://upload.example.com/bucket/{plan['object_filename']}?X-Amz-Signature=secret"
    result = upload.publish_reference(package, "https://media.example.com", signed)
    constructor.assert_called_once_with("upload.example.com", "93.184.216.34")
    connection.request.assert_called_once_with("PUT", f"/bucket/{plan['object_filename']}?X-Amz-Signature=secret",
        body=payload, headers={"Content-Type": "video/mp4", "If-None-Match": "*"})
    verify.assert_called_once_with(plan["public_url"], plan["sha256"])
    connection.close.assert_called_once()
    assert result["reused_existing"] is (status == 412)
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("status", [301, 307, 400, 403, 409, 500])
def test_redirects_conflicts_and_provider_errors_never_retry(package, monkeypatch, status):
    plan, _ = upload.prepare_upload(package)
    connection = Mock()
    connection.getresponse.return_value = SimpleNamespace(status=status)
    monkeypatch.setattr(upload, "_resolve_public_address", lambda host: "93.184.216.34")
    monkeypatch.setattr(upload, "_PinnedHTTPSConnection", lambda *args: connection)
    with pytest.raises(ValueError):
        upload.publish_reference(package, "https://media.example.com",
            f"https://upload.example.com/{plan['object_filename']}?signature=secret")
    assert connection.request.call_count == 1
    connection.close.assert_called_once()


@pytest.mark.parametrize("change", ["unsigned", "wrong_object", "private", "userinfo", "fragment", "newline"])
def test_bad_signed_url_stops_before_dns(package, change):
    plan, _ = upload.prepare_upload(package)
    url = f"https://upload.example.com/{plan['object_filename']}?signature=secret"
    if change == "unsigned":
        url = url.split("?")[0]
    elif change == "wrong_object":
        url = url.replace(plan["object_filename"], "mutable.mp4")
    elif change == "private":
        url = url.replace("upload.example.com", "127.0.0.1")
    elif change == "userinfo":
        url = url.replace("upload.example.com", "user:secret@upload.example.com")
    elif change == "fragment":
        url += "#fragment"
    else:
        url += "\n"
    with pytest.raises(ValueError) as exc:
        upload.publish_reference(package, "https://media.example.com", url)
    assert "secret" not in str(exc.value)


def test_existing_wrong_bytes_not_accepted(package, monkeypatch):
    plan, _ = upload.prepare_upload(package)
    connection = Mock()
    connection.getresponse.return_value = SimpleNamespace(status=412)
    monkeypatch.setattr(upload, "_resolve_public_address", lambda host: "93.184.216.34")
    monkeypatch.setattr(upload, "_PinnedHTTPSConnection", lambda *args: connection)
    monkeypatch.setattr(upload, "verify_reference_video_url", Mock(side_effect=ValueError("hash mismatch")))
    with pytest.raises(ValueError, match="hash mismatch"):
        upload.publish_reference(package, "https://media.example.com",
            f"https://upload.example.com/{plan['object_filename']}?signature=secret")
    connection.close.assert_called_once()


def test_ambiguous_transport_failure_redacts_secret(package, monkeypatch, capsys):
    plan, _ = upload.prepare_upload(package)
    connection = Mock()
    connection.request.side_effect = OSError("secret-upload-url")
    monkeypatch.setattr(upload, "_resolve_public_address", lambda host: "93.184.216.34")
    monkeypatch.setattr(upload, "_PinnedHTTPSConnection", lambda *args: connection)
    monkeypatch.setenv("MOTION_UPLOAD_PUT_URL", f"https://upload.example.com/{plan['object_filename']}?signature=secret")
    assert upload.main(["--package", str(package), "--public-base-url", "https://media.example.com", "--execute"]) == 2
    assert "secret" not in capsys.readouterr().out
    connection.close.assert_called_once()
    assert connection.request.call_count == 1


def test_execute_requires_explicit_environment_authorization(package, monkeypatch, capsys):
    monkeypatch.delenv("MOTION_UPLOAD_PUT_URL", raising=False)
    assert upload.main(["--package", str(package), "--execute"]) == 2
    assert "未完成上传" in capsys.readouterr().out

"""Loopback-only live HTTP checks; no tunnels or external requests."""

import hashlib
import http.client
import threading

import pytest

from tools.serve_motion_reference import reference_server

PAYLOAD = b"test-only-mp4-fixture"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def server():
    with reference_server(PAYLOAD, DIGEST) as service:
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            yield service
        finally:
            service.shutdown()
            thread.join(timeout=2)


def request(server, method="GET", path=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(method, path or f"/{DIGEST}.mp4", headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_serves_only_memory_snapshot(server):
    assert server.server_address[0] == "127.0.0.1"
    status, headers, body = request(server)
    assert status == 200 and body == PAYLOAD
    assert headers["Content-Type"] == "video/mp4"
    assert headers["ETag"] == f'"{DIGEST}"'
    status, headers, body = request(server, "HEAD")
    assert status == 200 and not body and int(headers["Content-Length"]) == len(PAYLOAD)


@pytest.mark.parametrize("path", ["/", "/.env", "/../.env", "/%2e%2e/.env", f"/{DIGEST}.mp4?x=1", "/other.mp4"])
def test_arbitrary_files_and_listing_not_exposed(server, path):
    assert request(server, path=path)[0] == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_write_methods_unsupported(server, method):
    assert request(server, method)[0] == 501
    assert request(server)[2] == PAYLOAD


@pytest.mark.parametrize("value,expected", [("bytes=0-3", PAYLOAD[:4]), ("bytes=4-", PAYLOAD[4:]),
    ("bytes=-3", PAYLOAD[-3:]), ("bytes=0-999", PAYLOAD), ("bytes=-999", PAYLOAD)])
def test_single_byte_range(server, value, expected):
    status, headers, body = request(server, headers={"Range": value})
    assert status == 206 and body == expected
    assert headers["Content-Range"].endswith(f"/{len(PAYLOAD)}")


@pytest.mark.parametrize("value", ["bytes=999-", "bytes=4-1", "bytes=-0", "bytes=-", "bytes=0-1,4-5", "garbage"])
def test_invalid_range_bounded_and_rejected(server, value):
    status, headers, body = request(server, headers={"Range": value})
    assert status == 416 and not body
    assert headers["Content-Range"] == f"bytes */{len(PAYLOAD)}"

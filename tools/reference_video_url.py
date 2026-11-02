"""Public HTTPS reference verification without redirects, proxies, or URL logging."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import re
import socket
import ssl
import time
from urllib.parse import SplitResult, urlsplit


MAX_VIDEO_REFERENCE_BYTES = 32 * 1024 * 1024
TRANSFER_TIMEOUT = 30
SOCKET_TIMEOUT = 10


def _public_ip(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return address.is_global and not address.is_multicast


def validate_public_video_url(value: str) -> SplitResult:
    """Syntax-only validation: preparation must remain entirely offline."""
    message = "参考链接须为无登录、无查询参数的公网 HTTPS 文件地址（443 端口）"
    if not isinstance(value, str) or len(value) > 2048 or any(
        ord(char) < 33 or ord(char) > 126 for char in value
    ) or any(char in value for char in "\\?#"):
        raise ValueError(message)
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if (parts.scheme != "https" or not host or parts.username is not None
                or parts.password is not None or parts.port not in (None, 443)):
            raise ValueError
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}", host):
                raise ValueError
            if host.lower().endswith((".localhost", ".local", ".internal", ".test", ".invalid")):
                raise ValueError
        else:
            if not _public_ip(host):
                raise ValueError
    except ValueError:
        raise ValueError(message) from None
    return parts


def _resolve_public_address(host: str) -> str:
    pin = os.environ.get("MOTION_REFERENCE_DNS_PIN")
    if pin is not None:
        try:
            if len(pin) > 384 or pin.count("=") != 1:
                raise ValueError
            pinned_host, address = pin.split("=")
            parsed = validate_public_video_url(f"https://{pinned_host}")
            if parsed.hostname != pinned_host.lower() or not _public_ip(address):
                raise ValueError
        except ValueError:
            raise ValueError("公网 DNS 固定配置无效；必须为 hostname=public_ip") from None
        if host.lower() == pinned_host.lower():
            # 显式、单域名的公网地址固定用于 Fake-IP 网络；TLS 仍验证原域名，绝不接受私网地址。
            return address
    records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses = [record[4][0] for record in records]
    if not addresses or any(not _public_ip(address) for address in addresses):
        raise ValueError("参考链接解析到了非公网地址，已停止")
    return addresses[0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to the checked IP while retaining hostname TLS verification/SNI."""

    def __init__(self, host: str, address: str):
        super().__init__(host, timeout=SOCKET_TIMEOUT, context=ssl.create_default_context())
        self.address = address

    def connect(self) -> None:
        sock = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def verify_reference_video_url(url: str, expected_sha256: str) -> None:
    """GET and hash bounded bytes immediately before creating a provider task."""
    parts = validate_public_video_url(url)
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("公网动作参考必须绑定本地 SHA-256")
    connection = None
    started = time.monotonic()
    try:
        address = _resolve_public_address(parts.hostname)
        connection = _PinnedHTTPSConnection(parts.hostname, address)
        # 不使用环境代理，不二次解析域名，不跟随重定向，不携带认证信息。
        connection.request("GET", parts.path or "/", headers={"Accept-Encoding": "identity"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("参考链接不可直接读取（要求 HTTP 200，不接受登录或跳转）")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("参考链接不能返回压缩编码内容")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdecimal() or not 0 < int(length) <= MAX_VIDEO_REFERENCE_BYTES):
            raise ValueError("远端参考文件大小无效或超过 32 MiB")
        size, digest = 0, hashlib.sha256()
        while True:
            if time.monotonic() - started > TRANSFER_TIMEOUT:
                raise ValueError("参考链接校验超时；未提交生成任务")
            # 单次底层读取即返回，避免慢速滴流为凑满缓冲区而跳过期限检查。
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_REFERENCE_BYTES:
                raise ValueError("远端参考文件超过 32 MiB")
            digest.update(chunk)
        if not size or (length is not None and size != int(length)):
            raise ValueError("远端参考文件为空或下载不完整")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("远端参考视频与本地文件指纹不一致；未提交生成任务")
    except (OSError, http.client.HTTPException):
        # 网络异常可能包含完整 URL；对用户仅输出稳定、无凭据的诊断。
        raise ValueError("参考链接连接、证书或传输失败；未提交生成任务") from None
    finally:
        if connection is not None:
            connection.close()

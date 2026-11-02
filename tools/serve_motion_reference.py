"""Serve one validated, immutable in-memory MP4 on loopback for a temporary tunnel."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tools.publish_motion_reference import prepare_upload


def reference_server(payload: bytes, digest: str, port: int = 0) -> HTTPServer:
    if not payload or len(payload) > 32 * 1024 * 1024 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("无效参考视频")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("端口无效")
    resource = f"/{digest}.mp4"

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def version_string(self):
            return "MotionReference"

        def log_message(self, format, *args):
            # 公网请求不写入日志，避免保存访客地址或恶意路径。
            pass

        def do_HEAD(self):
            self.send_reference(head=True)

        def do_GET(self):
            self.send_reference(head=False)

        def send_reference(self, *, head: bool):
            if self.path != resource:
                self.send_error(404)
                return
            start, end = 0, len(payload) - 1
            requested_range = self.headers.get("Range")
            if requested_range is not None:
                match = re.fullmatch(r"bytes=(\d{0,10})-(\d{0,10})", requested_range)
                valid = match is not None and bool(match.group(1) or match.group(2))
                if valid:
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), end) if right else end
                    else:
                        size = int(right)
                        start = max(0, len(payload) - size)
                        valid = size > 0
                    valid = valid and 0 <= start <= end < len(payload)
                if not valid:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if requested_range is not None else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{digest}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if requested_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            if not head:
                try:
                    self.wfile.write(payload[start:end + 1])
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return

    return HTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="仅在本机提供一个经校验的参考视频，无目录浏览；需另行启动临时隧道")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    plan, payload = prepare_upload(args.package)
    with reference_server(payload, plan["sha256"], args.port) as server:
        print(json.dumps({"local_url": f"http://127.0.0.1:{server.server_port}/{plan['object_filename']}",
                          "sha256": plan["sha256"], "bytes": len(payload), "public": False}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

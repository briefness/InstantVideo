"""Plan or explicitly execute a conditional S3-compatible upload of one reference."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from pipeline.motion_plan import MotionPlan, compile_motion_previs, motion_source_binding
from pipeline.motion_reference import probe_motion_reference, read_reference_bytes
from tools.motion_previs import read_previs_json
from tools.motion_media_validation import verify_rendered_motion
from tools.reference_video_url import (
    _PinnedHTTPSConnection, _resolve_public_address, validate_public_video_url,
    verify_reference_video_url,
)


def prepare_upload(package: Path, public_base_url: str | None = None) -> tuple[dict, bytes]:
    manifest = read_previs_json(package / "manifest.json")
    if manifest.get("version") != "motion-previs-v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("不支持的预演包或文件清单")
    if file_hash := manifest["files"].get("motion_plan.json"):
        if file_hash != hashlib.sha256(read_reference_bytes(package / "motion_plan.json")).hexdigest():
            raise ValueError("预演计划指纹不一致")
    else:
        raise ValueError("预演计划缺少指纹")
    plan = MotionPlan.model_validate(read_previs_json(package / "motion_plan.json"))
    source = read_previs_json(package / "source_shot.json")
    compiled = compile_motion_previs(source["shots"][0], source["ratio"], plan)
    if (compiled["diagnostics"]["issues"] or manifest.get("state_kind") != "planned_only"
            or manifest.get("source_sha256") != motion_source_binding(source["shots"][0], source["ratio"])):
        raise ValueError("预演包未通过检查")
    payload = read_reference_bytes(package / "reference.mp4")
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("files", {}).get("reference.mp4") != digest:
        raise ValueError("参考视频指纹与预演包不一致")
    probe_motion_reference(payload, source["shots"][0]["duration"], source["ratio"])
    verify_rendered_motion(payload, plan, compiled)
    public_url = None
    if public_base_url is not None:
        validate_public_video_url(public_base_url)
        public_url = f"{public_base_url.rstrip('/')}/{digest}.mp4"
        validate_public_video_url(public_url)
    return {"status": "planned_offline", "sha256": digest, "bytes": len(payload),
            "object_filename": f"{digest}.mp4", "public_url": public_url,
            "overwrite_allowed": False, "remote_calls": 0}, payload


def publish_reference(package: Path, public_base_url: str, signed_put_url: str) -> dict:
    plan, payload = prepare_upload(package, public_base_url)
    # 签名 URL 仅驻留内存；错误不含该值，不持久化签名、令牌或访问密钥。
    if not isinstance(signed_put_url, str) or len(signed_put_url) > 8192 or any(
        ord(char) < 33 or ord(char) > 126 for char in signed_put_url
    ):
        raise ValueError("上传授权链接格式无效")
    try:
        parts = urlsplit(signed_put_url)
        validate_public_video_url(parts._replace(query="").geturl())
        if not parts.query or not parts.path.endswith("/" + plan["object_filename"]):
            raise ValueError
    except ValueError:
        raise ValueError("上传授权必须指向以本地 SHA-256 命名的 HTTPS 对象") from None
    connection = None
    try:
        address = _resolve_public_address(parts.hostname)
        connection = _PinnedHTTPSConnection(parts.hostname, address)
        connection.request("PUT", parts.path + "?" + parts.query, body=payload, headers={
            "Content-Type": "video/mp4", "If-None-Match": "*",
        })
        status = connection.getresponse().status
        if status not in {200, 201, 204, 412}:
            raise ValueError("上传未确认成功；不自动重试或覆盖，请检查存储服务与原对象")
    except (OSError, http.client.HTTPException):
        raise ValueError("上传结果未知；保留本地文件，核对同名对象后再执行条件上传") from None
    finally:
        if connection is not None:
            connection.close()
    # 412 表示对象已存在，只有公开读取的字节相同才可作为幂等成功。
    verify_reference_video_url(plan["public_url"], plan["sha256"])
    return {"status": "published_verified", "sha256": plan["sha256"],
            "public_url": plan["public_url"], "reused_existing": status == 412}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="动作参考上传：默认离线计划，--execute 才写入对象存储")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--public-base-url")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            if not args.public_base_url or not os.environ.get("MOTION_UPLOAD_PUT_URL"):
                raise ValueError("执行上传需要公开读取基址和环境变量 MOTION_UPLOAD_PUT_URL")
            result = publish_reference(args.package, args.public_base_url, os.environ["MOTION_UPLOAD_PUT_URL"])
        else:
            result, _ = prepare_upload(args.package, args.public_base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        print("未完成上传；检查预演包、公开读取基址及条件 PUT 授权。不会输出签名链接或自动覆盖对象。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

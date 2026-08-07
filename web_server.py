#!/usr/bin/env python3
"""Local web workbench for the existing Seedance pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from aiohttp import web

import config
from pipeline.models import RunOptions
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import MANIFEST_FILENAME, RunWorkspace


WEB_ROOT = Path(__file__).parent / "web"
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT_KEY = web.AppKey("output_root", Path)
ACTIVE_RUNS_KEY = web.AppKey("active_runs", dict[str, asyncio.Task[None]])
PIPELINE_FACTORY_KEY = web.AppKey("pipeline_factory", object)

STYLE_OPTIONS = (
    "cinematic",
    "energetic",
    "warm",
    "cold",
    "dramatic",
    "futuristic",
    "premium",
)
MEDIA_SUFFIXES = {".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".webp", ".srt"}


def _json_error(message: str, *, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _workspace_path(output_root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise web.HTTPNotFound()
    workspace = (output_root / run_id).resolve()
    if workspace.parent != output_root.resolve():
        raise web.HTTPNotFound()
    if not (workspace / MANIFEST_FILENAME).is_file():
        raise web.HTTPNotFound()
    return workspace


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _media_url(run_id: str, workspace: Path, value: str | None) -> str | None:
    if not value or value.startswith(("http://", "https://", "data:")):
        return None
    path = Path(value)
    candidate = (path if path.is_absolute() else workspace / path).resolve()
    try:
        relative = candidate.relative_to(workspace.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in MEDIA_SUFFIXES:
        return None
    return f"/media/{quote(run_id)}/{quote(relative.as_posix(), safe='/')}"


def _run_summary(
    workspace: Path,
    *,
    active: bool = False,
    include_storyboard: bool = True,
) -> dict[str, Any] | None:
    manifest = _read_json(workspace / MANIFEST_FILENAME)
    if not manifest:
        return None

    run_id = str(manifest.get("run_id") or workspace.name)
    options = manifest.get("options") if isinstance(manifest.get("options"), dict) else {}
    shots_state = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
    storyboard = _read_json(workspace / "storyboard.json") if include_storyboard else None
    storyboard_shots = {
        str(shot.get("shot_id")): shot
        for shot in (storyboard or {}).get("shots", [])
        if isinstance(shot, dict) and shot.get("shot_id") is not None
    }

    shots: list[dict[str, Any]] = []
    for shot_id in sorted(
        shots_state,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    ):
        state = shots_state.get(shot_id)
        if not isinstance(state, dict):
            continue
        detail = storyboard_shots.get(str(shot_id), {})
        shots.append(
            {
                "shot_id": state.get("shot_id", shot_id),
                "status": state.get("status", "pending"),
                "duration": detail.get("duration"),
                "scene_description": detail.get("scene_description", ""),
                "primary_action": detail.get("primary_action", ""),
                "camera": (detail.get("camera") or {}).get("composition", "")
                if isinstance(detail.get("camera"), dict)
                else "",
                "quality_score": state.get("quality_score", 0),
                "technical_quality_score": state.get("technical_quality_score", 0),
                "semantic_accepted": state.get("semantic_accepted"),
                "attempts": state.get("attempts", 0),
                "errors": state.get("errors", []),
                "video_url": _media_url(run_id, workspace, state.get("local_path")),
                "poster_url": _media_url(run_id, workspace, state.get("last_frame_url")),
            }
        )

    successful_shots = sum(shot["status"] == "success" for shot in shots)
    reservations = (manifest.get("paid_take_budget") or {}).get("reservations", [])
    used_takes = sum(
        isinstance(item, dict) and item.get("status") in {"submitted", "reconciled"}
        for item in reservations
    )
    final_url = _media_url(run_id, workspace, manifest.get("final_path"))
    exports = {
        platform: _media_url(run_id, workspace, f"exports/{platform}.mp4")
        for platform in options.get("platforms", [])
    }
    exports = {name: url for name, url in exports.items() if url}
    poster_url = next((shot["poster_url"] for shot in reversed(shots) if shot["poster_url"]), None)
    status = str(manifest.get("status", "created"))

    return {
        "run_id": run_id,
        "status": status,
        "stage": manifest.get("stage", "initialized"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "request": options.get("request", ""),
        "options": {
            "resolution": options.get("resolution", config.DEFAULT_RESOLUTION),
            "aspect_ratio": options.get("aspect_ratio", config.DEFAULT_RATIO),
            "style": options.get("style", "cinematic"),
            "music": Path(options["music_path"]).name if options.get("music_path") else None,
            "platforms": options.get("platforms", []),
            "paid_take_budget": options.get("paid_take_budget"),
        },
        "title": (storyboard or {}).get("title", "") if include_storyboard else "",
        "mood": (storyboard or {}).get("mood", "") if include_storyboard else "",
        "total_duration": (storyboard or {}).get("total_duration") if include_storyboard else None,
        "shots": shots if include_storyboard else [],
        "progress": {"completed": successful_shots, "total": len(shots)},
        "budget": {
            "used": used_takes,
            "limit": options.get("paid_take_budget"),
            "estimated": (manifest.get("paid_take_budget") or {}).get("estimated_takes", 0),
        },
        "assets": {
            "final_url": final_url,
            "poster_url": poster_url,
            "exports": exports,
        },
        "error": manifest.get("error"),
        "active": active,
        "can_resume": not active and status in {"created", "failed", "interrupted"},
    }


def _music_options() -> list[dict[str, str]]:
    if not config.MUSIC_DIR.is_dir():
        return []
    return [
        {"value": path.name, "label": path.stem.replace("_", " ")}
        for path in sorted(config.MUSIC_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac"}
    ]


def _parse_create_payload(payload: Any) -> RunOptions:
    if not isinstance(payload, dict):
        raise ValueError("请求格式无效")
    request = str(payload.get("request", "")).strip()
    if not request:
        raise ValueError("视频需求不能为空")
    if len(request) > 4000:
        raise ValueError("视频需求不能超过 4000 个字符")

    music_name = payload.get("music")
    music_path: str | None = None
    if music_name:
        if Path(str(music_name)).name != str(music_name):
            raise ValueError("背景音乐无效")
        candidate = (config.MUSIC_DIR / str(music_name)).resolve()
        if candidate.parent != config.MUSIC_DIR.resolve() or not candidate.is_file():
            raise ValueError("背景音乐不存在")
        music_path = str(candidate)

    platforms = payload.get("platforms", ["youtube", "tiktok"])
    if not isinstance(platforms, list) or not platforms:
        raise ValueError("至少选择一个导出平台")

    budget = payload.get("paid_take_budget")
    if budget in ("", None):
        budget = None
    elif isinstance(budget, bool):
        raise ValueError("付费 Take 预算必须是非负整数")
    else:
        try:
            budget = int(budget)
        except (TypeError, ValueError) as exc:
            raise ValueError("付费 Take 预算必须是非负整数") from exc

    return RunOptions(
        request=request,
        resolution=str(payload.get("resolution", config.DEFAULT_RESOLUTION)),
        aspect_ratio=str(payload.get("aspect_ratio", config.DEFAULT_RATIO)),
        style=str(payload.get("style", "cinematic")).strip(),
        music_path=music_path,
        platforms=[str(platform) for platform in platforms],
        paid_take_budget=budget,
    )


async def _execute_pipeline(app: web.Application, run_id: str, pipeline: Any) -> None:
    try:
        await asyncio.to_thread(lambda: asyncio.run(pipeline.run()))
    except Exception:
        LOGGER.exception("Seedance run %s stopped", run_id)
    finally:
        app[ACTIVE_RUNS_KEY].pop(run_id, None)


def _start_pipeline(app: web.Application, run_id: str, pipeline: Any) -> None:
    task = asyncio.create_task(_execute_pipeline(app, run_id, pipeline))
    app[ACTIVE_RUNS_KEY][run_id] = task


async def index(_request: web.Request) -> web.StreamResponse:
    return web.FileResponse(WEB_ROOT / "index.html")


async def get_config(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "resolutions": list(config.SUPPORTED_RESOLUTIONS),
            "aspect_ratios": list(config.SUPPORTED_ASPECT_RATIOS),
            "platforms": list(config.SUPPORTED_PLATFORMS),
            "styles": list(STYLE_OPTIONS),
            "music": _music_options(),
            "api_ready": bool(config.ARK_API_KEY),
            "semantic_review_enabled": config.SEMANTIC_REVIEW_ENABLED,
        }
    )


async def list_runs(request: web.Request) -> web.Response:
    output_root = request.app[OUTPUT_ROOT_KEY]
    active_runs = request.app[ACTIVE_RUNS_KEY]
    runs: list[dict[str, Any]] = []
    if output_root.is_dir():
        for workspace in output_root.iterdir():
            if not workspace.is_dir():
                continue
            summary = _run_summary(
                workspace,
                active=workspace.name in active_runs,
                include_storyboard=False,
            )
            if summary:
                runs.append(summary)
    runs.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return web.json_response({"runs": runs})


async def get_run(request: web.Request) -> web.Response:
    workspace = _workspace_path(request.app[OUTPUT_ROOT_KEY], request.match_info["run_id"])
    summary = _run_summary(
        workspace,
        active=workspace.name in request.app[ACTIVE_RUNS_KEY],
    )
    if not summary:
        raise web.HTTPNotFound()
    return web.json_response(summary)


async def create_run(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        options = _parse_create_payload(payload)
    except json.JSONDecodeError:
        return _json_error("请求格式无效")
    except ValueError as exc:
        return _json_error(str(exc))

    workspace = RunWorkspace.create(request.app[OUTPUT_ROOT_KEY], options)
    pipeline_factory = request.app[PIPELINE_FACTORY_KEY]
    pipeline = pipeline_factory(
        resolution=options.resolution,
        aspect_ratio=options.aspect_ratio,
        style=options.style,
        music_path=options.music_path,
        platforms=options.platforms,
        paid_take_budget=options.paid_take_budget,
        run_workspace=workspace,
    )
    pipeline._resuming = False
    _start_pipeline(request.app, workspace.path.name, pipeline)
    return web.json_response(
        _run_summary(workspace.path, active=True),
        status=202,
    )


async def resume_run(request: web.Request) -> web.Response:
    workspace = _workspace_path(request.app[OUTPUT_ROOT_KEY], request.match_info["run_id"])
    run_id = workspace.name
    if run_id in request.app[ACTIVE_RUNS_KEY]:
        return _json_error("该任务正在运行", status=409)

    pipeline_factory = request.app[PIPELINE_FACTORY_KEY]
    pipeline = pipeline_factory.from_workspace(workspace)
    _start_pipeline(request.app, run_id, pipeline)
    return web.json_response(_run_summary(workspace, active=True), status=202)


async def get_media(request: web.Request) -> web.StreamResponse:
    workspace = _workspace_path(request.app[OUTPUT_ROOT_KEY], request.match_info["run_id"])
    relative = request.match_info["path"]
    candidate = (workspace / relative).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise web.HTTPNotFound() from None
    if not candidate.is_file() or candidate.suffix.lower() not in MEDIA_SUFFIXES:
        raise web.HTTPNotFound()
    return web.FileResponse(candidate)


def create_app(
    *,
    output_root: Path | None = None,
    pipeline_factory: Any = VideoPipeline,
) -> web.Application:
    app = web.Application(client_max_size=128 * 1024)
    app[OUTPUT_ROOT_KEY] = (output_root or config.OUTPUT_DIR).resolve()
    app[ACTIVE_RUNS_KEY] = {}
    app[PIPELINE_FACTORY_KEY] = pipeline_factory
    app.router.add_get("/", index)
    app.router.add_get("/api/config", get_config)
    app.router.add_get("/api/runs", list_runs)
    app.router.add_post("/api/runs", create_run)
    app.router.add_get("/api/runs/{run_id}", get_run)
    app.router.add_post("/api/runs/{run_id}/resume", resume_run)
    app.router.add_get("/media/{run_id}/{path:.+}", get_media)
    app.router.add_static("/static", WEB_ROOT, show_index=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Seedance Web 工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

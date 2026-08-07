"""Offline checks for the local Seedance web adapter."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from web_server import create_app


class FakePipeline:
    def __init__(self, *, run_workspace: RunWorkspace, **_kwargs):
        self.run_workspace = run_workspace
        self._resuming = True

    @classmethod
    def from_workspace(cls, workspace: Path):
        return cls(run_workspace=RunWorkspace.resume(workspace))

    async def run(self):
        self.run_workspace.checkpoint("generating")
        final_path = self.run_workspace.path / "final.mp4"
        final_path.write_bytes(b"test-video")
        self.run_workspace.mark_succeeded(str(final_path))
        return str(final_path)


@pytest_asyncio.fixture
async def web_client(tmp_path: Path):
    server = TestServer(create_app(output_root=tmp_path, pipeline_factory=FakePipeline))
    client = TestClient(server)
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_lists_existing_runs_and_serves_details(web_client: TestClient, tmp_path: Path):
    workspace = RunWorkspace.create(
        tmp_path,
        RunOptions(request="15秒咖啡广告", platforms=["youtube"]),
    )
    workspace.save_storyboard(
        {
            "title": "咖啡时刻",
            "aspect_ratio": "16:9",
            "resolution": "480p",
            "shots": [
                {
                    "shot_id": 1,
                    "duration": 5,
                    "scene_description": "咖啡豆落入研磨机",
                    "prompt_en": "Coffee beans fall into a grinder",
                }
            ],
        }
    )

    response = await web_client.get("/api/runs")
    assert response.status == 200
    runs = (await response.json())["runs"]
    assert runs[0]["run_id"] == workspace.path.name
    assert runs[0]["request"] == "15秒咖啡广告"

    response = await web_client.get(f"/api/runs/{workspace.path.name}")
    detail = await response.json()
    assert detail["title"] == "咖啡时刻"
    assert detail["shots"][0]["scene_description"] == "咖啡豆落入研磨机"


@pytest.mark.asyncio
async def test_create_run_uses_existing_pipeline_contract(web_client: TestClient):
    response = await web_client.post(
        "/api/runs",
        json={
            "request": "制作一个15秒产品视频",
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "style": "cinematic",
            "platforms": ["tiktok"],
            "paid_take_budget": 3,
        },
    )
    assert response.status == 202
    created = await response.json()

    for _ in range(20):
        await asyncio.sleep(0.01)
        response = await web_client.get(f"/api/runs/{created['run_id']}")
        detail = await response.json()
        if detail["status"] == "succeeded":
            break

    assert detail["status"] == "succeeded"
    assert detail["options"]["aspect_ratio"] == "9:16"
    assert detail["options"]["paid_take_budget"] == 3
    assert detail["assets"]["final_url"].endswith("/final.mp4")


@pytest.mark.asyncio
async def test_rejects_invalid_options_and_media_traversal(web_client: TestClient):
    response = await web_client.post(
        "/api/runs",
        json={"request": "test", "resolution": "1080p", "platforms": ["youtube"]},
    )
    assert response.status == 400
    assert "unsupported" in (await response.json())["error"]

    response = await web_client.get("/media/not-a-run/../../.env")
    assert response.status == 404

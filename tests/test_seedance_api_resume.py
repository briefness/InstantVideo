"""Ark task submission and recovery tests."""

from types import SimpleNamespace

import pytest

from tools.seedance_api import SeedanceAPI


class FakeTasks:
    def __init__(self):
        self.created = 0
        self.polled: list[str] = []

    def create(self, **_kwargs):
        self.created += 1
        return SimpleNamespace(id="ark-task-1")

    def get(self, *, task_id: str):
        self.polled.append(task_id)
        return SimpleNamespace(
            status="succeeded",
            content=SimpleNamespace(
                video_url="https://example.com/video.mp4",
                last_frame_url="https://example.com/frame.jpg",
                audio_url=None,
            ),
        )


def api_with_fake_tasks() -> tuple[SeedanceAPI, FakeTasks]:
    tasks = FakeTasks()
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(
        content_generation=SimpleNamespace(tasks=tasks)
    )
    return api, tasks


@pytest.mark.asyncio
async def test_new_task_is_persistable_before_polling():
    api, tasks = api_with_fake_tasks()
    submitted: list[str] = []

    result = await api.generate(
        prompt="test",
        timeout=1,
        on_submitted=submitted.append,
    )

    assert submitted == ["ark-task-1"]
    assert tasks.created == 1
    assert tasks.polled == ["ark-task-1"]
    assert result["provider_task_id"] == "ark-task-1"


@pytest.mark.asyncio
async def test_resume_polls_existing_task_without_resubmitting():
    api, tasks = api_with_fake_tasks()

    result = await api.generate(
        prompt="ignored while resuming",
        timeout=1,
        task_id="ark-task-existing",
    )

    assert tasks.created == 0
    assert tasks.polled == ["ark-task-existing"]
    assert result["provider_task_id"] == "ark-task-existing"


@pytest.mark.asyncio
async def test_poll_error_preserves_existing_task_identity():
    api, tasks = api_with_fake_tasks()
    tasks.get = lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("offline"))

    result = await api.generate(
        prompt="ignored while resuming",
        timeout=1,
        task_id="ark-task-existing",
    )

    assert tasks.created == 0
    assert result["error_type"] == "poll_error"
    assert result["provider_task_id"] == "ark-task-existing"


@pytest.mark.asyncio
async def test_checkpoint_failure_never_submits_a_second_task():
    api, tasks = api_with_fake_tasks()

    def fail_checkpoint(_task_id: str) -> None:
        raise OSError("disk full")

    with pytest.raises(RuntimeError, match="ark-task-1"):
        await api.generate(
            prompt="test",
            timeout=1,
            on_submitted=fail_checkpoint,
        )

    assert tasks.created == 1
    assert tasks.polled == []

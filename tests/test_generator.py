"""角色一致性 + 画面衔接逻辑测试"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from pipeline.generator import (
    RemoteTaskPendingError,
    ShotResult,
    VideoGenerator,
)
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from pipeline.semantic_review import SEMANTIC_REVIEW_VERSION, SemanticReview


@pytest.fixture
def generator():
    return VideoGenerator(tempfile.mkdtemp())


def _record_progress(workspace: RunWorkspace):
    def record(result: ShotResult) -> None:
        workspace.record_shot(
            shot_id=result.shot_id,
            status=result.status,
            provider_task_id=result.provider_task_id,
            prompt_profile=result.prompt_profile,
            prompt_fingerprint=result.prompt_fingerprint,
            compiled_contract_version=result.compiled_contract_version,
            compiled_contract_fingerprint=result.compiled_contract_fingerprint,
            prompt_attempts=result.prompt_attempts,
            attempts=result.attempts,
            errors=result.errors,
        )
    return record


class TestShotHasCharacter:
    def test_with_characters(self, generator):
        assert generator._shot_has_character({"characters": ["hero"]})

    def test_with_extract_flag(self, generator):
        assert generator._shot_has_character({"extract_character_ref": True})

    def test_no_character(self, generator):
        assert not generator._shot_has_character({"subtitle_text": "x"})


@pytest.mark.asyncio
async def test_pending_remote_task_stops_before_submitting_later_shots(
    generator, monkeypatch
):
    calls = []

    async def fake_generate(shot, **_kwargs):
        calls.append(shot["shot_id"])
        return ShotResult(
            shot_id=shot["shot_id"],
            status="running",
            provider_task_id="ark-task-running",
            errors=["Timeout (600s)"],
        )

    monkeypatch.setattr(generator, "_generate_single_shot", fake_generate)
    storyboard = {
        "characters": [],
        "shots": [
            {"shot_id": 1, "duration": 4, "prompt_en": "first shot"},
            {"shot_id": 2, "duration": 5, "prompt_en": "second shot"},
        ],
    }

    with pytest.raises(RemoteTaskPendingError, match="ark-task-running"):
        await generator.generate_all(storyboard)

    assert calls == [1]


@pytest.mark.asyncio
async def test_failed_shot_stops_before_generating_later_shots(
    generator, monkeypatch
):
    calls = []

    async def fake_generate(shot, **_kwargs):
        calls.append(shot["shot_id"])
        if shot["shot_id"] == 1:
            return ShotResult(
                shot_id=1,
                status="success",
                last_frame_url="accepted-tail.jpg",
            )
        return ShotResult(
            shot_id=2,
            status="failed",
            errors=["semantic contract failed"],
        )

    monkeypatch.setattr(generator, "_generate_single_shot", fake_generate)
    storyboard = {
        "characters": [],
        "shots": [
            {"shot_id": 1, "duration": 4, "prompt_en": "first shot"},
            {"shot_id": 2, "duration": 5, "prompt_en": "second shot"},
            {"shot_id": 3, "duration": 6, "prompt_en": "third shot"},
        ],
    }

    with pytest.raises(RuntimeError, match="Shot 2.*停止后续镜头"):
        await generator.generate_all(storyboard)

    assert calls == [1, 2]


class TestBuildImageRefs:
    """_build_image_refs 返回 (image_urls, role)"""

    def test_no_refs_t2v(self, generator):
        """无参考图 → T2V"""
        urls, role = generator._build_image_refs({"shot_id": 1}, None)
        assert urls == []
        assert role is None

    def test_prev_frame_only_first_frame(self, generator):
        """仅上一帧 → first_frame I2V"""
        urls, role = generator._build_image_refs(
            {"shot_id": 2}, "http://a/frame.jpg"
        )
        assert urls == ["http://a/frame.jpg"]
        assert role == "first_frame"

    def test_char_ref_only(self, generator):
        """有角色参考帧, 无上一帧 → reference_image"""
        # 模拟已提取角色参考帧
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {"shot_id": 2, "characters": ["hero"]}, None
        )
        assert urls == [ref_path]
        assert role == "reference_image"

    def test_seamless_continuation_uses_only_previous_frame(self, generator):
        """无缝续接时尾帧独占 first_frame 职责。"""
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 3,
                "scene_id": "street",
                "characters": ["hero"],
                "continuity_from_previous": "seamless",
            },
            "http://a/last.jpg",
        )
        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_same_scene_intentional_cut_uses_official_first_frame_state_role(self, generator):
        """同场景状态交接使用 Seedance 官方 first_frame 职责。"""
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 3,
                "scene_id": "street",
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "small",
            },
            "http://a/last.jpg",
            {"shot_id": 2, "scene_id": "street"},
        )
        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_large_same_scene_cut_keeps_state_and_identity_separate(
        self, generator
    ):
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 3,
                "scene_id": "street",
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "large",
            },
            "http://a/last.jpg",
            {"shot_id": 2, "scene_id": "street"},
        )

        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_medium_same_scene_cut_without_identity_ref_falls_back_to_state_tail(
        self, generator
    ):
        urls, role = generator._build_image_refs(
            {
                "shot_id": 2,
                "scene_id": "street",
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "medium",
            },
            "http://a/last.jpg",
            {"shot_id": 1, "scene_id": "street", "output_reference_depth": 0},
        )

        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_medium_same_scene_cut_uses_state_and_identity_refs(
        self, generator
    ):
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 2,
                "scene_id": "street",
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "medium",
            },
            "http://a/last.jpg",
            {"shot_id": 1, "scene_id": "street", "output_reference_depth": 0},
        )

        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_medium_same_scene_object_handoff_uses_tail_as_state_reference(
        self, generator
    ):
        urls, role = generator._build_image_refs(
            {
                "shot_id": 2,
                "scene_id": "workbench",
                "characters": ["domino_group"],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "medium",
            },
            "http://a/last.jpg",
            {"shot_id": 1, "scene_id": "workbench", "output_reference_depth": 0},
        )

        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_cross_scene_intentional_cut_uses_only_character_reference(self, generator):
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 3,
                "scene_id": "rooftop",
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
            },
            "http://a/last.jpg",
            {"shot_id": 2, "scene_id": "street"},
        )

        assert urls == [ref_path]
        assert role == "reference_image"

    def test_no_char_in_shot_uses_first_frame(self, generator):
        """镜头无角色但有上一帧 → first_frame"""
        # 即使 generator 有角色参考帧, 如果镜头没标角色就不用
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {"shot_id": 4, "characters": []}, "http://a/last.jpg"
        )
        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_same_scene_insert_inherits_previous_state_reference(self, generator):
        urls, role = generator._build_image_refs(
            {
                "shot_id": 4,
                "scene_id": "street",
                "characters": [],
                "continuity_from_previous": "intentional_cut",
                "composition_change": "small",
            },
            "http://a/last.jpg",
            {"shot_id": 3, "scene_id": "street"},
        )
        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_local_prev_frame_supported(self, generator):
        """恢复时提取的本地尾帧可由 API 转成 data URI 继续衔接"""
        local_file = str(Path(__file__).resolve())
        urls, role = generator._build_image_refs({"shot_id": 2}, local_file)
        assert urls == [local_file]
        assert role == "first_frame"


@pytest.mark.asyncio
async def test_dependent_cached_take_without_matching_provenance_stops(
    generator, monkeypatch
):
    previous_tail = generator.output_dir / "shots" / "shot_001_lastframe.jpg"
    previous_tail.write_bytes(b"accepted-tail")
    cached = generator.output_dir / "shots" / "shot_002.mp4"
    cached.write_bytes(b"legacy-take-without-state-reference")

    class NoGenerationAPI:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            pytest.fail("provenance mismatch must not authorize a paid task")

    generator.api = NoGenerationAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: str(previous_tail)
    )
    shot = {
        "shot_id": 2,
        "duration": 5,
        "scene_id": "workbench",
        "scene_description": "same workbench",
        "prompt_en": "Continue the visible physical process.",
        "continuity_from_previous": "intentional_cut",
        "composition_change": "medium",
        "characters": [],
        "production_slot": {"reference_policy": "state_and_identity"},
    }
    previous = {"shot_id": 1, "scene_id": "workbench"}

    result = await generator._generate_single_shot(
        shot,
        str(previous_tail),
        previous,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert cached.read_bytes() == b"legacy-take-without-state-reference"
    assert "缓存冲突本身不授权" in result.errors[-1]


@pytest.mark.asyncio
async def test_resume_provenance_mismatch_stops_without_paid_generation(
    generator,
):
    previous_tail = generator.output_dir / "shots" / "shot_001_lastframe.jpg"
    previous_tail.write_bytes(b"accepted-tail")
    accepted = generator.output_dir / "shots" / "shot_002.mp4"
    accepted.write_bytes(b"accepted-take")

    class NoGenerationAPI:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            pytest.fail("resume provenance mismatch must not submit a paid task")

    generator.api = NoGenerationAPI()
    generator.semantic_reviewer = None
    generator.accepted_shot_artifacts = {
        2: {"local_path": str(accepted), "semantic_accepted": True}
    }
    shot = {
        "shot_id": 2,
        "duration": 5,
        "scene_id": "workbench",
        "scene_description": "same workbench",
        "prompt_en": "Continue the visible physical process.",
        "continuity_from_previous": "intentional_cut",
        "composition_change": "medium",
        "characters": [],
        "production_slot": {"reference_policy": "state_and_identity"},
    }

    result = await generator._generate_single_shot(
        shot,
        str(previous_tail),
        {"shot_id": 1, "scene_id": "workbench"},
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert accepted.read_bytes() == b"accepted-take"
    assert "不会在 resume 中隐式提交" in result.errors[-1]


@pytest.mark.asyncio
async def test_pending_task_with_missing_compiled_contract_stops_before_api_poll(
    generator,
):
    class NoPollAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            pytest.fail("mismatched pending task must not be polled or resubmitted")

    generator.api = NoPollAPI()
    generator.resume_task_ids = {1: "old-remote-task"}
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A neutral legacy shot.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == 0
    assert "未轮询、未提交" in result.errors[-1]


@pytest.mark.asyncio
async def test_normal_pending_descriptor_polls_same_task_without_new_submission(
    generator,
):
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral legacy shot."}
    fingerprint = "a" * 64
    descriptor = {
        "task_id": "normal-pending-task",
        "prompt_profile": "normal",
        "prompt_fingerprint": fingerprint,
        "compiled_contract_version": "action-contract-v2",
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }
    provenance = generator._generation_provenance(
        shot, [], None, prompt_profile="normal", prompt_fingerprint=fingerprint
    )
    generator._write_generation_provenance(1, provenance, descriptor["task_id"])

    class PollOnlyAPI:
        supports_last_frame = True

        def __init__(self):
            self.task_ids = []
            self.creates = 0

        async def generate(self, **kwargs):
            self.task_ids.append(kwargs["task_id"])
            if kwargs["task_id"] is None:
                self.creates += 1
                pytest.fail("pending resume must not create a task")
            return {
                "status": "failed",
                "error_type": "poll_timeout",
                "provider_task_id": kwargs["task_id"],
                "error": "still pending",
            }

    generator.api = PollOnlyAPI()
    generator.semantic_reviewer = None
    generator.resume_tasks = {1: descriptor}

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "running"
    assert generator.api.task_ids == ["normal-pending-task"]
    assert generator.api.creates == 0


@pytest.mark.asyncio
async def test_paid_take_budget_stops_fresh_submission_before_fake_provider_call(
    tmp_path: Path,
):
    workspace = RunWorkspace.create(
        tmp_path,
        RunOptions(request="A test video", paid_take_budget=0),
    )
    generator = VideoGenerator(
        str(workspace.path),
        reserve_paid_take=workspace.reserve_paid_take,
        confirm_paid_take_submission=workspace.confirm_paid_take_submission,
        reconcile_paid_take=workspace.reconcile_paid_take,
        release_unsubmitted_paid_take=workspace.release_unsubmitted_paid_take,
    )

    class FakeProvider:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            pytest.fail("exhausted budget must stop before provider submission")

    generator.api = FakeProvider()
    generator.semantic_reviewer = None
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral test shot."}

    result = await generator._generate_single_shot(
        shot, None, None, {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    )

    assert result.status == "failed"
    assert generator.api.calls == 0
    assert "预算已耗尽" in result.errors[-1]


@pytest.mark.asyncio
async def test_pending_descriptor_poll_does_not_reserve_a_paid_take(tmp_path: Path):
    reservations: list[int] = []
    generator = VideoGenerator(
        str(tmp_path),
        reserve_paid_take=lambda shot_id: reservations.append(shot_id) or "1:1",
    )
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral legacy shot."}
    fingerprint = "b" * 64
    descriptor = {
        "task_id": "existing-task",
        "prompt_profile": "normal",
        "prompt_fingerprint": fingerprint,
        "compiled_contract_version": "action-contract-v2",
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }
    generator.resume_tasks = {1: descriptor}
    generator._write_generation_provenance(
        1,
        generator._generation_provenance(
            shot, [], None, prompt_profile="normal", prompt_fingerprint=fingerprint
        ),
        "existing-task",
    )

    class FakeProvider:
        supports_last_frame = True

        async def poll_task(self, task_id, *, timeout):
            assert task_id == "existing-task"
            return {
                "status": "failed",
                "error_type": "poll_timeout",
                "provider_task_id": task_id,
                "error": "still pending",
            }

    generator.api = FakeProvider()
    generator.semantic_reviewer = None

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "running"
    assert reservations == []


def test_identity_reanchor_privacy_fallback_preserves_tail_state(tmp_path: Path):
    generator = object.__new__(VideoGenerator)
    shot = {
        "continuity_from_previous": "intentional_cut",
        "production_slot": {"reference_policy": "identity_only"},
    }

    assert generator._build_state_only_refs(
        shot,
        str(tmp_path / "tail.jpg"),
        {"scene_id": "street"},
    ) == ([str(tmp_path / "tail.jpg")], "first_frame")


@pytest.mark.asyncio
async def test_pending_descriptor_uses_poll_api_without_submitting_prompt(generator):
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral legacy shot."}
    fingerprint = "d" * 64
    descriptor = {
        "task_id": "poll-only-task",
        "prompt_profile": "normal",
        "prompt_fingerprint": fingerprint,
        "compiled_contract_version": "action-contract-v2",
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }
    generator._write_generation_provenance(
        1,
        generator._generation_provenance(
            shot, [], None, prompt_profile="normal", prompt_fingerprint=fingerprint
        ),
        descriptor["task_id"],
    )

    class PollOnlyAPI:
        supports_last_frame = True

        def __init__(self):
            self.task_ids = []

        async def poll_task(self, task_id, *, timeout):
            self.task_ids.append((task_id, timeout))
            return {
                "status": "failed",
                "error_type": "poll_timeout",
                "provider_task_id": task_id,
                "error": "still pending",
            }

        async def generate(self, **_kwargs):
            pytest.fail("an existing task must not enter the submit API")

    generator.resume_tasks = {1: descriptor}
    generator.api = PollOnlyAPI()
    generator.semantic_reviewer = None

    result = await generator._generate_single_shot(
        shot, None, None, {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    )

    assert result.status == "running"
    assert generator.api.task_ids == [("poll-only-task", config.GENERATION_TIMEOUT)]


@pytest.mark.asyncio
async def test_download_outage_resumes_the_same_succeeded_task_without_resubmit(
    tmp_path: Path, monkeypatch
):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    storyboard = workspace.save_storyboard({
        "title": "Test",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "A neutral scene",
            "prompt_en": "A neutral scene",
        }],
    })
    shot = storyboard["shots"][0]

    class SubmitThenDownloadOutage:
        supports_last_frame = True

        def __init__(self):
            self.creates = 0

        async def generate(self, **kwargs):
            assert kwargs["task_id"] is None
            self.creates += 1
            return {
                "status": "succeeded",
                "provider_task_id": "succeeded-task",
                "video_url": "remote-video",
            }

        async def download_video(self, _url, _path):
            raise OSError("temporary local storage outage")

    first = VideoGenerator(str(workspace.path), on_progress=_record_progress(workspace))
    first.api = SubmitThenDownloadOutage()
    first.semantic_reviewer = None
    first_result = await first._generate_single_shot(shot, None, None, storyboard)

    assert first_result.status == "running"
    assert first.api.creates == 1
    descriptor = RunWorkspace.resume(workspace.path).resumable_pending_tasks()[1]
    assert descriptor["task_id"] == "succeeded-task"

    class PollThenDownload:
        supports_last_frame = True

        def __init__(self):
            self.polls = []

        async def poll_task(self, task_id, *, timeout):
            self.polls.append((task_id, timeout))
            return {
                "status": "succeeded",
                "provider_task_id": task_id,
                "video_url": "remote-video",
            }

        async def generate(self, **_kwargs):
            pytest.fail("download recovery must not submit a second task")

        async def download_video(self, _url, path):
            Path(path).write_bytes(b"video")

    tail = workspace.path / "shots" / "shot_001_lastframe.jpg"
    tail.write_bytes(b"tail")
    second = VideoGenerator(
        str(workspace.path),
        on_progress=_record_progress(workspace),
        resume_tasks=RunWorkspace.resume(workspace.path).resumable_pending_tasks(),
    )
    second.api = PollThenDownload()
    second.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(second, "_extract_local_tail_frame", lambda *_args: str(tail))

    resumed = await second._generate_single_shot(shot, None, None, storyboard)

    assert resumed.status == "success"
    assert second.api.polls == [("succeeded-task", config.GENERATION_TIMEOUT)]


@pytest.mark.asyncio
async def test_policy_safe_pending_descriptor_round_trips_workspace_without_resubmit(
    tmp_path: Path,
    monkeypatch,
):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    storyboard = {
        "title": "Test",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "A neutral test scene",
            "prompt_en": "A neutral test scene",
        }],
    }
    storyboard = workspace.save_storyboard(storyboard)
    shot = storyboard["shots"][0]

    class SubmitThenTimeoutAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "status": "failed",
                    "error_type": "moderation",
                    "error_locus": "input_text",
                    "error": "moderated",
                }
            kwargs["on_submitted"]("policy-pending-task")
            return {
                "status": "failed",
                "error_type": "poll_timeout",
                "provider_task_id": "policy-pending-task",
                "error": "still pending",
            }

    first = VideoGenerator(str(workspace.path), on_progress=_record_progress(workspace))
    first.api = SubmitThenTimeoutAPI()
    first.semantic_reviewer = None
    first_result = await first._generate_single_shot(
        shot, None, None, storyboard
    )

    assert first_result.status == "running"
    descriptor = RunWorkspace.resume(workspace.path).resumable_pending_tasks()[1]
    assert descriptor["prompt_profile"] == "policy_safe"

    class PollOnlyAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["task_id"] == "policy-pending-task"
            return {
                "status": "failed",
                "error_type": "poll_timeout",
                "provider_task_id": "policy-pending-task",
                "error": "still pending",
            }

    second = VideoGenerator(
        str(workspace.path),
        resume_tasks=RunWorkspace.resume(workspace.path).resumable_pending_tasks(),
    )
    second.api = PollOnlyAPI()
    second.semantic_reviewer = None
    resumed = await second._generate_single_shot(shot, None, None, storyboard)

    assert resumed.status == "running"
    assert len(second.api.calls) == 1
    assert second.api.calls[0]["task_id"] == "policy-pending-task"


@pytest.mark.asyncio
async def test_tampered_pending_descriptor_stops_before_poll_or_submit(generator):
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral legacy shot."}
    descriptor = {
        "task_id": "tampered-pending-task",
        "prompt_profile": "normal",
        "prompt_fingerprint": "f" * 64,
        "compiled_contract_version": "action-contract-v2",
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }

    class NoAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            pytest.fail("tampered descriptor must not poll or submit")

    generator.api = NoAPI()
    generator.resume_tasks = {1: descriptor}
    result = await generator._generate_single_shot(
        shot, None, None, {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    )

    assert result.status == "failed"
    assert generator.api.calls == 0
    assert "提交描述符" in result.errors[-1]


@pytest.mark.asyncio
async def test_unbound_accepted_take_stops_without_paid_generation(generator):
    accepted = generator.output_dir / "shots" / "shot_001.mp4"
    accepted.write_bytes(b"accepted")

    class NoGenerationAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            pytest.fail("unbound accepted take must not authorize a generation")

    generator.api = NoGenerationAPI()
    generator.semantic_reviewer = None
    generator.accepted_shot_artifacts = {
        1: {
            "local_path": str(accepted),
            "semantic_accepted": True,
        },
    }
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A neutral legacy shot.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == 0
    assert "编译合同、有效提示词" in result.errors[-1]


@pytest.mark.asyncio
async def test_offline_revalidated_take_is_reused_on_next_resume_with_tail(
    generator, monkeypatch
):
    accepted = generator.output_dir / "shots" / "shot_001.mp4"
    accepted.write_bytes(b"accepted")
    (generator.output_dir / "shots" / "shot_001_generation.json").write_text(
        json.dumps({
            "image_role": None,
            "reference_fingerprints": [],
        }),
        encoding="utf-8",
    )

    class NoGenerationAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            pytest.fail("local revalidation must not submit a video task")

    class AcceptingReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, *_args, **_kwargs):
            self.calls += 1
            return SemanticReview(
                accepted=True,
                required_entities_visible={},
                action_geometry_valid=True,
                primary_action_completed=True,
                observed_end_state={"action_phase": "accepted"},
            )

        def accepted_identity_crop_boxes(self, _path):
            return {}

    def persist_tail(shot_id, _path):
        tail = generator.output_dir / "shots" / f"shot_{shot_id:03d}_lastframe.jpg"
        tail.write_bytes(b"tail")
        return str(tail)

    generator.api = NoGenerationAPI()
    reviewer = AcceptingReviewer()
    generator.semantic_reviewer = reviewer
    generator.accepted_shot_artifacts = {
        1: {"local_path": str(accepted), "semantic_accepted": True},
    }
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(generator, "_extract_local_tail_frame", persist_tail)
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral legacy shot."}
    storyboard = {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}

    first = await generator._generate_single_shot(shot, None, None, storyboard)

    assert first.status == "success"
    assert reviewer.calls == 1
    assert generator.api.calls == 0
    assert Path(first.last_frame_url).is_file()

    second_generator = VideoGenerator(str(generator.output_dir))
    second_generator.api = NoGenerationAPI()
    second_reviewer = AcceptingReviewer()
    second_generator.semantic_reviewer = second_reviewer
    second_generator.accepted_shot_artifacts = {
        1: {
            "local_path": first.local_path,
            "last_frame_url": first.last_frame_url,
            "semantic_accepted": first.semantic_accepted,
            "accepted_contract_version": first.accepted_contract_version,
            "accepted_contract_fingerprint": first.accepted_contract_fingerprint,
            "semantic_evaluator_version": first.semantic_evaluator_version,
            "acceptance_policy": first.acceptance_policy,
        },
    }

    second = await second_generator._generate_single_shot(shot, None, None, storyboard)

    assert second.status == "success"
    assert second_reviewer.calls == 0
    assert second_generator.api.calls == 0


@pytest.mark.asyncio
async def test_accepted_resume_restores_cached_identity_crops_without_review(
    generator,
    monkeypatch,
):
    accepted = generator.output_dir / "shots" / "shot_001.mp4"
    accepted.write_bytes(b"accepted-take")
    crop_boxes = {"combat_cleaner_robot": (0.1, 0.2, 0.8, 0.9)}
    registered = []

    class CachedCropReviewer:
        def __init__(self):
            self.cache_reads = []

        def accepted_identity_crop_boxes(self, video_path):
            self.cache_reads.append(video_path)
            return crop_boxes

        def review(self, *_args, **_kwargs):
            pytest.fail("accepted resume must not re-review its local video")

    reviewer = CachedCropReviewer()
    generator.semantic_reviewer = reviewer
    generator.accepted_shot_artifacts = {
        1: {"local_path": str(accepted), "semantic_accepted": True}
    }
    monkeypatch.setattr(
        generator,
        "_extract_local_tail_frame",
        lambda *_args: "accepted-tail.jpg",
    )
    monkeypatch.setattr(
        generator,
        "_register_identity_crops",
        lambda **kwargs: registered.append(kwargs),
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "The cleaner clears the marked area.",
        "characters": ["combat_cleaner_robot"],
    }
    (generator.output_dir / "shots" / "shot_001_generation.json").write_text(
        json.dumps({
            "image_role": None,
            "reference_fingerprints": [],
            "acceptance_context": generator._acceptance_context(shot, [], None),
        }),
        encoding="utf-8",
    )
    generator.accepted_shot_artifacts[1].update({
        "accepted_contract_version": "action-contract-v2",
        "accepted_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
        "semantic_evaluator_version": SEMANTIC_REVIEW_VERSION,
        "acceptance_policy": "semantic_reviewed",
    })

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "success"
    assert reviewer.cache_reads == [str(accepted)]
    assert registered == [{
        "shot_id": 1,
        "video_path": str(accepted),
        "crop_boxes": crop_boxes,
    }]


@pytest.mark.asyncio
async def test_observed_end_state_does_not_replace_planned_contract(
    generator, monkeypatch
):
    planned_end = {"prop_state": "planned endpoint"}
    observed_end = {"prop_state": "observed endpoint"}
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A physical process reaches its endpoint.",
        "end_state": planned_end.copy(),
    }

    async def accepted_result(*_args, **_kwargs):
        return ShotResult(
            shot_id=1,
            status="success",
            local_path="shot.mp4",
            last_frame_url="tail.jpg",
            semantic_accepted=True,
            observed_end_state=observed_end,
        )

    monkeypatch.setattr(generator, "_generate_single_shot", accepted_result)
    monkeypatch.setattr(
        "pipeline.generator.ensure_storyboard_ready", lambda _value: None
    )

    await generator.generate_all({"shots": [shot]})

    assert shot["end_state"] == planned_end
    assert shot["observed_end_state"] == observed_end


class TestInjectCharacterDescription:
    def test_injects_description(self, generator):
        storyboard = {
            "characters": [
                {"name": "hero", "description": "tall muscular warrior with golden armor"}
            ]
        }
        shot = {"characters": ["hero"]}
        result = generator._inject_character_description("A warrior fights.", shot, storyboard)
        assert "tall muscular warrior with golden armor" in result
        assert "A warrior fights." in result

    def test_no_characters_in_shot(self, generator):
        storyboard = {
            "characters": [
                {"name": "hero", "description": "tall warrior"}
            ]
        }
        shot = {"characters": []}
        result = generator._inject_character_description("A scene.", shot, storyboard)
        assert result == "A scene."


class TestAcceptedTailFrame:
    def test_local_tail_frame_uses_actual_downloaded_duration(
        self, generator, monkeypatch
    ):
        calls = {}
        monkeypatch.setattr(
            "tools.ffmpeg_ops.get_video_duration", lambda path: 4.75
        )

        def fake_extract(video_path, output_path, timestamp=None):
            calls.update({
                "video_path": video_path,
                "output_path": output_path,
                "timestamp": timestamp,
            })
            return output_path

        monkeypatch.setattr("pipeline.generator.extract_frame", fake_extract)

        result = generator._extract_local_tail_frame(3, "accepted.mp4")

        assert result.endswith("shots/shot_003_lastframe.jpg")
        assert calls["video_path"] == "accepted.mp4"
        assert calls["timestamp"] == pytest.approx(4.65)

    @pytest.mark.asyncio
    async def test_identity_reference_uses_semantically_reviewed_midpoint(
        self, generator, monkeypatch
    ):
        video_path = str(Path(generator.output_dir) / "shot.mp4")
        Path(video_path).write_bytes(b"video")
        calls = {}
        monkeypatch.setattr(
            "tools.ffmpeg_ops.get_video_duration", lambda _path: 5.2
        )

        def fake_extract(_video_path, output_path, timestamp=None):
            calls["timestamp"] = timestamp
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"identity-frame")
            return output_path

        monkeypatch.setattr("pipeline.generator.extract_frame", fake_extract)

        await generator._extract_character_ref(
            {"shot_id": 1, "characters": ["hero"]},
            video_path,
            {
                "characters": [{
                    "name": "hero",
                    "reference_mode": "identity",
                }]
            },
        )

        assert calls["timestamp"] == pytest.approx(2.6)
        assert generator.character_refs["hero"].endswith("hero.jpg")

    def test_reviewed_midpoint_crop_becomes_project_identity_ref(
        self, generator, monkeypatch
    ):
        import cv2
        import numpy as np

        video_path = str(Path(generator.output_dir) / "shot.mp4")
        Path(video_path).write_bytes(b"video")
        monkeypatch.setattr(
            "tools.ffmpeg_ops.get_video_duration", lambda _path: 5.0
        )

        def fake_extract(_video_path, output_path, timestamp=None):
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[:, :100] = (220, 220, 220)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            return output_path

        monkeypatch.setattr("pipeline.generator.extract_frame", fake_extract)

        generator._register_identity_crops(
            shot_id=1,
            video_path=video_path,
            crop_boxes={"hero": (0.0, 0.0, 0.5, 1.0)},
        )

        ref_path = Path(generator.character_refs["hero"])
        crop = cv2.imread(str(ref_path))
        assert ref_path.is_file()
        assert crop.shape[0] == 100
        assert 95 <= crop.shape[1] <= 115

    def test_invalid_reviewed_crop_does_not_create_identity_ref(
        self, generator, monkeypatch
    ):
        import cv2
        import numpy as np

        video_path = str(Path(generator.output_dir) / "shot.mp4")
        Path(video_path).write_bytes(b"video")
        monkeypatch.setattr(
            "tools.ffmpeg_ops.get_video_duration", lambda _path: 5.0
        )

        def fake_extract(_video_path, output_path, timestamp=None):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, np.zeros((100, 200, 3), dtype=np.uint8))
            return output_path

        monkeypatch.setattr("pipeline.generator.extract_frame", fake_extract)

        generator._register_identity_crops(
            shot_id=1,
            video_path=video_path,
            crop_boxes={"hero": (0.0, 0.0, 0.05, 0.05)},
        )

        assert generator.character_refs == {}

    @pytest.mark.asyncio
    async def test_review_receives_previous_state_and_identity_crop_candidates(
        self, generator
    ):
        captured = {}

        class CapturingReviewer:
            def review(self, _video_path, _shot, **context):
                captured.update(context)
                return SemanticReview(
                    accepted=True,
                    required_entities_visible={"hero": True},
                    action_geometry_valid=True,
                    primary_action_completed=True,
                    observed_end_state={
                        "location": "street",
                        "subject": "hero",
                        "action_phase": "running",
                    },
                )

        generator.semantic_reviewer = CapturingReviewer()
        previous = {
            "scene_id": "street",
            "observed_end_state": {
                "location": "street corner",
                "subject": "hero facing right",
                "action_phase": "mid-stride",
            },
        }
        storyboard = {
            "characters": [
                {"name": "hero", "reference_mode": "identity"},
                {"name": "crowd", "reference_mode": "group"},
            ]
        }

        await generator._review_take(
            "shot.mp4",
            {
                "shot_id": 2,
                "scene_id": "street",
                "characters": ["hero", "crowd"],
            },
            previous_frame_path="tail.jpg",
            previous_shot=previous,
            storyboard=storyboard,
        )

        assert captured["boundary_context"] == {
            "same_scene": True,
            "previous_scene_id": "street",
            "state_reference_role": None,
            "previous_observed_end_state": previous["observed_end_state"],
        }
        assert captured["identity_crop_entities"] == ["hero"]

    def test_no_characters_in_storyboard(self, generator):
        shot = {"characters": ["hero"]}
        result = generator._inject_character_description("A scene.", shot, {})
        assert result == "A scene."

    def test_multiple_characters(self, generator):
        storyboard = {
            "characters": [
                {"name": "hero", "description": "tall warrior"},
                {"name": "villain", "description": "dark mage"},
            ]
        }
        shot = {"characters": ["hero", "villain"]}
        result = generator._inject_character_description("Battle.", shot, storyboard)
        assert "tall warrior" in result
        assert "dark mage" in result


@pytest.mark.asyncio
async def test_semantic_retake_is_bounded_across_resume(generator, monkeypatch):
    class RejectingReviewer:
        def review(self, _video_path, _shot, **_context):
            return SemanticReview(
                accepted=False,
                required_entities_visible={"robot": True, "zombies": False},
                action_geometry_valid=False,
                primary_action_completed=False,
                observed_end_state={},
                failure_reason="target is not visible in the line of fire",
            )

    class SuccessfulAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            return {"status": "succeeded", "video_url": f"take-{self.calls}"}

        async def download_video(self, video_url, save_path):
            Path(save_path).write_bytes(video_url.encode())

    generator.api = SuccessfulAPI()
    generator.semantic_reviewer = RejectingReviewer()
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "scene_description": "firefight",
        "prompt_en": "robot fires at zombies",
        "primary_action": "robot fires at zombies",
    }
    storyboard = {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}

    first = await generator._generate_single_shot(shot, None, None, storyboard)
    second = await generator._generate_single_shot(shot, None, None, storyboard)

    assert first.status == "failed"
    assert generator.api.calls == 2
    assert len(list((generator.output_dir / "shots").glob("*_rejected_*.mp4"))) == 2
    assert second.status == "failed"
    assert "已达到上限" in second.errors[-1]


@pytest.mark.asyncio
async def test_rejected_cached_take_submits_new_provider_task(generator, monkeypatch):
    cached = generator.output_dir / "shots" / "shot_001.mp4"
    cached.write_bytes(b"rejected-cached-take")
    generator.resume_task_ids = {1: "rejected-provider-task"}

    class RejectThenAcceptReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, _video_path, _shot, **_context):
            self.calls += 1
            accepted = self.calls == 2
            return SemanticReview(
                accepted=accepted,
                required_entities_visible={"robot": True},
                action_geometry_valid=accepted,
                primary_action_completed=accepted,
                observed_end_state={"action_phase": "complete"} if accepted else {},
                failure_reason="target effect appears during setup" if not accepted else "",
            )

    class CapturingAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "succeeded",
                "provider_task_id": "new-provider-task",
                "video_url": "new-take",
            }

        async def download_video(self, video_url, save_path):
            Path(save_path).write_bytes(video_url.encode())

    generator.api = CapturingAPI()
    generator.semantic_reviewer = RejectThenAcceptReviewer()
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A robot holds position while its sensor activates.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "success"
    assert generator.api.calls[0]["task_id"] is None
    assert result.provider_task_id == "new-provider-task"
    assert result.prompt_attempts[-1]["provider_task_id"] == "new-provider-task"


@pytest.mark.asyncio
async def test_rejected_downloaded_pending_take_consumes_descriptor_before_retake(
    generator, monkeypatch
):
    """A rejected local result turns its pending remote task into a new retake."""
    cached = generator.output_dir / "shots" / "shot_001.mp4"
    cached.write_bytes(b"downloaded-pending-take")
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A machine pauses before a marked obstacle.",
    }
    fingerprint = "b" * 64
    descriptor = {
        "task_id": "completed-but-rejected-task",
        "prompt_profile": "normal",
        "prompt_fingerprint": fingerprint,
        "compiled_contract_version": "action-contract-v2",
        "compiled_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
    }
    generator._write_generation_provenance(
        1,
        generator._generation_provenance(
            shot, [], None, prompt_profile="normal", prompt_fingerprint=fingerprint
        ),
        descriptor["task_id"],
    )

    class RejectThenAcceptReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, _video_path, _shot, **_context):
            self.calls += 1
            accepted = self.calls == 2
            return SemanticReview(
                accepted=accepted,
                required_entities_visible={},
                action_geometry_valid=accepted,
                primary_action_completed=accepted,
                observed_end_state={},
                failure_reason="endpoint is not visibly established" if not accepted else "",
            )

    class RetakeOnlyAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "succeeded",
                "provider_task_id": "retake-task",
                "video_url": "retake-video",
            }

        async def download_video(self, _video_url, save_path):
            Path(save_path).write_bytes(b"retake-video")

    generator.resume_tasks = {1: descriptor}
    generator.api = RetakeOnlyAPI()
    generator.semantic_reviewer = RejectThenAcceptReviewer()
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )

    result = await generator._generate_single_shot(
        shot, None, None, {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    )

    assert result.status == "success"
    assert len(generator.api.calls) == 1
    assert generator.api.calls[0]["task_id"] is None
    assert generator.api.calls[0]["prompt"] != "Resume the already submitted provider task."
    assert "endpoint is not visibly established" in generator.api.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_offline_revalidation_persists_latest_acceptance_context_for_resume(
    generator, monkeypatch
):
    """An accepted re-review binds to the current boundary without rewriting lineage."""
    old_tail = generator.output_dir / "old_tail.jpg"
    new_tail = generator.output_dir / "new_tail.jpg"
    accepted_tail = generator.output_dir / "accepted_tail.jpg"
    old_tail.write_bytes(b"old boundary")
    new_tail.write_bytes(b"new boundary")
    accepted_tail.write_bytes(b"accepted boundary")
    video = generator.output_dir / "shots" / "shot_002.mp4"
    video.write_bytes(b"accepted local take")
    shot = {
        "shot_id": 2,
        "duration": 5,
        "scene_id": "same_scene",
        "prompt_en": "Continue from the observed starting state.",
        "continuity_from_previous": "seamless",
    }
    storyboard = {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    fingerprint = "c" * 64
    generator._write_generation_provenance(
        2,
        generator._generation_provenance(
            shot,
            [str(old_tail)],
            "first_frame",
            prompt_profile="normal",
            prompt_fingerprint=fingerprint,
        ),
        "original-task",
    )
    artifact = {
        "local_path": str(video),
        "last_frame_url": str(accepted_tail),
        "semantic_accepted": True,
        "accepted_contract_version": "action-contract-v2",
        "accepted_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
        "semantic_evaluator_version": SEMANTIC_REVIEW_VERSION,
        "acceptance_policy": "semantic_reviewed",
    }

    class AcceptCurrentBoundaryReviewer:
        calls = 0

        def review(self, _video_path, _shot, **_context):
            self.calls += 1
            return SemanticReview(
                accepted=True,
                required_entities_visible={},
                action_geometry_valid=True,
                primary_action_completed=True,
                observed_end_state={"boundary": "current"},
            )

    reviewer = AcceptCurrentBoundaryReviewer()
    generator.accepted_shot_artifacts = {2: artifact}
    generator.semantic_reviewer = reviewer
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: str(accepted_tail)
    )

    revalidated = await generator._generate_single_shot(
        shot,
        str(new_tail),
        {"shot_id": 1, "scene_id": "same_scene"},
        storyboard,
    )

    stored = json.loads(
        (generator.output_dir / "shots" / "shot_002_generation.json").read_text()
    )
    assert reviewer.calls == 1
    assert stored["reference_fingerprints"] == [
        generator._reference_fingerprint(str(old_tail))
    ]
    assert stored["acceptance_context"]["reference_fingerprints"] == [
        generator._reference_fingerprint(str(new_tail))
    ]

    class MustNotReview:
        def review(self, *_args, **_kwargs):
            pytest.fail("latest acceptance context must be reusable without review")

        def accepted_identity_crop_boxes(self, _video_path):
            return []

    class MustNotGenerate:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            pytest.fail("latest accepted take must not create or poll a task")

    resumed = VideoGenerator(
        str(generator.output_dir),
        accepted_shot_artifacts={
            2: {
                **artifact,
                "last_frame_url": revalidated.last_frame_url,
                "semantic_accepted": revalidated.semantic_accepted,
                "observed_end_state": revalidated.observed_end_state,
                "accepted_contract_version": revalidated.accepted_contract_version,
                "accepted_contract_fingerprint": revalidated.accepted_contract_fingerprint,
                "semantic_evaluator_version": revalidated.semantic_evaluator_version,
            }
        },
    )
    resumed.semantic_reviewer = MustNotReview()
    resumed.api = MustNotGenerate()

    restored = await resumed._generate_single_shot(
        shot,
        str(new_tail),
        {"shot_id": 1, "scene_id": "same_scene"},
        storyboard,
    )

    assert restored.status == "success"


@pytest.mark.asyncio
async def test_technical_only_take_is_reusable_without_claiming_semantic_acceptance(
    generator, monkeypatch
):
    """Disabled semantic review has a distinct, durable acceptance policy."""
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A neutral test shot."}
    storyboard = {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]}
    tail = generator.output_dir / "shots" / "shot_001_lastframe.jpg"
    tail.write_bytes(b"tail")

    class SuccessfulAPI:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            return {"status": "succeeded", "video_url": "technical-only"}

        async def download_video(self, _url, path):
            Path(path).write_bytes(b"video")

    generator.api = SuccessfulAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: str(tail)
    )

    first = await generator._generate_single_shot(shot, None, None, storyboard)

    assert first.status == "success"
    assert first.semantic_accepted is None
    assert first.acceptance_policy == "technical_only"
    assert first.semantic_evaluator_version is None

    class MustNotGenerate:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            pytest.fail("technical-only accepted take must not submit or poll")

    resumed = VideoGenerator(
        str(generator.output_dir),
        accepted_shot_artifacts={
            1: {
                "local_path": first.local_path,
                "last_frame_url": first.last_frame_url,
                "semantic_accepted": first.semantic_accepted,
                "accepted_contract_version": first.accepted_contract_version,
                "accepted_contract_fingerprint": first.accepted_contract_fingerprint,
                "semantic_evaluator_version": first.semantic_evaluator_version,
                "acceptance_policy": first.acceptance_policy,
            }
        },
    )
    resumed.api = MustNotGenerate()
    resumed.semantic_reviewer = None

    restored = await resumed._generate_single_shot(shot, None, None, storyboard)

    assert restored.status == "success"
    assert restored.semantic_accepted is None
    assert restored.acceptance_policy == "technical_only"


def test_local_cache_acceptance_context_is_persisted_without_generation_lineage(
    generator,
):
    video = generator.output_dir / "shots" / "shot_001.mp4"
    video.write_bytes(b"local-cache")
    shot = {
        "shot_id": 1,
        "interaction_geometry": {"effect_phase": "none"},
    }

    generator._write_acceptance_context(
        1,
        str(video),
        shot,
        [],
        None,
        policy="technical_only",
    )

    sidecar = video.with_name("shot_001_generation.json")
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["version"] == "acceptance-only-v1"
    assert stored["acceptance_context"]["policy"] == "technical_only"
    assert generator._restored_provenance_matches(
        {
            "acceptance_policy": "technical_only",
            "accepted_contract_version": "action-contract-v2",
            "accepted_contract_fingerprint": generator._compiled_contract_fingerprint(shot),
        },
        str(video),
        shot,
        [],
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_task_id", "expected_api_task_id"),
    [
        ("rejected-provider-task", None),
        ("already-submitted-retake-task", "already-submitted-retake-task"),
    ],
)
async def test_duplicate_rejected_files_consume_one_take_budget(
    generator, monkeypatch, resume_task_id, expected_api_task_id
):
    shots_dir = generator.output_dir / "shots"
    first = shots_dir / "shot_001_rejected_1.mp4"
    duplicate = shots_dir / "shot_001_rejected_2.mp4"
    first.write_bytes(b"same-provider-take")
    duplicate.write_bytes(b"same-provider-take")
    (shots_dir / "shot_001_rejected_1_generation.json").write_text(
        json.dumps({"provider_task_id": "rejected-provider-task"}),
        encoding="utf-8",
    )
    (shots_dir / "shot_001_rejected_2_generation.json").write_text(
        json.dumps({"provider_task_id": None}),
        encoding="utf-8",
    )
    generator.resume_task_ids = {1: resume_task_id}

    class RejectHistoryAcceptRetakeReviewer:
        def review(self, video_path, _shot, **_context):
            accepted = "_rejected_" not in video_path
            return SemanticReview(
                accepted=accepted,
                required_entities_visible={"robot": True},
                action_geometry_valid=accepted,
                primary_action_completed=accepted,
                observed_end_state={"action_phase": "complete"} if accepted else {},
                failure_reason="physical result begins before the active phase"
                if not accepted else "",
            )

    class OneRetakeAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "status": "succeeded",
                "provider_task_id": "actual-retake-task",
                "video_url": "actual-retake",
            }

        async def download_video(self, video_url, save_path):
            Path(save_path).write_bytes(video_url.encode())

    generator.api = OneRetakeAPI()
    generator.semantic_reviewer = RejectHistoryAcceptRetakeReviewer()
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A robot holds position while its sensor activates.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    if expected_api_task_id:
        assert result.status == "failed"
        assert generator.api.calls == []
        assert "待恢复远端任务" in result.errors[-1]
    else:
        assert result.status == "success"
        assert len(generator.api.calls) == 1
        assert generator.api.calls[0]["task_id"] is None
        assert "physical result begins before the active phase" in generator.api.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_provider_success_result_restores_provider_identity(generator, monkeypatch):
    class ProviderIdentityAPI:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            return {
                "status": "succeeded",
                "provider_task_id": "provider-task-from-result",
                "video_url": "take",
            }

        async def download_video(self, video_url, save_path):
            Path(save_path).write_bytes(video_url.encode())

    generator.api = ProviderIdentityAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {"shot_id": 1, "duration": 5, "prompt_en": "A quiet city street."}

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    provenance = json.loads(
        (generator.output_dir / "shots" / "shot_001_generation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.status == "success"
    assert result.provider_task_id == "provider-task-from-result"
    assert result.prompt_attempts[-1]["provider_task_id"] == "provider-task-from-result"
    assert provenance["provider_task_id"] == "provider-task-from-result"


@pytest.mark.asyncio
async def test_privacy_rejection_without_references_stops_without_duplicate_requests(
    generator,
):
    class PrivacyRejectingAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            return {
                "status": "failed",
                "error_type": "privacy",
                "error": "privacy review rejected text-only request",
            }

    generator.api = PrivacyRejectingAPI()
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "a fully mechanical machine crosses an empty street",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == 1
    assert "停止重试" in result.errors[-1]


@pytest.mark.asyncio
async def test_copyright_policy_retries_once_without_changing_reference_duties(
    generator, monkeypatch
):
    class CopyrightThenSuccessAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "status": "failed",
                    "error_type": "copyright_policy",
                    "error_code": (
                        "OutputVideoSensitiveContentDetected.PolicyViolation"
                    ),
                    "error": "The output may be related to copyright restrictions.",
                }
            return {"status": "succeeded", "video_url": "accepted-take"}

        async def download_video(self, _video_url, save_path):
            Path(save_path).write_bytes(b"accepted")

    generator.api = CopyrightThenSuccessAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "Two original fictional travelers cross a mountain pass.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "success"
    assert len(generator.api.calls) == 2
    assert generator.api.calls[0]["image_urls"] is None
    assert generator.api.calls[1]["image_urls"] is None
    assert "Copyright boundary clarification" in generator.api.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_repeated_copyright_policy_rejection_stops_after_one_safe_retry(
    generator,
):
    class CopyrightRejectingAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            return {
                "status": "failed",
                "error_type": "copyright_policy",
                "error_code": "OutputVideoSensitiveContentDetected.PolicyViolation",
                "error": "The output may be related to copyright restrictions.",
            }

    generator.api = CopyrightRejectingAPI()
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "An original abstract kinetic sculpture rotates.",
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == 2
    assert "版权策略" in result.errors[-1]
    assert "隐私" not in result.errors[-1]


@pytest.mark.asyncio
async def test_input_text_moderation_recompiles_once_without_changing_contract(
    generator, monkeypatch
):
    class InputModerationThenSuccessAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "status": "failed",
                    "error_type": "moderation",
                    "error_locus": "input_text",
                    "error_code": "InputTextSensitiveContentDetected",
                    "error": "The input text may contain sensitive information.",
                }
            return {"status": "succeeded", "video_url": "accepted-take"}

        async def download_video(self, _video_url, save_path):
            Path(save_path).write_bytes(b"accepted")

    generator.api = InputModerationThenSuccessAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "tail.jpg"
    )
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "RAW_FREEFORM_PROMPT rejected by the provider.",
        "primary_action": "RAW_PRIMARY_ACTION",
        "required_visible_entities": ["combat_robot", "zombie_horde"],
        "camera": {
            "start_framing": "wide shot",
            "primary_movement": "fixed",
            "screen_positions": {
                "combat_robot": "left foreground",
                "zombie_horde": "right background",
            },
        },
        "action_beats": [{
            "phase": "peak",
            "actor": "combat_robot",
            "action": "RAW_BEAT_ACTION",
            "target": "zombie_horde",
            "visible_result": "RAW_VISIBLE_RESULT",
        }],
        "interaction_geometry": {
            "actor": "combat_robot",
            "target": "zombie_horde",
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
            "source": "RAW_EFFECT_SOURCE",
            "effect_region": "RAW_EFFECT_REGION",
            "reaction_scope": "RAW_REACTION_SCOPE",
            "unaffected_behavior": "RAW_UNAFFECTED_BEHAVIOR",
            "must_share_frame": True,
            "line_of_action_visible": True,
        },
        "narrative_beat": {
            "function": "setup",
            "state_before": "RAW_NARRATIVE_BEFORE",
            "state_change": "RAW_NARRATIVE_CHANGE",
            "state_after": "RAW_NARRATIVE_AFTER",
        },
        "negative_prompt": "RAW_NEGATIVE_PROMPT",
    }
    storyboard = {
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "style": "cinematic",
        "characters": [{
            "name": "combat_robot",
            "description": "brushed steel shell with an amber optical sensor",
        }],
        "shots": [shot],
    }

    result = await generator._generate_single_shot(
        shot, None, None, storyboard
    )

    assert result.status == "success"
    assert len(generator.api.calls) == 2
    first_prompt = generator.api.calls[0]["prompt"]
    safe_prompt = generator.api.calls[1]["prompt"]
    assert "RAW_PRIMARY_ACTION" in first_prompt
    assert "RAW_BEAT_ACTION" in first_prompt
    assert "RAW_FREEFORM_PROMPT" not in first_prompt
    assert "RAW_VISIBLE_RESULT" in first_prompt
    assert "RAW_NARRATIVE_BEFORE" not in first_prompt
    assert "RAW_NARRATIVE_CHANGE" not in first_prompt
    assert "RAW_NARRATIVE_AFTER" not in first_prompt
    assert "RAW_NEGATIVE_PROMPT" not in first_prompt
    assert "brushed steel shell with an amber optical sensor" in first_prompt
    assert "directed_path" in safe_prompt
    assert "combat robot, zombie horde" in safe_prompt
    for raw_marker in (
        "RAW_FREEFORM_PROMPT",
        "RAW_PRIMARY_ACTION",
        "RAW_BEAT_ACTION",
        "RAW_EFFECT_SOURCE",
        "RAW_EFFECT_REGION",
        "RAW_REACTION_SCOPE",
        "RAW_UNAFFECTED_BEHAVIOR",
        "RAW_NARRATIVE_BEFORE",
        "RAW_NARRATIVE_CHANGE",
        "RAW_NARRATIVE_AFTER",
        "RAW_NEGATIVE_PROMPT",
        "RAW_CHARACTER_DESCRIPTION",
    ):
        assert raw_marker not in safe_prompt
    assert "brushed steel shell with an amber optical sensor" in safe_prompt
    assert "RAW_VISIBLE_RESULT" in safe_prompt
    assert generator.api.calls[0]["image_urls"] == generator.api.calls[1]["image_urls"]
    assert result.prompt_profile == "policy_safe"
    assert result.prompt_fingerprint
    assert result.recovery_actions == ["recompile_input_text_policy_safe"]
    assert [attempt["profile"] for attempt in result.prompt_attempts] == [
        "normal",
        "policy_safe",
    ]
    assert [attempt["outcome"] for attempt in result.prompt_attempts] == [
        "failed",
        "succeeded",
    ]
    assert len({attempt["fingerprint"] for attempt in result.prompt_attempts}) == 2


@pytest.mark.asyncio
async def test_repeated_input_text_moderation_stops_after_one_recompile(generator):
    class InputModerationAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = 0

        async def generate(self, **_kwargs):
            self.calls += 1
            return {
                "status": "failed",
                "error_type": "moderation",
                "error_locus": "input_text",
                "error_code": "InputTextSensitiveContentDetected",
                "error": "The input text may contain sensitive information.",
            }

    generator.api = InputModerationAPI()
    shot = {
        "shot_id": 1,
        "duration": 5,
        "prompt_en": "A fictional scene.",
        "primary_action": "subject crosses the room",
        "required_visible_entities": ["subject"],
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == 2
    assert "已停止重试" in result.errors[-1]
    assert result.provider_error_locus == "input_text"
    assert result.recovery_actions == ["recompile_input_text_policy_safe"]
    assert len(result.prompt_attempts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy_response", [
    {
        "status": "failed",
        "error_type": "privacy",
        "error": "privacy review rejected first frame",
    },
    {
        "status": "failed",
        "error": "privacy information rejected first frame",
    },
])
async def test_privacy_rejection_of_state_only_first_frame_does_not_repeat_request(
    generator, monkeypatch, privacy_response
):
    previous_tail = generator.output_dir / "shots" / "shot_001_lastframe.jpg"
    previous_tail.write_bytes(b"accepted-tail")

    class PrivacyRejectingAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append((kwargs.get("image_urls"), kwargs.get("image_role")))
            return privacy_response

    async def no_sleep(_seconds):
        return None

    generator.api = PrivacyRejectingAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr("pipeline.generator.asyncio.sleep", no_sleep)
    shot = {
        "shot_id": 2,
        "duration": 5,
        "scene_id": "mountain_peak",
        "scene_description": "same mountain peak",
        "prompt_en": "Continue the accepted duel state.",
        "continuity_from_previous": "intentional_cut",
        "composition_change": "medium",
        "characters": ["white_cultivator", "blue_cultivator"],
    }

    result = await generator._generate_single_shot(
        shot,
        str(previous_tail),
        {"shot_id": 1, "scene_id": "mountain_peak"},
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == [([str(previous_tail)], "first_frame")]
    assert "没有可移除的身份参考" in result.errors[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy_response", [
    {
        "status": "failed",
        "error_type": "privacy",
        "error": "privacy review rejected reference",
    },
    {
        "status": "failed",
        "error": "privacy information rejected reference",
    },
])
async def test_privacy_rejection_retries_once_when_identity_refs_are_removed(
    generator, monkeypatch, privacy_response
):
    identity_ref = generator.output_dir / "character_refs" / "hero.jpg"
    identity_ref.write_bytes(b"identity")
    generator.character_refs["hero"] = str(identity_ref)

    class PrivacyRejectingAPI:
        supports_last_frame = True

        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append((kwargs.get("image_urls"), kwargs.get("image_role")))
            return privacy_response

    async def no_sleep(_seconds):
        return None

    generator.api = PrivacyRejectingAPI()
    generator.semantic_reviewer = None
    monkeypatch.setattr("pipeline.generator.asyncio.sleep", no_sleep)
    shot = {
        "shot_id": 1,
        "duration": 5,
        "scene_description": "new location",
        "prompt_en": "The hero enters a new location.",
        "characters": ["hero"],
    }

    result = await generator._generate_single_shot(
        shot,
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": [shot]},
    )

    assert result.status == "failed"
    assert generator.api.calls == [([str(identity_ref)], "reference_image"), (None, None)]
    assert "纯文本请求仍被隐私审核拒绝" in result.errors[-1]


@pytest.mark.asyncio
async def test_reviewer_upgrade_can_promote_latest_rejected_take_without_generation(
    generator, monkeypatch
):
    shots_dir = generator.output_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    (shots_dir / "shot_001_rejected_1.mp4").write_bytes(b"old-take")
    latest = shots_dir / "shot_001_rejected_2.mp4"
    latest.write_bytes(b"latest-take")

    class NoGenerationAPI:
        supports_last_frame = True

        async def generate(self, **_kwargs):
            pytest.fail("re-reviewing a local take must not call Seedance")

    class AcceptingReviewer:
        def review(self, _video_path, _shot, **_context):
            return SemanticReview(
                accepted=True,
                required_entities_visible={"combat_robot": True, "zombies": True},
                action_geometry_valid=True,
                primary_action_completed=True,
                observed_end_state={
                    "location": "ruined street",
                    "subject": "combat robot facing zombies",
                    "action_phase": "first burst completed",
                },
            )

    generator.api = NoGenerationAPI()
    generator.semantic_reviewer = AcceptingReviewer()
    monkeypatch.setattr(
        "pipeline.generator.check_video_quality",
        lambda _path: {"pass": True, "quality_score": 100, "issues": []},
    )
    monkeypatch.setattr(
        generator, "_extract_local_tail_frame", lambda *_args: "accepted-tail.jpg"
    )

    result = await generator._generate_single_shot(
        {
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "combat robot fires at zombies",
            "required_visible_entities": ["combat_robot", "zombies"],
        },
        None,
        None,
        {"resolution": "480p", "aspect_ratio": "16:9", "shots": []},
    )

    canonical = shots_dir / "shot_001.mp4"
    assert result.status == "success"
    assert result.local_path == str(canonical)
    assert canonical.read_bytes() == b"latest-take"
    assert not latest.exists()

class TestContinuityContract:
    def test_generation_boundary_normalizes_legacy_same_scene_shots(self, generator):
        shots = [
            {
                "shot_id": 1,
                "scene_id": "cafe",
                "scene_description": "【咖啡店】建立镜头",
                "camera": {"start_framing": "wide shot", "end_framing": "wide shot"},
            },
            {
                "shot_id": 2,
                "scene_id": "cafe",
                "scene_description": "【咖啡店】杯子特写",
                "camera": {
                    "start_framing": "extreme close-up",
                    "end_framing": "extreme close-up",
                },
            },
        ]

        generator._normalize_continuity(shots)

        assert [shot["continuity_from_previous"] for shot in shots] == [
            "none",
            "intentional_cut",
        ]

    def test_normalized_same_scene_insert_depends_on_previous_tail(self, generator):
        shots = [
            {
                "shot_id": 1,
                "scene_description": "【咖啡店吧台】建立镜头",
                "characters": ["hero"],
            },
            {
                "shot_id": 2,
                "scene_description": "【咖啡店吧台】产品特写",
                "continuity_from_previous": "none",
                "characters": [],
            },
        ]

        generator._normalize_continuity(shots)

        assert generator._find_independent_shots(shots) == set()

    def test_same_scene_id_injects_continuity_when_labels_differ(self, generator):
        previous = {
            "scene_id": "ruined_intersection",
            "scene_description": "【废墟十字路口·地面】",
            "lighting": "orange dusk side light",
            "key_props": ["combat robot"],
            "end_state": {
                "location": "intersection center",
                "subject": "robot facing forward",
                "action_phase": "zombies closing in",
                "camera": "wide shot, pan ending right",
            },
        }
        current = {
            "scene_id": "ruined_intersection",
            "scene_description": "【近距离交火·中近景】",
            "continuity_from_previous": "seamless",
            "key_props": ["combat robot"],
        }

        result = generator._inject_scene_continuity("Robot fires.", current, previous)

        assert "orange dusk side light" in result
        assert "combat robot" in result

    def test_shot_contract_limits_prompt_to_one_action(self, generator):
        shot = {
            "primary_action": "robot fires one controlled burst",
            "start_state": {
                "location": "intersection center",
                "subject": "robot braced",
                "action_phase": "weapon aimed",
                "camera": "locked medium shot",
            },
            "end_state": {
                "location": "intersection center",
                "subject": "robot still braced",
                "action_phase": "burst completed",
                "camera": "locked medium shot",
            },
        }

        result = generator._inject_shot_contract("Robot combat scene.", shot)

        assert "robot fires one controlled burst" in result
        assert "weapon aimed" in result
        assert "burst completed" in result

    def test_shot_contract_compiles_blocking_and_causal_beats(self, generator):
        shot = {
            "primary_action": "robot fires one controlled burst",
            "blocking": {
                "robot": {
                    "frame_position": "left third",
                    "body_orientation": "profile toward frame right",
                    "facing_target": "zombies",
                    "eyeline_target": "zombies",
                    "travel_direction": "holds position",
                    "action_target": "zombies",
                }
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "front rank recoils",
            }],
        }

        result = generator._inject_shot_contract("Robot combat scene.", shot)

        assert "body oriented profile toward frame right" in result
        assert "facing zombies" in result
        assert "action directed at zombies" in result
        assert "peak: robot fires one burst toward zombies" in result

    def test_setup_shot_contract_does_not_reinject_forbidden_target_outcomes(
        self, generator
    ):
        shot = {
            "primary_action": "RAW_PRIMARY_ACTION",
            "interaction_geometry": {
                "actor": "robot",
                "target": "target_group",
                "effect_phase": "setup",
                "interaction_mode": "none",
                "outcome_scope": "none",
                "effect_motion": "none",
                "unaffected_behavior": "TARGETS_KEEP_PRIOR_MOTION",
            },
            "action_beats": [{
                "phase": "trigger",
                "actor": "robot",
                "action": "RAW_BEAT_ACTION",
                "target": "target_group",
                "visible_result": "RAW_TARGET_RESULT",
            }],
            "narrative_beat": {
                "function": "setup",
                "state_before": "RAW_BEFORE",
                "state_change": "RAW_CHANGE",
                "state_after": "RAW_OUTCOME",
            },
            "end_state": {
                "subject": "robot",
                "action_phase": "RAW_ACTION_PHASE",
                "pose_and_gaze": "ROBOT_READY_POSE",
                "prop_state": "RAW_PROP_STATE",
                "open_motion": "RAW_TARGET_MOTION",
            },
        }

        result = generator._inject_shot_contract("A restrained setup shot.", shot)

        assert "robot visibly prepares toward target_group" in result
        assert "ROBOT_READY_POSE" in result
        assert "TARGETS_KEEP_PRIOR_MOTION" in result
        for leaked in (
            "RAW_TARGET_RESULT",
            "RAW_PRIMARY_ACTION",
            "RAW_BEAT_ACTION",
            "RAW_ACTION_PHASE",
            "RAW_PROP_STATE",
            "RAW_CHANGE",
            "RAW_OUTCOME",
            "RAW_TARGET_MOTION",
        ):
            assert leaked not in result

    def test_shot_contract_compiles_generic_causal_scope(self, generator):
        shot = {
            "primary_action": "source projects an effect through the scene",
            "interaction_geometry": {
                "interaction_mode": "directed_path",
                "source": "a visible origin on the source",
                "effect_region": "the narrow path from source to target",
                "reaction_scope": "only entities intersecting that path",
                "unaffected_behavior": "entities outside the path remain unaffected",
            },
        }

        result = generator._inject_shot_contract("Abstract interaction.", shot)

        assert "directed_path cause-and-effect" in result
        assert "a visible origin on the source" in result
        assert "the narrow path from source to target" in result
        assert "only entities intersecting that path" in result
        assert "entities outside the path remain unaffected" in result

    def test_shot_contract_compiles_narrative_state_change(self, generator):
        shot = {
            "primary_action": "the subject reveals the result",
            "narrative_beat": {
                "function": "payoff",
                "state_before": "the outcome is uncertain",
                "state_change": "the result becomes visible",
                "state_after": "the goal is visibly achieved",
            },
        }

        result = generator._inject_shot_contract("A final reveal.", shot)

        assert "narrative payoff" in result
        assert "the outcome is uncertain" in result
        assert "the result becomes visible" in result
        assert "the goal is visibly achieved" in result

    def test_shot_contract_compiles_required_composition_change(self, generator):
        result = generator._inject_shot_contract(
            "A welding action.",
            {
                "composition_change": "medium",
                "camera": {
                    "start_framing": "medium shot",
                    "end_framing": "medium shot",
                },
            },
        )

        assert "use a medium shot with a clearly different shot size or angle" in result
        assert "do not copy the previous framing" in result

    def test_seamless_contract_trusts_real_first_frame_over_planned_state(
        self, generator
    ):
        shot = {
            "continuity_from_previous": "seamless",
            "primary_action": "robot takes one step",
            "start_state": {"subject": "planned pose that may not have happened"},
            "end_state": {"subject": "robot one step forward"},
        }

        result = generator._inject_shot_contract("Robot advances.", shot)

        assert "supplied first frame" in result
        assert "planned pose that may not have happened" not in result

    def test_character_reference_is_scoped_to_identity_not_composition(
        self, generator
    ):
        result = generator._inject_reference_scope(
            "Low-angle robot at the left third.",
            "reference_image",
        )

        assert "identity and appearance only" in result
        assert "ignore its pose, framing, background, and camera angle" in result

    def test_first_frame_keeps_visible_state_authority(self, generator):
        prompt = "Continue the robot action."

        assert generator._inject_reference_scope(prompt, "first_frame") == prompt

    def test_same_scene_cut_assigns_separate_reference_responsibilities(
        self, generator
    ):
        result = generator._inject_reference_scope(
            "Continue the robot action.",
            "reference_image",
            reference_count=2,
            has_state_reference=True,
        )

        assert "Image 1" in result
        assert "observed physical and scene state" in result
        assert "Image 2" in result
        assert "identity and appearance only" in result
        assert "current shot contract controls camera composition" in result


@pytest.mark.asyncio
async def test_extractor_rejects_character_hidden_in_structured_fields(
    generator, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "pipeline.generator.extract_frame",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ]
    }
    shot = {
        "shot_id": 1,
        "characters": ["robot"],
        "extract_character_ref": True,
        "camera": {
            "screen_positions": {
                "robot": "left foreground",
                "zombies": "center background",
            }
        },
    }

    await generator._extract_character_ref(shot, "shot.mp4", storyboard)

    assert calls == []
    assert generator.character_refs == {}

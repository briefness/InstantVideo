"""角色一致性 + 画面衔接逻辑测试"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.generator import (
    RemoteTaskPendingError,
    ShotResult,
    VideoGenerator,
)
from pipeline.semantic_review import SemanticReview


@pytest.fixture
def generator():
    return VideoGenerator(tempfile.mkdtemp())


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

    def test_same_scene_intentional_cut_uses_state_and_identity_refs(self, generator):
        """同场景切镜在同一多模态模式中分离状态与身份职责。"""
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
        assert urls == ["http://a/last.jpg", ref_path]
        assert role == "reference_image"

    def test_large_same_scene_cut_uses_tail_as_state_not_composition(
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

        assert urls == ["http://a/last.jpg", ref_path]
        assert role == "reference_image"

    def test_medium_same_scene_cut_without_identity_ref_still_uses_state_tail(
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
        assert role == "reference_image"

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
        assert role == "reference_image"

    def test_local_prev_frame_supported(self, generator):
        """恢复时提取的本地尾帧可由 API 转成 data URI 继续衔接"""
        local_file = str(Path(__file__).resolve())
        urls, role = generator._build_image_refs({"shot_id": 2}, local_file)
        assert urls == [local_file]
        assert role == "first_frame"


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

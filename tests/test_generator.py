"""角色一致性 + 画面衔接逻辑测试"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.generator import VideoGenerator


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
                "characters": ["hero"],
                "continuity_from_previous": "seamless",
            },
            "http://a/last.jpg",
        )
        assert urls == ["http://a/last.jpg"]
        assert role == "first_frame"

    def test_intentional_cut_uses_only_character_reference(self, generator):
        """有意切镜时角色图负责身份，尾帧不得混入相同 role。"""
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {
                "shot_id": 3,
                "characters": ["hero"],
                "continuity_from_previous": "intentional_cut",
            },
            "http://a/last.jpg",
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

    def test_no_char_intentional_cut_does_not_inherit_previous_frame(self, generator):
        urls, role = generator._build_image_refs(
            {
                "shot_id": 4,
                "characters": [],
                "continuity_from_previous": "intentional_cut",
            },
            "http://a/last.jpg",
        )
        assert urls == []
        assert role is None

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


class TestContinuityContract:
    def test_legacy_same_scene_insert_is_not_prefetched(self, generator):
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

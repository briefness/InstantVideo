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

    def test_char_ref_plus_prev_frame(self, generator):
        """角色参考帧 + 上一帧 → 都用 reference_image"""
        ref_path = str(Path(generator.output_dir) / "character_refs" / "hero.jpg")
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_bytes(b"fake_image")
        generator.character_refs["hero"] = ref_path

        urls, role = generator._build_image_refs(
            {"shot_id": 3, "characters": ["hero"]}, "http://a/last.jpg"
        )
        assert len(urls) == 2
        assert ref_path in urls
        assert "http://a/last.jpg" in urls
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

    def test_local_prev_frame_ignored(self, generator):
        """本地文件路径的 prev_last_frame 不应被传给 API"""
        local_file = str(Path(__file__).resolve())
        urls, role = generator._build_image_refs({"shot_id": 2}, local_file)
        assert urls == []
        assert role is None


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

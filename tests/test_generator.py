"""角色一致性 + 断链兜底逻辑测试 (P1.3 + P1.4)"""

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
    def test_only_prev_frame(self, generator):
        refs = generator._build_image_refs(
            {"shot_id": 2}, "http://a/frame.jpg", None
        )
        assert refs == ["http://a/frame.jpg"]

    def test_dual_anchor(self, generator):
        refs = generator._build_image_refs(
            {"shot_id": 2}, "http://a/frame.jpg", "http://a/anchor.jpg"
        )
        assert refs == ["http://a/frame.jpg", "http://a/anchor.jpg"]

    def test_dedup_same_url(self, generator):
        refs = generator._build_image_refs(
            {"shot_id": 2}, "http://a/same.jpg", "http://a/same.jpg"
        )
        assert refs == ["http://a/same.jpg"]

    def test_all_none(self, generator):
        refs = generator._build_image_refs({"shot_id": 2}, None, None)
        assert refs == []

    def test_local_path_ignored(self, generator):
        """本地文件路径（实际存在的文件）不应被传给 API"""
        # 用测试文件自身的路径，确保文件存在
        local_file = str(Path(__file__).resolve())
        refs = generator._build_image_refs(
            {"shot_id": 2}, local_file, None
        )
        assert refs == []

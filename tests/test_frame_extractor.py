from pathlib import Path

import cv2
import numpy as np

from tools.frame_extractor import (
    check_video_quality,
    composition_change_is_readable,
    frame_structure_similarity,
)


def _write_video(path: Path, frame_factory, frame_count: int = 48) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(frame_factory(index))
    writer.release()


def test_quality_rejects_black_video(tmp_path: Path):
    video = tmp_path / "black.mp4"
    _write_video(video, lambda _index: np.zeros((120, 160, 3), dtype=np.uint8))

    result = check_video_quality(str(video))

    assert result["pass"] is False
    assert result["dark_ratio"] > 0.9


def test_quality_rejects_frozen_video(tmp_path: Path):
    video = tmp_path / "frozen.mp4"
    frame = np.random.default_rng(7).integers(
        0, 256, (120, 160, 3), dtype=np.uint8
    )
    _write_video(video, lambda _index: frame)

    result = check_video_quality(str(video))

    assert result["pass"] is False
    assert result["frozen_pair_ratio"] > 0.8


def test_frame_structure_similarity_distinguishes_reframing(tmp_path: Path):
    first = np.zeros((120, 160, 3), dtype=np.uint8)
    first[20:100, 15:65] = 255
    reframed = np.zeros_like(first)
    reframed[10:70, 90:150] = 255
    first_path = tmp_path / "first.jpg"
    same_path = tmp_path / "same.jpg"
    reframed_path = tmp_path / "reframed.jpg"
    cv2.imwrite(str(first_path), first)
    cv2.imwrite(str(same_path), first)
    cv2.imwrite(str(reframed_path), reframed)

    assert frame_structure_similarity(first_path, same_path) > 0.99
    assert frame_structure_similarity(first_path, reframed_path) < 0.7


def test_composition_change_threshold_matches_declared_scale():
    assert composition_change_is_readable("large", 0.92) is False
    assert composition_change_is_readable("large", 0.75) is True
    assert composition_change_is_readable("medium", 0.95) is False
    assert composition_change_is_readable("medium", 0.88) is True

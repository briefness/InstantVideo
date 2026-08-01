from pathlib import Path

import cv2
import numpy as np

from tools.frame_extractor import check_video_quality


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

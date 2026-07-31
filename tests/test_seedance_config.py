"""Official Seedance 2.0 Mini capability contract."""

import config
from tools.ffmpeg_ops import PLATFORM_SPECS


def test_mini_defaults_are_official_480p_profile():
    assert config.SEEDANCE_MODEL == "doubao-seedance-2-0-mini-260615"
    assert config.DEFAULT_RESOLUTION == "480p"
    assert config.GENERATION_CHAINS["480p"] == [
        {
            "model": "doubao-seedance-2-0-mini-260615",
            "resolution": "480p",
            "max_duration": 15,
        }
    ]


def test_official_mini_dimensions_cover_every_supported_ratio():
    assert config.SEEDANCE_OUTPUT_DIMENSIONS["480p"] == {
        "16:9": "864:496",
        "4:3": "752:560",
        "1:1": "640:640",
        "3:4": "560:752",
        "9:16": "496:864",
        "21:9": "992:432",
    }
    assert set(config.SEEDANCE_OUTPUT_DIMENSIONS["480p"]) == set(
        config.SUPPORTED_ASPECT_RATIOS
    )


def test_platform_exports_are_480p_and_exclude_xiaohongshu():
    assert set(PLATFORM_SPECS) == set(config.SUPPORTED_PLATFORMS)
    assert "xiaohongshu" not in PLATFORM_SPECS
    assert PLATFORM_SPECS["youtube"]["resolution"] == "864:496"
    assert PLATFORM_SPECS["tiktok"]["resolution"] == "496:864"

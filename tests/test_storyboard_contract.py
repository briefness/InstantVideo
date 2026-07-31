"""Storyboard continuity and motion-budget contract tests."""

from pipeline.storyboard import (
    _apply_defaults,
    _scene_id,
    _validate_storyboard_richness,
)


def _shot(shot_id: int, **overrides) -> dict:
    shot = {
        "shot_id": shot_id,
        "duration": 5,
        "scene_description": "【废墟路口·远景】测试镜头",
        "prompt_en": "word " * 80,
        "camera": {"speed": "slow", "start_framing": "wide shot"},
        "mood": f"mood-{shot_id}",
        "characters": ["hero"],
        "key_props": [],
    }
    shot.update(overrides)
    return shot


def test_explicit_scene_id_is_independent_from_display_description():
    first = _shot(1, scene_id="intersection")
    second = _shot(
        2,
        scene_id="intersection",
        scene_description="【近距离交火·中近景】同一地点的新景别",
    )

    assert _scene_id(first) == _scene_id(second) == "intersection"


def test_defaults_infer_continuity_from_stable_scene_id():
    storyboard = {
        "shots": [
            _shot(1, scene_id="highway"),
            _shot(2, scene_id="intersection"),
            _shot(
                3,
                scene_id="intersection",
                scene_description="【近距离交火】同一地点继续",
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert [s["continuity_from_previous"] for s in storyboard["shots"]] == [
        "none",
        "intentional_cut",
        "seamless",
    ]


def test_fast_shot_over_five_seconds_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(1, characters=[]),
            _shot(
                2,
                duration=8,
                camera={"speed": "fast", "start_framing": "close-up"},
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("fast" in warning and "5s" in warning for warning in warnings)


def test_missing_motion_contract_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [_shot(1, characters=[]), _shot(2)],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("动作契约不完整" in warning for warning in warnings)

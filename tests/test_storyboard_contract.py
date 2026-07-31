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


def test_defaults_do_not_infer_seamless_from_stable_scene_id():
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
        "intentional_cut",
    ]


def test_explicit_compatible_seamless_continuation_is_preserved():
    storyboard = {
        "shots": [
            _shot(
                1,
                scene_id="intersection",
                camera={"start_framing": "wide shot", "end_framing": "wide shot"},
            ),
            _shot(
                2,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={"start_framing": "wide shot", "end_framing": "medium shot"},
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][1]["continuity_from_previous"] == "seamless"


def test_incompatible_framing_forces_intentional_cut_in_same_scene():
    storyboard = {
        "shots": [
            _shot(
                1,
                scene_id="intersection",
                camera={
                    "start_framing": "medium shot",
                    "end_framing": "extreme wide shot",
                },
            ),
            _shot(
                2,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={
                    "start_framing": "extreme close-up macro",
                    "end_framing": "extreme close-up macro",
                },
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][1]["continuity_from_previous"] == "intentional_cut"


def test_seamless_tail_frame_chain_is_bounded():
    storyboard = {
        "shots": [
            _shot(
                shot_id,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={"start_framing": "wide shot", "end_framing": "wide shot"},
            )
            for shot_id in range(1, 5)
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert [s["continuity_from_previous"] for s in storyboard["shots"]] == [
        "none",
        "seamless",
        "seamless",
        "intentional_cut",
    ]


def test_extreme_close_up_character_reference_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(
                1,
                characters=["hero"],
                extract_character_ref=True,
                camera={
                    "start_framing": "extreme close-up",
                    "end_framing": "close-up",
                },
                primary_action="hero activates one sensor",
                start_state={
                    "location": "intersection",
                    "subject": "hero hidden in smoke",
                    "action_phase": "inactive",
                    "camera": "extreme close-up",
                },
                end_state={
                    "location": "intersection",
                    "subject": "sensor visible",
                    "action_phase": "active",
                    "camera": "close-up",
                },
            ),
            _shot(
                2,
                characters=[],
                primary_action="dust settles",
                start_state={
                    "location": "intersection",
                    "subject": "empty street",
                    "action_phase": "dust moving",
                    "camera": "wide shot",
                },
                end_state={
                    "location": "intersection",
                    "subject": "empty street",
                    "action_phase": "dust settled",
                    "camera": "wide shot",
                },
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("角色参考镜头" in warning for warning in warnings)


def test_first_character_appearance_must_extract_reference():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(1, characters=["hero"], extract_character_ref=False),
            _shot(2, characters=["hero"], extract_character_ref=True),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("角色首次出现" in warning for warning in warnings)


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

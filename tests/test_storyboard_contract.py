"""Storyboard continuity and motion-budget contract tests."""

from pipeline.storyboard import (
    _apply_defaults,
    _infer_content_focus,
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


def _contract_shot(
    shot_id: int,
    duration: int,
    primary_action: str,
    **overrides,
) -> dict:
    shot = _shot(
        shot_id,
        duration=duration,
        primary_action=primary_action,
        extract_character_ref=shot_id == 1,
        camera={
            "speed": "slow",
            "start_framing": "wide shot",
            "end_framing": "medium shot",
        },
        start_state={
            "location": "arena",
            "subject": "fighters ready",
            "action_phase": "start",
            "camera": "wide shot",
        },
        end_state={
            "location": "arena",
            "subject": "fighters complete the beat",
            "action_phase": "end",
            "camera": "medium shot",
        },
    )
    shot.update(overrides)
    return shot


def test_content_focus_is_inferred_from_explicit_request():
    assert _infer_content_focus("制作一个30秒的孙悟空大战龟仙人") == "action"
    assert _infer_content_focus("智能手表产品宣传片") == "product"
    assert _infer_content_focus("制作一个猫咖日常短视频") == "balanced"


def test_camera_movement_cannot_be_primary_subject_action():
    storyboard = {
        "title": "fight",
        "shots": [
            _contract_shot(
                1,
                4,
                "Drone slowly pushes in toward two stationary fighters",
            ),
            _contract_shot(2, 5, "fighter throws one straight punch"),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="孙悟空大战龟仙人",
    )

    assert is_critical
    assert any("运镜不能充当 primary_action" in warning for warning in warnings)
    assert not any("动作节拍过载" in warning for warning in warnings)


def test_action_request_rejects_one_short_combat_beat_in_thirty_seconds():
    storyboard = {
        "title": "fight",
        "shots": [
            _contract_shot(1, 4, "fighters hold their stance"),
            _contract_shot(2, 5, "fighter raises his staff into guard position"),
            _contract_shot(3, 6, "opponent advances across the arena"),
            _contract_shot(4, 8, "fighter fires one blast that strikes the opponent"),
            _contract_shot(5, 7, "fighters walk away from the arena"),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="制作一个30秒的孙悟空大战龟仙人",
    )

    assert is_critical
    action_warning = next(
        warning for warning in warnings if "动作重心不足" in warning
    )
    assert "Shot 1=" in action_warning
    assert "Shot 5=" in action_warning


def test_chinese_title_uses_structured_theme_anchors_not_english_prompt_text():
    storyboard = {
        "title": "「末日清道夫」机器人清除丧尸",
        "characters": [
            {"name": "combat_bot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1, 4, "combat bot fires one burst at the zombie horde",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                2, 5, "combat bot crushes the lead zombie",
                characters=["combat_bot", "zombie_horde"],
            ),
            _contract_shot(
                3, 6, "combat bot scans the cleared street",
                characters=["combat_bot"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not any("主题锚定不足" in warning for warning in warnings)


def test_product_process_causality_does_not_run_for_action_story():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                1,
                5,
                "robot takes a sip while monitoring the street",
                prompt_en="The robot takes a sip while monitoring the ruined street. " * 8,
            ),
            _contract_shot(
                2,
                5,
                "robot prepares its weapon",
                prompt_en="The robot prepares its weapon before moving forward. " * 8,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("时序倒置" in warning for warning in warnings)


def test_explicit_product_process_still_checks_causal_order():
    storyboard = {
        "title": "coffee process",
        "shots": [
            _contract_shot(
                1,
                5,
                "customer drinks the finished coffee",
                prompt_en="The customer drinks and enjoys the finished coffee. " * 10,
            ),
            _contract_shot(
                2,
                5,
                "barista grinds the coffee beans",
                prompt_en="The barista grinds and prepares fresh coffee beans. " * 10,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="展示从咖啡豆到成品的完整制作过程"
    )

    assert any("时序倒置" in warning for warning in warnings)


def test_effects_and_environment_key_props_are_not_persistent_prop_continuity():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(1, 5, "robot enters the street", key_props=["rifle"]),
            _contract_shot(
                2,
                5,
                "robot fires the rifle",
                key_props=["rifle", "muzzle flash", "bullet casings", "zombie bodies"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("物件凭空出现" in warning for warning in warnings)


def test_unexplained_persistent_prop_still_fails_continuity_check():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                1,
                5,
                "robot enters the street",
                continuity_props=["rifle"],
            ),
            _contract_shot(
                2,
                5,
                "robot surveys the street",
                continuity_props=["rifle", "supply crate"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert any("物件凭空出现" in warning for warning in warnings)


def test_short_action_budget_does_not_require_a_separate_identity_intro_shot():
    storyboard = {
        "title": "robot cleanup",
        "characters": [
            {"name": "combat_bot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                5,
                "combat bot fires one burst at the zombie horde",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                2,
                5,
                "combat bot crushes the lead zombie",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                3,
                5,
                "combat bot scans the cleared street",
                characters=["combat_bot"],
                extract_character_ref=False,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not any("身份锚点" in warning for warning in warnings)
    assert not any("角色首次出现" in warning for warning in warnings)


def test_long_action_can_introduce_identity_character_inside_the_conflict():
    storyboard = {
        "title": "robot cleanup",
        "characters": [
            {"name": "combat_robot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                shot_id,
                6,
                "combat robot faces the zombie horde",
                characters=["combat_robot", "zombie_horde"],
                extract_character_ref=False,
            )
            for shot_id in range(1, 6)
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个30秒的视频，机器人末日清除丧尸"
    )

    assert not any("身份锚点" in warning for warning in warnings)
    assert not any("角色首次出现" in warning for warning in warnings)


def test_insert_shot_requirement_depends_on_content_focus():
    storyboard = {
        "title": "subject",
        "shots": [
            _contract_shot(1, 4, "fighter punches the opponent"),
            _contract_shot(2, 5, "opponent blocks the punch"),
            _contract_shot(3, 6, "fighter counters with one kick"),
        ],
    }

    action_warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="两名武术家擂台对决",
    )
    product_warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="运动手表产品宣传片",
    )

    assert not any("缺少 insert shot" in warning for warning in action_warnings)
    assert any("缺少 insert shot" in warning for warning in product_warnings)


def test_compound_contact_sequence_requires_shot_split():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "clearance_bot", "mobility": "tracked"},
            {"name": "zombie_horde", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(1, 4, "zombie charges toward the robot"),
            _contract_shot(
                2,
                6,
                "A zombie leaps onto the robot's shoulder; "
                "robot crushes its neck and tears it off",
                characters=["clearance_bot", "zombie_horde"],
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="机器人末日清除丧尸",
    )

    assert is_critical
    assert any("动作节拍过载" in warning for warning in warnings)


def test_causal_action_and_target_state_are_not_mistaken_for_three_beats():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "clearance_bot", "mobility": "tracked"},
            {"name": "zombie_horde", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(1, 4, "zombie horde charges toward the robot"),
            _contract_shot(
                2,
                6,
                "Robot fires plasma blasts at the charging horde, "
                "incinerating the front line attackers",
                characters=["clearance_bot", "zombie_horde"],
                camera={
                    "speed": "slow",
                    "start_framing": "wide shot",
                    "end_framing": "medium shot",
                    "screen_positions": {
                        "clearance_bot": "left",
                        "zombie_horde": "right",
                    },
                    "axis_change": "establish",
                },
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="机器人末日清除丧尸",
    )

    assert not any("动作节拍过载" in warning for warning in warnings)


def test_two_tightly_causal_action_stages_are_not_overloaded():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [{"name": "robot", "mobility": "tracked"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "robot advances toward the lead zombie then crushes it",
                characters=["robot"],
            )
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("动作节拍过载" in warning for warning in warnings)


def test_three_explicit_action_stages_are_overloaded():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [{"name": "robot", "mobility": "tracked"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "robot advances then stops then crushes the lead zombie",
                characters=["robot"],
            )
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert any("动作节拍过载" in warning for warning in warnings)


def test_action_must_match_single_character_mobility():
    storyboard = {
        "title": "clearance robot",
        "characters": [
            {
                "name": "clearance_bot",
                "description": "A heavy combat robot with a tracked base instead of legs",
                "mobility": "tracked",
            }
        ],
        "shots": [
            _contract_shot(
                1,
                5,
                "Robot fires one plasma blast",
                characters=["clearance_bot"],
            ),
            _contract_shot(
                2,
                5,
                "Robot steps over the remains and walks toward the skyline",
                characters=["clearance_bot"],
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="机器人清除丧尸",
    )

    assert is_critical
    assert any("移动形态冲突" in warning for warning in warnings)


def test_action_scene_cannot_flip_screen_axis_without_reestablishing():
    storyboard = {
        "title": "arena fight",
        "characters": [
            {"name": "hero", "mobility": "bipedal"},
            {"name": "villain", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "hero punches villain",
                characters=["hero", "villain"],
                camera={
                    "speed": "fast",
                    "start_framing": "wide shot",
                    "end_framing": "wide shot",
                    "screen_positions": {"hero": "left", "villain": "right"},
                    "axis_change": "establish",
                },
            ),
            _contract_shot(
                2,
                5,
                "villain blocks the punch",
                characters=["hero", "villain"],
                camera={
                    "speed": "fast",
                    "start_framing": "medium shot",
                    "end_framing": "medium shot",
                    "screen_positions": {"hero": "right", "villain": "left"},
                    "axis_change": "hold",
                },
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="两名武术家擂台对决",
    )

    assert is_critical
    assert any("空间轴反转" in warning for warning in warnings)


def test_action_scene_requires_facing_eyeline_and_action_target():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "robot", "mobility": "tracked"},
            {"name": "zombies", "mobility": "bipedal", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "robot fires one burst at zombies",
                characters=["robot", "zombies"],
                extract_character_ref=False,
                action_beats=[{
                    "phase": "peak",
                    "actor": "robot",
                    "action": "fires one burst",
                    "target": "zombies",
                    "visible_result": "front rank recoils",
                }],
                camera={
                    "speed": "fast",
                    "start_framing": "wide shot",
                    "end_framing": "wide shot",
                    "screen_positions": {"robot": "left", "zombies": "right"},
                    "axis_change": "establish",
                },
            )
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard, user_request="机器人清除丧尸"
    )

    assert is_critical
    assert any("角色调度不完整" in warning for warning in warnings)


def test_multi_character_frame_cannot_become_identity_reference():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "robot aims at zombies",
                characters=["robot", "zombies"],
                extract_character_ref=True,
            )
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("角色参考归属不明确" in warning for warning in warnings)


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


def test_defaults_clear_unusable_multi_character_identity_extraction():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _shot(
                1,
                characters=["robot", "zombies"],
                extract_character_ref=True,
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["extract_character_ref"] is False


def test_defaults_sync_structured_visible_participants_before_reference_check():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _shot(
                1,
                characters=["robot"],
                extract_character_ref=True,
                camera={
                    "screen_positions": {
                        "robot": "left foreground",
                        "zombies": "center background",
                    }
                },
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["characters"] == ["robot", "zombies"]
    assert storyboard["shots"][0]["extract_character_ref"] is False


def test_defaults_compile_visible_impact_into_shootable_geometry():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "【ruined street】 firefight",
            "prompt_en": "Robot fires one burst at the zombies.",
            "characters": ["robot", "zombies"],
            "camera": {
                "screen_positions": {"robot": "left", "zombies": "right"}
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "the front zombie recoils",
            }],
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["coverage_role"] == "interaction"
    assert shot["required_visible_entities"] == ["robot", "zombies"]
    assert shot["interaction_geometry"]["must_share_frame"] is True
    assert shot["interaction_geometry"]["line_of_action_visible"] is True


def test_short_robot_action_replay_keeps_only_noncritical_scene_warning():
    characters = ["robot", "zombies"]
    blocking = {
        "robot": {
            "frame_position": "left",
            "body_orientation": "profile right",
            "facing_target": "zombies",
            "eyeline_target": "zombies",
            "action_target": "zombies",
        },
        "zombies": {
            "frame_position": "right",
            "body_orientation": "profile left",
            "facing_target": "robot",
            "eyeline_target": "robot",
            "action_target": "robot",
        },
    }
    actions = (
        "robot fires one burst at zombies",
        "zombies charge at the robot",
        "robot advances toward the lead zombie then crushes it",
    )
    storyboard = {
        "title": "robot apocalypse cleanup",
        "content_focus": "action",
        "characters": [
            {"name": "robot", "mobility": "tracked", "reference_mode": "identity"},
            {"name": "zombies", "mobility": "bipedal", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                index,
                duration,
                action,
                scene_id="ruined_city_street",
                characters=characters,
                extract_character_ref=index == 1,
                camera={
                    "speed": "slow" if index == 3 else "fast",
                    "start_framing": framing,
                    "end_framing": framing,
                    "screen_positions": {"robot": "left", "zombies": "right"},
                    "axis_change": "establish" if index == 1 else "hold",
                },
                blocking=blocking,
                action_beats=[{
                    "phase": "peak",
                    "actor": "robot" if index != 2 else "zombies",
                    "action": action,
                    "target": "zombies" if index != 2 else "robot",
                    "visible_result": "the opposing side visibly recoils",
                }],
            )
            for index, duration, framing, action in zip(
                (1, 2, 3),
                (4, 5, 6),
                ("wide shot", "close-up", "medium shot"),
                actions,
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")
    warnings, is_critical = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not is_critical, warnings
    assert any("场景集中" in warning for warning in warnings)
    assert not any("动作重心不足" in warning for warning in warnings)
    assert not any("动作节拍过载" in warning for warning in warnings)
    assert not any("角色参考归属不明确" in warning for warning in warnings)


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


def test_later_single_character_shot_can_extract_reference():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(1, characters=["hero"], extract_character_ref=False),
            _shot(2, characters=["hero"], extract_character_ref=True),
        ],
    }

    warnings, _ = _validate_storyboard_richness(storyboard)

    assert not any("角色首次出现" in warning for warning in warnings)


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

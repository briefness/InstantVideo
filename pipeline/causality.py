"""Shared interaction-causality contract for planning, prompting, and review."""

from __future__ import annotations


CAUSAL_INTERACTION_MODES = frozenset({
    "direct_contact",
    "directed_path",
    "area_effect",
    "indirect_effect",
})
EFFECT_PHASES = frozenset({"none", "setup", "active", "aftermath"})
OUTCOME_SCOPES = frozenset({"none", "single", "subset", "all"})
EFFECT_MOTIONS = frozenset({"none", "static", "sweep", "expand", "propagate"})
_SCOPE_RANK = {"none": 0, "single": 1, "subset": 2, "all": 3}
PHYSICAL_EFFECT_EVIDENCE_RULE = (
    "physical_effect_visible is true only when the contracted effect visibly "
    "leaves its source, travels through its effect region, makes contact, or "
    "changes a target. physical_effect_visible is false for actor-only aiming, "
    "charging, sensor activation, weapon power-up, or source-local glow"
)


def interaction_geometry(shot: dict) -> dict:
    value = shot.get("interaction_geometry")
    return value if isinstance(value, dict) else {}


def interaction_mode(shot: dict) -> str:
    return str(interaction_geometry(shot).get("interaction_mode", "none")).strip()


def requires_causal_review(shot: dict) -> bool:
    phase = str(interaction_geometry(shot).get("effect_phase", "none")).strip()
    return phase in {"setup", "active", "aftermath"} or (
        interaction_mode(shot) in CAUSAL_INTERACTION_MODES
    )


def with_causal_mode_invariants(value: object) -> dict:
    """Derive booleans that are definitional consequences of the chosen mode."""
    geometry = dict(value) if isinstance(value, dict) else {}
    phase = str(geometry.get("effect_phase", "")).strip()
    if phase in {"none", "setup"}:
        geometry.update({
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
            "must_share_frame": False,
            "line_of_action_visible": False,
        })
        return geometry
    if phase == "aftermath":
        geometry.update({
            "interaction_mode": "none",
            "effect_motion": "none",
            "must_share_frame": False,
            "line_of_action_visible": False,
        })
        return geometry

    mode = str(geometry.get("interaction_mode", "none")).strip()
    scope = str(geometry.get("outcome_scope", "")).strip()
    effect_motion = str(geometry.get("effect_motion", "")).strip()
    if phase == "active" and mode == "directed_path" and scope == "all":
        geometry["effect_motion"] = "sweep"
    elif phase == "active" and effect_motion in {"", "unspecified"}:
        if mode in {"direct_contact", "directed_path"}:
            geometry["effect_motion"] = "sweep" if scope == "all" else "static"
        elif mode == "area_effect":
            geometry["effect_motion"] = "static"
        elif mode == "indirect_effect":
            geometry["effect_motion"] = "propagate"
    if mode == "directed_path":
        geometry["line_of_action_visible"] = True
    elif mode == "direct_contact":
        geometry["must_share_frame"] = True
    return geometry


def normalize_causal_scope(shots: list[dict]) -> list[str]:
    """Clamp aftermath to the largest result already established upstream."""
    established: dict[tuple[str, str], int] = {}
    corrections: list[str] = []
    for shot in shots:
        geometry = interaction_geometry(shot)
        phase = str(geometry.get("effect_phase", "")).strip()
        scope = str(geometry.get("outcome_scope", "")).strip()
        if phase not in {"active", "aftermath"} or scope not in _SCOPE_RANK:
            continue
        key = _effect_target_key(shot, geometry)
        if key is None:
            continue
        if phase == "active" and scope != "none":
            established[key] = max(established.get(key, 0), _SCOPE_RANK[scope])
            continue
        inherited = established.get(key, 0)
        if phase == "aftermath" and inherited and _SCOPE_RANK[scope] > inherited:
            normalized = next(
                name for name, rank in _SCOPE_RANK.items() if rank == inherited
            )
            geometry["outcome_scope"] = normalized
            corrections.append(
                f"Shot {shot.get('shot_id', '?')}: aftermath 结果范围 "
                f"{scope} -> {normalized}（沿用已建立结果）"
            )
    return corrections


def causality_readiness_issues(
    shot: dict,
    *,
    require_for_visible_interaction: bool = False,
) -> list[str]:
    """Return missing invariants shared by every visible cause-and-effect mode."""
    mode = interaction_mode(shot)
    if (
        mode not in CAUSAL_INTERACTION_MODES
        and require_for_visible_interaction
        and _has_visible_interaction(shot)
    ):
        return [
            f"Shot {shot.get('shot_id', '?')}: 可见因果交互缺少 interaction_mode"
        ]
    if mode not in CAUSAL_INTERACTION_MODES:
        return []

    shot_id = shot.get("shot_id", "?")
    geometry = interaction_geometry(shot)
    required = (
        ("source", "作用来源"),
        ("effect_region", "作用区域"),
        ("reaction_scope", "反应范围"),
        ("unaffected_behavior", "范围外行为"),
    )
    issues = [
        f"Shot {shot_id}: {mode} 交互缺少{label} {field}"
        for field, label in required
        if not str(geometry.get(field, "")).strip()
    ]
    if mode == "directed_path" and not geometry.get("line_of_action_visible"):
        issues.append(f"Shot {shot_id}: directed_path 必须让作用路径可见")
    if mode == "direct_contact" and not geometry.get("must_share_frame"):
        issues.append(f"Shot {shot_id}: direct_contact 必须让执行者与目标同框接触")
    return issues


def causal_storyboard_issues(
    storyboard: dict,
    *,
    required: bool = False,
) -> list[str]:
    """Validate effect phases and conserve affected scope across adjacent shots."""
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    if not required and not any(
        str(interaction_geometry(shot).get("effect_phase", "")).strip()
        not in {"", "unspecified"}
        for shot in shots
    ):
        return []

    issues: list[str] = []
    affected_scope: dict[tuple[str, str], int] = {}
    for shot in shots:
        shot_id = shot.get("shot_id", "?")
        geometry = interaction_geometry(shot)
        phase = str(geometry.get("effect_phase", "")).strip()
        scope = str(geometry.get("outcome_scope", "")).strip()
        motion = str(geometry.get("effect_motion", "")).strip()
        mode = interaction_mode(shot)
        for field, value in (
            ("effect_phase", phase),
            ("outcome_scope", scope),
            ("effect_motion", motion),
        ):
            if not value or value == "unspecified":
                issues.append(
                    f"Shot {shot_id}: interaction_geometry.{field} 必须显式声明"
                )
        if phase not in EFFECT_PHASES or scope not in OUTCOME_SCOPES or motion not in EFFECT_MOTIONS:
            continue

        if phase in {"none", "setup"}:
            if mode != "none":
                label = "准备阶段" if phase == "setup" else "无作用阶段"
                issues.append(
                    f"Shot {shot_id}: {label}必须使用 interaction_mode=none，"
                    "不能提前要求物理作用路径"
                )
            if scope != "none" or motion != "none":
                issues.append(
                    f"Shot {shot_id}: {phase} 阶段不得产生作用结果或作用运动"
                )
            continue

        if phase == "active":
            if mode not in CAUSAL_INTERACTION_MODES:
                issues.append(f"Shot {shot_id}: 生效阶段必须声明物理 interaction_mode")
            if scope == "none":
                issues.append(f"Shot {shot_id}: 生效阶段 outcome_scope 不能为 none")
            if motion == "none":
                issues.append(f"Shot {shot_id}: 生效阶段 effect_motion 不能为 none")
            if mode == "directed_path" and scope == "all" and motion != "sweep":
                issues.append(
                    f"Shot {shot_id}: directed_path 要影响全部目标必须以 "
                    "effect_motion=sweep 展示路径扫过全部目标"
                )
            key = _effect_target_key(shot, geometry)
            if key and scope != "none":
                affected_scope[key] = max(
                    affected_scope.get(key, 0),
                    _SCOPE_RANK[scope],
                )
            continue

        if mode != "none" or motion != "none":
            issues.append(
                f"Shot {shot_id}: aftermath 只能展示既有结果，"
                "必须使用 interaction_mode=none 且 effect_motion=none"
            )
        key = _effect_target_key(shot, geometry)
        inherited = affected_scope.get(key, 0) if key else 0
        if _SCOPE_RANK[scope] > inherited:
            issues.append(
                f"Shot {shot_id}: aftermath 结果范围从已建立的 "
                f"{_scope_name(inherited)} 无原因扩大为 {scope}"
            )
    return issues


def causal_evidence_issues(shot: dict, samples: object) -> list[str]:
    """Validate per-sample visual evidence instead of trusting one aggregate verdict."""
    phase = str(interaction_geometry(shot).get("effect_phase", "none")).strip()
    if phase not in {"setup", "active", "aftermath"}:
        return []
    if not isinstance(samples, list) or not samples:
        return ["缺少逐采样因果证据"]

    issues: list[str] = []
    physical_effect_seen = False
    in_scope_reaction_seen = False
    contracted_outcome_seen = False
    connected_outcome_seen = False
    first_effect_sample: int | None = None
    first_outcome_sample: int | None = None
    required_fields = (
        "physical_effect_visible",
        "reaction_visible",
        "effect_intersects_reaction",
        "out_of_scope_reaction_visible",
        "contracted_outcome_visible",
        "outcome_causally_connected",
    )
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict) or any(
            not isinstance(sample.get(field), bool) for field in required_fields
        ):
            issues.append(f"采样 {index} 因果证据字段缺失或类型无效")
            continue
        physical_effect_seen |= sample["physical_effect_visible"]
        contracted_outcome_seen |= sample["contracted_outcome_visible"]
        if sample["contracted_outcome_visible"]:
            if sample["outcome_causally_connected"]:
                connected_outcome_seen = True
            elif phase == "active":
                issues.append(
                    f"采样 {index} 的约定结果与物理作用之间缺少可见因果过渡"
                )
        if sample["physical_effect_visible"] and first_effect_sample is None:
            first_effect_sample = index
        if sample["contracted_outcome_visible"] and first_outcome_sample is None:
            first_outcome_sample = index
        if phase == "active" and sample["reaction_visible"]:
            if sample["effect_intersects_reaction"]:
                in_scope_reaction_seen = True
            else:
                issues.append(f"采样 {index} 的反应目标未与作用区域相交")
        if phase != "setup" and sample["out_of_scope_reaction_visible"]:
            issues.append(f"采样 {index} 出现范围外目标发生反应")

    if phase == "setup" and (
        physical_effect_seen
        or contracted_outcome_seen
    ):
        issues.append("准备阶段提前出现物理作用或约定结果")
    if phase == "active" and not physical_effect_seen:
        issues.append("生效阶段未看到物理作用")
    if phase == "active" and not in_scope_reaction_seen:
        issues.append("生效阶段未看到作用区域内目标的同步反应")
    if phase in {"active", "aftermath"} and not contracted_outcome_seen:
        issues.append(f"{phase} 阶段未看到约定结果及其完整作用范围")
    if phase == "active" and not connected_outcome_seen:
        issues.append("active 阶段未看到作用到约定结果的完整可见因果过渡")
    if (
        phase == "active"
        and first_effect_sample is not None
        and first_outcome_sample is not None
        and first_outcome_sample < first_effect_sample
    ):
        issues.append("active 阶段的约定结果先于物理原因出现")
    if phase == "aftermath" and physical_effect_seen:
        issues.append("aftermath 阶段出现新物理作用")
    return issues


def _effect_target_key(shot: dict, geometry: dict) -> tuple[str, str] | None:
    target = str(geometry.get("target", "")).strip()
    if not target or target.lower() == "none":
        return None
    scene_id = str(shot.get("scene_id", "")).strip()
    return scene_id, target


def _scope_name(rank: int) -> str:
    return next((name for name, value in _SCOPE_RANK.items() if value == rank), "none")


def blocking_geometry_issues(shot: dict) -> list[str]:
    """Reject explicit body directions that contradict actor/target placement."""
    geometry = interaction_geometry(shot)
    actor = str(geometry.get("actor", "")).strip()
    target = str(geometry.get("target", "")).strip()
    if not actor or not target or target.lower() == "none":
        return []

    camera = shot.get("camera", {})
    positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
    blocking = shot.get("blocking", {})
    actor_blocking = blocking.get(actor, {}) if isinstance(blocking, dict) else {}
    if not isinstance(actor_blocking, dict):
        return []

    actor_position = str(
        positions.get(actor)
        or actor_blocking.get("frame_position")
        or geometry.get("actor_screen_position", "")
    ).lower()
    target_position = str(
        positions.get(target)
        or geometry.get("target_screen_position", "")
    ).lower()
    orientation = str(actor_blocking.get("body_orientation", "")).lower()
    if not actor_position or not target_position or not orientation:
        return []

    shot_id = shot.get("shot_id", "?")
    issues = []
    actor_depth = _depth_rank(actor_position)
    target_depth = _depth_rank(target_position)
    if (
        actor_depth is not None
        and target_depth is not None
        and target_depth > actor_depth
        and any(term in orientation for term in (
            "toward camera", "facing camera", "front toward camera"
        ))
    ):
        issues.append(
            f"Shot {shot_id}: {actor} 身体朝向与目标景深矛盾；"
            f"{target} 位于更深处，不能同时面向镜头"
        )
    elif (
        actor_depth is not None
        and target_depth is not None
        and target_depth < actor_depth
        and any(term in orientation for term in (
            "away from camera", "toward background", "back to camera"
        ))
    ):
        issues.append(
            f"Shot {shot_id}: {actor} 身体朝向与目标景深矛盾；"
            f"{target} 位于更靠近镜头处，不能背向镜头"
        )

    actor_side = _horizontal_rank(actor_position)
    target_side = _horizontal_rank(target_position)
    if actor_side is not None and target_side is not None:
        if target_side > actor_side and "screen-left" in orientation:
            issues.append(
                f"Shot {shot_id}: {actor} 身体朝向与目标左右位置矛盾；"
                f"{target} 位于其右侧，身体不能明确朝向 screen-left"
            )
        elif target_side < actor_side and "screen-right" in orientation:
            issues.append(
                f"Shot {shot_id}: {actor} 身体朝向与目标左右位置矛盾；"
                f"{target} 位于其左侧，身体不能明确朝向 screen-right"
            )
    return issues


def compile_interaction_blocking(shot: dict) -> None:
    """Derive redundant facing fields from canonical participants and positions."""
    geometry = interaction_geometry(shot)
    actor = str(geometry.get("actor", "")).strip()
    target = str(geometry.get("target", "")).strip()
    if not actor or not target or target.casefold() == "none":
        return

    camera = shot.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    positions = camera.get("screen_positions")
    positions = positions if isinstance(positions, dict) else {}
    blocking = shot.get("blocking")
    blocking = dict(blocking) if isinstance(blocking, dict) else {}

    # Positions are canonical. Compile a stable two-sided axis when the LLM omits it.
    if actor not in positions:
        positions[actor] = "left foreground"
    if target not in positions:
        positions[target] = "right midground"
    camera["screen_positions"] = positions
    shot["camera"] = camera

    for subject, counterpart in ((actor, target), (target, actor)):
        intent = blocking.get(subject)
        intent = dict(intent) if isinstance(intent, dict) else {}
        subject_position = str(
            positions.get(subject) or intent.get("frame_position", "")
        ).strip()
        counterpart_intent = blocking.get(counterpart)
        counterpart_intent = (
            counterpart_intent if isinstance(counterpart_intent, dict) else {}
        )
        counterpart_position = str(
            positions.get(counterpart)
            or counterpart_intent.get("frame_position", "")
        ).strip()
        if subject_position:
            intent["frame_position"] = subject_position
        intent["body_orientation"] = _orientation_toward(
            subject_position,
            counterpart_position,
            counterpart,
        )
        intent["facing_target"] = counterpart
        intent["eyeline_target"] = counterpart
        intent["action_target"] = counterpart
        blocking[subject] = intent

    for subject in shot.get("characters", []):
        if subject in blocking:
            continue
        position = str(positions.get(subject, "right background")).strip()
        blocking[subject] = {
            "frame_position": position,
            "body_orientation": f"oriented toward {actor}",
            "facing_target": actor,
            "eyeline_target": actor,
            "action_target": actor,
            "travel_direction": "toward actor",
        }
    shot["blocking"] = blocking


def _orientation_toward(
    subject_position: str,
    target_position: str,
    target: str,
) -> str:
    subject_side = _horizontal_rank(subject_position.lower())
    target_side = _horizontal_rank(target_position.lower())
    side = ""
    if subject_side is not None and target_side is not None:
        if target_side > subject_side:
            side = "screen-right"
        elif target_side < subject_side:
            side = "screen-left"

    subject_depth = _depth_rank(subject_position.lower())
    target_depth = _depth_rank(target_position.lower())
    depth = ""
    camera_relation = ""
    if subject_depth is not None and target_depth is not None:
        if target_depth > subject_depth:
            depth = "background"
            camera_relation = "away from camera"
        elif target_depth < subject_depth:
            depth = "foreground"
            camera_relation = "toward camera"

    direction = " and ".join(value for value in (side, depth) if value)
    if direction and camera_relation:
        return f"three-quarter toward {direction}, {camera_relation}"
    if direction:
        return f"profile toward {direction}"
    return f"three-quarter toward {target}"


def _depth_rank(position: str) -> int | None:
    if "foreground" in position:
        return 0
    if "background" in position or "far end" in position or "distant" in position:
        return 2
    if "midground" in position:
        return 1
    return None


def _horizontal_rank(position: str) -> int | None:
    if "left" in position:
        return -1
    if "right" in position:
        return 1
    if "center" in position or "centre" in position:
        return 0
    return None


def _has_visible_interaction(shot: dict) -> bool:
    geometry = interaction_geometry(shot)
    phase = str(geometry.get("effect_phase", "")).strip()
    if phase not in {"", "unspecified"}:
        return phase == "active"
    if geometry.get("must_share_frame") and geometry.get("actor") and geometry.get("target"):
        return True
    return any(
        isinstance(beat, dict)
        and str(beat.get("actor", "")).strip()
        and str(beat.get("target", "")).strip()
        and str(beat.get("visible_result", "")).strip()
        for beat in shot.get("action_beats", [])
    )


def causality_prompt_constraints(shot: dict) -> str:
    """Compile the validated causal geometry without guessing from subject matter."""
    geometry = interaction_geometry(shot)
    phase = str(geometry.get("effect_phase", "none")).strip()
    scope = str(geometry.get("outcome_scope", "none")).strip()
    motion = str(geometry.get("effect_motion", "none")).strip()
    if phase == "setup":
        return (
            "effect phase setup: show preparation, aiming, or charging only; "
            "no emitted physical effect, impact, damage, or target reaction"
        )
    if phase == "aftermath":
        return (
            f"effect phase aftermath: show only the already established {scope} outcome; "
            "do not create a new effect or expand the affected scope"
        )
    mode = interaction_mode(shot)
    if mode not in CAUSAL_INTERACTION_MODES:
        return ""
    fields = (
        ("source", "show the cause originating from"),
        ("effect_region", "confine the physical effect to"),
        ("reaction_scope", "only this reaction scope may respond"),
        ("unaffected_behavior", "outside that scope"),
    )
    details = [
        f"{label} {str(geometry.get(field, '')).strip()}"
        for field, label in fields
        if str(geometry.get(field, "")).strip()
    ]
    return (
        f"effect phase {phase}, outcome scope {scope}, effect motion {motion}; "
        f"{mode} cause-and-effect: " + ", ".join(details)
    )


def causality_review_instruction(shot: dict) -> str:
    geometry = interaction_geometry(shot)
    phase = str(geometry.get("effect_phase", "none")).strip()
    if phase == "setup":
        return (
            f"This is a setup phase. {PHYSICAL_EFFECT_EVIDENCE_RULE}. "
            "Reject effect_path_valid and "
            "reaction_causality_valid if a physical effect leaves its source, an impact "
            "occurs, or any target reacts before the active phase. "
        )
    if phase == "aftermath":
        return (
            "This is an aftermath phase. It may show only the outcome scope already "
            "established by the prior active phase; reject any new unexplained effect or "
            "larger affected population. "
        )
    mode = interaction_mode(shot)
    if mode not in CAUSAL_INTERACTION_MODES:
        return ""
    mode_rule = {
        "direct_contact": (
            "Direct contact requires source and target to share the frame at the "
            "moment of physical contact."
        ),
        "directed_path": (
            "A directed path requires a readable origin and path intersecting every "
            "reacting subject."
        ),
        "area_effect": (
            "An area effect does not require a straight line or visible actor; its source "
            "may be offscreen when the contract says so, but every reacting subject must "
            "be visibly inside effect_region."
        ),
        "indirect_effect": (
            "An indirect effect requires the intermediary cause sequence to be readable; "
            "its original source may be offscreen when the contract says so."
        ),
    }[mode]
    return (
        f"For this {mode} interaction, {mode_rule} effect_path_valid means the "
        "visible origin and path, region, contact, or intermediary match the contract "
        "across time. reaction_causality_valid means only entities inside reaction_scope "
        "react while unaffected_behavior remains true outside it. "
    )

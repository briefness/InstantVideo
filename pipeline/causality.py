"""Shared interaction-causality contract for planning, prompting, and review."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from pipeline.narrative import narrative_prompt_constraint


CAUSAL_INTERACTION_MODES = frozenset({
    "direct_contact",
    "directed_path",
    "area_effect",
    "indirect_effect",
})
EFFECT_PHASES = frozenset({"none", "setup", "active", "aftermath"})
OUTCOME_SCOPES = frozenset({"none", "single", "subset", "all"})
EFFECT_MOTIONS = frozenset({"none", "static", "sweep", "expand", "propagate"})
ACTION_CONTRACT_VERSION = "action-contract-v2"
_SCOPE_RANK = {"none": 0, "single": 1, "subset": 2, "all": 3}
_NONE_LIKE_VALUES = frozenset({
    "", "none", "none yet", "n/a", "na", "not applicable", "no reaction",
})
ACTION_EVIDENCE_FIELDS = (
    "preparation_state_visible",
    "non_physical_cue_visible",
    "physical_effect_visible",
    "effect_reaches_target",
    "target_reaction_visible",
    "out_of_scope_reaction_visible",
    "phase_endpoint_visible",
    "narrative_outcome_visible",
    "outcome_causally_connected",
)
LEGACY_ACTION_EVIDENCE_FIELDS = (
    "physical_effect_visible",
    "reaction_visible",
    "effect_intersects_reaction",
    "out_of_scope_reaction_visible",
    "contracted_outcome_visible",
    "outcome_causally_connected",
)
PHYSICAL_EFFECT_EVIDENCE_RULE = (
    "physical_effect_visible is true only when the contracted effect visibly "
    "leaves its source, travels through its effect region, makes contact, or "
    "changes a target. physical_effect_visible is false for actor-only aiming, "
    "charging, sensor activation, weapon power-up, or source-local glow"
)


@dataclass(frozen=True)
class CompiledActionContract:
    """One authoritative action interpretation for every downstream caller."""

    phase: str
    mode: str
    outcome_scope: str
    effect_motion: str
    requires_evidence: bool
    prompt_constraint: str
    review_instruction: str
    prompt_parts: tuple[str, ...]
    review_projection: dict[str, object]
    canonical_geometry: dict[str, object]
    prompt_start_state: str
    contracted_visible_result: str
    evidence_fields: tuple[str, ...] = ACTION_EVIDENCE_FIELDS

    def evidence_prompt(self, sample_count: int) -> str:
        fields = ", ".join(self.evidence_fields)
        return (
            f"Return causal_sample_evidence as exactly {sample_count} objects in "
            f"current-sample order. Every object must contain strict booleans {fields}. "
            "preparation_state_visible means the contracted preparation visibly advances. "
            "non_physical_cue_visible means a guidance light, aim indicator, charge signal, "
            "sensor display, or other visible cue communicates intent without physically "
            "changing a target. physical_effect_visible follows this rule: "
            f"{PHYSICAL_EFFECT_EVIDENCE_RULE}. effect_reaches_target is true only when the "
            "same target visibly lies on the path, inside the region, at contact, or in the "
            "contracted intermediary chain in the current sample. target_reaction_visible means "
            "that target visibly changes state because of the Physical Effect; it may remain true "
            "when the reaction continues after a prior reach. Anticipation, aiming, defensive "
            "preparation, ordinary motion, disappearance, smoke, a cut, or a later empty frame "
            "is not a reaction. out_of_scope_reaction_visible is true only when an entity "
            "outside reaction_scope undergoes impact, injury, damage, forced displacement, "
            "or contracted outcome because of the Physical Effect. Continuing or beginning "
            "the contracted unaffected behavior, including independent approach, retreat, or "
            "ordinary motion, is false. phase_endpoint_visible means the endpoint of this phase is "
            "visible; it does not by itself imply impact or story payoff. "
            "narrative_outcome_visible means the full contracted story consequence and scope "
            "are visible; a preparation endpoint, flash, camera change, or partial result does "
            "not satisfy it. outcome_causally_connected is cumulative across current-sample "
            "order: it becomes true once ordered visible evidence from Physical Effect through "
            "target intersection and target reaction to Narrative Outcome has appeared at or "
            "before the current sample."
        )

    def evidence_issues(self, samples: object) -> list[str]:
        return _action_evidence_issues(self, samples)

    def evidence_schema_issues(
        self,
        samples: object,
        sample_count: int,
    ) -> list[str]:
        if not isinstance(samples, list):
            return ["causal_sample_evidence 必须是数组"]
        issues = []
        if len(samples) != sample_count:
            issues.append(
                f"causal_sample_evidence 应有 {sample_count} 项，实际 {len(samples)} 项"
            )
        for index, sample in enumerate(samples, start=1):
            if not isinstance(sample, dict):
                issues.append(f"因果证据第 {index} 项必须是对象")
                continue
            current_valid = all(
                isinstance(sample.get(field), bool)
                for field in self.evidence_fields
            )
            legacy_valid = all(
                isinstance(sample.get(field), bool)
                for field in LEGACY_ACTION_EVIDENCE_FIELDS
            )
            if not current_valid and not legacy_valid:
                invalid = [
                    field
                    for field in self.evidence_fields
                    if not isinstance(sample.get(field), bool)
                ]
                issues.append(
                    f"因果证据第 {index} 项缺少布尔字段: " + ", ".join(invalid)
                )
        return issues

    def evidence_json_schema(self, sample_count: int) -> dict:
        sample_schema = {
            "type": "object",
            "properties": {
                field: {"type": "boolean"}
                for field in self.evidence_fields
            },
            "required": list(self.evidence_fields),
            "additionalProperties": False,
        }
        return {
            "type": "array",
            "items": sample_schema,
            "minItems": sample_count,
            "maxItems": sample_count,
        }

    def retake_instruction(self, reason: str) -> str:
        mode = self.mode if self.mode in CAUSAL_INTERACTION_MODES else "non-effect"
        detail = str(reason or "the take did not satisfy its visible evidence").strip()
        return (
            f"[Targeted retake — enforce the {self.phase} {mode} contract. "
            f"The previous take failed only because: {detail}. Correct only that evidence "
            "failure while preserving accepted identity, environment, composition, and action "
            "state; do not add new actions.]"
        )

    def safe_retake_instruction(self) -> str:
        """Bounded retry direction for moderated prompts; never repeats failure prose."""
        mode = self.mode if self.mode in CAUSAL_INTERACTION_MODES else "non-effect"
        return (
            f"Targeted retake: enforce the contracted {self.phase} {mode} evidence only; "
            "preserve accepted scene state and do not add another action"
        )


def interaction_geometry(shot: dict) -> dict:
    value = shot.get("interaction_geometry")
    return value if isinstance(value, dict) else {}


def _is_none_like(value: object) -> bool:
    return str(value or "").strip().casefold().replace("_", " ") in _NONE_LIKE_VALUES


def interaction_mode(shot: dict) -> str:
    return str(interaction_geometry(shot).get("interaction_mode", "none")).strip()


def compile_action_contract(shot: dict) -> CompiledActionContract:
    """Compile one shot into the only action semantics consumed downstream."""
    raw_geometry = interaction_geometry(shot)
    raw_phase = str(raw_geometry.get("effect_phase", "")).strip()
    geometry = (
        with_causal_mode_invariants(raw_geometry)
        if raw_phase in EFFECT_PHASES
        else deepcopy(raw_geometry)
    )
    phase = str(geometry.get("effect_phase", "none")).strip()
    mode = str(geometry.get("interaction_mode", "none")).strip()
    scope = str(geometry.get("outcome_scope", "none")).strip()
    motion = str(geometry.get("effect_motion", "none")).strip()
    prompt_constraint = _compile_action_prompt(
        shot,
        geometry=geometry,
        phase=phase,
        mode=mode,
        scope=scope,
        motion=motion,
    )
    review_projection = _compile_phase_review_projection(shot, phase, geometry)
    return CompiledActionContract(
        phase=phase,
        mode=mode,
        outcome_scope=scope,
        effect_motion=motion,
        requires_evidence=phase in {"setup", "active", "aftermath"},
        prompt_constraint=prompt_constraint,
        review_instruction=_compile_action_review_instruction(phase, mode),
        prompt_parts=_compile_phase_prompt_parts(
            shot,
            phase=phase,
            prompt_constraint=prompt_constraint,
            narrative_constraint=narrative_prompt_constraint(review_projection),
        ),
        review_projection=review_projection,
        canonical_geometry=deepcopy(geometry),
        prompt_start_state=(
            _setup_state_summary(shot.get("start_state"))
            if phase == "setup"
            else _state_summary(shot.get("start_state"), include_open_motion=True)
        ),
        contracted_visible_result=_contracted_visible_result(shot, phase),
    )


def requires_causal_review(shot: dict) -> bool:
    contract = compile_action_contract(shot)
    return contract.requires_evidence or contract.mode in CAUSAL_INTERACTION_MODES


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
    if mode not in CAUSAL_INTERACTION_MODES:
        if geometry.get("line_of_action_visible"):
            mode = "directed_path"
            geometry["interaction_mode"] = mode
        elif geometry.get("must_share_frame"):
            mode = "direct_contact"
            geometry["interaction_mode"] = mode
    target = str(geometry.get("target", "")).strip()
    if not scope or scope in {"unspecified", "none"}:
        if target and target.casefold() != "none":
            scope = "single"
            geometry["outcome_scope"] = scope
    if not str(geometry.get("source", "")).strip():
        actor = str(geometry.get("actor", "")).strip()
        if actor:
            geometry["source"] = actor
    if not str(geometry.get("effect_region", "")).strip() and target:
        geometry["effect_region"] = target
    if not str(geometry.get("reaction_scope", "")).strip() and target:
        geometry["reaction_scope"] = target
    if not str(geometry.get("unaffected_behavior", "")).strip() and target:
        geometry["unaffected_behavior"] = (
            "entities outside the reaction scope remain unchanged"
        )
    if phase == "active" and mode == "directed_path" and scope == "all":
        geometry["effect_motion"] = "sweep"
    elif phase == "active" and scope == "single" and mode in {
        "direct_contact", "directed_path",
    }:
        geometry["effect_motion"] = "static"
    elif phase == "active" and effect_motion in {"", "unspecified", "none"}:
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
    if phase == "active" and scope == "single":
        # A target may be a collection identifier.  "single" is a relation in
        # the visible effect geometry, never an alias for that collection.
        geometry["reaction_scope"] = (
            "one clearly isolated intended target within the visible effect region"
        )
        geometry["unaffected_behavior"] = (
            "all entities outside that one intended target continue their prior motion"
        )
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
    duration = shot.get("duration")
    if (
        str(geometry.get("effect_phase", "")).strip() == "active"
        and mode == "directed_path"
        and str(geometry.get("effect_motion", "")).strip() == "sweep"
        and str(geometry.get("outcome_scope", "")).strip() in {"subset", "all"}
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and duration < 6
    ):
        issues.append(
            f"Shot {shot_id}: 短镜头多目标定向扫掠过载；改为 "
            "outcome_scope=single 且 effect_motion=static，或将 duration 提高到至少 6 秒"
        )
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
        if (
            phase not in EFFECT_PHASES
            or scope not in OUTCOME_SCOPES
            or motion not in EFFECT_MOTIONS
        ):
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
            if _is_none_like(geometry.get("reaction_scope")):
                issues.append(
                    f"Shot {shot_id}: 生效阶段 reaction_scope 必须声明实际可反应范围"
                )
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
    """Compatibility entry point for the deep Action Contract interface."""
    return compile_action_contract(shot).evidence_issues(samples)


def _action_evidence_issues(
    contract: CompiledActionContract,
    samples: object,
) -> list[str]:
    if not contract.requires_evidence:
        return []
    if not isinstance(samples, list) or not samples:
        return ["缺少逐采样因果证据"]

    normalized: list[dict[str, bool]] = []
    issues: list[str] = []
    for index, sample in enumerate(samples, start=1):
        evidence = _normalize_action_evidence_sample(contract.phase, sample)
        if evidence is None:
            issues.append(f"采样 {index} 因果证据字段缺失或类型无效")
        else:
            normalized.append(evidence)
    if issues:
        return issues

    seen = {field: False for field in ACTION_EVIDENCE_FIELDS}
    first: dict[str, int | None] = {
        "physical_effect_visible": None,
        "effect_reaches_target": None,
        "target_reaction_visible": None,
        "narrative_outcome_visible": None,
    }
    has_reached_target = False
    for index, sample in enumerate(normalized, start=1):
        for field in ACTION_EVIDENCE_FIELDS:
            seen[field] |= sample[field]
        for field in first:
            legacy_precontact_reaction = (
                field == "target_reaction_visible"
                and sample.get("_legacy_evidence", False)
                and not sample["effect_reaches_target"]
            )
            if (
                sample[field]
                and first[field] is None
                and not legacy_precontact_reaction
            ):
                first[field] = index

        if contract.phase == "active" and sample["target_reaction_visible"]:
            if (
                not sample["effect_reaches_target"]
                and not has_reached_target
                and (
                    not sample.get("_legacy_evidence", False)
                    or sample["out_of_scope_reaction_visible"]
                )
            ):
                issues.append(f"采样 {index} 的反应目标未与作用区域相交")
        if contract.phase != "setup" and sample["out_of_scope_reaction_visible"]:
            issues.append(f"采样 {index} 出现范围外目标发生反应")
        has_reached_target |= sample["effect_reaches_target"]

    narrative_index = first["narrative_outcome_visible"]
    valid_connection_index = next(
        (
            index
            for index, sample in enumerate(normalized, start=1)
            if narrative_index is not None
            and index >= narrative_index
            and sample["outcome_causally_connected"]
        ),
        None,
    )
    if contract.phase == "setup":
        if seen["physical_effect_visible"]:
            issues.append("准备阶段提前出现物理作用")
        if seen["effect_reaches_target"] or seen["target_reaction_visible"]:
            issues.append("准备阶段提前作用于目标")
        if seen["narrative_outcome_visible"]:
            issues.append("准备阶段提前出现约定结果（叙事结果）")
        if not seen["phase_endpoint_visible"]:
            issues.append("准备阶段未看到约定的准备端点")
        return issues

    if contract.phase == "active":
        if not seen["physical_effect_visible"]:
            issues.append("生效阶段未看到物理作用")
        if not seen["effect_reaches_target"]:
            issues.append("生效阶段未看到物理作用到达目标")
        if not seen["target_reaction_visible"] or not seen["effect_reaches_target"]:
            issues.append("生效阶段未看到作用区域内目标的同步反应")
        if not seen["phase_endpoint_visible"]:
            issues.append("active 阶段未看到约定的动作端点")
        if not seen["narrative_outcome_visible"]:
            issues.append("active 阶段未看到约定结果（叙事结果）及其完整作用范围")
        if valid_connection_index is None:
            issues.append(
                "active 阶段的约定结果与物理作用之间缺少可见因果过渡"
            )
        if (
            first["physical_effect_visible"] is not None
            and first["narrative_outcome_visible"] is not None
            and first["narrative_outcome_visible"] < first["physical_effect_visible"]
        ):
            issues.append("active 阶段的叙事结果先于物理原因出现")
        if (
            first["effect_reaches_target"] is not None
            and first["physical_effect_visible"] is not None
            and first["effect_reaches_target"] < first["physical_effect_visible"]
        ):
            issues.append("active 阶段的接触先于物理作用出现")
        if (
            first["target_reaction_visible"] is not None
            and (
                first["effect_reaches_target"] is None
                or first["target_reaction_visible"] < first["effect_reaches_target"]
            )
        ):
            issues.append("active 阶段的目标反应先于可见接触出现")
        if (
            first["narrative_outcome_visible"] is not None
            and (
                first["target_reaction_visible"] is None
                or first["narrative_outcome_visible"] < first["target_reaction_visible"]
            )
        ):
            issues.append("active 阶段的叙事结果先于可见接触或目标反应出现")
        return issues

    if seen["physical_effect_visible"] or seen["effect_reaches_target"]:
        issues.append("aftermath 阶段出现新物理作用")
    if not seen["phase_endpoint_visible"]:
        issues.append("aftermath 阶段未看到约定的结果端点")
    if not seen["narrative_outcome_visible"]:
        issues.append("aftermath 阶段未看到约定结果（叙事结果）及其完整作用范围")
    return issues


def _normalize_action_evidence_sample(
    phase: str,
    sample: object,
) -> dict[str, bool] | None:
    if not isinstance(sample, dict):
        return None
    if all(isinstance(sample.get(field), bool) for field in ACTION_EVIDENCE_FIELDS):
        evidence = {
            **{field: sample[field] for field in ACTION_EVIDENCE_FIELDS},
            "_legacy_evidence": False,
        }
        return _apply_evidence_invariants(evidence)

    if not all(
        isinstance(sample.get(field), bool)
        for field in LEGACY_ACTION_EVIDENCE_FIELDS
    ):
        return None
    preparation_only = (
        phase == "setup"
        and not sample["physical_effect_visible"]
        and not sample["contracted_outcome_visible"]
    )
    evidence = {
        "preparation_state_visible": preparation_only,
        "non_physical_cue_visible": False,
        "physical_effect_visible": sample["physical_effect_visible"],
        "effect_reaches_target": sample["effect_intersects_reaction"],
        "target_reaction_visible": (
            sample["reaction_visible"] if phase != "setup" else False
        ),
        "out_of_scope_reaction_visible": sample["out_of_scope_reaction_visible"],
        "phase_endpoint_visible": (
            preparation_only or sample["contracted_outcome_visible"]
        ),
        "narrative_outcome_visible": sample["contracted_outcome_visible"],
        "outcome_causally_connected": sample["outcome_causally_connected"],
        "_legacy_evidence": True,
    }
    return _apply_evidence_invariants(evidence)


def _apply_evidence_invariants(evidence: dict[str, bool]) -> dict[str, bool]:
    """Enforce causal prerequisites before phase-specific verdicts consume evidence."""
    if not evidence["physical_effect_visible"]:
        evidence["effect_reaches_target"] = False
    return evidence


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

    characters = {
        str(name).strip() for name in shot.get("characters", []) if str(name).strip()
    }
    for subject, counterpart in ((actor, target), (target, actor)):
        if subject not in characters:
            continue
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
    if (
        geometry.get("must_share_frame")
        and geometry.get("actor")
        and geometry.get("target")
    ):
        return True
    return any(
        isinstance(beat, dict)
        and str(beat.get("actor", "")).strip()
        and str(beat.get("target", "")).strip()
        and str(beat.get("visible_result", "")).strip()
        for beat in shot.get("action_beats", [])
    )


def causality_prompt_constraints(shot: dict) -> str:
    """Compatibility entry point for the deep Action Contract interface."""
    return compile_action_contract(shot).prompt_constraint


def _compile_action_prompt(
    shot: dict,
    *,
    geometry: dict,
    phase: str,
    mode: str,
    scope: str,
    motion: str,
) -> str:
    parts: list[str] = []
    actor = str(geometry.get("actor", "")).strip()
    target = str(geometry.get("target", "")).strip()
    if actor and target:
        geometry_parts = [f"interaction geometry {actor} toward {target}"]
        if geometry.get("must_share_frame"):
            geometry_parts.append("actor and target must share the frame")
        if geometry.get("line_of_action_visible"):
            geometry_parts.append("keep the line of action clearly visible")
        if geometry.get("occlusion_policy") == "none":
            geometry_parts.append("neither subject may be occluded")
        parts.append(", ".join(geometry_parts))

    contracted_result = _contracted_visible_result(shot, phase)
    if phase == "setup":
        parts.append(
            "effect phase setup: show preparation, aiming, or charging only; "
            "a non-physical cue such as a guidance light, targeting indicator, charge "
            "signal, or sensor display may visibly confirm the preparation and phase "
            "endpoint; no emitted Physical Effect, impact, damage, target reaction, or "
            "Narrative Outcome"
        )
        unaffected = str(geometry.get("unaffected_behavior", "")).strip()
        if unaffected and not _is_none_like(unaffected):
            parts.append(
                "target motion is independent of the preparation and may only continue "
                f"this prior behavior: {unaffected}"
            )
    elif phase == "aftermath":
        parts.append(
            f"effect phase aftermath: show only the already established {scope} outcome; "
            "do not create a new effect or expand the affected scope"
        )
        if contracted_result:
            parts.append(
                "only the permitted scope may show this exact contracted result: "
                f"{contracted_result}"
            )
    elif mode in CAUSAL_INTERACTION_MODES:
        fields = [
            ("source", "show the cause originating from"),
            ("effect_region", "confine the Physical Effect to"),
        ]
        if scope == "single":
            fields.extend([
                ("reaction_scope", "only the single intended target"),
                ("unaffected_behavior", "outside that single target"),
            ])
        else:
            fields.extend([
                ("reaction_scope", "only this reaction scope may respond"),
                ("unaffected_behavior", "outside that scope"),
            ])
        details = [
            f"{label} {str(geometry.get(field, '')).strip()}"
            for field, label in fields
            if str(geometry.get(field, "")).strip()
        ]
        parts.append(
            f"effect phase {phase}, outcome scope {scope}, effect motion {motion}; "
            f"{mode} cause-and-effect: " + ", ".join(details)
        )
        if contracted_result:
            parts.append(
                "only the permitted scope may show this exact contracted result: "
                f"{contracted_result}"
            )

    if phase not in {"setup", "aftermath"}:
        compiled_beats = []
        for beat in shot.get("action_beats", []):
            if not isinstance(beat, dict):
                continue
            text = f"{beat.get('phase')}: {beat.get('actor')} {beat.get('action')}"
            if beat.get("target"):
                text += f" toward {beat['target']}"
            compiled_beats.append(text)
        if compiled_beats:
            parts.append("causal action phases " + "; ".join(compiled_beats))
    return "; ".join(parts)


def _contracted_visible_result(shot: dict, phase: str) -> str:
    """Select the one structured endpoint owned by an active/aftermath Contract."""
    if phase not in {"active", "aftermath"}:
        return ""
    results = [
        str(beat.get("visible_result", "")).strip()
        for beat in shot.get("action_beats", [])
        if isinstance(beat, dict) and str(beat.get("visible_result", "")).strip()
    ]
    return results[-1] if results else ""


def _compile_phase_prompt_parts(
    shot: dict,
    *,
    phase: str,
    prompt_constraint: str,
    narrative_constraint: str,
) -> tuple[str, ...]:
    """Project free-form shot fields through the selected Action Contract phase."""
    primary_action = str(shot.get("primary_action", "")).strip()
    parts: list[str] = []

    if phase == "setup":
        geometry = interaction_geometry(shot)
        actor = str(geometry.get("actor", "the primary subject")).strip()
        target = str(geometry.get("target", "the intended subject")).strip()
        parts.append(
            f"perform only this non-physical preparation: {actor} visibly prepares toward {target}"
        )
        if prompt_constraint:
            parts.append(prompt_constraint)
        if narrative_constraint:
            parts.append(narrative_constraint)
        endpoint = _setup_state_summary(shot.get("end_state"))
        if endpoint:
            parts.append(f"finish at this preparation phase endpoint: {endpoint}")
        return tuple(parts)

    if phase == "aftermath":
        if prompt_constraint:
            parts.append(prompt_constraint)
        if narrative_constraint:
            parts.append(narrative_constraint)
        parts.append("finish with only the already established contracted outcome")
        return tuple(parts)

    if primary_action:
        parts.append(f"perform only this primary action: {primary_action}")
    if prompt_constraint:
        parts.append(prompt_constraint)
    if narrative_constraint:
        parts.append(narrative_constraint)
    if phase == "active":
        parts.append("finish with only the contracted outcome and its permitted scope")
    else:
        endpoint = _state_summary(shot.get("end_state"), include_open_motion=True)
        if endpoint:
            parts.append(f"finish with {endpoint}")
    return tuple(parts)


def _compile_phase_review_projection(
    shot: dict,
    phase: str,
    geometry: dict,
) -> dict[str, object]:
    """Return the only phase-sensitive fields the reviewer may consume."""
    projection = {
        key: deepcopy(shot.get(key))
        for key in (
            "primary_action", "action_beats", "narrative_beat", "start_state", "end_state",
        )
    }
    projection["interaction_geometry"] = deepcopy(geometry)
    if phase == "setup":
        actor = str(interaction_geometry(shot).get("actor", "the actor")).strip()
        target = str(interaction_geometry(shot).get("target", "the intended subject")).strip()
        start_state = _setup_visual_state(shot.get("start_state"))
        end_state = _setup_visual_state(shot.get("end_state"))
        end_state["prop_state"] = "a non-physical preparation cue is active"
        endpoint = _setup_state_summary(end_state)
        projection.update({
            "primary_action": (
                f"{actor} visibly prepares toward {target} without a Physical Effect"
            ),
            "action_beats": [],
            "start_state": start_state,
            "narrative_beat": {
                "function": "setup",
                "state_before": "the preparation is not yet complete",
                "state_change": (
                    f"{actor} reaches the contracted non-physical preparation endpoint"
                ),
                "state_after": endpoint or f"{actor} is visibly prepared",
            },
            "end_state": end_state,
        })
    elif phase == "active":
        actor = str(geometry.get("actor", "the actor")).strip()
        target = str(geometry.get("target", "the intended target")).strip()
        contracted_result = _contracted_visible_result(shot, phase)
        beats = []
        for beat in shot.get("action_beats", []):
            if not isinstance(beat, dict):
                continue
            canonical_beat = {
                key: deepcopy(value)
                for key, value in beat.items()
                if key != "visible_result"
            }
            if contracted_result:
                canonical_beat["visible_result"] = contracted_result
            beats.append(canonical_beat)
        narrative = shot.get("narrative_beat")
        narrative = narrative if isinstance(narrative, dict) else {}
        projection.update({
            "action_beats": beats,
            "narrative_beat": {
                "function": str(narrative.get("function", "progress")),
                "state_before": "the contracted action has not yet completed",
                "state_change": (
                    f"{actor} produces the exact contracted result on only the permitted target scope"
                ),
                "state_after": (
                    contracted_result
                    if contracted_result
                    else f"the contracted result is visible only for {target} within the permitted scope"
                ),
            },
            "end_state": {
                "action_phase": (
                    contracted_result
                    if contracted_result
                    else "only the contracted outcome and permitted scope are visible"
                )
            },
        })
    elif phase == "aftermath":
        end_state = _filtered_state(shot.get("end_state"), include_open_motion=False)
        endpoint = _state_summary(end_state, include_open_motion=False)
        narrative = shot.get("narrative_beat")
        narrative = narrative if isinstance(narrative, dict) else {}
        projection.update({
            "primary_action": "show only the already established result",
            "action_beats": [],
            "start_state": _filtered_state(
                shot.get("start_state"), include_open_motion=False
            ),
            "narrative_beat": {
                "function": str(narrative.get("function", "payoff")),
                "state_before": str(narrative.get("state_before", "")),
                "state_change": "the already established result remains visible",
                "state_after": endpoint,
            },
            "end_state": end_state,
        })
    return projection


_STATE_PROMPT_KEYS = (
    "location",
    "subject",
    "action_phase",
    "camera",
    "screen_direction",
    "pose_and_gaze",
    "prop_state",
    "open_motion",
)


def _filtered_state(state: object, *, include_open_motion: bool) -> dict[str, object]:
    if not isinstance(state, dict):
        return {}
    allowed = set(_STATE_PROMPT_KEYS)
    if not include_open_motion:
        allowed.remove("open_motion")
    return {
        key: deepcopy(value)
        for key, value in state.items()
        if key in allowed
    }


_SETUP_VISUAL_STATE_KEYS = (
    "location", "subject", "camera", "screen_direction", "pose_and_gaze",
)


def _setup_visual_state(state: object) -> dict[str, object]:
    """Preserve visual continuity without allowing state text to define an effect."""
    if not isinstance(state, dict):
        return {}
    return {
        key: deepcopy(value)
        for key, value in state.items()
        if key in _SETUP_VISUAL_STATE_KEYS
    }


def _setup_state_summary(state: object) -> str:
    filtered = _setup_visual_state(state)
    return ", ".join(
        str(filtered.get(key, "")).strip()
        for key in _SETUP_VISUAL_STATE_KEYS
        if str(filtered.get(key, "")).strip()
    )


def _state_summary(state: object, *, include_open_motion: bool) -> str:
    filtered = _filtered_state(state, include_open_motion=include_open_motion)
    values = [str(filtered.get(key, "")).strip() for key in _STATE_PROMPT_KEYS]
    return ", ".join(value for value in values if value)


def causality_review_instruction(shot: dict) -> str:
    """Compatibility entry point for the deep Action Contract interface."""
    return compile_action_contract(shot).review_instruction


def _compile_action_review_instruction(phase: str, mode: str) -> str:
    if phase == "setup":
        return (
            "This is a setup phase. A visible preparation state, non-physical cue, and "
            "phase endpoint are permitted and do not count as a Physical Effect or "
            "narrative outcome. "
            f"{PHYSICAL_EFFECT_EVIDENCE_RULE}. Reject effect_path_valid and "
            "reaction_causality_valid if a Physical Effect reaches a target, an impact "
            "occurs, a target reacts, or a narrative outcome appears before active phase. "
        )
    if phase == "aftermath":
        return (
            "This is an aftermath phase. It may show only the outcome scope already "
            "established by the prior active phase; reject any new unexplained effect or "
            "larger affected population. "
        )
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

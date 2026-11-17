"""Validated data contracts shared by the generation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

import config
from pipeline.action_results import SceneChangeSpec, scene_change_issues
from pipeline.motion_timing import MotionTimingSpec
from pipeline.creative_direction import CreativeDirectionSpec
from pipeline.completion_intent import CompletionIntent, CompletionGoal
from pipeline.action_vocabulary import Activity, ResponseKind
from pipeline.intent_evidence import Evidence, RequiredEvidence, evidence_grounded, has_evidence
from pipeline.production_shape import DurationLimitsSpec


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_movement: str = "fixed"
    composition: str = ""
    start_framing: str = ""
    end_framing: str = ""
    speed: str = "fixed"
    screen_positions: dict[str, str] = Field(default_factory=dict)
    axis_change: Literal["establish", "hold", "reestablish"] = "hold"

    @field_validator("axis_change", mode="before")
    @classmethod
    def normalize_axis_change(cls, value: Any) -> Any:
        if value is None:
            return "hold"
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "none": "hold",
            "no change": "hold",
            "no_change": "hold",
            "same axis": "hold",
            "same_axis": "hold",
            "unchanged": "hold",
            "maintain": "hold",
            "maintain_axis": "hold",
            "hold_axis": "hold",
            "re-establish": "reestablish",
            "re_establish": "reestablish",
            "re establish": "reestablish",
            "reset_axis": "reestablish",
        }
        return aliases.get(normalized, normalized)


class CharacterBlockingSpec(BaseModel):
    """Visible spatial intent for one character in one shot."""

    model_config = ConfigDict(extra="forbid")

    frame_position: str = ""
    body_orientation: str = ""
    facing_target: str = ""
    eyeline_target: str = ""
    travel_direction: str = ""
    action_target: str = ""


class VisibleOutcomeSpec(BaseModel):
    """Observable state obligation, separate from the director's chosen maneuver."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    entity: str = Field(min_length=1, max_length=1600)
    state: str = Field(min_length=1, max_length=1600)


class ActionBeatSpec(BaseModel):
    """One causal phase inside a shot, kept separate from camera movement."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["trigger", "peak", "aftermath"]
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target: str = ""
    visible_result: str = ""
    outcomes: list[VisibleOutcomeSpec] | None = Field(default=None, min_length=1, max_length=16)
    scene_changes: list[SceneChangeSpec] | None = Field(default=None, max_length=14)
    result_directions: list[str] | None = Field(default=None, max_length=30)
    result_functions: list[str] | None = Field(default=None, max_length=30)
    completion_intent: CompletionIntent | None = None
    start_seconds: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    end_seconds: float | None = Field(default=None, gt=0, strict=True, allow_inf_nan=False)
    activity: Activity | None = None
    interaction_geometry: InteractionGeometrySpec | None = None
    effect_at_seconds: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    reaction_at_seconds: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    response_kind: ResponseKind | None = None
    flow_from_previous: str | None = Field(default=None, min_length=1, max_length=800)
    temporal_relation: Literal['sequential', 'overlap_previous'] | None = None

    @model_validator(mode='after')
    def owned_outcomes(self):
        if self.outcomes is not None:
            entities = [outcome.entity for outcome in self.outcomes]
            if (self.activity != 'interaction' or self.target not in entities
                    or (self.scene_changes is None and len(entities) != len(set(entities)))
                    or not set(entities) <= {self.actor, self.target}):
                raise ValueError('可见结果只能属于本次交互参与者且必须覆盖目标')
        return self

    @field_validator("actor", "action", "target", "visible_result", mode="before")
    @classmethod
    def strip_beat_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("interaction_geometry")
    @classmethod
    def canonicalize_empty_event_geometry(cls, value: InteractionGeometrySpec | None) -> InteractionGeometrySpec | None:
        # 显式全空的 none 对象与缺省表示同一含义；任何实际作用字段都必须保留供校验。
        empty = InteractionGeometrySpec(effect_phase="none", outcome_scope="none", effect_motion="none")
        return None if value == empty else value


class CharacterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    mobility: Literal[
        "unspecified", "bipedal", "quadruped", "tracked", "wheeled",
        "flying", "stationary", "other",
    ] = "unspecified"
    reference_mode: Literal["identity", "group", "none"] = "identity"

    @field_validator("mobility", mode="before")
    @classmethod
    def normalize_mobility(cls, value: Any) -> str:
        label = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
        canonical = {
            "unspecified", "bipedal", "quadruped", "tracked", "wheeled",
            "flying", "stationary", "other",
        }
        if label in canonical:
            return label
        mappings = (
            ("tracked", ("track", "tread", "crawler", "tank", "履带")),
            ("wheeled", ("wheel", "vehicle", "car", "轮式", "车轮")),
            ("quadruped", ("quadrup", "four leg", "animal", "四足")),
            ("bipedal", ("biped", "human", "humanoid", "two leg", "双足", "人形")),
            ("flying", ("fly", "airborne", "wing", "drone", "飞行")),
            ("stationary", ("stationary", "fixed", "turret", "immobile", "固定")),
        )
        for mobility, markers in mappings:
            if any(marker in label for marker in markers):
                return mobility
        return "other" if label else "unspecified"

    @field_validator("reference_mode", mode="before")
    @classmethod
    def normalize_reference_mode(cls, value: Any) -> str:
        label = (
            str(value or "identity")
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        if label in {"identity", "group", "none"}:
            return label
        if any(marker in label for marker in (
            "group", "crowd", "horde", "ensemble", "mob", "群体", "人群", "尸群",
        )):
            return "group"
        if any(marker in label for marker in (
            "none", "background", "environment", "extra", "transient", "无需", "背景",
        )):
            return "none"
        return "identity"


class ContinuityStateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str = ""
    subject: str = ""
    action_phase: str = ""
    camera: str = ""
    screen_direction: str = ""
    pose_and_gaze: str = ""
    prop_state: str = ""
    open_motion: str = ""
    lighting: str = ""


class InteractionGeometrySpec(BaseModel):
    """What the audience must be able to read in an interaction shot."""

    model_config = ConfigDict(extra="forbid")

    actor: str = ""
    target: str = ""
    interaction_mode: Literal[
        "none", "direct_contact", "directed_path", "area_effect", "indirect_effect"
    ] = "none"
    effect_phase: Literal[
        "unspecified", "none", "setup", "active", "aftermath"
    ] = "unspecified"
    outcome_scope: Literal[
        "unspecified", "none", "single", "subset", "all"
    ] = "unspecified"
    effect_motion: Literal[
        "unspecified", "none", "static", "sweep", "expand", "propagate"
    ] = "unspecified"
    source: str = ""
    effect_region: str = ""
    reaction_scope: str = ""
    unaffected_behavior: str = ""
    must_share_frame: bool = False
    line_of_action_visible: bool = False
    actor_screen_position: str = ""
    target_screen_position: str = ""
    occlusion_policy: Literal["none", "partial", "motivated"] = "none"

    @field_validator("interaction_mode", mode="before")
    @classmethod
    def normalize_interaction_mode(cls, value: Any) -> str:
        label = str(value or "none").strip().lower().replace("-", " ").replace("_", " ")
        aliases = {
            "none": "none",
            "direct contact": "direct_contact",
            "contact": "direct_contact",
            "directed path": "directed_path",
            "path": "directed_path",
            "area effect": "area_effect",
            "area": "area_effect",
            "indirect effect": "indirect_effect",
            "indirect": "indirect_effect",
        }
        if label in aliases:
            return aliases[label]
        semantic_aliases = (
            ("direct_contact", ("contact", "collision", "melee", "touch", "impact")),
            ("directed_path", ("projectile", "beam", "ray", "line", "ranged", "path")),
            ("area_effect", ("area", "aoe", "blast", "explosion", "wave", "field")),
            ("indirect_effect", ("indirect", "trap", "chain", "environment", "mediated")),
        )
        for canonical, markers in semantic_aliases:
            if any(marker in label for marker in markers):
                return canonical
        return "none"

    @field_validator("effect_phase", mode="before")
    @classmethod
    def normalize_effect_phase(cls, value: Any) -> str:
        label = str(value or "unspecified").strip().lower().replace("-", " ").replace("_", " ")
        if label in {"none", "no effect", "non causal"}:
            return "none"
        if any(marker in label for marker in (
            "setup", "prepare", "preparation", "aim", "target", "charge", "trigger"
        )):
            return "setup"
        if any(marker in label for marker in (
            "active", "fire", "impact", "contact", "peak", "attack"
        )):
            return "active"
        if any(marker in label for marker in (
            "aftermath", "result", "resolution", "payoff", "resolved"
        )):
            return "aftermath"
        return "unspecified"

    @field_validator("outcome_scope", mode="before")
    @classmethod
    def normalize_outcome_scope(cls, value: Any) -> str:
        label = str(value or "unspecified").strip().lower().replace("-", " ").replace("_", " ")
        if label in {"none", "no effect", "zero"}:
            return "none"
        if any(marker in label for marker in ("single", "one target", "one subject")):
            return "single"
        if any(marker in label for marker in ("subset", "partial", "some", "several")):
            return "subset"
        if any(marker in label for marker in ("all", "whole", "entire", "every")):
            return "all"
        return "unspecified"

    @field_validator("effect_motion", mode="before")
    @classmethod
    def normalize_effect_motion(cls, value: Any) -> str:
        label = str(value or "unspecified").strip().lower().replace("-", " ").replace("_", " ")
        if label in {"none", "no motion"}:
            return "none"
        if any(marker in label for marker in ("sweep", "scan", "pan across")):
            return "sweep"
        if any(marker in label for marker in ("expand", "radial", "spread outward")):
            return "expand"
        if any(marker in label for marker in ("propagate", "chain", "travel through")):
            return "propagate"
        if any(marker in label for marker in ("static", "fixed", "straight")):
            return "static"
        return "unspecified"

    @field_validator("occlusion_policy", mode="before")
    @classmethod
    def normalize_occlusion_policy(cls, value: Any) -> str:
        label = str(value or "none").strip().lower().replace("_", " ").replace("-", " ")
        if label in {"none", "partial", "motivated"}:
            return label
        if any(marker in label for marker in ("partial", "partly", "部分")):
            return "partial"
        if any(marker in label for marker in ("motivated", "intentional", "叙事")):
            return "motivated"
        return "none"


class StoryArcSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = ""
    stakes: str = ""
    turning_point: str = ""
    resolution: str = ""


class NarrativeBeatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Literal["setup", "progress", "turn", "payoff"] | None = None
    state_before: str = ""
    state_change: str = ""
    state_after: str = ""

    @field_validator("function", mode="before")
    @classmethod
    def normalize_function(cls, value: Any) -> Any:
        if value is None:
            return None
        label = str(value).strip().lower().replace("-", " ").replace("_", " ")
        aliases = (
            ("setup", ("setup", "opening", "establish", "inciting", "介绍", "建立")),
            ("turn", ("turn", "turning", "reversal", "reveal", "转折", "揭示")),
            ("payoff", ("payoff", "resolution", "resolve", "ending", "结果", "收束")),
            ("progress", ("progress", "development", "escalation", "过程", "推进")),
        )
        for canonical, markers in aliases:
            if any(marker in label for marker in markers):
                return canonical
        return "progress"


class DirectorIntentSpec(BaseModel):
    """Frozen semantic reading with evidence from the unmodified user request."""

    model_config = ConfigDict(extra="forbid")
    source_request: str = Field(min_length=1, max_length=16000)
    content_focus: Literal["action", "product", "balanced"]
    pacing: Literal["dense", "balanced", "atmospheric"]
    intent_evidence: RequiredEvidence
    ending_requirement: str = Field(default="", max_length=1200)
    ending_evidence: Evidence = ''
    motion_control: Literal["text_guided", "reference_motion"] = "text_guided"
    motion_control_evidence: Evidence = ''
    choreography: Literal["exact", "flexible"] | None = None
    requested_duration_seconds: int | None = Field(default=None, strict=True, ge=4, le=600)
    duration_evidence: RequiredEvidence | None = None
    duration_mode: Literal['exact', 'approximate'] | None = None
    requested_shot_count: int | None = Field(default=None, strict=True, ge=1, le=150)
    shot_count_evidence: RequiredEvidence | None = None
    motion_timing: MotionTimingSpec | None = None
    style_input: str | None = Field(default=None, min_length=1, max_length=1600)
    creative_direction: CreativeDirectionSpec | None = None
    completion_goals: list[CompletionGoal] | None = Field(default=None, min_length=1, max_length=150)

    # 程序冻结执行协议；缺省表示历史流程，不由模型选择或恢复时推断。
    authoring_protocol: Literal['scene-units-v1'] | None = None

    @model_serializer(mode='wrap')
    def serialize_intent(self, handler):
        data = handler(self)
        for name in ('motion_timing', 'style_input', 'creative_direction', 'authoring_protocol', 'completion_goals', 'duration_mode'):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    @model_validator(mode="after")
    def grounded_intent(self) -> "DirectorIntentSpec":
        if self.completion_goals:
            if any(not evidence_grounded(goal.evidence, self.source_request) for goal in self.completion_goals):
                raise ValueError('可观察目标必须逐字引用用户原需求')
        if self.creative_direction and not evidence_grounded(
            self.creative_direction.evidence, self.source_request, self.style_input):
            raise ValueError('创作方向必须逐字引用用户需求或风格输入')
        if self.motion_timing and not evidence_grounded(self.motion_timing.evidence, self.source_request):
            raise ValueError('动作速度要求必须逐字引用用户原文')
        for value, evidence in ((self.requested_duration_seconds, self.duration_evidence),
                                (self.requested_shot_count, self.shot_count_evidence)):
            if (value is None) != (evidence is None):
                raise ValueError('时长或段数要求与原文依据必须成对提供')
            if evidence is not None and not evidence_grounded(evidence, self.source_request):
                raise ValueError('时长或段数要求必须逐字引用原文')
        if not evidence_grounded(self.intent_evidence, self.source_request):
            raise ValueError("意图依据必须逐字引用用户原文")
        if self.pacing == "dense" and self.content_focus != "action":
            raise ValueError("密集交互节奏必须与动作重心一致")
        if self.ending_requirement:
            if not evidence_grounded(self.ending_evidence, self.source_request):
                raise ValueError("结局要求必须有原文依据")
        elif has_evidence(self.ending_evidence):
            raise ValueError("结局依据缺少对应要求")
        if self.motion_control == "reference_motion":
            if not evidence_grounded(self.motion_control_evidence, self.source_request):
                raise ValueError("动作参考要求必须有原文依据")
        elif has_evidence(self.motion_control_evidence):
            raise ValueError("动作参考依据缺少对应要求")
        return self


class ProductionSlotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
    duration: int = Field(
        ge=config.MIN_SHOT_DURATION,
        le=config.MAX_SHOT_DURATION,
    )
    narrative_function: Literal["setup", "progress", "turn", "payoff"]
    allowed_effect_phases: list[
        Literal["none", "setup", "active", "aftermath"]
    ] = Field(min_length=1)
    outcome_scope: Literal["none", "single", "subset", "all"] | None = None
    requires_visible_result: bool
    min_interaction_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    min_interaction_events: int = Field(default=0, ge=0, le=12, strict=True)
    result_hold_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    required_outcome: str | None = Field(default=None, max_length=1200)
    choreography: Literal["exact", "flexible"] | None = None
    creative_brief: str | None = Field(default=None, max_length=16000)
    authoring_protocol: Literal['scene-units-v1'] | None = None
    motion_timing: MotionTimingSpec | None = None
    creative_direction: CreativeDirectionSpec | None = None
    coverage_roles: list[
        Literal[
            "establish", "action_subject", "target_reaction",
            "interaction", "aftermath", "insert",
        ]
    ] = Field(min_length=1)
    framing_family: Literal["wide", "medium", "close_detail"]
    reference_policy: Literal[
        "independent", "state_if_same_scene", "state_and_identity",
        "identity_or_state", "identity_only",
    ]

    @model_serializer(mode='wrap')
    def serialize_slot(self, handler):
        data = handler(self)
        for name in ('motion_timing', 'creative_direction', 'authoring_protocol'):
            if data.get(name) is None:
                data.pop(name, None)
        return data


class ProductionPlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["production-plan-v1", "production-plan-v2-short-payoff", "production-plan-v3-intent-timing", "production-plan-v4-events", "production-plan-v5-payoff-events", "production-plan-v6-performance", "production-plan-v7-film-rhythm", "production-plan-v8-request-shape"]
    min_interaction_seconds: float | None = Field(default=None, ge=0, strict=True, allow_inf_nan=False)
    director_intent: DirectorIntentSpec | None = None
    content_focus: Literal["balanced", "action", "product"]
    target_duration: int = Field(gt=0)
    planned_duration: int = Field(gt=0)
    duration_allocation: list[Annotated[int, Field(strict=True, ge=4, le=15)]] | None = None
    duration_limits: DurationLimitsSpec | None = None
    slots: list[ProductionSlotSpec] = Field(min_length=1)

    @model_serializer(mode='wrap')
    def preserve_legacy_capacity(self, handler):
        value = handler(self)
        if self.duration_limits is None:
            value.pop('duration_limits', None)
        return value


class MotionReferenceSpec(BaseModel):
    """Local snapshot plus an explicitly bound provider asset; never observed state."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model: Literal[
        "doubao-seedance-2-0-mini-260615",
        "doubao-seedance-2-0-fast-260128",
        "doubao-seedance-2-0-260128",
        "doubao-seedance-2.0-mini",
        "doubao-seedance-2.0-fast",
        "doubao-seedance-2.0",
    ]
    asset_uri: str | None = Field(default=None, pattern=r"^asset://[A-Za-z0-9_-]{1,128}$")
    video_url: str | None = None

    @model_validator(mode="after")
    def require_one_reference_location(self) -> "MotionReferenceSpec":
        if self.asset_uri is not None and self.video_url is not None:
            raise ValueError("素材 ID 与公网链接只能选择一个")
        if self.video_url is not None:
            from tools.reference_video_url import validate_public_video_url

            validate_public_video_url(self.video_url)
        return self


class ShotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
    motion_reference: MotionReferenceSpec | None = None
    duration: int = Field(
        default=config.DEFAULT_DURATION,
        ge=config.MIN_SHOT_DURATION,
        le=config.MAX_SHOT_DURATION,
    )
    # Empty keeps persisted pre-contract storyboards resumable; new storyboards
    # receive a stable ID at the LLM boundary in _apply_defaults.
    scene_id: str = ""
    scene_description: str = Field(min_length=1)
    prompt_en: str = Field(min_length=1)
    # 独立视觉设计不能只藏在旧 prompt_en 中；缺省不迁移历史镜头。
    visual_description: str | None = Field(default=None, min_length=1, max_length=1600)
    continuity_from_previous: Literal[
        "none", "seamless", "intentional_cut"
    ] = "none"
    composition_change: Literal["small", "medium", "large"] = "medium"
    result_admission: Literal['review-required-v1', 'reviewed-results-v1'] | None = None
    result_policy: Literal['functional-v1'] | None = None
    completion_policy: Literal['admitted-intent-v1', 'source-goals-v1', 'goal-allocation-v1'] | None = None
    completion_goals: dict[str, CompletionGoal] | None = None
    # 新编译稿显式启用；历史任务缺省保持原提示、合同指纹和恢复语义。
    generation_rhythm: Literal['allocated-v1'] | None = None
    generation_direction: Literal['authored-staging-v1'] | None = None
    generation_execution: Literal['local-events-v1', 'local-events-v2', 'scene-instructions-v1', 'scene-instructions-v2'] | None = None
    action_semantics: Literal['authored-v1'] | None = None
    execution_semantics: Literal['intent-bounded-v1'] | None = None
    media_selection: Literal['planned-prefix-v1'] | None = None
    continuity_evidence: Literal['temporal-window-v1'] | None = None
    evidence_policy: Literal['witness-decisions-v1', 'grounded-requirements-v2'] | None = None
    spatial_control: Literal['grounded-diagram-v1', 'observed-layout-v1'] | None = None
    presentation_policy: Literal['local-reframe-v1'] | None = None
    coverage_role: Literal[
        "establish", "action_subject", "target_reaction",
        "interaction", "aftermath", "insert",
    ] = "establish"
    required_visible_entities: list[str] = Field(default_factory=list)
    interaction_geometry: InteractionGeometrySpec = Field(
        default_factory=InteractionGeometrySpec
    )
    narrative_beat: NarrativeBeatSpec | None = None
    production_slot: ProductionSlotSpec | None = None
    primary_action: str = ""
    action_beats: list[ActionBeatSpec] = Field(default_factory=list, max_length=12)
    start_state: ContinuityStateSpec = Field(default_factory=ContinuityStateSpec)
    end_state: ContinuityStateSpec = Field(default_factory=ContinuityStateSpec)
    observed_end_state: ContinuityStateSpec | None = None
    camera: CameraSpec = Field(default_factory=CameraSpec)
    blocking: dict[str, CharacterBlockingSpec] = Field(default_factory=dict)
    lighting: str = ""
    mood: str = "cinematic"
    negative_prompt: str = "avoid jitter, stable motion, no text artifacts"
    subtitle_text: str = ""
    transition_to_next: str = "crossfade"
    generate_audio: bool = True
    characters: list[str] = Field(default_factory=list)
    extract_character_ref: bool = False
    key_props: list[str] = Field(default_factory=list)
    continuity_props: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def registered_scene_results(self):
        from pipeline.completion_intent import admitted_shot_intents
        admitted_shot_intents(self.model_dump(mode='json'))
        if any(beat.scene_changes is not None or beat.result_directions is not None
               or beat.result_functions is not None for beat in self.action_beats):
            issues = scene_change_issues(self.model_dump(mode='json'))
            if issues:
                raise ValueError('；'.join(issues))
        return self

    @field_validator("action_beats", mode="before")
    @classmethod
    def normalize_action_beat_phases(cls, value: Any) -> Any:
        """Normalize open-ended LLM labels before strict beat validation."""
        if not isinstance(value, list):
            return value
        normalized = []
        for index, beat in enumerate(value):
            if not isinstance(beat, dict):
                normalized.append(beat)
                continue
            item = dict(beat)
            item["phase"] = _normalize_action_phase(
                item.get("phase"), index=index, count=len(value)
            )
            normalized.append(item)
        return normalized

    @field_validator("composition_change", mode="before")
    @classmethod
    def normalize_composition_change(cls, value: Any) -> str:
        label = str(value or "medium").strip().lower().replace("_", " ").replace("-", " ")
        if label in {"small", "medium", "large"}:
            return label
        if any(marker in label for marker in ("minor", "slight", "subtle", "same", "小")):
            return "small"
        if any(marker in label for marker in ("major", "drastic", "extreme", "大")):
            return "large"
        return "medium"

    @field_validator("coverage_role", mode="before")
    @classmethod
    def normalize_coverage_role(cls, value: Any) -> str:
        label = str(value or "establish").strip().lower().replace("_", " ").replace("-", " ")
        aliases = (
            ("target_reaction", ("target reaction", "reaction", "受击", "反应")),
            ("interaction", ("interaction", "two shot", "combat", "交互", "交战")),
            ("action_subject", ("action subject", "action", "主体动作")),
            ("aftermath", ("aftermath", "result", "resolution", "ending", "结果", "收束")),
            ("insert", ("insert", "detail", "cutaway", "macro", "特写", "细节")),
            ("establish", ("establish", "establishing", "setup", "建立")),
        )
        for canonical, markers in aliases:
            if any(marker in label for marker in markers):
                return canonical
        return "establish"

    @field_validator("required_visible_entities", mode="before")
    @classmethod
    def normalize_required_visible_entities(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return value

    @field_validator("scene_description", "prompt_en", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class StorySpineShotSpec(BaseModel):
    """Compact global intent for one production slot."""

    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
    scene_id: str = Field(min_length=1)
    narrative_function: Literal["setup", "progress", "turn", "payoff"]
    state_before: str = Field(min_length=1)
    state_change: str = Field(min_length=1)
    state_after: str = Field(min_length=1)
    primary_action: str = Field(min_length=1)
    characters: list[str] = Field(default_factory=list)


class StorySpineSpec(BaseModel):
    """Global story authority compiled before detailed shot payloads."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    mood: str = "cinematic"
    music_style: str = "cinematic orchestral"
    theme_elements: list[str] = Field(default_factory=list)
    story_arc: StoryArcSpec
    characters: list[CharacterSpec] = Field(default_factory=list)
    shot_intents: list[StorySpineShotSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_ordered_shot_ids(self) -> "StorySpineSpec":
        ids = [intent.shot_id for intent in self.shot_intents]
        if ids != sorted(set(ids)):
            raise ValueError("shot_intents shot_id 必须唯一且递增")
        return self


def _normalize_action_phase(value: Any, *, index: int, count: int) -> str:
    """Map descriptive LLM phase labels to the three-phase production grammar."""
    label = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if label in {"trigger", "peak", "aftermath"}:
        return label

    semantic_markers = (
        (
            "aftermath",
            (
                "aftermath", "complete", "finish", "result", "resolve",
                "resolution", "settle", "recovery", "ending", "end",
                "完成", "结果", "收尾", "结束",
            ),
        ),
        (
            "trigger",
            (
                "trigger", "setup", "start", "begin", "opening", "init",
                "approach", "advance", "prepare", "windup", "起始", "触发",
                "准备", "推进",
            ),
        ),
        (
            "peak",
            (
                "peak", "impact", "climax", "contact", "strike", "attack",
                "execution", "main action", "高潮", "命中", "冲击",
            ),
        ),
    )
    for phase, markers in semantic_markers:
        if any(marker in label for marker in markers):
            return phase

    if count <= 1:
        return "peak"
    if index == 0:
        return "trigger"
    if index == count - 1:
        return "aftermath"
    return "peak"


class StoryboardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    total_duration: int | None = Field(default=None, gt=0)
    style: str = "cinematic"
    aspect_ratio: str = config.DEFAULT_RATIO
    resolution: str = config.DEFAULT_RESOLUTION
    mood: str = "cinematic"
    music_style: str = "cinematic orchestral"
    content_focus: Literal["balanced", "action", "product"] = "balanced"
    theme_elements: list[str] = Field(default_factory=list)
    story_arc: StoryArcSpec | None = None
    production_plan: ProductionPlanSpec | None = None
    characters: list[CharacterSpec] = Field(default_factory=list)
    shots: list[ShotSpec] = Field(min_length=1)

    @field_validator("aspect_ratio")
    @classmethod
    def require_supported_aspect_ratio(cls, value: str) -> str:
        if value not in config.SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {value}")
        return value

    @field_validator("resolution")
    @classmethod
    def require_supported_resolution(cls, value: str) -> str:
        if value not in config.SUPPORTED_RESOLUTIONS:
            raise ValueError(f"unsupported Seedance Mini resolution: {value}")
        return value

    @model_validator(mode="after")
    def require_unique_shot_ids(self) -> "StoryboardSpec":
        shot_ids = [shot.shot_id for shot in self.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("shot_id must be unique")
        return self


class RunStatus(str, Enum):
    created = "created"
    running = "running"
    failed = "failed"
    interrupted = "interrupted"
    succeeded = "succeeded"


class ShotStatus(str, Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1)
    resolution: str = config.DEFAULT_RESOLUTION
    aspect_ratio: str = config.DEFAULT_RATIO
    style: str = Field(default="cinematic", min_length=1)
    music_path: str | None = None
    platforms: list[str] = Field(default_factory=lambda: ["youtube", "tiktok"], min_length=1)
    paid_take_budget: int | None = Field(default=None, ge=0)
    # Missing on disk means the original submission policy; new entrypoints select explicitly.
    take_policy: Literal['legacy', 'completion_first', 'single_pass'] = 'legacy'
    # Preserve the existing review gate for workspaces saved before delivery modes.
    delivery_mode: Literal['reviewed', 'normal', 'preview'] = 'reviewed'
    planning_repair: bool = Field(default=True, strict=True)
    motion_control: Literal["auto", "text_guided", "reference_motion"] = "auto"

    @model_serializer(mode="wrap")
    def serialize_options(self, handler):
        data = handler(self)
        # Default mode must keep existing checkpoint hashes and experiment identities.
        if data.get("delivery_mode") == "reviewed":
            data.pop("delivery_mode")
        if data.get('planning_repair') is True:
            data.pop('planning_repair')
        return data

    @field_validator("resolution")
    @classmethod
    def require_supported_resolution(cls, value: str) -> str:
        if value not in config.SUPPORTED_RESOLUTIONS:
            raise ValueError(f"unsupported Seedance Mini resolution: {value}")
        return value

    @field_validator("aspect_ratio")
    @classmethod
    def require_supported_aspect_ratio(cls, value: str) -> str:
        if value not in config.SUPPORTED_ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {value}")
        return value

    @field_validator("platforms")
    @classmethod
    def require_supported_platforms(cls, value: list[str]) -> list[str]:
        unsupported = sorted(set(value) - set(config.SUPPORTED_PLATFORMS))
        if unsupported:
            raise ValueError(f"unsupported export platforms: {', '.join(unsupported)}")
        return value


class FailureEvidenceState(BaseModel):
    """Stable objective evidence; localized error text remains display-only."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "contract",
        "technical",
        "provider_policy",
        "provider_transient",
        "state",
        "unknown",
    ]
    code: str = Field(min_length=1)
    source: Literal[
        "provider",
        "technical_qa",
        "semantic_review",
        "pipeline_state",
        "legacy_manifest",
    ]
    details: dict[str, str] = Field(default_factory=dict)


class SubmissionDiagnosticState(BaseModel):
    """提交边界白名单元数据，不接收异常正文、请求头或素材 URL。"""
    model_config = ConfigDict(extra='forbid')
    version: Literal['submission-evidence-v1']
    phase: Literal['create_request', 'create_response']
    observed_at: str = Field(min_length=1, max_length=64)
    exception_type: str | None = Field(default=None, max_length=200)
    cause_type: str | None = Field(default=None, max_length=200)
    http_status: int | None = Field(default=None, ge=100, le=599, strict=True)
    request_id: str | None = Field(default=None, max_length=200)
    sdk_request_id: str | None = Field(default=None, max_length=200, pattern=r'^[A-Za-z0-9_.:-]+$')
    response_request_id: str | None = Field(default=None, max_length=200, pattern=r'^[A-Za-z0-9_.:-]+$')
    body_request_id: str | None = Field(default=None, max_length=200, pattern=r'^[A-Za-z0-9_.:-]+$')
    error_code: str | None = Field(default=None, max_length=200)


class PromptAttemptState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(gt=0)
    profile: Literal["normal", "policy_safe"]
    fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    outcome: Literal["pending", "succeeded", "failed"]
    provider_task_id: str | None = None
    provider_error_locus: str | None = None
    provider_error_code: str | None = None
    failure_evidence: list[FailureEvidenceState] = Field(default_factory=list)
    submission_diagnostic: SubmissionDiagnosticState | None = None


class PendingTaskDescriptor(BaseModel):
    """Immutable identity of one already-submitted provider request."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    prompt_profile: Literal["normal", "policy_safe"]
    prompt_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    compiled_contract_version: str = Field(min_length=1)
    compiled_contract_fingerprint: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )


class TakeRecordState(BaseModel):
    """Immutable observation record for one generated Take."""

    model_config = ConfigDict(extra="forbid")

    take_id: str = Field(min_length=1)
    disposition: Literal["accepted", "rejected"]
    local_path: str = Field(min_length=1)
    last_frame_url: str | None = None
    semantic_accepted: bool | None = None
    observed_end_state: dict[str, str] = Field(default_factory=dict)
    quality_score: int = 0
    technical_quality_score: int = 0
    model_used: str = ""
    resolution_used: str = ""
    prompt_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    compiled_contract_version: str | None = None
    compiled_contract_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    accepted_contract_version: str | None = None
    accepted_contract_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    semantic_evaluator_version: str | None = None
    acceptance_policy: Literal["semantic_reviewed", "reviewed_with_warnings", "technical_only"] | None = None
    review_warnings: list[str] = Field(default_factory=list)
    rejection_evidence: list[FailureEvidenceState] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ShotTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
    status: ShotStatus = ShotStatus.pending
    provider_task_id: str | None = None
    pending_task: PendingTaskDescriptor | None = None
    provider_error_type: str | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_error_locus: str | None = None
    prompt_profile: Literal["normal", "policy_safe"] | None = None
    prompt_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    compiled_contract_version: str | None = None
    compiled_contract_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    accepted_contract_version: str | None = None
    accepted_contract_fingerprint: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"
    )
    semantic_evaluator_version: str | None = None
    acceptance_policy: Literal["semantic_reviewed", "reviewed_with_warnings", "technical_only"] | None = None
    review_warnings: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    prompt_attempts: list[PromptAttemptState] = Field(default_factory=list)
    take_history: list[TakeRecordState] = Field(default_factory=list)
    canonical_take_id: str | None = None
    local_path: str | None = None
    last_frame_url: str | None = None
    quality_score: int = 0
    technical_quality_score: int = 0
    semantic_accepted: bool | None = None
    observed_end_state: dict[str, str] = Field(default_factory=dict)
    reference_chain_depth: int = Field(default=0, ge=0)
    model_used: str = ""
    resolution_used: str = ""
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)


class PaidTakeReservation(BaseModel):
    """Durable authorization for one provider submission."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str = Field(min_length=1)
    shot_id: int = Field(gt=0)
    take_number: int = Field(gt=0)
    status: Literal["reserved", "submitted", "reconciled", "released"] = "reserved"
    provider_task_id: str | None = None


class PaidTakeBudgetState(BaseModel):
    """The run-local paid-take ledger; absent limits remain explicitly unmetered."""

    model_config = ConfigDict(extra="forbid")

    estimated_takes: int = Field(default=0, ge=0)
    reservations: list[PaidTakeReservation] = Field(default_factory=list)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.created
    stage: str = "initialized"
    created_at: str
    updated_at: str
    options: RunOptions
    paid_take_budget: PaidTakeBudgetState = Field(default_factory=PaidTakeBudgetState)
    shots: dict[str, ShotTaskState] = Field(default_factory=dict)
    final_path: str | None = None
    delivery_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    delivery_base: dict[str, Any] | None = None
    error: str | None = None


def validate_storyboard(data: dict[str, Any]) -> dict[str, Any]:
    """Validate once at the LLM/persistence seam and preserve dict callers."""
    return StoryboardSpec.model_validate(data).model_dump(mode="json", exclude_none=True)


def validate_storyboard_draft(data: dict[str, Any]) -> dict[str, Any]:
    """Project an untrusted LLM draft onto the strict persisted schema."""
    return StoryboardSpec.model_validate(
        data,
        extra="ignore",
    ).model_dump(mode="json", exclude_none=True)


def validate_story_spine(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the compact global artifact before requesting shot details."""
    return StorySpineSpec.model_validate(data).model_dump(mode="json")

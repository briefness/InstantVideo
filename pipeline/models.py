"""Validated data contracts shared by the generation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config


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


class ActionBeatSpec(BaseModel):
    """One causal phase inside a shot, kept separate from camera movement."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["trigger", "peak", "aftermath"]
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target: str = ""
    visible_result: str = ""

    @field_validator("actor", "action", "target", "visible_result", mode="before")
    @classmethod
    def strip_beat_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


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
    must_share_frame: bool = False
    line_of_action_visible: bool = False
    actor_screen_position: str = ""
    target_screen_position: str = ""
    occlusion_policy: Literal["none", "partial", "motivated"] = "none"

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


class ShotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
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
    continuity_from_previous: Literal[
        "none", "seamless", "intentional_cut"
    ] = "none"
    composition_change: Literal["small", "medium", "large"] = "medium"
    coverage_role: Literal[
        "establish", "action_subject", "target_reaction",
        "interaction", "aftermath", "insert",
    ] = "establish"
    required_visible_entities: list[str] = Field(default_factory=list)
    interaction_geometry: InteractionGeometrySpec = Field(
        default_factory=InteractionGeometrySpec
    )
    primary_action: str = ""
    action_beats: list[ActionBeatSpec] = Field(default_factory=list, max_length=3)
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


class ShotTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_id: int = Field(gt=0)
    status: ShotStatus = ShotStatus.pending
    provider_task_id: str | None = None
    local_path: str | None = None
    last_frame_url: str | None = None
    quality_score: int = 0
    technical_quality_score: int = 0
    semantic_accepted: bool | None = None
    observed_end_state: dict[str, str] = Field(default_factory=dict)
    model_used: str = ""
    resolution_used: str = ""
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.created
    stage: str = "initialized"
    created_at: str
    updated_at: str
    options: RunOptions
    shots: dict[str, ShotTaskState] = Field(default_factory=dict)
    final_path: str | None = None
    error: str | None = None


def validate_storyboard(data: dict[str, Any]) -> dict[str, Any]:
    """Validate once at the LLM/persistence seam and preserve dict callers."""
    return StoryboardSpec.model_validate(data).model_dump(mode="json", exclude_none=True)

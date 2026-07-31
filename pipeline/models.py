"""Validated data contracts shared by the generation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_movement: str = "fixed"
    composition: str = ""
    start_framing: str = ""
    end_framing: str = ""
    speed: str = "fixed"


class CharacterSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    description: str = ""


class ContinuityStateSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: str = ""
    subject: str = ""
    action_phase: str = ""
    camera: str = ""


class ShotSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    shot_id: int = Field(gt=0)
    duration: int = Field(default=5, ge=4, le=15)
    # Empty keeps persisted pre-contract storyboards resumable; new storyboards
    # receive a stable ID at the LLM boundary in _apply_defaults.
    scene_id: str = ""
    scene_description: str = Field(min_length=1)
    prompt_en: str = Field(min_length=1)
    continuity_from_previous: Literal[
        "none", "seamless", "intentional_cut"
    ] = "none"
    primary_action: str = ""
    start_state: ContinuityStateSpec = Field(default_factory=ContinuityStateSpec)
    end_state: ContinuityStateSpec = Field(default_factory=ContinuityStateSpec)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    lighting: str = ""
    mood: str = "cinematic"
    negative_prompt: str = "avoid jitter, stable motion, no text artifacts"
    subtitle_text: str = ""
    transition_to_next: str = "crossfade"
    generate_audio: bool = True
    characters: list[str] = Field(default_factory=list)
    extract_character_ref: bool = False
    key_props: list[str] = Field(default_factory=list)

    @field_validator("scene_description", "prompt_en", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class StoryboardSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = ""
    total_duration: int | None = Field(default=None, gt=0)
    style: str = "cinematic"
    aspect_ratio: str = config.DEFAULT_RATIO
    resolution: str = config.DEFAULT_RESOLUTION
    mood: str = "cinematic"
    music_style: str = "cinematic orchestral"
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

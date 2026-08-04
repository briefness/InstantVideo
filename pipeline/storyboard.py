"""Stage 1: LLM 分镜生成 — 创意策略 + 运镜公式 + 丰富度校验"""

from __future__ import annotations

import re
import json
from copy import deepcopy
from pathlib import Path

from openai import OpenAI

import config
from pipeline.models import (
    validate_story_spine,
    validate_storyboard,
    validate_storyboard_draft,
)
from pipeline.participants import (
    canonical_entity_id,
    canonical_participant_id,
    canonical_target_id,
    normalize_shot_participants,
    normalize_structured_entity_references,
    shot_entity_registry,
    visible_character_names,
)
from pipeline.narrative import normalize_narrative_handoffs
from pipeline.production_plan import (
    apply_production_plan,
    build_production_plan,
    classify_framing,
    format_production_plan,
    reference_policy,
)
from pipeline.causality import (
    CAUSAL_INTERACTION_MODES,
    compile_interaction_blocking,
    normalize_causal_scope,
    with_causal_mode_invariants,
)


_STORYBOARD_MAX_COMPLETION_TOKENS = 16_384
_STORYBOARD_SHOTS_PER_BATCH = 3
_STORY_SPINE_SYSTEM_PROMPT = """你是影视故事规划器。只规划全局故事脊柱，不写完整视频提示词。
只输出一个合法 JSON 对象，不要 Markdown、解释或未定义字段：
{
  "title": "中文标题",
  "mood": "整体情绪",
  "music_style": "音乐方向",
  "theme_elements": ["stable_english_id"],
  "story_arc": {
    "goal": "可见目标",
    "stakes": "未完成时仍存在的问题",
    "turning_point": "改变局势的可见变化",
    "resolution": "最终可见结果"
  },
  "characters": [{
    "name": "stable_character_id",
    "description": "跨镜稳定外观",
    "mobility": "unspecified|bipedal|quadruped|tracked|wheeled|flying|stationary|other",
    "reference_mode": "identity|group|none"
  }],
  "shot_intents": [{
    "shot_id": 1,
    "scene_id": "stable_location_id",
    "narrative_function": "setup|progress|turn|payoff",
    "state_before": "镜头前故事状态",
    "state_change": "本镜可见变化",
    "state_after": "镜头后故事状态",
    "primary_action": "一个主体动作",
    "characters": ["stable_character_id"]
  }]
}
相邻 shot_intents 的 state_before 必须逐字复用前一项 state_after。严格按 ProductionPlan 的槽位数量与顺序输出；不要写 prompt_en、camera、blocking 或 interaction_geometry。"""
_JSON_REPAIR_SYSTEM_PROMPT = (
    "你是 JSON 语法修复器。只修复 JSON 语法，保留所有字段和值；"
    "只输出一个完整 JSON 对象，不要解释、不要 Markdown、不要新增字段。"
)
_CORRECTION_SYSTEM_PROMPT = (
    "你是影视分镜契约校正器。输入包含一个已经结构化的分镜和明确问题列表。"
    "只修改导致问题的字段，保留其余叙事、角色、场景、时长和字段结构。"
    "只能使用输入中已有的字段以及分镜契约允许的枚举值；只输出完整 JSON 对象。"
)
_MISPLACED_SHOT_METADATA = (
    "title", "total_duration", "aspect_ratio", "resolution", "style",
    "music_style", "content_focus", "theme_elements", "story_arc",
)
_ACTION_FOCUS_TERMS = (
    "大战", "打斗", "战斗", "对决", "决斗", "格斗", "武打", "搏斗",
    "动作片", "动作视频", "交战", "追逐", "追杀", "清剿", "清除丧尸",
    "清除僵尸",
    "fight", "battle", "combat", "duel", "versus", "martial arts",
    "action sequence", "chase",
)
_PRODUCT_FOCUS_TERMS = (
    "产品", "商品", "广告", "品牌", "带货", "种草", "开箱",
    "product", "commercial", "advertisement", "unboxing",
)
_ACTION_DEEMPHASIS_TERMS = (
    "氛围为主", "对峙为主", "慢节奏为主", "情绪为主",
    "slow burn", "standoff focused", "atmosphere focused",
)
_ACTION_PROGRESS_PATTERNS = (
    r"\bpunch(?:es|ed|ing)?\b", r"\bkick(?:s|ed|ing)?\b",
    r"\bstrike(?:s|ing)?\b|\bstruck\b", r"\bslash(?:es|ed|ing)?\b",
    r"\bstab(?:s|bed|bing)?\b", r"\bblock(?:s|ed|ing)?\b",
    r"\bparr(?:y|ies|ied|ying)\b", r"\bdodg(?:e|es|ed|ing)\b",
    r"\bevad(?:e|es|ed|ing)\b", r"\bcounter(?:s|ed|ing|attacks?)?\b",
    r"\bgrappl(?:e|es|ed|ing)\b", r"\bthrow(?:s|ing)?\b|\bthrew\b",
    r"\bslam(?:s|med|ming)?\b", r"\bhit(?:s|ting)?\b",
    r"\battack(?:s|ed|ing)?\b", r"\bfight(?:s|ing)?\b|\bfought\b",
    r"\bclash(?:es|ed|ing)?\b", r"\bcollid(?:e|es|ed|ing)\b",
    r"\bfire(?:s|d|ing)?\b", r"\bshoot(?:s|ing)?\b|\bshot\b",
    r"\bblast(?:s|ed|ing)?\b", r"\blung(?:e|es|ed|ing)\b",
    r"\btackl(?:e|es|ed|ing)\b", r"\bdeflect(?:s|ed|ing)?\b",
    r"\bimpact(?:s|ed|ing)?\b", r"\bknock(?:s|ed|ing)?\b",
    r"\bchas(?:e|es|ed|ing)\b", r"\bpursu(?:e|es|ed|ing)\b",
    r"\bcharg(?:e|es|ed|ing) at\b", r"\bswing(?:s|ing)?\b|\bswung\b",
    r"\bsmash(?:es|ed|ing)?\b", r"\bslic(?:e|es|ed|ing)\b",
    r"\bexchange(?:s|d|ing)? blows\b",
    r"\bland(?:s|ed|ing)? (?:a |the )?(?:blow|strike|kick|punch)\b",
)
_CAMERA_SUBJECT_PREFIXES = (
    "camera ", "the camera ", "drone ", "the drone ", "lens ", "the lens ",
)
_CAMERA_ACTION_TERMS = (
    "push", "pull", "dolly", "pan", "tilt", "orbit", "crane", "zoom",
    "track", "move", "sweep", "reveal", "follow",
)
_PHYSICAL_ACTION_PATTERNS = (
    ("advance", r"\badvanc(?:e|es|ed|ing)\b"),
    ("stop", r"\bstop(?:s|ped|ping)?\b"),
    ("power up", r"\bpower(?:s|ed|ing)? up\b"),
    ("emerge", r"\bemerg(?:e|es|ed|ing)\b"),
    ("charge", r"\bcharg(?:e|es|ed|ing)\b"),
    ("fire", r"\bfire(?:s|d|ing)?\b"),
    ("shoot", r"\bshoot(?:s|ing)?\b|\bshot\b"),
    ("incinerate", r"\bincinerat(?:e|es|ed|ing)\b"),
    ("leap", r"\bleap(?:s|ed|ing)?\b"),
    ("jump", r"\bjump(?:s|ed|ing)?\b"),
    ("step", r"\bstep(?:s|ped|ping)?\b"),
    ("walk", r"\bwalk(?:s|ed|ing)?\b"),
    ("run", r"\brun(?:s|ning)?\b|\bran\b"),
    ("punch", r"\bpunch(?:es|ed|ing)?\b"),
    ("kick", r"\bkick(?:s|ed|ing)?\b"),
    ("strike", r"\bstrike(?:s|ing)?\b|\bstruck\b"),
    ("block", r"\bblock(?:s|ed|ing)?\b"),
    ("dodge", r"\bdodg(?:e|es|ed|ing)\b"),
    ("grapple", r"\bgrappl(?:e|es|ed|ing)\b"),
    ("tackle", r"\btackl(?:e|es|ed|ing)\b"),
    ("crush", r"\bcrush(?:es|ed|ing)?\b"),
    ("tear", r"\btear(?:s|ing)?\b|\btore\b"),
    ("rip", r"\brip(?:s|ped|ping)?\b"),
    (
        "throw",
        r"\b(?:throw(?:s|ing)?|threw)\b"
        r"(?!\s+(?:\w+\s+){0,3}(?:punch|kick|strike|blow)s?\b)",
    ),
    ("slam", r"\bslam(?:s|med|ming)?\b"),
)
_CONTACT_ACTIONS = {
    "punch", "kick", "strike", "grapple", "tackle", "crush", "tear",
    "rip", "throw", "slam",
}
_SEQUENCE_SEPARATORS = (";", " then ", " followed by ", " before it ", " after it ")
_MOBILITY_CONFLICT_PATTERNS = {
    "tracked": r"\b(?:step|steps|stepped|stepping|walk|walks|walked|walking|run|runs|running|ran|jump|jumps|jumped|jumping|leap|leaps|leaped|leaping|kick|kicks|kicked|kicking)\b",
    "wheeled": r"\b(?:step|steps|stepped|stepping|walk|walks|walked|walking|run|runs|running|ran|jump|jumps|jumped|jumping|leap|leaps|leaped|leaping|kick|kicks|kicked|kicking)\b",
    "stationary": r"\b(?:advance|advances|advanced|advancing|move|moves|moved|moving|step|steps|stepped|stepping|walk|walks|walked|walking|run|runs|running|ran|jump|jumps|jumped|jumping|leap|leaps|leaped|leaping)\b",
}
_SCREEN_SIDES = {"left", "right"}
_PROCESS_ORDER_TERMS = (
    "制作过程", "制作流程", "完整过程", "全过程", "从头到尾", "从零开始",
    "教程", "步骤", "如何制作", "从咖啡豆到", "从原料到",
    "process", "workflow", "step by step", "how to make", "from scratch",
    "from bean to", "from ingredients to",
)


def _infer_content_focus(user_request: str) -> str:
    """从用户明确措辞推断叙事重心；不明确时保持均衡。"""
    request = user_request.lower()
    product_score = sum(term in request for term in _PRODUCT_FOCUS_TERMS)
    action_score = sum(term in request for term in _ACTION_FOCUS_TERMS)
    if action_score and any(term in request for term in _ACTION_DEEMPHASIS_TERMS):
        action_score = 0
    if action_score > product_score:
        return "action"
    if product_score > action_score:
        return "product"
    return "balanced"


def _content_focus_guidance(focus: str) -> str:
    if focus == "action":
        return (
            "动作驱动：建立和收尾保持简洁，用多个可执行的主体攻防节拍推进冲突；"
            "运镜跟随动作，不得代替主体动作；不强制广告式 insert shot"
        )
    if focus == "product":
        return (
            "产品展示驱动：产品始终是视觉锚点，可使用材质微距、使用过程和英雄镜头；"
            "至少保留一个有叙事价值的 insert shot"
        )
    return "均衡叙事：根据用户强调的主体、情绪与节奏分配动作、运镜和留白"


def _requires_process_order(user_request: str) -> bool:
    request = user_request.lower()
    return any(term in request for term in _PROCESS_ORDER_TERMS)


def _camera_is_primary_subject(action: str) -> bool:
    value = action.strip().lower()
    return value.startswith(_CAMERA_SUBJECT_PREFIXES) and any(
        term in value for term in _CAMERA_ACTION_TERMS
    )


def _advances_action_conflict(action: str) -> bool:
    value = action.lower()
    return any(re.search(pattern, value) for pattern in _ACTION_PROGRESS_PATTERNS)


def _physical_action_beats(action: str) -> list[str]:
    """Return ordered physical verbs so a string cannot hide a whole sequence."""
    matches = []
    value = action.lower()
    for name, pattern in _PHYSICAL_ACTION_PATTERNS:
        matches.extend((match.start(), name) for match in re.finditer(pattern, value))
    return [name for _, name in sorted(matches)]


def _character_mobility(character: dict) -> str:
    mobility = str(character.get("mobility", "unspecified")).lower()
    if mobility != "unspecified":
        return mobility
    description = str(character.get("description", "")).lower()
    inferred = (
        ("tracked", ("tracked base", "treaded base", "tank treads", "crawler tracks")),
        ("wheeled", ("wheeled base", "on wheels")),
        ("stationary", ("fixed pedestal", "bolted to", "stationary turret")),
    )
    for candidate, markers in inferred:
        if any(marker in description for marker in markers):
            return candidate
    return "unspecified"


def _screen_axis_flipped(previous: dict, current: dict) -> bool:
    previous_camera = previous.get("camera", {})
    current_camera = current.get("camera", {})
    if not isinstance(previous_camera, dict) or not isinstance(current_camera, dict):
        return False
    if current_camera.get("axis_change") == "reestablish":
        return False
    previous_positions = previous_camera.get("screen_positions", {})
    current_positions = current_camera.get("screen_positions", {})
    if not isinstance(previous_positions, dict) or not isinstance(current_positions, dict):
        return False
    shared = set(previous_positions) & set(current_positions)
    return any(
        str(previous_positions[name]).lower() in _SCREEN_SIDES
        and str(current_positions[name]).lower() in _SCREEN_SIDES
        and str(previous_positions[name]).lower() != str(current_positions[name]).lower()
        for name in shared
    )


def generate_storyboard(
    user_request: str,
    target_duration: int = 30,
    aspect_ratio: str = config.DEFAULT_RATIO,
    resolution: str = config.DEFAULT_RESOLUTION,
    style: str = "cinematic",
) -> dict:
    """
    调用 LLM 根据用户需求生成结构化分镜脚本

    Returns:
        完整的 storyboard dict (含 shots, music_style, mood 等)
    """
    client = OpenAI(
        api_key=config.ARK_API_KEY,
        base_url=config.ARK_BASE_URL,
    )

    system_prompt = _load_system_prompt()

    content_focus = _infer_content_focus(user_request)
    production_plan = build_production_plan(content_focus, target_duration)
    user_prompt = f"""用户需求: {user_request}

目标参数:
- 总时长: ~{target_duration} 秒
- 画面比例: {aspect_ratio}
- 分辨率: {resolution}
- 整体风格: {style}
- 叙事重心: {_content_focus_guidance(content_focus)}

生产计划（先于创意内容确定，属于不可改写的执行输入）:
{format_production_plan(production_plan)}

请生成完整的分镜脚本 (JSON 格式)。"""

    storyboard = _generate_storyboard_draft(
        client, system_prompt, user_prompt, production_plan
    )
    storyboard = _compile_storyboard_contract(
        storyboard, aspect_ratio, resolution, style, production_plan=production_plan
    )

    # 丰富度校验 + 自动修正 (严重问题触发 LLM 重跑)
    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request=user_request,
        require_narrative_contract=True,
    )
    if warnings:
        print("\n   [丰富度校验]")
        for w in warnings:
            print(f"   {w}")

    if is_critical:
        print("\n   [自动修正] 检测到严重问题, 重新生成分镜...")
        storyboard = _correct_storyboard_draft(
            client,
            user_prompt,
            storyboard,
            warnings,
            production_plan,
        )
        storyboard = _compile_storyboard_contract(
            storyboard,
            aspect_ratio,
            resolution,
            style,
            production_plan=production_plan,
        )

        # 二次校验: 不再重试，但严重缺陷也不能进入付费生成。
        warnings_2, is_critical_2 = _validate_storyboard_richness(
            storyboard,
            user_request=user_request,
            require_narrative_contract=True,
        )
        if warnings_2:
            print("\n   [二次校验]")
            for w in warnings_2:
                print(f"   {w}")
            if is_critical_2:
                raise ValueError(
                    "分镜自动修正 1 次后仍未通过生成就绪要求；"
                    "已停止，未调用视频生成接口"
                )
        else:
            print("   ✓ 修正后通过所有校验")

    return storyboard


def _generate_storyboard_draft(
    client,
    system_prompt: str,
    user_prompt: str,
    production_plan: dict,
) -> dict:
    """Generate one bounded artifact at a time while keeping one global spine."""
    slots = production_plan["slots"]
    if len(slots) <= _STORYBOARD_SHOTS_PER_BATCH:
        return _call_llm_for_storyboard(client, system_prompt, user_prompt)

    spine_prompt = _build_story_spine_prompt(user_prompt, production_plan)
    spine = validate_story_spine(
        _call_llm_for_storyboard(
            client,
            _STORY_SPINE_SYSTEM_PROMPT,
            spine_prompt,
            temperature=0.3,
        )
    )
    _project_spine_onto_plan(spine, production_plan)

    shots: list[dict] = []
    for start in range(0, len(slots), _STORYBOARD_SHOTS_PER_BATCH):
        batch_slots = slots[start:start + _STORYBOARD_SHOTS_PER_BATCH]
        batch_prompt = _build_shot_batch_prompt(
            user_prompt, spine, production_plan, batch_slots
        )
        batch_result = _call_llm_for_storyboard(
            client, system_prompt, batch_prompt, temperature=0.5
        )
        batch_shots = batch_result.get("shots")
        if not isinstance(batch_shots, list) or len(batch_shots) != len(batch_slots):
            expected = [slot["shot_id"] for slot in batch_slots]
            raise ValueError(
                f"分镜批次 {expected} 返回镜头数量不匹配；"
                "已停止，未调用视频生成接口"
            )
        for shot, slot in zip(batch_shots, batch_slots):
            if not isinstance(shot, dict):
                raise ValueError(
                    f"Shot {slot['shot_id']} 详情必须是 JSON 对象；"
                    "已停止，未调用视频生成接口"
                )
            normalized = deepcopy(shot)
            normalized["shot_id"] = slot["shot_id"]
            normalized["duration"] = slot["duration"]
            shots.append(normalized)

    return {
        key: deepcopy(value)
        for key, value in spine.items()
        if key != "shot_intents"
    } | {"shots": shots}


def _correct_storyboard_draft(
    client,
    user_prompt: str,
    storyboard: dict,
    warnings: list[str],
    production_plan: dict,
) -> dict:
    """Correct one bounded artifact per call and merge it into the frozen story."""
    slots = production_plan["slots"]
    if len(slots) <= _STORYBOARD_SHOTS_PER_BATCH:
        corrected = _call_llm_for_storyboard(
            client,
            _CORRECTION_SYSTEM_PROMPT,
            _build_correction_prompt(user_prompt, storyboard, warnings),
            temperature=0.2,
        )
        _preserve_correction_contract(storyboard, corrected)
        return corrected

    original_by_id = {
        shot.get("shot_id"): shot
        for shot in storyboard.get("shots", [])
        if isinstance(shot, dict) and shot.get("shot_id") is not None
    }
    expected_ids = [slot["shot_id"] for slot in slots]
    if len(original_by_id) != len(slots) or any(
        shot_id not in original_by_id for shot_id in expected_ids
    ):
        raise ValueError(
            "长分镜纠错前的镜头集合与 ProductionPlan 不一致；"
            "已停止，未调用视频生成接口"
        )

    merged = deepcopy(storyboard)
    merged["shots"] = []
    frozen_story = {
        key: deepcopy(value)
        for key, value in storyboard.items()
        if key != "shots"
    }

    for start in range(0, len(slots), _STORYBOARD_SHOTS_PER_BATCH):
        batch_slots = slots[start:start + _STORYBOARD_SHOTS_PER_BATCH]
        batch_ids = [slot["shot_id"] for slot in batch_slots]
        batch_original = deepcopy(frozen_story) | {
            "shots": [deepcopy(original_by_id[shot_id]) for shot_id in batch_ids]
        }
        batch_warnings = _correction_warnings_for_batch(warnings, set(batch_ids))
        if not batch_warnings:
            merged["shots"].extend(batch_original["shots"])
            continue
        correction_prompt = _build_batch_correction_prompt(
            user_prompt,
            batch_original,
            batch_warnings,
            production_plan,
            batch_slots,
        )
        corrected = _call_llm_for_storyboard(
            client,
            _CORRECTION_SYSTEM_PROMPT,
            correction_prompt,
            temperature=0.2,
        )
        corrected_shots = corrected.get("shots")
        if (
            not isinstance(corrected_shots, list)
            or len(corrected_shots) != len(batch_slots)
        ):
            raise ValueError(
                f"分镜纠错批次 {batch_ids} 返回镜头数量不匹配；"
                "已停止，未调用视频生成接口"
            )

        _preserve_correction_contract(batch_original, corrected)
        for shot, slot in zip(corrected_shots, batch_slots):
            if not isinstance(shot, dict):
                raise ValueError(
                    f"Shot {slot['shot_id']} 纠错结果必须是 JSON 对象；"
                    "已停止，未调用视频生成接口"
                )
            normalized = deepcopy(shot)
            normalized["shot_id"] = slot["shot_id"]
            normalized["duration"] = slot["duration"]
            merged["shots"].append(normalized)

    return merged


def _correction_warnings_for_batch(
    warnings: list[str], batch_ids: set[int]
) -> list[str]:
    """Route local issues only to batches that own every referenced shot."""
    scoped = []
    for warning in warnings:
        referenced_ids = {
            int(value) for value in re.findall(r"\bShot\s+(\d+)\b", warning)
        }
        if not referenced_ids or referenced_ids.issubset(batch_ids):
            scoped.append(warning)
    return scoped


def _build_batch_correction_prompt(
    user_prompt: str,
    storyboard: dict,
    warnings: list[str],
    production_plan: dict,
    batch_slots: list[dict],
) -> str:
    batch_plan = deepcopy(production_plan)
    batch_plan["slots"] = deepcopy(batch_slots)
    batch_plan["planned_duration"] = sum(slot["duration"] for slot in batch_slots)
    scoped_prompt = f"""[SHOT_CORRECTION_BATCH]
{_storyboard_request_context(user_prompt)}

本批次 ProductionPlan：
{format_production_plan(batch_plan)}

这是长分镜的一部分。只修正并返回本批次 shots；顶层故事、角色和其他批次均已冻结，
不得生成、概述或改写其他批次镜头。调用方会按 ProductionPlan 合并结果。"""
    return _build_correction_prompt(
        scoped_prompt,
        storyboard,
        warnings,
        batch_scoped=True,
    )


def _build_story_spine_prompt(user_prompt: str, production_plan: dict) -> str:
    return f"""[STORY_SPINE]
{_storyboard_request_context(user_prompt)}

只生成紧凑全局故事脊柱。ProductionPlan 是不可改写的结构事实：
{format_production_plan(production_plan)}
"""


def _build_shot_batch_prompt(
    user_prompt: str,
    spine: dict,
    production_plan: dict,
    batch_slots: list[dict],
) -> str:
    batch_ids = {slot["shot_id"] for slot in batch_slots}
    batch_plan = deepcopy(production_plan)
    batch_plan["slots"] = deepcopy(batch_slots)
    batch_plan["planned_duration"] = sum(slot["duration"] for slot in batch_slots)
    intents = [
        intent for intent in spine["shot_intents"]
        if intent["shot_id"] in batch_ids
    ]
    global_spine = {
        key: value for key, value in spine.items() if key != "shot_intents"
    }
    return f"""[SHOT_DETAIL_BATCH]
{_storyboard_request_context(user_prompt)}

全局故事脊柱已经冻结，不得改写：
{json.dumps(global_spine, ensure_ascii=False, indent=2)}

只为下列 intent 生成完整镜头字段：
{json.dumps(intents, ensure_ascii=False, indent=2)}

本批次 ProductionPlan：
{format_production_plan(batch_plan)}

输出一个 JSON 对象，shots 只能包含本批次镜头。可以重复冻结的顶层字段，
但不得输出其他批次的镜头。调用方会按 shot_id 合并全部批次。
"""


def _project_spine_onto_plan(spine: dict, production_plan: dict) -> None:
    intents = spine["shot_intents"]
    slots = production_plan["slots"]
    if len(intents) != len(slots):
        raise ValueError(
            f"Story Spine 计划 {len(slots)} 个镜头，实际返回 {len(intents)} 个；"
            "已停止，未调用视频生成接口"
        )
    for intent, slot in zip(intents, slots):
        intent["shot_id"] = slot["shot_id"]
        intent["narrative_function"] = slot["narrative_function"]


def _storyboard_request_context(user_prompt: str) -> str:
    """Remove the full embedded plan before adding one scoped plan."""
    return user_prompt.split("\n生产计划（", 1)[0].rstrip()


def _call_llm_for_storyboard(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
) -> dict:
    """调用 LLM 获取分镜 JSON"""
    try:
        response = _create_storyboard_completion(
            client, system_prompt, user_prompt, temperature=temperature
        )
    except Exception as e:
        print(f"   [ERROR] LLM 调用失败: {e}")
        raise
    raw_content = response.choices[0].message.content
    _reject_incomplete_completion(response, raw_content)

    try:
        storyboard = _parse_json_response(raw_content)
    except json.JSONDecodeError as parse_error:
        print("   [格式修复] 分镜 JSON 语法异常，正在修复（最多 1 次）...")
        repair_prompt = _build_json_repair_prompt(raw_content, parse_error)
        try:
            response = _create_storyboard_completion(
                client, _JSON_REPAIR_SYSTEM_PROMPT, repair_prompt, temperature=0
            )
        except Exception as e:
            print(f"   [ERROR] LLM JSON 修复调用失败: {e}")
            raise
        repaired_content = response.choices[0].message.content
        _reject_incomplete_completion(response, repaired_content)
        try:
            storyboard = _parse_json_response(repaired_content)
        except json.JSONDecodeError as repair_error:
            raise ValueError(
                f"LLM 分镜 JSON 修复失败（已重试 1 次）: {repair_error}"
            ) from repair_error

    if not isinstance(storyboard, dict):
        raise ValueError("LLM 分镜响应必须是 JSON 对象")
    return storyboard


def _create_storyboard_completion(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float,
):
    """使用足够容纳中长分镜的显式输出预算调用 Ark。"""
    return client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_completion_tokens=_STORYBOARD_MAX_COMPLETION_TOKENS,
    )


def _reject_incomplete_completion(response, content: str | None) -> None:
    """截断或被拦截的响应不是可修复的 JSON 语法错误。"""
    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason in (None, "stop"):
        return

    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    token_detail = (
        f", 已生成 {completion_tokens} tokens"
        if completion_tokens is not None
        else ""
    )
    content_length = len(content or "")
    if finish_reason == "length":
        raise ValueError(
            "LLM 分镜输出被截断"
            f"（finish_reason=length{token_detail}, 响应 {content_length} 字符）；"
            "未尝试无效的 JSON 语法修复，请缩短分镜输出或更换支持更大输出的模型"
        )
    raise ValueError(
        f"LLM 分镜未正常结束（finish_reason={finish_reason}{token_detail}, "
        f"响应 {content_length} 字符）；未进入 JSON 语法修复"
    )


def _build_json_repair_prompt(raw_content: str, error: json.JSONDecodeError) -> str:
    """要求模型仅修复自身响应中的 JSON 语法，不重写分镜内容。"""
    return f"""上一次响应不是合法 JSON。
解析器错误 (JSONDecodeError): {error}

请只修复 JSON 语法（例如缺失逗号、引号或转义），保留原有字段和值，不要改写分镜。
只输出修复后的完整 JSON 对象，不要输出 Markdown 或解释。

<invalid_json>
{raw_content}
</invalid_json>"""


_MAX_SEAMLESS_TRANSITIONS = 2
_FRAMING_RANKS = (
    ("extreme close-up", 0),
    ("extreme close up", 0),
    ("macro", 0),
    ("medium close-up", 2),
    ("medium close up", 2),
    ("close-up", 1),
    ("close up", 1),
    ("medium shot", 3),
    ("medium framing", 3),
    ("waist-level", 3),
    ("waist level", 3),
    ("full-body", 4),
    ("full body", 4),
    ("full shot", 4),
    ("extreme wide", 6),
    ("establishing", 6),
    ("wide shot", 5),
    ("wide framing", 5),
    ("long shot", 5),
)


def _framing_rank(value: object) -> int | None:
    text = str(value or "").lower()
    for phrase, rank in _FRAMING_RANKS:
        if phrase in text:
            return rank
    return {
        "close_detail": 1,
        "medium": 3,
        "wide": 5,
    }.get(classify_framing(value))


def _framing_family(value: object) -> str | None:
    return classify_framing(value)


def _boundary_framing(shot: dict, boundary: str) -> int | None:
    camera = shot.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    state = shot.get(f"{boundary}_state")
    state = state if isinstance(state, dict) else {}
    values = (
        camera.get(f"{boundary}_framing"),
        state.get("camera"),
    )
    for value in values:
        rank = _framing_rank(value)
        if rank is not None:
            return rank
    return None


def _identity_reference_issue(
    shot: dict,
    character_modes: dict[str, str],
) -> str | None:
    """Return why this shot cannot safely become an identity reference."""
    shot_characters = visible_character_names(shot, list(character_modes))
    identity_characters = [
        name
        for name in shot_characters
        if character_modes.get(name, "identity") == "identity"
    ]
    if len(shot_characters) != 1 or len(identity_characters) != 1:
        return (
            "角色参考归属不明确: 只能从单一 identity 角色镜头提取身份参考，"
            "群体和多人整帧不可复用"
        )

    ranks = [
        _boundary_framing(shot, "start"),
        _boundary_framing(shot, "end"),
    ]
    known_ranks = [rank for rank in ranks if rank is not None]
    if known_ranks and max(known_ranks) <= 1:
        return (
            "角色参考镜头只有特写，无法稳定锚定完整外观；"
            "必须使用清晰无遮挡的中景、全身或全景"
        )
    return None


def _seamless_cut_reason(
    previous: dict,
    current: dict,
    seamless_transitions: int,
) -> str | None:
    if _scene_id(previous) != _scene_id(current):
        return "物理场景不同"

    previous_end = _boundary_framing(previous, "end")
    current_start = _boundary_framing(current, "start")
    if (
        previous_end is not None
        and current_start is not None
        and abs(previous_end - current_start) >= 2
    ):
        return "相邻镜头边界景别不兼容"

    if seamless_transitions >= _MAX_SEAMLESS_TRANSITIONS:
        return f"连续尾帧链已达到 {_MAX_SEAMLESS_TRANSITIONS} 次上限"

    return None


def _normalize_continuity_contract(shots: list[dict]) -> list[str]:
    """归一化剪辑契约；scene_id 只表示地点，不推导摄影机连续性。"""
    corrections: list[str] = []
    seamless_transitions = 0

    for index, shot in enumerate(shots):
        if index == 0:
            shot["continuity_from_previous"] = "none"
            continue

        continuity = shot.get("continuity_from_previous")
        if continuity not in {"seamless", "intentional_cut"}:
            continuity = "intentional_cut"

        if continuity == "seamless":
            reason = _seamless_cut_reason(
                shots[index - 1], shot, seamless_transitions
            )
            if reason:
                continuity = "intentional_cut"
                corrections.append(
                    f"Shot {shot.get('shot_id', index + 1)}: {reason}"
                )

        shot["continuity_from_previous"] = continuity
        if continuity == "seamless":
            seamless_transitions += 1
        else:
            seamless_transitions = 0

    return corrections


def _infer_composition_change(previous: dict, current: dict) -> str:
    """Classify editorial coverage changes without comparing prompt prose."""
    if current.get("continuity_from_previous") == "seamless":
        return "small"
    previous_end = _boundary_framing(previous, "end")
    current_start = _boundary_framing(current, "start")
    if previous_end is None or current_start is None:
        return "medium"
    delta = abs(previous_end - current_start)
    if delta <= 1:
        return "small"
    if delta <= 3:
        return "medium"
    return "large"


def _apply_coverage_defaults(
    shots: list[dict],
    character_names: list[str] | None = None,
    theme_elements: list[str] | None = None,
) -> None:
    """Compile LLM action fields into one deterministic production contract."""
    character_names = character_names or list(dict.fromkeys(
        str(name).strip()
        for shot in shots
        for name in shot.get("characters", [])
        if str(name).strip()
    ))
    theme_elements = theme_elements or []
    for index, shot in enumerate(shots):
        registry = shot_entity_registry(shot, character_names, theme_elements)
        normalize_structured_entity_references(shot, registry)
        if index == 0:
            shot.setdefault("composition_change", "large")
        elif shot.get("continuity_from_previous") == "seamless":
            shot["composition_change"] = "small"
        else:
            shot.setdefault(
                "composition_change",
                _infer_composition_change(shots[index - 1], shot),
            )

        camera = shot.get("camera")
        camera = camera if isinstance(camera, dict) else {}
        positions = camera.get("screen_positions")
        positions = dict(positions) if isinstance(positions, dict) else {}
        blocking = shot.get("blocking")
        blocking = blocking if isinstance(blocking, dict) else {}
        for name, intent in blocking.items():
            frame_position = (
                str(intent.get("frame_position", "")).strip()
                if isinstance(intent, dict) else ""
            )
            if frame_position and name not in positions:
                positions[name] = frame_position
        if positions:
            camera["screen_positions"] = positions
        shot["camera"] = camera

        beats = [beat for beat in shot.get("action_beats", []) if isinstance(beat, dict)]

        interaction = next(
            (
                beat for beat in beats
                if str(beat.get("actor", "")).strip()
                and str(beat.get("target", "")).strip()
            ),
            None,
        )
        if interaction:
            if shot.get("coverage_role") in (None, "", "establish"):
                shot["coverage_role"] = "interaction"
            actor = str(interaction["actor"]).strip()
            target = str(interaction["target"]).strip()
            camera = shot.get("camera", {})
            positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
            geometry = shot.get("interaction_geometry")
            geometry = dict(geometry) if isinstance(geometry, dict) else {}
            if not geometry.get("actor"):
                geometry["actor"] = actor
            if not geometry.get("target"):
                geometry["target"] = target
            if not geometry.get("actor_screen_position"):
                geometry["actor_screen_position"] = str(positions.get(actor, ""))
            if not geometry.get("target_screen_position"):
                geometry["target_screen_position"] = str(positions.get(target, ""))
            if not geometry.get("occlusion_policy"):
                geometry["occlusion_policy"] = "none"
            phase = str(geometry.get("effect_phase", "")).strip()
            if not phase and str(interaction.get("visible_result", "")).strip():
                geometry["must_share_frame"] = True
                geometry["line_of_action_visible"] = True
            elif not phase:
                geometry.setdefault("must_share_frame", False)
                geometry.setdefault("line_of_action_visible", False)
            shot["interaction_geometry"] = geometry
        else:
            shot.setdefault("coverage_role", "establish" if index == 0 else "aftermath")
            shot.setdefault("interaction_geometry", {})
        shot["interaction_geometry"] = with_causal_mode_invariants(
            shot.get("interaction_geometry")
        )
        geometry = shot["interaction_geometry"]
        registry = shot_entity_registry(shot, character_names, theme_elements)
        actor = canonical_entity_id(geometry.get("actor"), registry.characters)
        target = canonical_target_id(
            geometry.get("target"),
            registry,
            exclude=[actor],
        )
        if actor:
            geometry["actor"] = actor
        if target:
            geometry["target"] = target
        if (
            str(geometry.get("effect_phase", "")).strip() == "active"
            and str(geometry.get("interaction_mode", "")).strip() == "directed_path"
            and str(geometry.get("effect_motion", "")).strip() == "sweep"
            and str(geometry.get("outcome_scope", "")).strip() in {"subset", "all"}
            and isinstance(shot.get("duration"), (int, float))
            and not isinstance(shot.get("duration"), bool)
            and shot["duration"] < 6
        ):
            geometry["outcome_scope"] = "single"
            geometry["effect_motion"] = "static"
            print(
                f"   [因果负载校正] Shot {shot.get('shot_id', '?')} "
                "短镜头多目标扫掠改为 single/static"
            )
        canonical_actor = actor if actor in registry.characters else ""
        canonical_target = target if target in registry.target_ids else ""
        for beat in beats:
            beat_actor = canonical_entity_id(beat.get("actor"), registry.characters)
            if beat_actor in registry.characters:
                beat["actor"] = beat_actor
            elif canonical_actor:
                beat["actor"] = canonical_actor
            beat_target = canonical_target_id(
                beat.get("target"),
                registry,
                exclude=[str(beat.get("actor", "")).strip()],
            )
            if beat_target in registry.target_ids:
                beat["target"] = beat_target
            elif canonical_target:
                beat["target"] = canonical_target
        actors = [str(beat.get("actor", "")).strip() for beat in beats]
        targets = [str(beat.get("target", "")).strip() for beat in beats]
        participants = list(dict.fromkeys(
            name for name in actors + targets
            if name and name.casefold() != "none"
        ))
        if not shot.get("required_visible_entities"):
            shot["required_visible_entities"] = (
                participants or list(shot.get("characters", []))
            )
        compile_interaction_blocking(shot)
        if str(geometry.get("effect_phase", "")).strip() == "active":
            mode = str(geometry.get("interaction_mode", "none")).strip()
            actor = str(geometry.get("actor", "")).strip()
            target = str(geometry.get("target", "")).strip()
            causal_entities = [target]
            if mode in {"direct_contact", "directed_path"} or geometry.get(
                "must_share_frame"
            ):
                causal_entities.insert(0, actor)
            shot["required_visible_entities"] = list(dict.fromkeys([
                *shot.get("required_visible_entities", []),
                *(
                    name for name in causal_entities
                    if name and name.casefold() != "none"
                ),
            ]))


def _apply_defaults(storyboard: dict, aspect_ratio: str, resolution: str, style: str):
    """补充默认值"""
    # CLI parameters are authoritative; the LLM cannot silently override them.
    storyboard["aspect_ratio"] = aspect_ratio
    storyboard["resolution"] = resolution
    storyboard["style"] = style
    storyboard.setdefault("mood", "cinematic")
    storyboard.setdefault("music_style", "cinematic orchestral")
    storyboard.setdefault("content_focus", "balanced")
    storyboard.setdefault("theme_elements", [])
    character_modes = {
        character.get("name"): character.get("reference_mode", "identity")
        for character in storyboard.get("characters", [])
    }
    character_names = list(character_modes)

    for index, shot in enumerate(storyboard["shots"]):
        for field in _MISPLACED_SHOT_METADATA:
            shot.pop(field, None)
        shot.setdefault("duration", config.DEFAULT_DURATION)
        normalized_duration = _normalize_positive_duration(shot["duration"])
        if normalized_duration != shot["duration"]:
            print(
                f"   [镜头时长校正] Shot {shot.get('shot_id', index + 1)} "
                f"{shot['duration']}s → {normalized_duration}s"
            )
        shot["duration"] = normalized_duration
        shot.setdefault("generate_audio", config.DEFAULT_GENERATE_AUDIO)
        shot.setdefault("transition_to_next", "crossfade")
        shot.setdefault("negative_prompt", "avoid jitter, stable motion, no text artifacts")
        shot.setdefault("key_props", [])
        shot.setdefault("continuity_props", [])
        shot["scene_id"] = _scene_id(shot) or f"scene_{index + 1:03d}"
        shot.setdefault("primary_action", "")
        shot.setdefault("action_beats", [])
        shot.setdefault("start_state", {})
        shot.setdefault("end_state", {})
        shot.setdefault("blocking", {})
        normalize_shot_participants(
            shot,
            character_names,
            storyboard.get("theme_elements", []),
        )
        visible_characters = visible_character_names(shot, character_names)
        if visible_characters != shot.get("characters", []):
            added = [
                name for name in visible_characters
                if name not in shot.get("characters", [])
            ]
            shot["characters"] = visible_characters
            if added:
                print(
                    f"   [角色参与者校正] Shot {shot.get('shot_id', index + 1)} "
                    f"补充结构化字段中的可见角色: {', '.join(added)}"
                )
        if shot.get("extract_character_ref"):
            reference_issue = _identity_reference_issue(shot, character_modes)
            if reference_issue:
                shot["extract_character_ref"] = False
                print(
                    f"   [角色参考校正] Shot {shot.get('shot_id', index + 1)} "
                    f"{reference_issue}，已禁用身份参考提取"
                )

    for correction in _normalize_continuity_contract(storyboard["shots"]):
        print(f"   [连续性校正] {correction}，改为 intentional_cut")
    _apply_coverage_defaults(
        storyboard["shots"],
        character_names,
        storyboard.get("theme_elements", []),
    )
    for shot_id, previous_id in normalize_narrative_handoffs(storyboard):
        print(
            f"   [故事状态校正] Shot {shot_id} state_before "
            f"复用 Shot {previous_id} state_after"
        )
    for correction in normalize_causal_scope(storyboard["shots"]):
        print(f"   [因果范围校正] {correction}")


def _compile_storyboard_contract(
    storyboard: dict,
    aspect_ratio: str,
    resolution: str,
    style: str,
    *,
    production_plan: dict | None = None,
) -> dict:
    """Normalize LLM labels before compiling their deterministic consequences."""
    if production_plan is not None:
        apply_production_plan(storyboard, production_plan)
    _apply_defaults(storyboard, aspect_ratio, resolution, style)
    normalized = validate_storyboard_draft(storyboard)
    if production_plan is not None:
        apply_production_plan(normalized, production_plan)
    _apply_defaults(normalized, aspect_ratio, resolution, style)
    return validate_storyboard(normalized)


def _normalize_positive_duration(value):
    """Project a positive whole-second LLM hint onto the executable API range."""
    duration = None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        duration = value
    elif isinstance(value, float) and value.is_integer():
        duration = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\+?\d+", value.strip()):
        duration = int(value)
    if duration is None or duration <= 0:
        return value
    return min(
        config.MAX_SHOT_DURATION,
        max(config.MIN_SHOT_DURATION, duration),
    )


def _build_correction_prompt(
    original_prompt: str,
    storyboard: dict,
    warnings: list[str],
    *,
    batch_scoped: bool = False,
) -> str:
    """构建修正 prompt — 将校验问题作为 feedback 让 LLM 修正"""
    issues_text = "\n".join(f"- {w}" for w in warnings)
    storyboard_json = json.dumps(storyboard, ensure_ascii=False, indent=2)
    action_repair = ""
    if any("动作重心不足" in warning for warning in warnings):
        action_repair = """
- 本批次只按 ProductionPlan 中的 narrative_function、allowed_effect_phases、coverage_roles 和 requires_visible_result 修正动作职责；不得引用或补写批次外镜头
- primary_action 必须直接执行该槽位的冲突推进或可见结果，运镜、扫描和准备不得代替槽位职责
- 每镜仍只保留一个主动作，对手反应写入 visible_result，不得把全片动作配额机械塞进每个批次
""" if batch_scoped else """
- 至少 2 个镜头的 primary_action 必须直接攻击、防守、受击、反击或追逐，并在 action_beats 写明 actor、target 和 visible_result
- 纯建立与纯收尾只能保留一个：若 Shot 1 是纯环境、扫描或准备，最后一镜必须“动作中收束”；若最后一镜保留简短余韵，Shot 1 必须“冲突中建立”
- 每镜仍只保留一个主动作，不得把动作配额塞进同一镜头
"""
    coverage_repair = ""
    if any(
        marker in warning
        for warning in warnings
        for marker in (
            "多个主动动作执行者", "极近景无法", "远处目标与近景浅焦",
            "同框交互", "交互镜头缺少",
        )
    ):
        coverage_repair = """
- 先确定 coverage_role 和唯一主动 actor；对手反应只写入 visible_result
- 要求命中可见时，将 actor/target 都列入 required_visible_entities，使用中景或全景让双方同框并保持攻击线清晰
- 武器细节必须使用独立 insert，目标受击使用独立 target_reaction；不得在极近景浅焦镜头同时要求远处目标清晰命中
"""
    if any(
        marker in warning
        for warning in warnings
        for marker in (
            "动作结果视角缺失", "动作镜头职责集中", "动作景别层次不足",
        )
    ):
        coverage_repair += """
- 长动作段落只调整 coverage_role、景别和对应 prompt_en：保留角色、场景、空间轴、时长、动作因果与连续性字段
- coverage 必须服务因果：在空间/交互、动作主体、目标反应或关键细节、结果收束中选择适合当前剧情的职责，不机械套模板
- 至少用 wide、medium、close/detail 三类景别建立空间、看清交互并展示一次结果；近景/细节镜头不得承担远距离双方同框命中
"""
    causality_repair = ""
    if any(
        marker in warning
        for warning in warnings
        for marker in (
            "interaction_mode", "作用来源", "作用区域",
            "反应范围", "范围外行为", "作用路径", "effect_phase",
            "outcome_scope", "effect_motion", "结果范围", "准备阶段",
            "生效阶段", "aftermath", "全部目标",
        )
    ):
        causality_repair = """
- 只按物理传播方式选择 interaction_mode：直接接触 direct_contact、有方向路径 directed_path、区域作用 area_effect、中介触发 indirect_effect
- effect_phase 必须按真实时间阶段选择：准备/瞄准/充能用 setup，物理作用发生用 active，只展示既有结果用 aftermath，无交互用 none
- setup/none 必须使用 interaction_mode=none、outcome_scope=none、effect_motion=none，不得提前画出有效攻击路径或目标反应
- active 必须声明 outcome_scope=single/subset/all 和 effect_motion=static/sweep/expand/propagate；静止 directed_path 不能宣称影响全部目标，全部目标必须实际展示 sweep
- aftermath 不得创建新作用，也不得把上一 active 已影响的 single/subset 无原因扩大成 all
- 明确 source、effect_region、reaction_scope、unaffected_behavior；只有作用区域内主体可产生结果，范围外主体必须保持契约指定的行为
- 不得把 area_effect 强制改成直线命中，也不得用武器名或具体题材代替作用范围描述
"""
    participant_repair = ""
    if any(
        marker in warning
        for warning in warnings
        for marker in (
            "未定义角色",
            "characters 未包含实际可见角色",
            "引用未注册实体",
        )
    ):
        participant_repair = """
- actor 和角色 target 必须逐字复用顶层 characters.name，包括 characters、action_beats、interaction_geometry 与 blocking 的角色引用
- 物件 target 必须逐字复用本镜 key_props/continuity_props；阶段性角色描述只能写入动作或画面的自然语言字段，不得创建新的结构化角色 ID
"""
    narrative_repair = ""
    if any(
        marker in warning
        for warning in warnings
        for marker in ("story_arc", "narrative_beat", "故事状态")
    ):
        narrative_repair = """
- 完整填写 story_arc 的 goal/stakes/turning_point/resolution，使结局是画面可见的状态而非抽象评价
- 每镜 narrative_beat 必须以一个可见动作、信息或结果改变故事状态；运镜和气氛不能充当 state_change
- 后镜 state_before 必须逐字复用前镜 state_after；只修复断裂处，不重写已连贯的角色、场景和动作
"""

    return f"""{original_prompt}

---

⚠️ 上一次生成的分镜存在以下问题, 请修正后重新输出完整 JSON:

{issues_text}

修正原则:
- 只修改导致告警的创意字段；ProductionPlan 及其镜头数量、shot_id、duration、阶段、结果槽位、景别族和参考策略不可改写
- 保留已经通过校验的角色、场景、空间轴与连续故事状态
{action_repair}
{coverage_repair}
{causality_repair}
{participant_repair}
{narrative_repair}

上次生成的分镜 (需要修正):
```json
{storyboard_json}
```

请根据上述问题修正分镜, 输出修正后的完整 JSON。只输出 JSON, 不要解释。"""


def _merge_missing_mapping(original: object, corrected: object) -> dict:
    source = original if isinstance(original, dict) else {}
    current = corrected if isinstance(corrected, dict) else {}
    merged = deepcopy(source)
    for key, value in current.items():
        if value not in (None, "", [], {}):
            merged[key] = deepcopy(value)
    return merged


def _preserve_correction_contract(original: dict, corrected: dict) -> None:
    """Keep accepted spatial/continuity data when a surgical LLM correction omits it."""
    if not isinstance(corrected, dict):
        return

    for key in ("characters", "theme_elements", "music_style"):
        if key not in corrected or corrected[key] in (None, "", [], {}):
            if key in original:
                corrected[key] = deepcopy(original[key])
    if isinstance(original.get("story_arc"), dict):
        corrected["story_arc"] = _merge_missing_mapping(
            original["story_arc"], corrected.get("story_arc", {})
        )

    original_shots = {
        shot.get("shot_id"): shot
        for shot in original.get("shots", [])
        if isinstance(shot, dict) and shot.get("shot_id") is not None
    }
    for shot in corrected.get("shots", []):
        if not isinstance(shot, dict):
            continue
        source = original_shots.get(shot.get("shot_id"))
        if not source:
            continue

        for key in ("shot_id", "duration", "scene_id", "continuity_from_previous"):
            if key not in shot or shot[key] in (None, "", [], {}):
                if key in source:
                    shot[key] = deepcopy(source[key])

        for key in ("start_state", "end_state"):
            if isinstance(source.get(key), dict):
                shot[key] = _merge_missing_mapping(source[key], shot.get(key, {}))
        if isinstance(source.get("narrative_beat"), dict):
            shot["narrative_beat"] = _merge_missing_mapping(
                source["narrative_beat"], shot.get("narrative_beat", {})
            )

        role = str(shot.get("coverage_role", "")).strip()
        current_characters = shot.get("characters")
        single_focus = (
            role in {"target_reaction", "insert", "aftermath"}
            and isinstance(current_characters, list)
            and len(current_characters) == 1
        )
        if not single_focus:
            if isinstance(source.get("characters"), list):
                current_characters = current_characters if isinstance(current_characters, list) else []
                shot["characters"] = list(dict.fromkeys([
                    *source["characters"],
                    *current_characters,
                ]))
            if isinstance(source.get("required_visible_entities"), list):
                current_visible = shot.get("required_visible_entities")
                current_visible = current_visible if isinstance(current_visible, list) else []
                shot["required_visible_entities"] = list(dict.fromkeys([
                    *source["required_visible_entities"],
                    *current_visible,
                ]))

        source_camera = source.get("camera")
        current_camera = shot.get("camera")
        if not isinstance(current_camera, dict) or not current_camera:
            shot["camera"] = deepcopy(source_camera) if isinstance(source_camera, dict) else {}
            if single_focus:
                shot["camera"].pop("screen_positions", None)
            current_camera = shot["camera"]
        elif isinstance(source_camera, dict):
            if not single_focus:
                source_positions = source_camera.get("screen_positions")
                current_positions = current_camera.get("screen_positions")
                if isinstance(source_positions, dict):
                    current_camera["screen_positions"] = {
                        **deepcopy(source_positions),
                        **(deepcopy(current_positions) if isinstance(current_positions, dict) else {}),
                    }
            if not str(current_camera.get("axis_change", "")).strip():
                current_camera["axis_change"] = source_camera.get("axis_change", "hold")

        if not single_focus:
            source_blocking = source.get("blocking")
            current_blocking = shot.get("blocking")
            if isinstance(source_blocking, dict):
                current_blocking = current_blocking if isinstance(current_blocking, dict) else {}
                shot["blocking"] = {
                    name: _merge_missing_mapping(intent, current_blocking.get(name, {}))
                    for name, intent in source_blocking.items()
                }
                shot["blocking"].update({
                    name: value
                    for name, value in current_blocking.items()
                    if name not in shot["blocking"]
                })

            source_geometry = source.get("interaction_geometry")
            if isinstance(source_geometry, dict):
                shot["interaction_geometry"] = _merge_missing_mapping(
                    source_geometry,
                    shot.get("interaction_geometry", {}),
                )


def _parse_json_response(text: str) -> dict:
    """从 LLM 响应中提取 JSON (兼容 markdown 包裹)"""
    if not text or not text.strip():
        raise json.JSONDecodeError("LLM 返回空响应", text or "", 0)

    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 中的内容
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # 尝试找第一个 { 到最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise json.JSONDecodeError(f"无法从响应中提取 JSON", text[:200], 0)


def _load_system_prompt() -> str:
    """加载分镜系统 prompt"""
    prompt_file = config.PROMPTS_DIR / "storyboard_system.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    # Fallback 内置 prompt
    return _BUILTIN_SYSTEM_PROMPT


_BUILTIN_SYSTEM_PROMPT = """你是一个题材适应能力很强的影视导演兼分镜师。你的目标不是"描述画面"，而是设计一段让观众看完就想再看一遍的视觉体验。

## 最高优先级规则

1. 主题锚定: ≥ 60% 的镜头必须包含与用户指定主题直接相关的视觉元素 (产品本体/使用过程/产出物/原材料), 产品不能只在 insert shot 出现
2. 场景连续性: scene_id 只表示物理地点; seamless 只表示无剪切续拍, 同一地点换机位/插入特写仍用 intentional_cut
3. 景别与剪辑: seamless 保持相同/相邻景别且 composition_change=small；intentional_cut 再按实际覆盖变化填写 small/medium/large，不强迫跳级
4. 情绪弧线: mood 随叙事阶段变化，允许连续动作保持一致基调，不为字段多样性制造突变
5. Insert Shot: 产品/品牌展示类至少 1 个; 动作/剧情/氛围类按叙事需要决定
6. prompt_en 词数 80-150 词
7. 每个镜头时长 4-15 秒 (整数)
8. 物件连续性: key_props 描述可见元素; continuity_props 只追踪同场景跨镜持续的可移动叙事道具, 新增持久道具必须有引入动作或前镜铺垫
9. 环境一致性: 同场景连续镜头的天气/光线/背景物不得突变
11. 镜头数量: 10s=2镜, 15s=3镜, 20s=3-4镜, 30s=4-6镜, 60s=8-10镜
12. 动作契约: 每镜只有一个 primary_action, 最多包含两个紧密因果动作动词; 多次接触/破坏动作必须拆镜, 并填写 start_state/end_state
13. 题材适配: 动作请求用多个主体攻防节拍推进; 运镜只写入 camera, 不得充当 primary_action
14. 物理能力: characters 必须声明 mobility; 动作必须符合角色移动形态, 履带/轮式角色不得迈步、跑跳或踢击
15. 空间轴: 多角色动作镜头必须在 camera.screen_positions 声明角色左右位置并保持轴线; axis_change 只用 establish/hold/reestablish, 没有越轴写 hold
16. 动作调度: 动作镜头用 action_beats 写 1-3 个 trigger/peak/aftermath 因果阶段; 同一镜头的 beat 只能有一个主动 actor，目标反应写 visible_result; 多角色动作镜头还必须在 blocking 为每个角色声明 frame_position/body_orientation/facing_target/eyeline_target/action_target，武器和身体必须朝向实际目标
17. 可拍摄性: 每镜填写 composition_change、coverage_role、required_visible_entities 和 interaction_geometry；接触要求双方同框，定向作用要求路径可见，区域/间接作用要求作用范围与反应目标可读
18. 身份参考预算: 动作片不得为了提取身份单独占用静态镜头; 角色可以在多人冲突镜头首次出现且 extract_character_ref=false，只有自然的单一 identity 角色清晰镜头才可提取，其他镜头由生成阶段从顶层 characters 自动注入完整外观
19. 结构化主题: 顶层 theme_elements 使用稳定英文ID; 不得用中文标题与英文 prompt 做字符串匹配
20. 长动作 coverage: 24 秒以上且至少 4 镜时，根据剧情组合空间/交互、动作主体、目标反应或关键细节、结果收束，并覆盖 wide、medium、close/detail 三类景别；变化服务因果，不机械套固定顺序或运镜
21. 交互因果: 每镜 interaction_geometry 必须填写 effect_phase/outcome_scope/effect_motion；setup 不得产生物理作用，active 才能选择 direct_contact/directed_path/area_effect/indirect_effect，aftermath 不得扩大上一镜已建立的结果范围
22. 作用守恒: 静止 directed_path 只能影响路径相交的 single/subset；要影响 all 必须清楚展示 sweep，只有作用范围内主体可反应
23. 故事状态: 顶层 story_arc 填写 goal/stakes/turning_point/resolution；每镜 narrative_beat 填写 function/state_before/state_change/state_after，后镜 state_before 必须逐字复用前镜 state_after
24. 实体目录: actor 和角色 target 必须逐字复用顶层 characters.name，包括 characters、action_beats、interaction_geometry 与 blocking 的角色引用；物件 target 必须逐字复用本镜 key_props/continuity_props；阶段性角色描述只能写入自然语言字段，不得创建新 ID

## Seedance 模型局限 (必须规避)

- 禁止 prompt 中要求画面出现文字/Logo/数字 (用 subtitle_text 代替)
- 单镜头最多 2 个角色互动, 超过用远景
- 手部特写追加 negative_prompt: anatomically correct hands
- duration > 10s 的镜头用 fixed 运镜 + 简单动作
- fast speed 镜头 duration ≤ 5s

## 运镜 6 步公式
[Subject] -> [Action] -> [Environment] -> [Camera] -> [Style] -> [Constraints]

## 道具字段 (必须)
每个 shot 必须包含 key_props 和 continuity_props。key_props 列出可见关键元素；continuity_props 只列出同场景跨镜持续的可移动叙事道具，不含环境、尸体状态、烟尘、弹壳、枪口火焰等效果。
同场景新增 continuity_props 必须在 prompt_en 中包含引入动作。

## 输出格式 (严格 JSON, 参考 storyboard_system.md)
注意: 身份参考是可选增强，不是首次出场硬约束；动作片允许角色在多人冲突镜头首次出现，只有自然的单一 identity 角色清晰镜头才提取
subtitle_text 必须中文; prompt_en 必须英文
顶层必须包含 content_focus、theme_elements 和 story_arc；每个 shot 必须包含 scene_id, continuity_from_previous, composition_change, coverage_role, required_visible_entities, interaction_geometry, narrative_beat, primary_action, action_beats, start_state, end_state, blocking, key_props, continuity_props
最多连续 2 次 seamless; 插入特写、换机位或大跨度景别变化必须 intentional_cut
"""


def _extract_scene_name(scene_description: str) -> str:
    """从 scene_description 中提取【】包裹的场景名称"""
    m = re.search(r'[\u3010\[][^\u3011\]]+[\u3011\]]', scene_description)
    return m.group(0) if m else scene_description[:20]


def _scene_id(shot: dict) -> str:
    """返回稳定场景 ID；旧分镜才回退到展示名称。"""
    explicit = shot.get("scene_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _extract_scene_name(shot.get("scene_description", ""))


def _should_use_previous_tail_reference(
    current: dict,
    previous: dict | None,
    *,
    has_identity_reference: bool | None = None,
) -> bool:
    """Return whether accepted footage should carry state into this shot."""
    if previous is None or _scene_id(current) != _scene_id(previous):
        return False
    policy = reference_policy(current)
    if policy in {"independent", "identity_only"}:
        return False
    if policy in {
        "state_if_same_scene", "state_and_identity", "identity_or_state",
    }:
        # identity_or_state is a persisted v1 policy. Treat it as the corrected
        # state-and-identity behavior so old workspaces do not lose scene state.
        return True
    continuity = current.get("continuity_from_previous")
    if continuity == "seamless":
        return True
    return continuity == "intentional_cut"


# 用于检测 prompt 中是否包含物件引入动作的关键词
_INTRODUCTION_VERBS = {
    "placing", "places", "place",
    "bringing", "brings", "bring",
    "carrying", "carries", "carry",
    "picking", "picks", "pick",
    "pulling", "pulls", "pull",
    "pouring", "pours", "pour",
    "handing", "hands", "hand",
    "setting", "sets", "set",
    "putting", "puts", "put",
    "grabbing", "grabs", "grab",
    "reaching", "reaches", "reach",
    "lifting", "lifts", "lift",
    "opening", "opens", "open",
    "removing", "removes", "remove",
    "sliding", "slides", "slide",
    "dropping", "drops", "drop",
    "tossing", "tosses", "toss",
    "holding", "holds", "hold",
    "unwrapping", "unwraps",
    "revealing", "reveals",
}


def _check_temporal_causality(shots: list[dict]) -> list[str]:
    """时序因果校验 — 检测"先享用后制作"等因果倒置

    通过 prompt_en 中的关键词语义分析, 检测消费行为出现在生产行为之前的情况。
    仅对相邻镜头检测 (非线性叙事可能跨越多镜头, 此处保守检测)。
    """
    # 语义类别: 每个关键词归属一个阶段
    # 阶段权重: 制作准备 < 成品产出 < 享用消费
    _PHASE_KEYWORDS = {
        # 制作/准备 (phase=1)
        1: {
            "grinding", "grind", "grinds",
            "brewing", "brew", "brews",
            "roasting", "roast", "roasts",
            "extracting", "extract", "extracts",
            "steaming", "frothing", "froth",
            "tamping", "tamp",
            "preparing", "prepare", "prepares",
            "crafting", "craft", "crafts",
            "mixing", "mix", "mixes",
            "kneading", "knead",
            "chopping", "chop",
            "cooking", "cook", "cooks",
            "assembling", "assemble",
            "pouring into",  # 倒入 (制作阶段)
        },
        # 成品/完成 (phase=2)
        2: {
            "freshly brewed", "freshly made",
            "finished", "completed", "ready",
            "plating", "plate", "plates",
            "garnishing", "garnish",
        },
        # 享用/消费 (phase=3)
        3: {
            "drinking", "drink", "drinks",
            "sipping", "sip", "sips",
            "tasting", "taste", "tastes",
            "enjoying", "enjoy", "enjoys",
            "savoring", "savor", "savors",
            "eating", "eat", "eats",
            "biting", "bite", "bites",
            "lifting cup to lips",
            "raising the cup",
            "takes a sip",
        },
    }

    warnings: list[str] = []

    def _detect_phase(prompt: str) -> int | None:
        """检测 prompt 主要属于哪个阶段 (返回最高阶段)"""
        prompt_lower = prompt.lower()
        detected = None
        for phase, keywords in _PHASE_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    if detected is None or phase > detected:
                        detected = phase
                    break  # 该阶段已匹配, 继续检查更高阶段
        return detected

    for i in range(1, len(shots)):
        prev_prompt = shots[i - 1].get("prompt_en", "")
        curr_prompt = shots[i].get("prompt_en", "")

        prev_phase = _detect_phase(prev_prompt)
        curr_phase = _detect_phase(curr_prompt)

        if prev_phase is not None and curr_phase is not None:
            if prev_phase > curr_phase:
                phase_names = {1: "制作/准备", 2: "成品/完成", 3: "享用/消费"}
                warnings.append(
                    f"⚠ 时序倒置: Shot {shots[i-1].get('shot_id', i)} "
                    f"是 '{phase_names[prev_phase]}' 阶段, "
                    f"但 Shot {shots[i].get('shot_id', i+1)} "
                    f"退回到 '{phase_names[curr_phase]}' 阶段 — 因果顺序可能不合理"
                )

    return warnings


def _check_prop_continuity(shots: list[dict]) -> list[str]:
    """物件连续性检测 — 检查同场景内新增道具是否有引入动作

    检测逻辑:
    1. 提取每个 shot 的 continuity_props 和场景名称
    2. 如果当前镜头与上一镜头是同场景，检查新增的持久道具
    3. 对于新增的 prop，检查 prompt_en 中是否包含引入动作词
    4. 缺少引入动作的新增 prop → 预警
    """
    warnings: list[str] = []

    for i in range(1, len(shots)):
        prev_shot = shots[i - 1]
        curr_shot = shots[i]

        # 提取场景名
        prev_scene = _scene_id(prev_shot)
        curr_scene = _scene_id(curr_shot)

        # 场景切换豁免: 不同场景的物件可以直接存在
        if prev_scene != curr_scene:
            continue

        # 同场景: 检查新增 props
        prev_props = set(
            p.lower().strip() for p in prev_shot.get("continuity_props", [])
        )
        curr_props = set(
            p.lower().strip() for p in curr_shot.get("continuity_props", [])
        )
        new_props = curr_props - prev_props

        if not new_props:
            continue

        # 检查 prompt_en 中是否有引入动作
        prompt = curr_shot.get("prompt_en", "").lower()
        prompt_words = set(prompt.split())
        has_intro_verb = bool(prompt_words & _INTRODUCTION_VERBS)

        if not has_intro_verb:
            prop_list = ", ".join(sorted(new_props))
            warnings.append(
                f"⚠ 物件凭空出现: Shot {curr_shot.get('shot_id', i+1)} "
                f"在同场景 '{curr_scene}' 中新增了 [{prop_list}], "
                f"但 prompt 中未检测到引入动作 (placing/bringing/carrying 等)"
            )

    return warnings


def _theme_anchor_coverage(storyboard: dict) -> tuple[int, int, list[str]]:
    """Measure subject presence from structured IDs, never translated title text."""
    shots = storyboard.get("shots", [])
    character_names = {
        str(character.get("name", "")).strip().lower()
        for character in storyboard.get("characters", [])
        if str(character.get("name", "")).strip()
    }
    theme_elements = {
        str(element).strip().lower()
        for element in storyboard.get("theme_elements", [])
        if str(element).strip()
    }
    anchors = sorted(character_names | theme_elements)
    if not anchors:
        return len(shots), len(shots), []

    anchored_count = 0
    for shot in shots:
        shot_characters = {
            str(name).strip().lower() for name in shot.get("characters", [])
        }
        if shot_characters & character_names:
            anchored_count += 1
            continue
        searchable = " ".join([
            str(shot.get("prompt_en", "")).lower(),
            " ".join(str(prop).lower() for prop in shot.get("key_props", [])),
            " ".join(
                str(prop).lower() for prop in shot.get("continuity_props", [])
            ),
        ])
        if any(anchor.replace("_", " ") in searchable for anchor in theme_elements):
            anchored_count += 1
    return anchored_count, len(shots), anchors


def _shot_advances_action_conflict(shot: dict) -> bool:
    """Use both the compact action and structured beats to detect real conflict."""
    geometry = shot.get("interaction_geometry")
    if (
        isinstance(geometry, dict)
        and geometry.get("effect_phase") == "active"
        and geometry.get("interaction_mode") in CAUSAL_INTERACTION_MODES
    ):
        return True
    if _advances_action_conflict(str(shot.get("primary_action", ""))):
        return True
    for beat in shot.get("action_beats", []):
        if not isinstance(beat, dict):
            continue
        action_text = " ".join([
            str(beat.get("action", "")),
            str(beat.get("visible_result", "")),
        ])
        if _advances_action_conflict(action_text):
            return True
    return False


def _validate_storyboard_richness(
    storyboard: dict,
    user_request: str = "",
    *,
    require_narrative_contract: bool = False,
) -> tuple[list[str], bool]:
    """分镜丰富度校验 — 返回 (warnings, is_critical)

    is_critical=True 时触发 LLM 自动修正。
    严重问题 (触发重试): 情绪扁平、题材意图未覆盖、prompt 词数严重越界
    轻微问题 (仅警告): 连续相同景别、物件连续性、时序因果

    检测维度:
    1. 场景多样性
    2. 景别跳跃
    3. 情绪弧线
    4. 题材相关的 Insert shot
    5. 物件连续性
    6. 时序因果
    7. 主题锚定度
    8. prompt_en 词数
    """
    shots = storyboard.get("shots", [])
    if not shots:
        return [], False

    warnings: list[str] = []
    critical_count = 0  # 严重问题计数
    content_focus = _infer_content_focus(user_request)

    from pipeline.readiness import storyboard_readiness_issues

    for issue in storyboard_readiness_issues(storyboard):
        warnings.append(f"🚨 {issue}")
        critical_count += 1

    # --- 1. 场景连续性 ---
    scene_names = [
        _scene_id(s)
        for s in shots
    ]

    unique_scenes = set(scene_names)
    if len(unique_scenes) == 1 and len(shots) >= 3:
        warnings.append(
            f"⚠ 场景集中: 所有 {len(shots)} 个镜头都在 '{scene_names[0]}', "
            "请确认每镜头有不同动作或叙事推进"
        )

    # --- 2. 景别跳跃 ---
    framings = [
        s.get("camera", {}).get("start_framing", "unknown")
        for s in shots
    ]
    for i in range(1, len(framings)):
        if framings[i] == framings[i - 1] and framings[i] != "unknown":
            warnings.append(
                f"⚠ 景别重复: Shot {i} 和 Shot {i + 1} "
                f"都是 '{framings[i]}', 建议景别跳跃"
            )

    # --- 3. 情绪弧线 ---
    moods = [s.get("mood", "unknown") for s in shots]
    unique_moods = set(moods)
    if len(unique_moods) == 1 and len(shots) >= 2:
        warnings.append(
            f"🚨 情绪扁平: 所有镜头情绪都是 '{moods[0]}', "
            f"缺乏弧线变化, 必须为每个镜头设置不同情绪"
        )
        critical_count += 1
    for i in range(1, len(moods)):
        if moods[i] == moods[i - 1] and moods[i] != "unknown":
            warnings.append(
                f"⚠ 连续相同情绪: Shot {i} 和 Shot {i + 1} "
                f"都是 '{moods[i]}'"
            )

    # --- 4. 题材相关的 Insert shot ---
    has_insert = any(not s.get("characters") for s in shots)
    if content_focus == "product" and not has_insert and len(shots) >= 3:
        warnings.append(
            "🚨 缺少 insert shot: 产品展示请求的所有镜头都包含角色, "
            "必须加入至少 1 个产品细节或纯产品镜头"
        )
        critical_count += 1

    # --- 5. 时序因果校验 ---
    if _requires_process_order(user_request):
        warnings.extend(_check_temporal_causality(shots))

    # --- 6. 物件连续性 ---
    warnings.extend(_check_prop_continuity(shots))

    # --- 7. 主题锚定度 ---
    if len(shots) >= 3:
        anchored_count, shot_count, theme_anchors = _theme_anchor_coverage(storyboard)
        ratio = anchored_count / shot_count if shot_count else 1.0
        if ratio < 0.6:
            warnings.append(
                f"🚨 主题锚定不足: 仅 {anchored_count}/{shot_count} 个镜头"
                f"包含结构化主题元素 [{', '.join(theme_anchors)}] "
                f"(占比 {ratio:.0%}, 要求 ≥60%)"
            )
            critical_count += 1

    # --- 8. prompt_en 词数校验 ---
    for s in shots:
        prompt = s.get("prompt_en", "")
        word_count = len(prompt.split())
        shot_id = s.get("shot_id", "?")
        if word_count < 50:
            warnings.append(
                f"🚨 Shot {shot_id} prompt_en 过短: {word_count} 词 "
                f"(要求 80-150 词), 画面细节严重不足"
            )
            critical_count += 1
        elif word_count < 80:
            warnings.append(
                f"⚠ Shot {shot_id} prompt_en 偏短: {word_count} 词 "
                f"(建议 80-150 词), 建议补充环境/光线/真实感细节"
            )
        elif word_count > 180:
            warnings.append(
                f"⚠ Shot {shot_id} prompt_en 过长: {word_count} 词 "
                f"(建议 80-150 词), 过长可能导致模型忽略部分描述"
            )

    # --- 9. 单镜动作契约 ---
    state_keys = ("location", "subject", "action_phase", "camera")
    for shot in shots:
        missing = []
        primary_action = str(shot.get("primary_action", "")).strip()
        if not primary_action:
            missing.append("primary_action")
        for state_name in ("start_state", "end_state"):
            state = shot.get(state_name)
            if not isinstance(state, dict) or any(
                not str(state.get(key, "")).strip() for key in state_keys
            ):
                missing.append(state_name)
        if missing:
            warnings.append(
                f"🚨 Shot {shot.get('shot_id', '?')} 动作契约不完整: "
                f"{', '.join(missing)}, 必须明确一个主动作及完整起止状态"
            )
            critical_count += 1
        elif _camera_is_primary_subject(primary_action):
            warnings.append(
                f"🚨 Shot {shot.get('shot_id', '?')} 运镜不能充当 primary_action；"
                "必须改写为主体可见动作，并把摄影机运动留在 camera 字段"
            )
            critical_count += 1

        beats = _physical_action_beats(primary_action)
        contact_beats = [beat for beat in beats if beat in _CONTACT_ACTIONS]
        has_explicit_sequence = any(
            separator in primary_action.lower()
            for separator in _SEQUENCE_SEPARATORS
        )
        if (
            len(contact_beats) > 1
            or (has_explicit_sequence and len(beats) > 2)
            or len(beats) > 3
        ):
            warnings.append(
                f"🚨 Shot {shot.get('shot_id', '?')} 动作节拍过载: "
                f"primary_action 包含 {len(beats)} 个物理动作阶段 "
                f"({', '.join(beats)})；每镜最多保留两个紧密因果阶段，"
                "多次接触/破坏必须拆成独立镜头"
            )
            critical_count += 1

        shot_characters = shot.get("characters", [])
        if len(shot_characters) == 1:
            character = next(
                (
                    candidate for candidate in storyboard.get("characters", [])
                    if candidate.get("name") == shot_characters[0]
                ),
                None,
            )
            if character:
                mobility = _character_mobility(character)
                conflict_pattern = _MOBILITY_CONFLICT_PATTERNS.get(mobility)
                if conflict_pattern and re.search(
                    conflict_pattern, primary_action.lower()
                ):
                    warnings.append(
                        f"🚨 Shot {shot.get('shot_id', '?')} 移动形态冲突: "
                        f"角色 '{shot_characters[0]}' 是 {mobility}，"
                        f"不能执行 '{primary_action}'；请改写为符合其结构的动作"
                    )
                    critical_count += 1

    # --- 10. 动作场景空间轴 ---
    if content_focus == "action":
        for previous, current in zip(shots, shots[1:]):
            if _scene_id(previous) != _scene_id(current):
                continue
            if _screen_axis_flipped(previous, current):
                warnings.append(
                    f"🚨 Shot {current.get('shot_id', '?')} 空间轴反转: "
                    "相邻镜头交换了角色左右位置；保持原轴线，或先用 "
                    "camera.axis_change='reestablish' 的建立镜头重新定向"
                )
                critical_count += 1

    # --- 11. 动作题材意图覆盖 ---
    if content_focus == "action":
        total_duration = sum(
            max(0, int(shot.get("duration", 0) or 0))
            for shot in shots
        )
        action_flags = [
            (shot, _shot_advances_action_conflict(shot))
            for shot in shots
        ]
        action_shots = [shot for shot, advances in action_flags if advances]
        action_duration = sum(
            max(0, int(shot.get("duration", 0) or 0))
            for shot in action_shots
        )
        enough_runtime_for_sequence = total_duration >= 15 and len(shots) >= 3
        action_ratio = action_duration / total_duration if total_duration else 0
        if (
            not action_shots
            or (
                enough_runtime_for_sequence
                and (len(action_shots) < 2 or action_ratio < 0.4)
            )
        ):
            assessment = "; ".join(
                f"Shot {shot.get('shot_id', '?')}="
                f"{'推进' if advances else '非推进'}"
                f"({str(shot.get('primary_action', '')).strip()[:80]})"
                for shot, advances in action_flags
            )
            warnings.append(
                f"🚨 动作重心不足: 直接推进攻防/追逐的镜头仅 "
                f"{len(action_shots)}/{len(shots)} 个、{action_duration}/{total_duration}s；"
                "应压缩纯建立、准备和离场镜头，拆成多个单一且可执行的动作节拍。"
                f"当前判定: {assessment}"
            )
            critical_count += 1

        if total_duration >= 24 and len(shots) >= 4:
            coverage_roles = [
                str(shot.get("coverage_role", "")).strip()
                for shot in shots
                if str(shot.get("coverage_role", "")).strip()
            ]
            result_roles = {"target_reaction", "insert"}
            if not result_roles.intersection(coverage_roles):
                warnings.append(
                    "🚨 动作结果视角缺失: 24 秒以上动作段落必须至少用一个 "
                    "target_reaction 或有叙事价值的 insert 展示受击、关键细节或局势变化"
                )
                critical_count += 1

            if coverage_roles:
                dominant_count = max(coverage_roles.count(role) for role in set(coverage_roles))
                if dominant_count / len(coverage_roles) > 0.75:
                    warnings.append(
                        f"🚨 动作镜头职责集中: {dominant_count}/{len(coverage_roles)} 个镜头"
                        "承担同一种 coverage_role；应按动作因果分配主体、交互、反应或结果视角"
                    )
                    critical_count += 1

            framing_families = {
                family
                for shot in shots
                if (family := _framing_family(
                    shot.get("camera", {}).get("start_framing")
                    if isinstance(shot.get("camera"), dict) else ""
                ))
            }
            if len(framing_families) < 3:
                warnings.append(
                    "🚨 动作景别层次不足: 24 秒以上动作段落应按因果覆盖 "
                    "wide、medium、close/detail 三类景别；近景只用于可读的反应或关键细节"
                )
                critical_count += 1

    # --- 12. 角色参考镜头可用性 ---
    character_modes = {
        character.get("name"): character.get("reference_mode", "identity")
        for character in storyboard.get("characters", [])
    }
    for shot in shots:
        if not shot.get("extract_character_ref"):
            continue
        reference_issue = _identity_reference_issue(shot, character_modes)
        if reference_issue:
            warnings.append(
                f"🚨 Shot {shot.get('shot_id', '?')} {reference_issue}"
            )
            critical_count += 1

    # --- 13. Seedance 快动作预算 ---
    for shot in shots:
        speed = str(shot.get("camera", {}).get("speed", "")).lower()
        movement = str(
            shot.get("camera", {}).get("primary_movement", "")
        ).lower()
        duration = shot.get("duration", config.DEFAULT_DURATION)
        if duration > 5 and (
            speed in {"fast", "quick", "rapid"} or "fast" in movement
        ):
            warnings.append(
                f"🚨 Shot {shot.get('shot_id', '?')} fast 动作时长为 {duration}s, "
                "超过 Seedance 稳定预算 5s, 必须缩短或降速"
            )
            critical_count += 1

    # 判定是否达到「严重」阈值 (≥1 个严重问题触发重试)
    is_critical = critical_count >= 1

    return warnings, is_critical

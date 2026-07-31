"""Stage 1: LLM 分镜生成 — 创意策略 + 运镜公式 + 丰富度校验"""

from __future__ import annotations

import re
import json
from pathlib import Path

from openai import OpenAI

import config
from pipeline.models import validate_storyboard


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

    user_prompt = f"""用户需求: {user_request}

目标参数:
- 总时长: ~{target_duration} 秒
- 画面比例: {aspect_ratio}
- 分辨率: {resolution}
- 整体风格: {style}

请生成完整的分镜脚本 (JSON 格式)。"""

    storyboard = _call_llm_for_storyboard(client, system_prompt, user_prompt)

    # 补充默认值
    _apply_defaults(storyboard, aspect_ratio, resolution, style)
    storyboard = validate_storyboard(storyboard)

    # 丰富度校验 + 自动修正 (严重问题触发 LLM 重跑)
    warnings, is_critical = _validate_storyboard_richness(storyboard)
    if warnings:
        print("\n   [丰富度校验]")
        for w in warnings:
            print(f"   {w}")

    if is_critical:
        print("\n   [自动修正] 检测到严重问题, 重新生成分镜...")
        correction_prompt = _build_correction_prompt(user_prompt, storyboard, warnings)
        storyboard = _call_llm_for_storyboard(client, system_prompt, correction_prompt)
        _apply_defaults(storyboard, aspect_ratio, resolution, style)
        storyboard = validate_storyboard(storyboard)

        # 二次校验 (仅打印, 不再重试)
        warnings_2, _ = _validate_storyboard_richness(storyboard)
        if warnings_2:
            print("\n   [二次校验]")
            for w in warnings_2:
                print(f"   {w}")
        else:
            print("   ✓ 修正后通过所有校验")

    return storyboard


def _call_llm_for_storyboard(client: OpenAI, system_prompt: str, user_prompt: str) -> dict:
    """调用 LLM 获取分镜 JSON"""
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        print(f"   [ERROR] LLM 调用失败: {e}")
        raise

    try:
        storyboard = _parse_json_response(raw_content)
    except json.JSONDecodeError as parse_error:
        print("   [格式修复] 分镜 JSON 语法异常，正在修复（最多 1 次）...")
        repair_prompt = _build_json_repair_prompt(raw_content, parse_error)
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            storyboard = _parse_json_response(response.choices[0].message.content)
        except json.JSONDecodeError as repair_error:
            raise ValueError(
                f"LLM 分镜 JSON 修复失败（已重试 1 次）: {repair_error}"
            ) from repair_error
        except Exception as e:
            print(f"   [ERROR] LLM JSON 修复调用失败: {e}")
            raise

    if not isinstance(storyboard, dict):
        raise ValueError("LLM 分镜响应必须是 JSON 对象")
    return storyboard


def _build_json_repair_prompt(raw_content: str, error: json.JSONDecodeError) -> str:
    """要求模型仅修复自身响应中的 JSON 语法，不重写分镜内容。"""
    return f"""上一次响应不是合法 JSON。
解析器错误 (JSONDecodeError): {error}

请只修复 JSON 语法（例如缺失逗号、引号或转义），保留原有字段和值，不要改写分镜。
只输出修复后的完整 JSON 对象，不要输出 Markdown 或解释。

<invalid_json>
{raw_content}
</invalid_json>"""


def _apply_defaults(storyboard: dict, aspect_ratio: str, resolution: str, style: str):
    """补充默认值"""
    # CLI parameters are authoritative; the LLM cannot silently override them.
    storyboard["aspect_ratio"] = aspect_ratio
    storyboard["resolution"] = resolution
    storyboard["style"] = style
    storyboard.setdefault("mood", "cinematic")
    storyboard.setdefault("music_style", "cinematic orchestral")

    previous_scene_id = None
    for index, shot in enumerate(storyboard["shots"]):
        shot.setdefault("duration", config.DEFAULT_DURATION)
        shot.setdefault("generate_audio", config.DEFAULT_GENERATE_AUDIO)
        shot.setdefault("transition_to_next", "crossfade")
        shot.setdefault("negative_prompt", "avoid jitter, stable motion, no text artifacts")
        shot.setdefault("key_props", [])
        shot["scene_id"] = _scene_id(shot) or f"scene_{index + 1:03d}"

        inferred_continuity = (
            "none"
            if index == 0
            else (
                "seamless"
                if shot["scene_id"] == previous_scene_id
                else "intentional_cut"
            )
        )
        continuity = shot.get("continuity_from_previous", inferred_continuity)
        if index == 0:
            continuity = "none"
        elif continuity not in {"seamless", "intentional_cut"}:
            continuity = inferred_continuity
        elif continuity == "seamless" and shot["scene_id"] != previous_scene_id:
            continuity = "intentional_cut"
        shot["continuity_from_previous"] = continuity
        shot.setdefault("primary_action", "")
        shot.setdefault("start_state", {})
        shot.setdefault("end_state", {})
        previous_scene_id = shot["scene_id"]


def _build_correction_prompt(original_prompt: str, storyboard: dict, warnings: list[str]) -> str:
    """构建修正 prompt — 将校验问题作为 feedback 让 LLM 修正"""
    issues_text = "\n".join(f"- {w}" for w in warnings)
    storyboard_json = json.dumps(storyboard, ensure_ascii=False, indent=2)

    return f"""{original_prompt}

---

⚠️ 上一次生成的分镜存在以下问题, 请修正后重新输出完整 JSON:

{issues_text}

上次生成的分镜 (需要修正):
```json
{storyboard_json}
```

请根据上述问题修正分镜, 输出修正后的完整 JSON。只输出 JSON, 不要解释。"""


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


_BUILTIN_SYSTEM_PROMPT = """你是一个顶尖的电影广告导演兼分镜师。你的目标不是"描述画面"，而是设计一段让观众看完就想再看一遍的视觉体验。

## 最高优先级规则

1. 主题锚定: ≥ 60% 的镜头必须包含与用户指定主题直接相关的视觉元素 (产品本体/使用过程/产出物/原材料), 产品不能只在 insert shot 出现
2. 场景连续性: 每个物理地点使用稳定 scene_id; 同一地点续接复用 ID, 不为多样性强行换场
3. 景别跳跃: 连续镜头景别至少跳 2 级 (特写→全景 OK, 特写→中近景 NG)
4. 情绪弧线: 每镜头 mood 不得与相邻镜头重复
5. Insert Shot: 全片至少 1 个无人物的纯产品/纯环境镜头
6. prompt_en 词数 80-150 词
7. 每个镜头时长 4-15 秒 (整数)
8. 物件连续性: 同场景内新物件必须有引入动作(placing/bringing等)或前镜铺垫, 场景切换时豁免
9. 环境一致性: 同场景连续镜头的天气/光线/背景物不得突变
10. 时长分配: 禁止所有镜头时长相同, 开场 4s, 高潮最长 6-8s, insert 4-5s
11. 镜头数量: 10s=2镜, 15s=3镜, 20s=3-4镜, 30s=4-6镜, 60s=8-10镜
12. 动作契约: 每镜只有一个 primary_action, 并填写 start_state/end_state; 快速复杂动作不得超过 5s

## Seedance 模型局限 (必须规避)

- 禁止 prompt 中要求画面出现文字/Logo/数字 (用 subtitle_text 代替)
- 单镜头最多 2 个角色互动, 超过用远景
- 手部特写追加 negative_prompt: anatomically correct hands
- duration > 10s 的镜头用 fixed 运镜 + 简单动作
- fast speed 镜头 duration ≤ 5s

## 运镜 6 步公式
[Subject] -> [Action] -> [Environment] -> [Camera] -> [Style] -> [Constraints]

## key_props 字段 (必须)
每个 shot 必须包含 key_props 数组, 列出画面中可见的关键道具/物件英文名。
同场景新增道具必须在 prompt_en 中包含引入动作。

## 输出格式 (严格 JSON, 参考 storyboard_system.md)
注意: shot 1 含角色时设 extract_character_ref: true
subtitle_text 必须中文; prompt_en 必须英文
每个 shot 必须包含 scene_id, continuity_from_previous, primary_action, start_state, end_state
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
    1. 提取每个 shot 的 key_props 和场景名称
    2. 如果当前镜头与上一镜头是同场景，检查新增的 props
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
            p.lower().strip() for p in prev_shot.get("key_props", [])
        )
        curr_props = set(
            p.lower().strip() for p in curr_shot.get("key_props", [])
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


def _validate_storyboard_richness(storyboard: dict) -> tuple[list[str], bool]:
    """分镜丰富度校验 — 返回 (warnings, is_critical)

    is_critical=True 时触发 LLM 自动修正。
    严重问题 (触发重试): 场景全部相同、情绪全部扁平、缺少 insert shot、prompt 词数严重越界
    轻微问题 (仅警告): 连续相同景别、物件连续性、时序因果

    检测维度:
    1. 场景多样性
    2. 景别跳跃
    3. 情绪弧线
    4. Insert shot 缺失
    5. 物件连续性
    6. 时序因果
    7. 主题锚定度
    8. prompt_en 词数
    """
    shots = storyboard.get("shots", [])
    if len(shots) < 2:
        return [], False

    warnings: list[str] = []
    critical_count = 0  # 严重问题计数

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

    # --- 4. Insert shot 缺失 ---
    has_insert = any(not s.get("characters") for s in shots)
    if not has_insert and len(shots) >= 3:
        warnings.append(
            "🚨 缺少 insert shot: 所有镜头都包含角色, "
            "必须加入至少 1 个纯产品/纯环境镜头"
        )
        critical_count += 1

    # --- 5. 时序因果校验 ---
    warnings.extend(_check_temporal_causality(shots))

    # --- 6. 物件连续性 ---
    warnings.extend(_check_prop_continuity(shots))

    # --- 7. 主题锚定度 ---
    title = storyboard.get("title", "")
    title_keywords = set(
        w.lower() for w in re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]+', title)
        if len(w) >= 2
    )
    if title_keywords and len(shots) >= 3:
        anchored_count = 0
        for s in shots:
            prompt_lower = s.get("prompt_en", "").lower()
            props_lower = " ".join(
                p.lower() for p in s.get("key_props", [])
            )
            combined = prompt_lower + " " + props_lower
            if any(kw in combined for kw in title_keywords):
                anchored_count += 1
        ratio = anchored_count / len(shots)
        if ratio < 0.6:
            warnings.append(
                f"🚨 主题锚定不足: 仅 {anchored_count}/{len(shots)} 个镜头"
                f"包含与标题 '{title}' 相关的元素 "
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
        if not str(shot.get("primary_action", "")).strip():
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

    # --- 10. 时长均匀性检测 ---
    durations = [s.get("duration", 5) for s in shots]
    if len(set(durations)) == 1 and len(shots) >= 3:
        warnings.append(
            f"🚨 时长全部相同: 所有镜头都是 {durations[0]}s, "
            f"缺乏节奏变化, 必须为不同叙事位置分配不同时长"
        )
        critical_count += 1

    # --- 11. Seedance 快动作预算 ---
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

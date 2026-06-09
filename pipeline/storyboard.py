"""Stage 1: LLM 分镜生成 — 创意策略 + 运镜公式 + 丰富度校验"""

from __future__ import annotations

import re
import json
from pathlib import Path

from openai import OpenAI

import config

SYSTEM_PROMPT = Path(config.PROMPTS_DIR / "storyboard_system.md").read_text

def generate_storyboard(
    user_request: str,
    target_duration: int = 30,
    aspect_ratio: str = "16:9",
    resolution: str = "1080p",
    style: str = "cinematic",
) -> dict:
    """
    调用 LLM 根据用户需求生成结构化分镜脚本

    Returns:
        完整的 storyboard dict (含 shots, music_style, mood 等)
    """
    client = OpenAI(
        api_key=config.ARK_API_KEY,
        base_url=config.ARK_BASE_URL + "/chat/completions" if False else config.ARK_BASE_URL,
    )

    system_prompt = _load_system_prompt()

    user_prompt = f"""用户需求: {user_request}

目标参数:
- 总时长: ~{target_duration} 秒
- 画面比例: {aspect_ratio}
- 分辨率: {resolution}
- 整体风格: {style}

请生成完整的分镜脚本 (JSON 格式)。"""

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        raw_content = response.choices[0].message.content
        print(f"   [DEBUG] LLM 原始响应前 200 字: {raw_content[:200]}")

        # 尝试提取 JSON (处理 markdown 包裹的情况)
        storyboard = _parse_json_response(raw_content)

    except json.JSONDecodeError as e:
        print(f"   [ERROR] JSON 解析失败: {e}")
        print(f"   [ERROR] 原始响应: {raw_content[:500]}")
        raise
    except Exception as e:
        print(f"   [ERROR] LLM 调用失败: {e}")
        print(f"   [DEBUG] model={config.LLM_MODEL}, base_url={config.ARK_BASE_URL}")
        raise

    # 验证必要字段
    assert "shots" in storyboard, "分镜缺少 shots 字段"
    assert len(storyboard["shots"]) > 0, "分镜 shots 为空"

    # 补充默认值
    storyboard.setdefault("aspect_ratio", aspect_ratio)
    storyboard.setdefault("resolution", resolution)
    storyboard.setdefault("style", style)
    storyboard.setdefault("mood", "cinematic")
    storyboard.setdefault("music_style", "cinematic orchestral")

    for shot in storyboard["shots"]:
        shot.setdefault("duration", config.DEFAULT_DURATION)
        shot.setdefault("generate_audio", config.DEFAULT_GENERATE_AUDIO)
        shot.setdefault("transition_to_next", "crossfade")
        shot.setdefault("negative_prompt", "avoid jitter, stable motion, no text artifacts")

    # 丰富度校验 (非阻断, 仅输出预警)
    _validate_storyboard_richness(storyboard)

    return storyboard


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

1. 场景多样性: 3+ 镜头时至少 2 个不同空间, 连续 2 镜头禁止相同场景
2. 景别跳跃: 连续镜头景别至少跳 2 级 (特写→全景 OK, 特写→中近景 NG)
3. 情绪弧线: 每镜头 mood 不得与相邻镜头重复
4. Insert Shot: 全片至少 1 个无人物的纯产品/纯环境镜头
5. prompt_en 词数 80-150 词
6. 每个镜头时长 4-15 秒 (整数)

## 运镜 6 步公式
[Subject] -> [Action] -> [Environment] -> [Camera] -> [Style] -> [Constraints]

## 输出格式 (严格 JSON, 参考 storyboard_system.md)
注意: shot 1 含角色时设 extract_character_ref: true
subtitle_text 必须中文; prompt_en 必须英文
"""


def _validate_storyboard_richness(storyboard: dict) -> None:
    """分镜丰富度校验 - 非阻断, 仅输出预警日志

    检测维度:
    1. 场景多样性 (scene_description 中的场景名称)
    2. 景别跳跃 (start_framing 连续相同)
    3. 情绪弧线 (mood 连续相同)
    4. Insert shot 缺失 (全部镜头都有角色)
    """
    shots = storyboard.get("shots", [])
    if len(shots) < 2:
        return

    warnings: list[str] = []

    # --- 1. 场景多样性 ---
    scene_names = []
    for s in shots:
        desc = s.get("scene_description", "")
        # 尝试提取【】中的场景名
        m = re.search(r'[\u3010\[][^\u3011\]]+[\u3011\]]', desc)
        if m:
            scene_names.append(m.group(0))
        else:
            scene_names.append(desc[:20])

    unique_scenes = set(scene_names)
    if len(unique_scenes) == 1 and len(shots) >= 3:
        warnings.append(
            f"\u26a0 \u573a\u666f\u5355\u4e00: \u6240\u6709 {len(shots)} \u4e2a\u955c\u5934\u90fd\u5728\u540c\u4e00\u573a\u666f "
            f"'{scene_names[0]}', \u5efa\u8bae\u589e\u52a0\u573a\u666f\u5207\u6362"
        )

    # 检测连续相同场景
    for i in range(1, len(scene_names)):
        if scene_names[i] == scene_names[i - 1]:
            warnings.append(
                f"\u26a0 \u8fde\u7eed\u76f8\u540c\u573a\u666f: Shot {i} \u548c Shot {i + 1} "
                f"\u90fd\u5728 '{scene_names[i]}'"
            )

    # --- 2. 景别跳跃 ---
    framings = [
        s.get("camera", {}).get("start_framing", "unknown")
        for s in shots
    ]
    for i in range(1, len(framings)):
        if framings[i] == framings[i - 1] and framings[i] != "unknown":
            warnings.append(
                f"\u26a0 \u666f\u522b\u91cd\u590d: Shot {i} \u548c Shot {i + 1} "
                f"\u90fd\u662f '{framings[i]}', \u5efa\u8bae\u666f\u522b\u8df3\u8dc3"
            )

    # --- 3. 情绪弧线 ---
    moods = [s.get("mood", "unknown") for s in shots]
    unique_moods = set(moods)
    if len(unique_moods) == 1 and len(shots) >= 2:
        warnings.append(
            f"\u26a0 \u60c5\u7eea\u6241\u5e73: \u6240\u6709\u955c\u5934\u60c5\u7eea\u90fd\u662f '{moods[0]}', "
            f"\u7f3a\u4e4f\u5f27\u7ebf\u53d8\u5316"
        )
    for i in range(1, len(moods)):
        if moods[i] == moods[i - 1] and moods[i] != "unknown":
            warnings.append(
                f"\u26a0 \u8fde\u7eed\u76f8\u540c\u60c5\u7eea: Shot {i} \u548c Shot {i + 1} "
                f"\u90fd\u662f '{moods[i]}'"
            )

    # --- 4. Insert shot 缺失 ---
    has_insert = any(not s.get("characters") for s in shots)
    if not has_insert and len(shots) >= 3:
        warnings.append(
            "\u26a0 \u7f3a\u5c11 insert shot: \u6240\u6709\u955c\u5934\u90fd\u5305\u542b\u89d2\u8272, "
            "\u5efa\u8bae\u52a0\u5165\u7eaf\u4ea7\u54c1/\u7eaf\u73af\u5883\u955c\u5934\u63d0\u5347\u9ad8\u7aef\u611f"
        )

    # --- 输出 ---
    if warnings:
        print("\n   [\u4e30\u5bcc\u5ea6\u6821\u9a8c]")
        for w in warnings:
            print(f"   {w}")
        print()

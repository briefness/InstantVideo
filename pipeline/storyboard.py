"""Stage 1: LLM 分镜生成 — 含电影级运镜 6 步公式"""

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


_BUILTIN_SYSTEM_PROMPT = """你是一个专业的电影级视频分镜师。根据用户需求，生成结构化的分镜脚本。

## 核心规则

1. 每个镜头时长 4-15 秒 (整数)
2. 总时长尽量接近用户要求
3. 使用英文 prompt (Seedance 2.0 英文效果更好)
4. 每个镜头的 prompt_en 必须 60-100 词

## 运镜 6 步公式 (每个 prompt_en 必须包含)

[Subject 主体] → [Action 动作] → [Environment 环境] → [Camera 运镜] → [Style 风格] → [Constraints 约束]

## 运镜铁律

- 每个镜头只用 1 个主运镜指令 (不要叠加多个)
- 用节奏词: slow, gentle, smooth, controlled (不要用 fps, ISO, 焦距等技术参数)
- 分离运镜和主体运动: "dancer spins slowly. Camera holds fixed."
- 复合运镜最多 2 层: "slow push-in then subtle rise"

## 8 类运镜 (选择 1 个)

- push-in (推): 发现感/重要性
- pull-out (拉): 环境揭示/规模感  
- pan (横摇): 扫描/展示广度
- tracking (跟踪): 沉浸/跟随
- orbit (环绕): 产品360°/肖像
- crane/drone (升降): 史诗感
- handheld (手持): 纪录片真实感 (必须加 subtle/controlled)
- fixed (固定): 让主体运动说话

## 叙事节奏

- 开场: 航拍/pan/push-in (建立世界)
- 发展: tracking/固定 (展示动作)
- 高潮: push-in/orbit/特写 (情感聚焦)
- 结尾: crane-up/pull-back (升华)

## 负面提示 (negative_prompt 必须包含)

所有镜头: "avoid jitter, stable motion, no text artifacts"
人物镜头追加: "avoid bent limbs, no identity drift"
产品镜头追加: "no morphing shapes, keep geometry stable"

## 输出格式 (严格 JSON)

{
  "title": "视频标题",
  "total_duration": 30,
  "style": "cinematic",
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "mood": "premium / energetic / dramatic / warm / cold / ...",
  "music_style": "描述音乐风格",
  "characters": [
    {"name": "main", "description": "角色外观描述 (英文)"}
  ],
  "shots": [
    {
      "shot_id": 1,
      "duration": 5,
      "scene_description": "中文场景描述 (给人看的)",
      "prompt_en": "完整英文 prompt (给 Seedance 的, 60-100词, 含6步公式)",
      "camera": {
        "primary_movement": "slow dolly push-in",
        "start_framing": "wide shot",
        "end_framing": "close-up",
        "speed": "slow"
      },
      "lighting": "soft golden hour, rim light",
      "mood": "premium",
      "negative_prompt": "avoid jitter, stable motion, no text artifacts",
      "subtitle_text": "字幕文案 (中文)",
      "transition_to_next": "crossfade",
      "generate_audio": true,
      "characters": ["main"],
      "extract_character_ref": false
    }
  ]
}

注意: shot 1 如果包含角色, 设置 "extract_character_ref": true (用于后续镜头的角色一致性)
"""

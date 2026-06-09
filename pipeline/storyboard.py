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
4. 每个镜头的 prompt_en 必须 60-120 词

## 运镜 6 步公式 (每个 prompt_en 必须包含)

[Subject 主体] → [Action 动作] → [Environment 环境] → [Camera 运镜] → [Style 风格] → [Constraints 约束]

## 构图多样性 (禁止所有镜头居中构图)

- 三分法偏置: 主体在画面左/右三分之一处
- 前景遮挡: 通过门框/树枝/废墟窥视主体
- 过肩镜头: 从角色A肩膀后方拍角色B (多角色必用)
- 低角度仰拍: 威压感、英雄感
- 极端特写: 眼睛/手/物件占满画面
- 全片不超过 50% 的镜头使用居中构图
- camera 字段中必须包含 composition 属性

## 叙事与互动 (视频不是幻灯片)

- 动作-反应链: A做什么 → B如何反应 → 推动下一镜头
- 禁止"站-做-站"流水账
- 角色之间必须有互动 (对峙/追逐/协作/冲突)
- 角色必须与环境互动 (踢起碎片/穿越烟雾/撞碎物体)
- 情绪弧线必须变化, 不能全片同一种情绪
- 加入微叙事细节 (远处烟柱/衣角飘动/手指颤抖)
- ❗ 物理逻辑: 动作效果方向必须正确(向前开枪→前方敌人倒), 镜头间空间位置一致, 动作必须承接上一镜头结果

## 真实感增强 (每个 prompt_en 至少包含 2 种)

- 环境微粒: dust particles, smoke wisps, ash falling, sparks
- 物理细节: fabric rippling, hair swaying, debris scattering
- 表面质感: scratched metal, wet pavement, peeling paint
- 光影互动: light rays through windows, flickering reflections
- 环境活力: distant birds, swaying grass, dripping water

## 运镜铁律

- 每个镜头只用 1 个主运镜指令 (不要叠加多个)
- 用节奏词: slow, gentle, smooth, controlled (不要用 fps, ISO, 焦距等技术参数)
- 分离运镜和主体运动: "dancer spins slowly. Camera holds fixed."
- 复合运镜最多 2 层: "slow push-in then subtle rise"
- ❗ 动静搭配: 至少 30% 的镜头必须是 fixed (固定机位), 禁止连续 3 个以上动态运镜
- fixed 不等于无聊: 固定机位让主体的动作/表演/光线变化成为焦点

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

- 开场: 航拍/pan/push-in (建立世界) 🎬动态
- 发展: **fixed/微推** (角色表演、沉浸观察) 🔒优先静态
- 发展: tracking/固定 (展示动作) 🎬动态
- 高潮: **fixed 特写** (情感爆发点，让表演说话) 🔒优先静态
- 高潮: orbit/快推 (视觉冲击) 🎬动态
- 结尾: crane-up/pull-back (升华) 🎬动态
- 动静交替形成张弛节奏，连续运镜会让观众疲劳

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
      "scene_description": "中文场景描述 (含叙事意图)",
      "prompt_en": "完整英文 prompt (60-120词, 含6步公式+构图+真实感细节)",
      "camera": {
        "primary_movement": "slow dolly push-in",
        "composition": "subject at left third, foreground debris",
        "start_framing": "wide shot",
        "end_framing": "close-up",
        "speed": "slow"
      },
      "lighting": "soft golden hour, rim light",
      "mood": "premium",
      "negative_prompt": "avoid jitter, stable motion, no text artifacts",
      "subtitle_text": "字幕/口播文案 (必须中文)",
      "transition_to_next": "crossfade",
      "generate_audio": true,
      "characters": ["main"],
      "extract_character_ref": false
    }
  ]
}

注意: shot 1 如果包含角色, 设置 "extract_character_ref": true (用于后续镜头的角色一致性)
重要: `subtitle_text` 必须用中文 (面向观众); 仅 `prompt_en` 用英文 (给 Seedance 模型)
禁止: 所有镜头居中构图 / "站-做-站"流水账 / 缺少真实感细节
"""

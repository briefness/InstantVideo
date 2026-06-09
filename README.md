# 🎬 Seedance — 一句话生成电影级 AI 短视频

给一句中文描述，自动生成带分镜、调色、配乐、口播和字幕的完整短视频。

基于字节跳动 Seedance 2.0 视频生成模型，通过火山引擎 Ark Agent Plan 统一调用。

## 核心能力

- **分镜自动生成** — LLM 理解需求 → 拆解为电影级分镜（含运镜 / 光线 / 角色 / 情绪）
- **分镜丰富度校验** — 场景多样性、景别跳跃、情绪弧线、insert shot 四维度检测
- **角色一致性** — 首镜提取参考帧 + 显式 `image_role` + 文本描述注入，图文双保障
- **智能转场** — 基于运镜方向和速度自动推导转场类型（cut / dissolve / fade_to_black / crossfade）
- **电影级运镜** — 8 类运镜 × 6 步公式，自动组装 Prompt
- **LUT 调色** — 根据分镜 mood 自动匹配 LUT，视觉风格连贯
- **音乐自动匹配** — 根据视频情绪匹配 BGM + BPM 卡点
- **TTS 口播** — 可插拔引擎（macOS say / 火山语音合成），逐镜头合成 + sidechain ducking 混音
- **字幕烧录** — 字幕时长与 TTS 语音同步，无口播时退回镜头时长
- **多平台导出** — 一键输出 YouTube / TikTok / B站 / 小红书
- **三层容错** — 429 限流退避 / 参考图异常降级 / 断链尾帧兜底

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
brew install ffmpeg  # macOS

# 2. 配置 .env
cp .env.example .env
# 编辑 .env, 填入你的火山引擎 Ark API Key

# 3. 运行
python main.py "制作一个15秒的电影短片：机器人在末日清理丧尸"
```

运行完成后终端输出：

```
📋 分镜脚本     ✓ 3 个镜头, 风格: grim, tense
🎥 视频生成     ✓ 成功 3/3 个镜头
🎞️ 规格统一     ✓ 3 个视频已统一规格
🔗 拼接转场     ✓ 15.3s
🎨 自动调色     ✓ 已应用 LUT
🔊 背景音乐     ✓ 已添加
🎙️ 口播合成     ✓ 3 段, ducking 混入
💬 字幕烧录     ✓ 完成
📦 导出         📤 YouTube  📤 TikTok

✅ 成片 · 17.5s · 1080p
```

## 环境变量 (.env)

```bash
# 火山引擎 Ark (Agent Plan) — 一个 Key 搞定一切
ARK_API_KEY=ark-xxxxx
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3

# 视频生成模型
SEEDANCE_MODEL=doubao-seedance-2.0-fast

# LLM 文本模型 (生成分镜脚本)
LLM_MODEL=doubao-seed-2.0-lite

# TTS 引擎 (可选, 默认 macos)
TTS_ENGINE=macos  # macos | volcano
```

## 使用示例

```bash
# 基础使用
python main.py "15秒咖啡品牌广告，从咖啡豆到拿铁的过程"

# 指定风格和分辨率
python main.py "科技感产品介绍" --style futuristic --resolution 1080p

# 竖版 + 指定平台
python main.py "旅行短视频" --ratio 9:16 --platforms tiktok xiaohongshu

# 带背景音乐 (自动卡点)
python main.py "运动品牌宣传" --music music/upbeat_electronic.mp3
```

## 流水线

```
用户输入
  → Stage 1:   LLM 分镜 (运镜 + 光线 + 角色 + 丰富度校验)
  → Stage 1.5: 音乐匹配 + BPM 卡点 (可选)
  → Stage 2:   Seedance 2.0 生成 (角色一致 + 降级链 + 即时下载)
  → Stage 2.5: 规格统一 (帧率 / 分辨率 / 编码 / 补静音轨)
  → Stage 3:   拼接 + 智能转场 (xfade + acrossfade)
  → Stage 4:   LUT 调色
  → Stage 5:   BGM 混合
  → Stage 5.5: TTS 口播合成 + sidechain ducking
  → Stage 6:   字幕烧录 (与口播时长同步)
  → Stage 7:   片头片尾 + 多平台导出
```

## 项目结构

```
seedance/
├── main.py                  ← CLI 入口
├── config.py                ← 配置 (API Key, 模型名, 参数)
├── requirements.txt
├── .env                     ← 你的 Key (不提交到 git)
├── pipeline/                ← 核心流水线
│   ├── orchestrator.py      ← 主编排 (7+ Stage)
│   ├── storyboard.py        ← Stage 1: LLM 分镜 + 丰富度校验
│   └── generator.py         ← Stage 2: 视频生成 + 角色一致性
├── tools/                   ← 底层工具
│   ├── seedance_api.py      ← Seedance API 封装 (显式 image_role)
│   ├── ffmpeg_ops.py        ← FFmpeg 操作 (转场 / 调色 / 混音 / 字幕)
│   ├── tts.py               ← TTS 引擎 (Protocol + macOS say + 火山占位)
│   ├── beat_analyzer.py     ← 音乐节拍分析
│   └── frame_extractor.py   ← 帧提取 + 质量检测
├── prompts/                 ← Prompt 模板
│   └── storyboard_system.md ← 分镜系统 Prompt (运镜知识库)
├── luts/                    ← .cube LUT 调色文件
├── music/                   ← 背景音乐库
├── tests/                   ← 测试
├── docs/                    ← 设计文档
│   └── design.md            ← 完整技术设计
└── output/                  ← 生成产物
    └── 20260609_124525/
        ├── storyboard.json
        ├── shots/
        ├── character_refs/  ← 角色参考帧
        ├── voiceover/       ← TTS 口播音频
        ├── final.mp4
        └── exports/
            ├── youtube.mp4
            └── tiktok.mp4
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 视频生成 | Seedance 2.0 (火山引擎 Ark Agent Plan) |
| 分镜生成 | 豆包 Seed 2.0 LLM |
| 视频处理 | FFmpeg (xfade + acrossfade + sidechaincompress) |
| TTS 口播 | macOS say (默认) / 火山语音合成 (可选) |
| 音乐分析 | librosa |
| 质量检测 | OpenCV |
| CLI | argparse + rich |

## 素材准备

```bash
# 查看需要哪些素材 + 下载指引
python setup_assets.py
```

- **LUT 文件** — 放入 `luts/` 目录（`.cube` 格式，可选，没有会跳过调色）
- **背景音乐** — 放入 `music/` 目录（`.mp3` 格式，可选，没有用 Seedance 原生音效）

## 运镜体系

本项目内置电影级运镜知识库（8 类运镜 × 6 步公式），在分镜生成阶段自动应用：

| 运镜 | 效果 | 适用场景 |
|------|------|---------|
| Push-in | 发现 / 重要 / 亲密 | 产品特写、情绪 |
| Pull-out | 揭示 / 规模 / 结尾 | 环境、收尾 |
| Pan | 扫描 / 广度 | 风景、空间 |
| Tracking | 跟随 / 沉浸 | 人物、运动 |
| Orbit | 环绕 / 立体 | 产品 360° |
| Crane/Drone | 史诗 / 建立 | 开场、壮观 |
| Handheld | 真实 / 纪录片 | Vlog、幕后 |
| Fixed | 主体运动 | 时尚、舞蹈 |


## License

MIT

# 🎬 Seedance — 电影级 AI 视频自动生成工具

给一个主题或脚本，自动生成电影级质感的完整视频。

基于字节跳动 Seedance 2.0 视频生成模型，通过火山引擎 Ark 平台调用。

## 核心能力

- **分镜自动生成** — LLM 理解需求 → 拆解为电影级分镜 (含运镜/光线/角色)
- **电影级运镜** — 8 类运镜 × 6 步公式，自动组装 Prompt
- **角色一致性** — 首镜提取参考帧 → 后续镜头 I2V 锁定角色
- **调色统一** — LUT 调色，视觉风格连贯
- **音乐自动匹配** — 根据视频情绪匹配 BGM + BPM 卡点
- **字幕烧录** — 根据分镜自动生成时间轴字幕
- **多平台导出** — 一键输出 YouTube / TikTok / B站 / 小红书

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
brew install ffmpeg  # macOS

# 2. 配置 .env
cp .env.example .env
# 编辑 .env, 填入你的火山引擎 Ark API Key

# 3. 运行
python main.py "制作一个30秒的智能手表宣传视频，展示外观设计和运动功能"
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

## 流水线 7 个 Stage

```
用户输入
  → Stage 1:   LLM 分镜 (6步公式 + 运镜 + 光线 + 角色)
  → Stage 1.5: 音乐匹配 + BPM 卡点 (可选)
  → Stage 2:   Seedance 2.0 生成 (角色一致 + 降级链 + 即时下载)
  → Stage 2.5: 规格统一 (帧率/分辨率/编码)
  → Stage 3:   拼接 + 转场 (FFmpeg xfade)
  → Stage 4:   LUT 调色
  → Stage 5:   音频混合 (环境音 + BGM)
  → Stage 6:   字幕烧录
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
│   ├── orchestrator.py      ← 主编排 (7 Stage)
│   ├── storyboard.py        ← Stage 1: LLM 分镜
│   └── generator.py         ← Stage 2: Seedance 生成
├── tools/                   ← 底层工具
│   ├── seedance_api.py      ← Seedance API 封装
│   ├── ffmpeg_ops.py        ← FFmpeg 操作合集
│   ├── beat_analyzer.py     ← 音乐节拍分析
│   └── frame_extractor.py   ← 帧提取 + 质量检测
├── prompts/                 ← Prompt 模板
│   └── storyboard_system.md ← 分镜系统 Prompt (运镜知识库)
├── luts/                    ← .cube LUT 调色文件
├── music/                   ← 背景音乐库
├── docs/                    ← 设计文档
│   └── design.md            ← 完整技术设计 (75KB)
└── output/                  ← 生成产物
    └── 20260605_171300/
        ├── storyboard.json
        ├── shots/
        ├── final.mp4
        └── exports/
            ├── youtube.mp4
            └── tiktok.mp4
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 视频生成 | Seedance 2.0 (火山引擎 Ark) |
| 分镜生成 | 豆包 Seed 2.0 LLM |
| 视频处理 | FFmpeg |
| 音乐分析 | librosa |
| 质量检测 | OpenCV |
| CLI | argparse + rich |

## 素材准备

```bash
# 查看需要哪些素材 + 下载指引
python setup_assets.py
```

- **LUT 文件** — 放入 `luts/` 目录 (`.cube` 格式，可选，没有会跳过调色)
- **背景音乐** — 放入 `music/` 目录 (`.mp3` 格式，可选，没有用 Seedance 原生音效)

## 运镜体系

本项目内置电影级运镜知识库 (8 类运镜 × 6 步公式)，在分镜生成阶段自动应用：

| 运镜 | 效果 | 适用场景 |
|------|------|---------|
| Push-in | 发现/重要/亲密 | 产品特写、情绪 |
| Pull-out | 揭示/规模/结尾 | 环境、收尾 |
| Pan | 扫描/广度 | 风景、空间 |
| Tracking | 跟随/沉浸 | 人物、运动 |
| Orbit | 环绕/立体 | 产品 360° |
| Crane/Drone | 史诗/建立 | 开场、壮观 |
| Handheld | 真实/纪录片 | Vlog、幕后 |
| Fixed | 主体运动 | 时尚、舞蹈 |

详细运镜文档: `docs/design.md` 第 9 章

## License

MIT

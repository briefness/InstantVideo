# 🎬 Seedance — 一句话生成电影级 AI 短视频

给一句中文描述，自动生成带分镜、转场、调色、配乐、口播和字幕的完整短视频。

基于字节跳动 Seedance 2.0 视频生成模型，通过火山引擎 Ark API 统一调用。

## 核心能力

- **分镜自动生成** — LLM 理解需求 → 拆解为电影级分镜（含运镜 / 光线 / 角色 / 情绪）
- **分镜丰富度校验 + 自动修正** — 9 维度检测（场景多样性、景别跳跃、情绪弧线、insert shot、物件连续性、时序因果、主题锚定、prompt 词数、时长均匀性），严重问题自动触发 LLM 重跑修正
- **Seedance 模型局限规避** — prompt 生成时自动规避文字渲染、多人互动、手部畸变、长镜头不稳定等 8 项已知局限
- **智能时长/数量分配** — 根据叙事位置自动分配镜头时长，根据总时长推荐镜头数量
- **竖版 (9:16) 专属策略** — 构图、运镜、安全区自动适配 TikTok/小红书等竖屏平台
- **角色一致性** — 首镜提取参考帧 + 显式 `image_role` + 文本描述注入 + 场景连贯性注入，图文双保障
- **智能转场** — 基于运镜方向和速度自动推导转场类型（cut / dissolve / fade_to_black / crossfade / wipe）
- **电影级运镜** — 8 类运镜 × 6 步公式，自动组装 Prompt
- **LUT 调色** — 根据分镜 mood 自动匹配 LUT，调色 + 字幕合并为单次编码减少画质损失
- **音频设计** — 智能决策 generate_audio（有环境音的场景自动开启），音乐风格具体到乐器+节奏+动态变化
- **音乐自动匹配** — 根据视频情绪匹配 BGM + BPM 卡点对齐镜头时长
- **TTS 口播** — 可插拔引擎（macOS say / 火山语音合成），逐镜头合成 + sidechain ducking 混音
- **字幕烧录** — 字幕时长与 TTS 语音同步，无口播时退回镜头时长
- **多平台导出** — 一键输出 YouTube / TikTok / B站 / 小红书 / Instagram
- **断点续传** — 已生成的镜头本地缓存（含尾帧提取），中断后重跑自动跳过已完成片段
- **并行预生成** — 识别无依赖的 Insert Shot，提前提交 API 节省等待时间
- **零损耗拼接** — 编码一致时优先 `-c copy` 直拼，避免不必要的重编码
- **三层容错** — 429 限流指数退避 / 参考图异常降级（隐私审核 → 尾帧衔接 → 纯文本 T2V）/ 断链尾帧兜底

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
📋 Stage 1: 生成分镜脚本...
   ✓ 3 个镜头, 风格: grim, tense, 目标时长: 15s
🎥 Stage 2: 生成视频片段...
   ✓ 成功 3/3 个镜头
🎞️ Stage 2.5: 统一视频规格...
   ✓ 3 个视频已统一规格
🔗 Stage 3: 拼接 & 转场...
   转场: fade_to_black(0.8s), crossfade(0.5s)
   ✓ 拼接完成 (15.3s)
🔊 Stage 4: 音频处理...
   ✓ 背景音乐已添加
🎙️ Stage 4.5: 口播合成...
   ✓ 合成 3 段口播
   ✓ 口播已混入 (ducking)
🎨 Stage 5: 调色 & 字幕...
   ✓ LUT 调色: IWLTBAP Coronado - Standard.cube
   ✓ 字幕已烧录
   ✓ 视觉滤镜合并完成 (单次编码)
📦 Stage 7: 包装 & 导出...
   📤 youtube  📤 tiktok

🎉 完成! 视频已保存到: output/20260609_124525/final.mp4
```

## 环境变量 (.env)

```bash
# 火山引擎 Ark — 一个 Key 搞定一切
ARK_API_KEY=your_ark_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 视频生成模型
SEEDANCE_MODEL=doubao-seedance-2-0-260128

# LLM 文本模型 (生成分镜脚本)
LLM_MODEL=doubao-1-5-pro-256k

# TTS 引擎 (可选, 默认 macos)
TTS_ENGINE=macos  # macos | volcano
```

> **注意**: `config.py` 中有内置默认值（`doubao-seedance-2.0-fast` / `doubao-seed-2.0-lite`），`.env` 中的配置会覆盖默认值。`doubao-seedance-2.0-fast` 最高支持 720p 分辨率。

## 使用示例

```bash
# 基础使用
python main.py "15秒咖啡品牌广告，从咖啡豆到拿铁的过程"

# 指定风格和分辨率
python main.py "科技感产品介绍" --style futuristic --resolution 1080p

# 竖版 + 指定平台
python main.py "旅行短视频" --ratio 9:16 --platforms tiktok xiaohongshu

# 带背景音乐 (自动 BPM 卡点)
python main.py "运动品牌宣传" --music music/upbeat_electronic.mp3

# 支持的画面比例
# 16:9 (默认) | 9:16 | 4:3 | 1:1 | 21:9
```

## 流水线

```
用户输入
  → Stage 1:   LLM 分镜 (9 维丰富度校验 + 严重问题自动修正)
  → Stage 1.5: 音乐匹配 + BPM 卡点 (可选, 指定 --music 时启用)
  → Stage 2:   Seedance 2.0 生成 (角色一致 + 降级链 + 断点续传 + 独立镜头并行预生成)
  → Stage 2.5: 规格统一 (帧率 / 分辨率 / 编码 / 补静音轨)
  → Stage 3:   拼接 + 智能转场 (xfade + acrossfade, 优先零损耗直拼)
  → Stage 4:   BGM 混合 (自动匹配或用户指定)
  → Stage 4.5: TTS 口播合成 + sidechain ducking (时间计算减去转场重叠)
  → Stage 5:   LUT 调色 + 字幕烧录 (合并为单次编码, 减少画质损失)
  → Stage 7:   片头定版 + 多平台导出
```

## 项目结构

```
seedance/
├── main.py                  ← CLI 入口
├── config.py                ← 配置 (API Key, 模型名, 降级链, LUT/音乐映射)
├── requirements.txt
├── setup_assets.py          ← 素材检查 + 下载指引
├── .env.example             ← 环境变量模板
├── .env                     ← 你的 Key (不提交到 git)
├── pipeline/                ← 核心流水线
│   ├── orchestrator.py      ← 主编排 (7 Stage, 口播时间减去转场重叠)
│   ├── storyboard.py        ← Stage 1: LLM 分镜 + 9 维校验 + 自动修正
│   └── generator.py         ← Stage 2: 视频生成 + 角色一致性 + 降级策略 + 并行预生成
├── tools/                   ← 底层工具
│   ├── seedance_api.py      ← Seedance API 封装 (显式 image_role + base64)
│   ├── ffmpeg_ops.py        ← FFmpeg 操作 (转场 / 调色 / 混音 / 字幕 / 导出)
│   ├── tts.py               ← TTS 引擎 (Protocol + macOS say + 火山占位)
│   ├── beat_analyzer.py     ← 音乐节拍分析 (BPM + 卡点)
│   └── frame_extractor.py   ← 帧提取 + 质量检测
├── prompts/                 ← Prompt 模板
│   └── storyboard_system.md ← 分镜系统 Prompt (运镜知识库 + 模型局限 + 时长/数量策略 + 竖版规则 + 音频设计)
├── luts/                    ← .cube LUT 调色文件
├── music/                   ← 背景音乐库
├── tests/                   ← 测试
│   ├── test_generator.py    ← 视频生成测试
│   ├── test_transitions.py  ← 转场逻辑测试
│   └── test_ffmpeg_audio.py ← 音频处理测试
├── docs/                    ← 设计文档
│   └── design.md            ← 完整技术设计 (110KB)
└── output/                  ← 生成产物
    └── 20260609_124525/
        ├── storyboard.json
        ├── shots/
        ├── character_refs/  ← 角色参考帧
        ├── normalized/      ← 规格统一后的视频
        ├── voiceover/       ← TTS 口播音频
        ├── final.mp4
        └── exports/
            ├── youtube.mp4
            └── tiktok.mp4
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 视频生成 | Seedance 2.0 (火山引擎 Ark API) |
| 分镜生成 | 豆包 LLM (OpenAI 兼容接口) |
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
- **背景音乐** — 放入 `music/` 目录（`.mp3` 格式，可选，没有则用 Seedance 原生音效或自动匹配）

## 降级策略

```
角色参考帧正常  →  reference_image (角色 + 尾帧)
       ↓ 隐私审核拒绝
尾帧衔接模式    →  first_frame (仅尾帧, 角色靠 prompt 描述)
       ↓ 再次拒绝
纯文本 T2V      →  无参考图, 仅 prompt 驱动

分辨率降级: 720p → 480p
限流退避: 30s → 60s → 120s (指数退避, 最多 3 次)
```

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

## 多平台导出规格

| 平台 | 分辨率 | 比例 |
|------|--------|------|
| YouTube | 1920×1080 | 16:9 |
| TikTok | 1080×1920 | 9:16 |
| B站 | 1920×1080 | 16:9 |
| 小红书 | 1080×1440 | 3:4 |
| Instagram Reels | 1080×1920 | 9:16 |
| Instagram Feed | 1080×1080 | 1:1 |

## License

MIT

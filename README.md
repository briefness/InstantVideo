# 🎬 Seedance — 一句话生成电影级 AI 短视频

给一句中文描述，自动生成带分镜、转场、调色、配乐、口播和字幕的完整短视频。

基于字节跳动 Seedance 2.0 Mini 视频生成模型，通过火山引擎 Ark API 统一调用。模型开通、API 参数和计费规则以[火山方舟官方文档](https://www.volcengine.com/docs/82379)为准。

## 核心能力

- **分镜自动生成** — LLM 理解需求 → 拆解为电影级分镜（含故事状态、运镜、光线、角色与情绪）
- **故事状态契约** — 顶层 `story_arc` 定义目标、风险、局势变化与结果；每镜 `narrative_beat` 必须产生可见状态变化，后镜精确接续前镜结果。15 秒使用简单微剧情，长视频增加推进层次，不机械套三幕式
- **通用交互因果** — 接触、定向路径、区域作用和间接作用共用 `source` / `effect_region` / `reaction_scope` / `unaffected_behavior`；`setup` 禁止提前生效，`active` 明确 single/subset/all 结果范围及 static/sweep/expand/propagate 运动，`aftermath` 不得无原因扩大既有结果。规则不依赖机器人、枪械、丧尸等题材关键词
- **分镜契约校验 + 自动修正** — 检查稳定场景 ID、故事状态、交互因果、景别、情绪、物件/时序、主题、prompt、时长、单镜动作负载、角色移动形态和动作空间轴，严重问题只触发一次 LLM 修正；相位必然字段和相邻故事状态交接由代码确定性编译，校正边界保留已通过的叙事、场景、连续性和空间字段，不依赖模型重新抄写
- **ProductionPlan 执行拓扑** — 先由代码确定镜头数量、时长、叙事槽位、结果槽位、景别族和参考职责，LLM 只填写题材内容；未知字段在草稿边界忽略，持久化和恢复仍使用严格模型校验
- **题材自适应节奏** — 根据用户需求区分动作驱动、产品展示与均衡叙事；24 秒以上动作片按因果组合空间/交互、动作主体、反应或关键细节、结果视角，并覆盖 wide / medium / close-detail 三类景别；产品片保留细节 Insert Shot，明显失衡时只修正一次
- **Seedance 模型局限规避** — prompt 生成时自动规避文字渲染、多人互动、手部畸变、长镜头不稳定等 8 项已知局限
- **智能时长/数量分配** — 根据叙事位置自动分配镜头时长，根据总时长推荐镜头数量
- **竖版 (9:16) 专属策略** — 构图、运镜、安全区自动适配 TikTok 等竖屏平台
- **镜头连续性** — `scene_id` 只区分物理地点；只有无剪切、边界景别兼容的直接续拍才使用 `seamless`，切镜按 `small` / `medium` / `large` 标记构图变化
- **参考职责分离** — 无缝续接和所有同场景状态交接都使用已验收真实尾帧作为官方 `first_frame`；身份参考只在跨场景或没有状态依赖时使用 `reference_image`，不把身份图和状态图混为二选一；状态链不按镜头数硬截断，每次只继承上一条已验收尾帧
- **角色参考归属** — 角色按 `identity` / `group` / `none` 分类；`characters`、`screen_positions`、`blocking`、`action_beats` 共用完整可见参与者集合。身份图只从已通过语义验收的中点帧按角色裁剪，裁剪框必须排除其他角色、群体、倒影和轮廓；缺失或无效裁剪框不会回退为整帧
- **结构化动作调度** — 动作镜头用 `action_beats` 表达 trigger / peak / aftermath，用 `blocking` 固定位置、身体朝向、视线、移动和攻击目标；生成前确定性检查前/中/后景及左右位置与身体朝向是否相容，避免角色背对目标或武器朝错方向
- **生成输入溯源** — 依赖上一镜的缓存必须记录镜头契约、参考角色和参考帧指纹；计划合同与验收观测分开保存，验收结果不会反向改变缓存指纹；任何已有片段的溯源不匹配都会保留文件并立即停止，缓存冲突不会隐式授权新的付费任务
- **生成前就绪门** — 在任何付费视频调用前统一检查 prompt、时长、可见角色归属、身份参考、同场景连续性尾帧和本地参考文件；确定性缺陷直接停止，不进入重试链。供应商隐私拒绝后，只有实际移除 `reference_image` 身份资产才允许一次降级重试；`first_frame` 状态参考被拒会立即停止，不重复相同请求或退化为不连续 T2V
- **可拍摄性契约** — `coverage_role`、必需可见主体和交互几何按模式约束接触同框、定向路径、作用区域与遮挡；角色 `blocking.frame_position` 会确定性编译为相机屏幕位置，拒绝“极近景浅焦却要求远处目标清晰反应”等互相冲突的镜头
- **生成后语义验收** — 普通镜头使用五个时序点；结构化因果镜头自适应增加到九点，在同一次审核中为每个采样返回作用是否可见、目标是否反应、作用与反应是否相交、范围外是否误反应、约定结果及完整范围是否达成五项证据。`setup` 中角色自身的瞄准、充能、传感器激活和源内光效属于准备，只有作用离开来源、穿过作用区域、发生接触或改变目标才算物理作用；任何提前发射、接触、目标反应或结果仍会拒绝。同场景边界额外要求上一状态保留、进度不倒退、开放动作交接、持续主体/道具和场景身份五项证据；程序不采信与逐点证据矛盾的聚合结论。验收同时结合上一条已接受尾帧、上一镜实际终态和身份图检查主体、动作结果、环境地标、动作交接、屏幕方向/空间轴、持续道具/武器；任一失败都不能被运镜质感抵消。空、截断或畸形 JSON 只复核同一视频一次，仍异常则暂停，明确审核拒绝不重试
- **有限定向重拍** — 语义失败最多补拍一次，第二次失败立即停止；被拒绝片段保留但不会成为后镜参考，恢复运行也不会重置预算；验收器升级后可先复核最后一条本地 take，通过时直接恢复，不重新调用 Seedance
- **智能转场** — 基于运镜方向和速度自动推导转场类型（cut / dissolve / fade_to_black / crossfade / wipe）
- **电影级运镜** — 8 类运镜 × 6 步公式，自动组装 Prompt
- **LUT 调色** — 根据分镜 mood 自动匹配 LUT，调色 + 字幕合并为单次编码减少画质损失
- **音频设计** — 智能决策 generate_audio（有环境音的场景自动开启），音乐风格具体到乐器+节奏+动态变化
- **音乐自动匹配** — 根据视频情绪匹配 BGM + BPM 卡点对齐镜头时长
- **TTS 口播** — 支持 macOS `say` 和豆包语音合成模型 1.0；根据整份分镜自动选择统一旁白音色，通过 sidechain ducking 混音
- **字幕烧录** — 字幕时长与 TTS 语音同步，无口播时退回镜头时长
- **中文片头字形校验** — 按片头完整字符集解析本机可用中文字体；找不到覆盖字体或 FFmpeg 渲染失败时明确停止，不输出缺字框或纯黑替代片头
- **多平台导出** — 默认按 480p 输出 YouTube / TikTok / B站 / Instagram
- **断点续传** — 溯源匹配的已验收镜头及其尾帧、实际终态是恢复时的 canonical 输入；中断后不会重新生成或重新判定这些片段，溯源冲突则 fail-closed
- **远端任务恢复** — 提交后立即持久化 Ark 任务 ID；观察超时或查询异常时立即暂停后续镜头，恢复时继续轮询同一任务，避免误报失败、重复提交和重复计费
- **严格顺序生成** — 只有当前镜头生成且通过技术、语义验收后才提交下一镜头；失败或未决任务立即暂停，避免错误上下文和额外计费
- **安全拼接** — 编码、分辨率、帧率、时间基和音频规格全部兼容时使用 `-c copy`；否则统一时间戳和音视频规格后重新编码，避免成片时长异常或画面冻结
- **确定性时间线** — 片段真实时长、转场重叠、口播与字幕起点、片头和最终发布时长共用同一套计算规则，最终通过 ffprobe 校验后才交付
- **三层容错** — 429 限流指数退避 / 身份参考隐私降级 / 断链尾帧兜底；同场景状态参考不可用时停止，不以 T2V 掩盖连续性丢失

## 快速开始

运行要求：

- Python 3.10-3.13（推荐 3.11/3.12；火山 Ark SDK 当前在 Python 3.14 会发出 Pydantic V1 兼容警告）
- [FFmpeg](https://ffmpeg.org/download.html)（macOS 可执行 `brew install ffmpeg`，Ubuntu/Debian 可执行 `sudo apt install ffmpeg`）
- 已开通 Seedance 2.0 Mini 的火山方舟 API Key
- TTS 默认使用 macOS `say`；Windows 和 Linux 可配置豆包语音合成模型 1.0

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 配置 .env
cp .env.example .env
# 编辑 .env, 填入你的火山引擎 Ark API Key

# 3. 运行
python main.py "制作一个15秒的电影短片：机器人在末日清理丧尸"
```

每次运行都会在 `output/` 下创建带 `run_manifest.json` 的独立工作区。发生中断、限流或临时网络错误后，使用原工作区继续：

```bash
python main.py --resume output/20260730_153000_123456
```

恢复运行会复用原始需求、画幅、风格、分镜和远端任务 ID，不重新调用分镜 LLM，也不允许在恢复时悄悄覆盖参数。

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
# 火山引擎 Ark — 默认统一 Key
ARK_API_KEY=your_ark_api_key

# LLM 分镜接口
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3

# Seedance 视频生成接口；独立 Key 可选，未配置时复用 ARK_API_KEY
# ARK_API_KEY_SEEDANCE=your_seedance_api_key
ARK_BASE_URL_SEEDANCE=https://ark.cn-beijing.volces.com/api/v3

# 视频生成模型
SEEDANCE_MODEL=doubao-seedance-2-0-mini-260615

# LLM 文本模型 (生成分镜脚本)
LLM_MODEL=doubao-seed-2.0-lite

# 生成后自适应 5/9 点语义验收 (默认复用 LLM_MODEL)
SEMANTIC_REVIEW_MODEL=doubao-seed-2.0-lite
SEMANTIC_REVIEW_ENABLED=true
SEMANTIC_REVIEW_IMAGE_DETAIL=high

# TTS 引擎 (可选, 默认 macos)
TTS_ENGINE=macos
TTS_VOICE=Tingting

# 使用豆包语音合成模型 1.0
# TTS_ENGINE=volcano
# VOLCANO_TTS_API_KEY=your_doubao_speech_api_key
```

分镜接口使用精简的强类型契约，并显式申请 16,384 个 completion tokens，以容纳 30-60 秒的结构化分镜。JSON 语法修复与分镜契约校正各自职责单一且最多执行一次；若服务端以 `finish_reason=length` 结束，程序会立即停止，不会把截断内容当作普通语法错误反复修复。目标时长用于节奏规划，最终时长允许随完整动作和实际生成素材合理浮动。LLM 返回的正整数单镜时长会在严格校验前收敛到 Seedance 可执行的 `4-15s`，零、负数和非数值仍会被拒绝，不会通过额外 LLM 重试碰运气。参数语义以[火山方舟对话 API 官方文档](https://docs.volcengine.com/docs/82379/1494384?lang=zh)为准。

> **注意**: 视频生成默认使用官方模型 `doubao-seedance-2-0-mini-260615` 和 480p；Mini 仅支持 480p、720p，不支持 1080p。
>
> 调用 LLM、Seedance 视频生成、自适应语义验收和豆包语音合成接口会产生费用；语义响应异常最多增加一次同视频验收调用，不会生成新视频；视频内容未通过语义契约时可能触发最多一次 Seedance 定向重拍。断点恢复请使用 `--resume`，避免重复提交任务。
> 豆包语音的 API Key 获取方式见[WebSocket 单向流式 V3 官方文档](https://docs.volcengine.com/docs/6561/1719100?lang=zh)。程序会根据整份分镜的主题、风格和情绪，从[官方 1.0 音色列表](https://docs.volcengine.com/docs/6561/1257544?lang=zh)自动选择统一旁白音色，无需配置音色 ID。

## 使用示例

```bash
# 基础使用
python main.py "15秒咖啡品牌广告，从咖啡豆到拿铁的过程"

# 指定风格；视频默认生成并导出 480p
python main.py "科技感产品介绍" --style futuristic

# 竖版 + 指定平台
python main.py "旅行短视频" --ratio 9:16 --platforms tiktok

# 带背景音乐 (自动 BPM 卡点)
python main.py "运动品牌宣传" --music music/upbeat_electronic.mp3

# 支持的画面比例
# 16:9 (默认) | 9:16 | 4:3 | 1:1 | 3:4 | 21:9
```

## 流水线

```
用户输入
  → Stage 1:   LLM 分镜 (故事状态/连续性/交互因果/动作契约校验 + 严重问题有限修正)
  → Stage 1.5: 音乐匹配 + BPM 卡点 (可选, 指定 --music 时启用)
  → Stage 2:   Seedance 2.0 Mini 严格顺序生成 → 技术 QA → 自适应 5/9 点语义 QA → 实际终态交接 (最多一次定向重拍)
  → Stage 2.5: 规格统一 (帧率 / 分辨率 / 编码 / 补静音轨)
  → Stage 3:   拼接 + 智能转场 (xfade + acrossfade, 兼容时零损耗直拼，否则规范时间戳后重编码)
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
│   ├── storyboard.py        ← Stage 1: LLM 分镜 + 连续性/动作契约校验 + 自动修正
│   ├── generator.py         ← Stage 2: 严格顺序生成 + canonical 状态 + 角色一致性 + 有限降级
│   ├── models.py            ← 分镜、运行与镜头任务的强类型契约
│   ├── participants.py      ← 结构化字段共用的完整可见角色契约
│   ├── causality.py         ← 题材无关的交互模式、作用范围和生成/验收契约
│   ├── narrative.py         ← 故事弧、逐镜状态变化和跨镜交接契约
│   ├── readiness.py         ← 付费调用前的分镜与运行时资产就绪门
│   ├── semantic_review.py   ← 自适应 5/9 点跨镜语义验收 + evaluator/reference 哈希缓存
│   └── run_state.py         ← 原子 manifest + 工作区恢复
├── tools/                   ← 底层工具
│   ├── seedance_api.py      ← Seedance API 封装 (显式 image_role + base64)
│   ├── ffmpeg_ops.py        ← FFmpeg 操作 (转场 / 调色 / 混音 / 字幕 / 导出)
│   ├── tts.py               ← macOS say + 豆包语音合成模型 1.0
│   ├── beat_analyzer.py     ← 音乐节拍分析 (BPM + 卡点)
│   └── frame_extractor.py   ← 帧提取 + 质量检测
├── prompts/                 ← Prompt 模板
│   └── storyboard_system.md ← 分镜系统 Prompt (运镜知识库 + 模型局限 + 时长/数量策略 + 竖版规则 + 音频设计)
├── luts/                    ← .cube LUT 调色文件
├── music/                   ← 背景音乐库
├── tests/                   ← 测试
│   ├── test_generator.py    ← 视频生成测试
│   ├── test_storyboard_contract.py ← 分镜连续性与动作预算测试
│   ├── test_readiness.py    ← 生成前就绪门测试
│   ├── test_transitions.py  ← 转场逻辑测试
│   ├── test_ffmpeg_audio.py ← 音频处理测试
│   ├── test_seedance_config.py     ← 官方参数约束测试
│   ├── test_seedance_api_resume.py ← 远端任务恢复测试
│   ├── test_tts.py                  ← 豆包语音 TTS 1.0 协议测试
│   ├── test_run_state.py            ← 本地运行状态测试
│   └── test_orchestrator_resume.py  ← 流水线恢复测试
├── docs/                    ← 设计文档
│   └── design.md            ← 完整技术设计 (110KB)
└── output/                  ← 生成产物
    └── 20260609_124525/
        ├── run_manifest.json       ← 运行参数、阶段、镜头状态、远端任务 ID
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
| 视频生成 | Seedance 2.0 Mini (火山引擎 Ark API) |
| 分镜生成 | 豆包 LLM (OpenAI 兼容接口) |
| 视频处理 | FFmpeg (xfade + acrossfade + sidechaincompress) |
| TTS 口播 | macOS say / 豆包语音合成模型 1.0（WebSocket V3） |
| 音乐分析 | librosa |
| 技术质量检测 | OpenCV |
| 镜头语义验收 | 豆包 Seed 多模态 LLM（普通五点、因果九点 + 上一尾帧 + 身份图） |
| CLI | argparse + rich |

## 架构借鉴与取舍

本项目借鉴的是经过交叉验证的生产契约，不复制外部项目的产品形态或技术栈。当前落地映射如下：

| 参考项目 | 借鉴的核心能力 | 在本项目中的落点 | 明确未引入 |
|---------|---------------|------------------|------------|
| FireRed-OpenStoryline | 分阶段故事产物、场景内聚和可恢复任务状态 | `story_arc` → `narrative_beat` → take 的分层产物、稳定 `scene_id`、有限修正、`RunWorkspace` 原子状态 | 素材搜索代理和会话服务 |
| Remotion | 先定义组合、状态和时间线，再执行确定性渲染 | 分镜契约先于付费执行；真实片段时长、转场重叠、字幕/口播起点和发布校验共用时间线 | React/Node 渲染运行时 |
| MoneyPrinterTurbo | 素材时长与最终播放时长分离、任务产物可追踪 | 片段规范化、断点恢复、成片时长校验 | 素材下载、多 Provider 和发布平台自动上传 |
| moyin-creator | 叙事骨架、视觉描述、拍摄控制、首帧和运动/终帧分阶段校准；角色资产归属于具体角色和项目 | 故事状态 → 调度/交互 → camera → 生成 prompt 的分阶段编译，以及 `reference_mode` 和单一角色身份锚点 | Electron 编辑器和素材库 UI |
| ViMax | 规划、角色/事件提取和执行阶段分离 | 整体故事弧、逐镜状态变化、结构化动作/交互阶段和生成就绪门 | 多代理运行时和长篇小说工作流 |
| OpenMontage | 生成前质量门、帧采样复核、确定性编辑计划 | `readiness.py`、按镜头风险自适应 5/9 点语义验收、有限定向重拍和发布校验 | 审批看板、成本治理和第二套渲染器 |
| Emily2040/seedance-2.0 | canonical 与 transient 状态分离、参考职责、accepted footage 优先 | 身份/尾帧分责、构图变化分级、实际终态交接与生成输入溯源；失败片段不能污染后镜参考 | 把未经语义验收的计划状态写入 canon |

这些项目没有可直接复制的通用物理因果图；本项目的 `effect_phase`、`outcome_scope`、`effect_motion` 和逐采样证据门，是结合上述分阶段规划、确定性时间线、质量门与 Seedance 参考职责后形成的本地契约。当前项目只有一个 Seedance Provider 和一个 FFmpeg 后期实现，因此不为假设中的第二个适配器增加接口，也不引入能由现有模块完成的依赖。

## 素材准备

```bash
# 查看需要哪些素材 + 下载指引
python setup_assets.py
```

- **LUT 文件** — 放入 `luts/` 目录（`.cube` 格式，可选，没有会跳过调色）
- **背景音乐** — 放入 `music/` 目录（`.mp3` 格式，可选，没有则用 Seedance 原生音效或自动匹配）

## 降级策略

```
seamless                → first_frame (仅已验收真实尾帧，锁定起始画面)
同场景 intentional_cut  → first_frame (已验收真实尾帧，镜头契约负责后续构图变化)
跨场景 intentional_cut → reference_image (仅角色参考帧，不继承旧环境)
无职责参考素材          → 纯文本 T2V

scene_id 只表示物理地点，不自动推导 seamless；同地点换机位、插入特写或边界景别跨度过大时使用 intentional_cut。
每次只继承上一条已验收尾帧，不把更早镜头的参考图串入当前请求；依赖镜头还必须通过生成输入溯源校验。
身份参考是可选增强：语义验收器只可在其已经检查的中点帧上，为尚无 canonical 资产的 `identity` 角色返回归一化裁剪框。裁剪必须容纳清晰、基本完整的单一角色并排除背景群体、对手、倒影和轮廓；无框、越界或尺寸过小只会跳过身份资产，不会否决合格视频，不会增加语义调用，也不会触发重拍。未启用语义验收时，才保留严格单一可见角色的整帧兼容路径。

官方 API 的首帧、首尾帧和多模态参考模式互斥，因此同场景状态交接统一使用 `first_frame`，不把尾帧降级为可被忽略的 `reference_image`。跨场景才使用 canonical 角色 `reference_image`。同场景状态参考被拒绝或不可用时直接停止该依赖镜头，不能退化为纯文本 T2V 伪造连续性。

分辨率策略: 默认 480p；显式选择 720p 时可降级到 480p
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

所有平台导出固定使用 480p 规格；当前暂不支持小红书。

| 平台 | 分辨率 | 比例 |
|------|--------|------|
| YouTube | 864×496 | 16:9 |
| TikTok | 496×864 | 9:16 |
| B站 | 864×496 | 16:9 |
| Instagram Reels | 496×864 | 9:16 |
| Instagram Feed | 640×640 | 1:1 |

## License

[MIT](LICENSE)

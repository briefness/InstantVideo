# Seedance

这是一个非官方 AI 短视频流水线：输入一句中文需求，即可在可恢复的工作区中完成分镜、视频片段、转场、调色、配乐、口播、字幕和多平台导出。

项目使用火山引擎 Ark 调用 Seedance 2.0 Mini。模型可用性、接口参数和价格以[火山方舟官方文档](https://www.volcengine.com/docs/82379)为准。

## 项目定位

Seedance 的目标不是把长提示词直接交给视频模型，而是把用户需求编译为可验证的生产计划：每个镜头只有一个明确的叙事职责和动作因果，再由前一条已接受镜头的真实状态驱动下一镜头。

它适用于短剧情、产品展示、旅行片段和风格化短片。生成式视频仍有不确定性；本项目通过分镜约束、状态交接与验收降低失败传播，不承诺完全消除模型失真。

## 项目结构

```text
seedance/
├── main.py                 # CLI 入口
├── config.py               # 支持规格、模型和运行默认值
├── .env.example            # 环境变量模板
├── pipeline/               # 分镜、合约、生成、验收、恢复与编排
├── tools/                  # Ark、FFmpeg、TTS、节拍与帧处理
├── prompts/                # 分镜系统提示词
├── luts/                   # LUT 调色文件
├── music/                  # 本地背景音乐库
├── tests/                  # 离线测试
├── docs/adr/               # 架构决策记录
└── output/                 # 每次运行的独立工作区
```

主要入口：`main.py`、`pipeline/orchestrator.py`、`pipeline/storyboard.py`、`pipeline/generator.py`、`pipeline/run_state.py`。

## 核心能力

- 从自然语言需求生成有时长、故事状态、动作与镜头职责的分镜。
- 用统一 Action Contract 表达接触、定向路径、区域与间接作用，限制结果范围并检查因果顺序。
- 同场景镜头使用已接受尾帧作为 `first_frame` 交接；跨场景才使用独立身份参考。
- 对生成片段执行技术质量检查和可选的时序语义验收；失败只允许有限、定向的处理。
- 生成任务、提示词指纹、合约签名、参考职责和验收上下文均持久化到工作区，支持安全恢复。
- 自动执行规格统一、转场、BGM、TTS、LUT、字幕、片头和平台导出。

## 工作原理

```text
用户需求
  -> 生产计划与故事骨架
  -> 分镜详情与生成前就绪校验
  -> 严格顺序视频生成
  -> 技术 QA + 可选语义验收
  -> 已接受终态交接到下一镜
  -> 规格统一 / 转场 / 音频 / 调色 / 字幕 / 导出
```

对动作镜头，系统把准备、作用、结果与范围外主体拆到结构化合同中。Provider 提示词、安全重编译、语义验收和定向重拍共享同一合同投影，避免为某个题材添加关键词补丁。

恢复时，已接受本地片段只在当前验收上下文匹配时复用；上下文变化先离线复核。远端未决任务只轮询原任务，不重新提交。已返回但本地下载或技术 QA 失败的任务会停止，避免隐式重复扣费。

## 要求

- Python 3.11 或 3.12（Python 3.14 目前会触发 Ark SDK 的 Pydantic V1 兼容警告）。
- [FFmpeg](https://ffmpeg.org/download.html) 与 `ffprobe`。macOS 可使用 `brew install ffmpeg`；Ubuntu/Debian 可使用 `sudo apt install ffmpeg`。
- 已开通 Seedance 2.0 Mini 的火山方舟 API Key。
- macOS 默认使用系统 `say` 生成口播；其他系统可配置豆包语音合成。

安装完成后可先检查本地命令是否可用：

```bash
python --version
ffmpeg -version
python main.py --help
```

## 快速开始

```bash
# 安装依赖
python -m pip install -r requirements.txt

# 创建并填写本地配置
cp .env.example .env
# 至少填写 ARK_API_KEY

# 生成一个视频
python main.py "制作一个15秒的视频，机器人在末日城市清除丧尸"
```

首次生成会创建 `output/<run_id>/` 工作区。不要把 `.env`、`output/` 或其中可能包含的本地媒体提交到仓库。

## 配置表

在 `.env` 中配置。默认值与 [`.env.example`](.env.example) 和 [`config.py`](config.py) 保持一致。

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `ARK_API_KEY` | 是 | Ark 的默认 API Key，LLM 与视频接口都会使用。 |
| `ARK_API_KEY_SEEDANCE` | 否 | 视频生成专用 Key；未设置时复用 `ARK_API_KEY`。 |
| `ARK_BASE_URL` | 否 | 分镜 LLM 接口，默认 `https://ark.cn-beijing.volces.com/api/plan/v3`。 |
| `ARK_BASE_URL_SEEDANCE` | 否 | 视频生成接口，默认 `https://ark.cn-beijing.volces.com/api/v3`。 |
| `SEEDANCE_MODEL` | 否 | 视频模型，默认 `doubao-seedance-2-0-mini-260615`。 |
| `LLM_MODEL` | 否 | 分镜文本模型，默认 `doubao-seed-2.0-lite`。 |
| `SEMANTIC_REVIEW_MODEL` | 否 | 语义验收模型，默认复用 `LLM_MODEL`。 |
| `SEMANTIC_REVIEW_ENABLED` | 否 | 是否启用语义验收，默认 `true`。关闭后仅记录技术验收，不会声称语义已通过。 |
| `SEMANTIC_REVIEW_IMAGE_DETAIL` | 否 | 语义验收图片细节级别，默认 `high`。 |
| `TTS_ENGINE` | 否 | `macos` 或 `volcano`，默认 `macos`。 |
| `TTS_VOICE` | 否 | macOS 口播音色，默认 `Tingting`。 |
| `VOLCANO_TTS_API_KEY` | 否 | 使用 `TTS_ENGINE=volcano` 时需要的语音 API Key。 |

## CLI 示例

```bash
# 基础生成
python main.py "15秒咖啡品牌广告，从咖啡豆到拿铁的过程"

# 720p 与风格
python main.py "科技感产品介绍" --resolution 720p --style futuristic

# 竖版与指定导出平台
python main.py "旅行短视频" --ratio 9:16 --platforms tiktok instagram_reels

# 指定背景音乐
python main.py "运动品牌宣传" --music music/upbeat_electronic.mp3

# 从原工作区恢复
python main.py --resume output/20260730_153000_123456
```

支持分辨率：`480p`、`720p`。默认 `480p`；请求 `720p` 时，生成链可降级到 `480p`。

支持比例：`16:9`、`9:16`、`4:3`、`1:1`、`3:4`、`21:9`。

支持平台：`youtube`、`tiktok`、`bilibili`、`instagram_reels`、`instagram_feed`。未指定时默认导出 `youtube` 与 `tiktok`。

完整参数以以下命令为准：

```bash
python main.py --help
```

## 恢复与费用安全

真实生成会调用付费的 LLM、视频生成、可选语义验收和可选 TTS 服务。

- `--resume` 必须指向已有工作区，使用原始需求、分辨率、比例、风格、音乐和平台；不能与这些覆盖参数同时使用。
- 已提交但未决的远端视频任务会记录不可变提交描述符，恢复时仅轮询同一任务 ID。
- 下载暂时失败后，系统保留原任务谱系并在恢复时继续获取同一结果；不会创建第二个任务。
- 已接受镜头会复用；验收上下文变化时仅离线复核本地视频。复核失败或不可用时停止，不降低质量门强行放行。
- 语义失败最多触发一次定向重拍；被拒片段不会成为后续镜头的状态或身份参考。
- 本地技术 QA 或物化失败会显式停止。该状态不授权恢复流程自动提交新的付费视频任务。

费用与可用性以[火山方舟官方文档](https://www.volcengine.com/docs/82379)为准。自动化测试不应使用真实 Key 或真实视频生成任务。

## 输出目录

每次新运行都会创建一个独立目录：

```text
output/<run_id>/
├── storyboard.json          # 已确认的分镜
├── run_manifest.json        # 原始参数、镜头状态、恢复与验收元数据
├── generation_results.json  # 片段生成结果
├── shots/                   # 原始镜头、尾帧和生成谱系
├── normalized/              # 统一规格后的片段
├── semantic_reviews/        # 本地语义验收缓存
└── final.mp4                # 最终成片（成功时）
```

工作区是恢复的唯一依据。移动或清理其中的 `run_manifest.json`、镜头、尾帧或谱系文件会使安全恢复无法判断状态，从而停止而不是重新生成。

常见的中间文件还包括 `concat.mp4`、`with_audio.mp4`、口播文件和字幕文件。它们用于本地后处理；`final.mp4` 是最终交付文件。

## 测试

测试使用本地 fixture、模拟 Provider 和临时工作区，不会调用真实 Ark/Seedance API，也不会产生视频生成费用。

```bash
# 完整回归测试
python -m pytest -q

# Python 编译检查
python -m compileall -q pipeline tests
```

涉及生成、恢复或验收的改动应覆盖：同一远端任务恢复、已接受 take 复用、拒绝 take 不传播、以及失败时不创建额外付费任务。

测试覆盖重点包括分镜合同、时长与规格约束、Provider 参数、动作因果、状态恢复、转场和音频时间线。媒体处理测试仍可能要求本机已安装 FFmpeg。

## 架构取舍与参考项目

本项目借鉴下列开源项目的可迁移思想，而非复制其实现或技术栈。重点是把它们的内容编排、可复现渲染、工作流与镜头控制经验落到本项目的 Action Contract、运行账本和 FFmpeg 处理边界。

| 项目 | 可借鉴方向 | 本项目中的取舍 |
| --- | --- | --- |
| [FireRed-OpenStoryline](https://github.com/FireRedTeam/FireRed-OpenStoryline) | 长叙事拆分与故事状态 | 使用受限 Story Spine 与逐镜状态交接，不让长 JSON 直接决定运行状态。 |
| [Remotion](https://github.com/remotion-dev/remotion) | 可组合、可复现的视频时间线 | 使用确定性时长、转场与音频时间计算；不引入其 React 渲染栈。 |
| [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) | 自动化内容生产流程 | 保留可运行工作区和素材处理，不复制其产品工作流。 |
| [moyin-creator](https://github.com/MemeCalculate/moyin-creator) | 叙事骨架、视觉描述与拍摄控制分层 | 将角色资产、镜头调度、相机和生成提示词分阶段编译。 |
| [ViMax](https://github.com/HKUDS/ViMax) | 规划、角色/事件提取与执行阶段分离 | 使用整体故事弧、逐镜状态和生成就绪门，不引入多代理运行时。 |
| [OpenMontage](https://github.com/calesthio/OpenMontage) | 生成前质量门、帧采样复核与确定性编辑计划 | 落到就绪校验、时序语义验收、有限重拍和发布校验。 |
| [seedance-2.0](https://github.com/Emily2040/seedance-2.0) | Seedance 调用经验 | 以官方 Ark 接口和本项目的持久化恢复边界为准。 |

详细的单一 Action Contract 决策见 [`docs/adr/0001-single-action-contract-owner.md`](docs/adr/0001-single-action-contract-owner.md)。

## 已知限制

- Seedance 2.0 Mini 当前仅支持 `480p` 与 `720p`，不支持 `1080p`。
- 生成式视频可能出现主体漂移、复杂交互失真、文字渲染错误或物理因果不稳定；验收只能拦截已定义的违反项。
- 语义验收提高可靠性，但会增加模型调用、运行时间和费用。关闭它时，系统只保证技术质量路径，不保证动作语义。
- 同场景连续性依赖上一镜已接受尾帧。该尾帧不可用或被接口拒绝时，系统会停止依赖镜头，而不是伪装成不连续的纯文本生成。
- Linux/Windows 默认没有 macOS `say`；需要配置豆包语音或接受无口播输出。
- 背景音乐和 LUT 依赖本地文件是否存在；不存在时相应后处理会跳过或使用可用替代路径。
- 目标时长用于规划，转场重叠、实际生成片段与完整动作节奏会使最终时长在合理范围内浮动。

## 贡献

提交 Issue 请附上复现命令、完整错误信息以及可脱敏的 `output/<run_id>/run_manifest.json`。不要提交 `.env`、API Key、私有素材或完整付费生成产物。

提交 Pull Request 前请运行测试与编译检查。生成、恢复和验收的变更必须使用模拟 Provider 覆盖，不把真实付费视频生成作为自动化测试步骤。修复应落在通用合同、状态机或数据边界，不添加题材专用条件。

## License

[MIT](LICENSE)

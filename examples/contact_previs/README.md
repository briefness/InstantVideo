# 双人动作预演：离线验证切片

目标是让动作片创作者在付费生成之前看清一次接触的空间关系和时序，让开发者能检查数值输入与合约的对应关系。这个示例是人工编排的最小实验样本，不是真实失败 Take 的自动修复，也没有证明生成质量提高。

## 运行与检查

在项目根目录执行：

```bash
python -m tools.motion_previs inspect --storyboard examples/contact_previs/storyboard.json --shot 1
python -m tools.motion_previs validate --storyboard examples/contact_previs/storyboard.json --plan examples/contact_previs/motion.json
python -m tools.motion_previs render --storyboard examples/contact_previs/storyboard.json --plan examples/contact_previs/motion.json --output output/contact-previs
```

`output` 父目录须先存在；目标目录必须不存在。第一次安装、还没有 `output` 时先执行 `mkdir -p output`。重复导出请选择新目录，工具不覆盖上一版。`inspect` 和 `validate` 不写文件；`render` 只写指定的新目录。

示例为 640×360、24 fps、4 秒，共 96 帧：

- 第 0–39 帧准备与蓄势；第 39–48 帧出掌。
- 第 48 帧（2 秒）两只前手相遇。
- 第 55 帧（约 2.292 秒）目标开始收势；示例双脚位置保持固定。
- 第 95 帧为最后可见帧；视频时长仍为 96/24 = 4 秒。

骨架用固定长度和角度插值产生，避免直接插值关节坐标造成四肢伸缩。角度按最短圆弧插值；这不是动力学仿真。示例可以用来检查接触先于目标反应，不能据此判断武术技术、人体自然度或打击力度。

## 输出与事实边界

| 文件 | 用途 |
| --- | --- |
| `preview.mp4` | 带“计划动作”标记的离线预演 |
| `reference.mp4` | 无网格、文字、角色字母的动作参考；仅保留背景与双色骨架，按视频输入规格放大/补帧 |
| `contact_sheet.png`、`keyframe_*.png` | 准备、接触、反应和收势的抽帧检查 |
| `motion_plan.json` | 实际用于渲染的数值计划 |
| `source_shot.json` | 原镜头与画幅快照 |
| `keypoints.json` | 每一帧、每个人物的全部关节坐标 |
| `diagnostics.json` | 接触距离、首次接近/运动帧、出画数量、独立 ffprobe 结果 |
| `experiment.json` | 待执行的基线/预演对照协议；没有观测结果 |
| `manifest.json` | 文件 SHA-256、角色标签、合约版本与源绑定；最后发布 |

`manifest.json` 的 `state_kind` 固定为 `planned_only`。几何检查结果为 `valid_geometry` 时，`generation_quality` 和 `semantic_acceptance` 仍为 `not_evaluated`。不会写入生产账本，不会提供可绕过原有预算与恢复规则的付费入口。

`SeedanceAPI.generate` 新增可选 `reference_video` 参数，接受显式绑定的 `asset://` 素材或公网 HTTPS 文件直链。HTTPS 输入还必须传 `reference_video_sha256`，与远端文件逐字节哈希比对后才能创建任务。数值骨架 JSON 仍是中立交换格式，不能直接当作 Wan、ATI 或 Seedance 的专用控制输入。模型是否遵循双色骨架尚未实测。

## 离线准备生产工作区

```bash
python -m tools.prepare_motion_run \
  --storyboard examples/contact_previs/storyboard.json \
  --package output/contact-previs --output-root output \
  --model doubao-seedance-2-0-mini-260615
```

该命令核验预演包与规范化分镜，复制已核验的参考字节，创建普通 `RunWorkspace`，默认 `paid_take_budget: 0`。不会调用 LLM、生成、语义审查或上传。创建后的工作区可以正常读取，但缺少素材绑定时不能提交生成。失败时保留带失败状态的工作区供诊断，不覆盖原始预演包。

当前支持数据面版本号 `doubao-seedance-2-0-mini-260615`、`doubao-seedance-2-0-fast-260128`、`doubao-seedance-2-0-260128`，以及 Agent Plan 官方模型名 `doubao-seedance-2.0-mini`、`doubao-seedance-2.0-fast`、`doubao-seedance-2.0`。运行时配置必须与绑定名称完全一致；订阅模型名原样记录，不映射为固定版本，不能据此宣称底层版本恒定。

Agent Plan 共用 `.env` 中的 `ARK_API_KEY` 和 `ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3`，无需视频专用 Key。未配置独立视频 Key/URL 时，视频接口跟随共享网关。使用独立凭据时可显式配置 `ARK_API_KEY_SEEDANCE` 与 `ARK_BASE_URL_SEEDANCE`；原有独立 Key 部署默认保留 `/api/v3`。请保持 Key、网关和模型名属于同一服务；不会在认证失败时自动切换到按量计费接口。[官方 Agent Plan 视觉模型文档](https://docs.volcengine.com/docs/ark/agent-plan-personal-visual-models?lang=zh)。

根据 2026-10-27 核对的[官方创建任务接口](https://console.volcengine.com/ark/region:cn-beijing/docs/create-video-generation-task-api)，视频输入使用 URL 或素材 ID，**不使用视频 Base64**；首帧/首尾帧与多模态参考互斥。此实现进一步限定为单镜头、单个 H.264 MP4、无音轨、32 MiB 以内；用 ffprobe 核验时长、帧数、画幅和 24–60 fps。参考片保守保持至少 409600 像素、短边至少 300；12 fps 预演补帧至 24 fps，动作时间不变。

日后得到上传与付费授权后，可以选择以下一种方式：

- 公网直链：将 `reference.mp4` 原样放在已有对象存储或静态文件服务中，取得匿名可读的 HTTPS 文件地址。使用带 SHA-256 的文件名并禁止覆盖该对象；不需要先入库取得素材 ID。托管服务与模型生成的费用各自结算，项目不会自动购买或开通。
- 素材库：上传 `reference.mp4`，人工核对上传源文件的 SHA-256 与 `manifest.json` 一致，取得素材 ID，传入 `--asset-uri asset://实际ID`。**素材 ID 与云端字节的对应关系仍需上传记录/人工核验**。

```bash
# 替换为真实文件地址。准备本身不会下载，默认预算仍为 0。
python -m tools.prepare_motion_run \
  --storyboard examples/contact_previs/storyboard.json \
  --package output/contact-previs --output-root output \
  --model doubao-seedance-2-0-mini-260615 \
  --video-url https://media.example.com/motion/sha256-filename.mp4
```

实际地址须为 ASCII URL（中文路径请先百分号编码）、标准 443 端口，无用户密码、查询参数或片段；登录链接、临时签名链接、分享页面和重定向地址不支持。为避免签名令牌写入工作区，本版不接收查询参数。下载时检查全部 DNS 结果均为公网地址，固定连接已核验 IP 并验证原域名 TLS 证书，不使用系统代理、不跟随跳转，限制 32 MiB 与网络等待时间。仅在远端文件 SHA-256 与本地快照完全一致后提交；失败时停止，不自动降级为无参考生成，且释放本次未提交的预算预留。

准备命令只做离线校验，不证明链接可访问。实际生成需要用明确的 `--paid-take-budget 已授权数量` 另建工作区，再执行原有 `python main.py --resume <新工作区>`。预算为 0 时在下载之前停止；恢复已有远端任务只轮询原 ID，不重读链接或再提交。

**提交前哈希校验不能保证 provider 稍后读取时 URL 内容仍不变**，也不能保证 provider 网络同样可达，因此托管期间须保持文件不可覆盖、链接可读直到任务结束。已接受片段复用依赖原提交记录和本地快照，不要求远端链接永久在线。本轮只用模拟网络验证这些路径，没有上传、访问实际参考链接或付费生成。

文件指纹、源分镜指纹、模型、`reference_video` 输入模式与素材地址同时进入生成来源和验收上下文。添加、移除或更换动作参考后禁止复用旧 Take 或通过重新验收绕过；已有远端任务只允许在记录一致时轮询原 ID。已完成成片的快捷返回也先核验参考。旧的无动作参考记录保留兼容性。

## 自定义一个镜头

1. 准备已有格式的 `storyboard.json`。镜头必须明确包含 `active / direct_contact / single` 合约、两个具名角色、可见结果、固定机位和 `continuity_from_previous: none`。若全局角色元数据声明群体或非双足结构，会拒绝映射到单个人形骨架。
2. 用 `inspect --storyboard <文件> --shot <编号>` 查看合约投影与 `source_sha256`。人工确认动作、接触部位和角色映射，再参考 `motion.json` 编排数值计划。
3. 把绑定值写入计划。动作合约、时长、机位、角色或画幅变化时，旧计划会拒绝使用；不能只换哈希而不检查动作是否仍匹配。
4. 运行 `validate` 查看明确诊断，通过后再 `render`。数值正确后仍需人工观看动作，判断接触关节是否真的表达了原合约的作用部位与可见结果。

输入字段：

| 字段 | 约束与含义 |
| --- | --- |
| `version`、`shot_id`、`source_sha256` | 计划版本、镜头编号及源绑定 |
| `width`、`height` | 偶数尺寸，各 256–1280；乘积 ≤921600；画幅与分镜一致 |
| `fps` | 12、24 或 30；镜头时长沿用原分镜的 4–15 秒整数 |
| `rigs` | 恰好两个独立骨架，`entity` 与合约角色名完全匹配 |
| `facing` | `left` 或 `right`；镜像身体的横向伸展，整个镜头内固定 |
| `lengths` | 可选骨长覆盖；单位为画布短边；省略则用默认骨架 |
| `keyframes` | 2–64 个；从 0 严格递增，最后一帧必须是时长×fps−1 |
| `root` | 骨盆的 `[x,y]`，分别相对画布宽、高归一化到 0–1 |
| `angles` | 各段骨骼的屏幕平面角度，−180° 到 180°，向下为正；每个关键帧省略的项回到默认值，不继承上一帧 |
| `contact` | 一次接触的帧、双方接触关节、目标开始运动的帧 |

骨骼名称与默认长度/角度以 `pipeline/motion_plan.py` 的 `BONES` 为准。支持头、颈、双肩、双肘、双手、双膝和双脚，以骨盆为根。`lead` 表示前侧肢体，`rear` 表示后侧肢体，不代表真实人物的解剖学左右手。

接触容差固定为短边的 1.5%，不是可以调大以强行通过的输入参数。首次进入容差范围的帧与声明接触最多相差一帧；报告保留实际测量值。目标在约定反应帧前必须保持姿态；这个切片不支持目标提前主动迎击、独立移动或多次接触。

JSON 上限 2 MiB，拒绝重复字段、非有限数、非法类型、未知骨骼、重复关键帧与超大画布。所有帧检查出画及左右关系。编码限时 90 秒、验证限时 15 秒，使用单线程编码；编码失败、探测信息不完整、并发输出冲突或写入失败均不会发布一个成功包。

## 为什么先做这个范围

主要用户需要减少不可控重试；工程用户需要知道控制信号如何进入模型、何时失效、如何复现。先固定一次双人接触，把动作语义、几何计划、渲染结果和生成结果分开，才能测量问题发生在哪一段。

已有历史记录包含“期望影响两个目标，实际只影响一个”的验收反馈，但旧记录不全有逐 Take 结构化证据，且包含群体动作和连续镜头。它只能帮助选取研究方向，不能证明二维双人预演能修复这些失败。本例不把那些旧镜头重标为符合当前实验条件。

本实现复用 Action Contract 编译器，保持其单一语义来源。输入模型、几何计算、媒体渲染和命令行编排分开；无需新增依赖、数据库或工作流框架。未来只有证明某个 Provider 可消费动作控制且能改善实际结果后，才添加对应适配；适配还须参与提示词/参考指纹、付费提交描述符、恢复和验收上下文。

当前未覆盖：完整三维模型、多人/群体交互、移动摄影机、骨骼遮挡、碰撞动力学、脚底约束、解剖学检查、动作捕捉、身份保真，以及从自然语言自动生成可靠骨架计划。二维坐标相遇不等于三维空间发生真实接触。

## 对照实验如何证明收益

输出包准备同一动作的两组试验：现有合约流程，以及增加动作视频参考和固定解释指令的流程。这是组合干预，不能把差异归因于视频单一变量。各三次仅是小规模试验，不足以宣称普遍有效。两组固定模型及版本、基础提示词、身份参考、时长、画幅和分辨率，保留所有失败结果；有 seed 时记录，但不能假定相同 seed 必然可复现。

动作参考组目前为 `submission_ready: false`。接口格式与离线恢复规则已接入；执行前仍须绑定实际上传素材、固定合格标准与费用上限，并得到付费授权。本轮预算为零，没有实际提交。

评审应不知道分组，逐帧标注接触与反应、人物/作用范围、身份和屏幕方向、合约终态；几何/姿态检测仅作辅助。报告每个原始 Take、失败原因、样本量与不确定性，并计入准备与人工修正时间。没有合格片段时，每条合格片段成本不可定义，不能记为零。

## 完整实验操作

以下入口已实现；本轮仅执行离线模式。上传、真实生成及人工评审结果仍未验证。执行器使用 macOS/Linux 的进程文件锁；Windows 原生执行尚不支持。

### 1. 离线准备与检查

```bash
python -m tools.motion_experiment prepare \
  --storyboard examples/contact_previs/storyboard.json \
  --package output/contact-previs --output output/contact-experiment

SEEDANCE_MODEL=doubao-seedance-2-0-mini-260615 SEMANTIC_REVIEW_ENABLED=false \
  python -m tools.motion_experiment check --experiment output/contact-experiment

python -m tools.motion_experiment report --experiment output/contact-experiment
```

默认三对样本，独立工作区，每对轮换执行顺序。`prepare` 不访问网络，清单最后写入；已有目录不覆盖，半成品目录不允许执行。`check` 和未带 `--execute` 的 `run` 只检查本地配置，`blocked` / 退出码 2 是零预算或素材未绑定时的预期结果。不会读取 API 额度或声称账号权限已核验。

实验冻结源分镜、模型、运行参数与每份分镜指纹。真实实验须另建目录，每组 1–3 个样本，预算只能为 0 或总样本数。每个样本最多提交一个付费 Take；不补拍失败样本。不要手改清单、预算或工作区，也不要同时用普通 `main.py --resume` 操作实验子工作区：实验锁只协调本实验执行入口。

### 2. 托管同一参考文件

已有合格公网 HTTPS 地址或 `asset://` ID 时可直接用于下一步。上传工具是可选路径：

```bash
# 默认只输出文件 SHA-256、大小和建议对象名，不读取上传授权、不联网
python -m tools.publish_motion_reference --package output/contact-previs
```

工具要求已配置支持条件 PUT 的对象存储、公开读取地址和针对 `<SHA-256>.mp4` 的预签名上传 URL。签名须允许 `Content-Type: video/mp4` 和 `If-None-Match: *`。支持此语义的参考：[AWS S3 条件写入](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)、[Cloudflare R2 条件 PutObject](https://developers.cloudflare.com/r2/api/s3/extensions/)。不能假定所有兼容存储都支持此保护。

以后授权上传时，通过环境变量 `MOTION_UPLOAD_PUT_URL` 提供预签名 URL，再运行：

```bash
python -m tools.publish_motion_reference --package output/contact-previs \
  --public-base-url https://media.example.com/motion --execute
```

示例域名须换成真实公开基址。工具不创建云账号、购买服务、变更存储权限或自动生成签名；签名不作为命令行参数、不写入产物或日志。条件 PUT 禁止覆盖；已有对象仅在公开读取字节与本地一致时作为幂等成功。连接失败、冲突、重定向不会自动重试；上传成功但读回失败也不能用于生成，应核对原对象。重新执行仍做条件写入。

读取链接必须无查询参数、无鉴权、无跳转。上传和读回均限制公网 HTTPS 与固定 IP/TLS 校验，载荷最多 32 MiB；DNS 使用系统解析器，超时不构成整个 DNS/TLS/传输过程的统一绝对期限。需由存储策略保证对象在 Provider 完成前稳定且不可覆盖；本地校验不能保证 Provider 后续能访问或取到同样字节。

没有存储账号时，短期人工监督的试验可用 [Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)。安装官方 `cloudflared` 后：

```bash
python -m tools.serve_motion_reference --package output/contact-previs
# 依据上一命令输出的本机端口，另开终端：
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:实际端口
```

将隧道的 HTTPS 域名与本机输出的 `/<SHA256>.mp4` 路径组合为参考地址。服务只绑定 127.0.0.1，只提供启动时已核验、载入内存的单个视频；支持 GET/HEAD 与单段 Range，无目录浏览、写入或其他文件路由。不要使用整个项目目录启动普通文件服务器。临时隧道不需存储账号，但没有可用性保证；须在所有远端任务结束前保持连接，结束或阻塞时关闭服务与隧道。它不是长期素材库。

Fake-IP 网络可能把公网域名解析为 198.18.0.0/15，从而被校验器正确拒绝。可从可信公共 DNS 查询该域名的实际公网地址，仅对本次运行设置 `MOTION_REFERENCE_DNS_PIN=域名=公网IP`。固定只匹配一个域名，依然拒绝私网/保留地址、跳转和代理，TLS 仍核验原域名，下载后仍核对 SHA-256；不会修改系统 DNS。该设置不包含秘密，也不写入生产分镜。地址失效时应重新核验，不能关闭证书或公网检查。

### 3. 明确授权后的原始 Take 执行

以下命令会在 `run --execute` 时调用真实生成服务，只能在已授权预算内执行：

```bash
python -m tools.motion_experiment prepare \
  --storyboard examples/contact_previs/storyboard.json \
  --package output/contact-previs --output output/contact-experiment-real \
  --video-url https://media.example.com/motion/实际SHA256.mp4 \
  --repetitions 3 --paid-take-budget 6

SEEDANCE_MODEL=doubao-seedance-2-0-mini-260615 SEMANTIC_REVIEW_ENABLED=false \
  python -m tools.motion_experiment run --experiment output/contact-experiment-real --execute
```

素材 ID 路径用 `--asset-uri` 替换 `--video-url`。实际 URL 使用 ASCII；模型须与清单一致且账号有可用权限。配置凭据沿用项目现有环境变量。实验关闭远程语义审查，也不执行配音、插帧、拼接等后期；本地技术检查、尾帧和账本继续沿用现有流程。技术通过只表示素材可供评审。

实验禁用 Ark SDK 的隐式请求重试。创建请求超时、连接中断、HTTP 408/5xx 等结果不明确时保留预算预留并停止；没有任务 ID 不等于没有发生生成。HTTP 401/403 属于明确认证/权限拒绝，释放未提交预算并立即停止，不重试其他样本。有效 Key、账号权限和模型开通状态仍需真实服务验证，本地配置检查不能代替。401 时先核对 Key 对应的服务与接口地址，再检查凭据有效性；不把密钥写入命令行、报告或聊天。

再次运行相同 `run --execute` 会跳过已完成样本，保留失败样本，等待中的任务只轮询原 ID。提交结果未知、缺少恢复凭证、部分物化失败或账本保存失败时停止，核对原任务后处理；禁止通过删目录或加预算来绕过。不要修改正在执行的工作区文件。

### 4. 人工逐帧评审与报告

```bash
python -m tools.motion_experiment review-export \
  --experiment output/contact-experiment-real --output output/contact-review
python -m tools.motion_experiment report \
  --experiment output/contact-experiment-real --review-dir output/contact-review
```

仅将 `contact-review/reviewer/` 交给评审者；外层 `private_mapping.json` 留在本地。导出所有可读原始 Take（包含技术拒绝片段），随机排列并换成匿名文件名，不转码。损坏、缺失、超 32 MiB 或超 900 帧的本地片段列为不可评审，原账本仍保留。没有本地视频的生成失败计入计划样本与执行报告，不伪造视频。

新导出使用 `motion-review-v3`，同时携带动作合约与生成流程共享的 `visual_invariants`。评审者用支持逐帧播放的软件填写匿名 `reviewer_id`，记录从 0 开始的首次接触帧和首次反应帧。`assessment` 区分 `unreviewed`（未评）、`unjudgeable`（无法判断，须在 notes 说明）与 `reviewed`（已评）。已评须填写五项布尔值：身份/屏幕方向 `identity_ok`、作用对象/范围 `scope_ok`、终态 `endpoint_ok`、可见因果关联 `causal_link_ok`，以及身体朝向、固定机位等视觉要求 `visual_invariants_ok`。人物站在正确左右位置不等于朝向正确；接触与反应时序正确也不能弥补朝向错误。

五项均真且接触先于反应才记为通过；这仍是人工观察，不证明物理因果。事件不存在填 `null`；证据不清时选择无法判断。旧 v1/v2 仍按原标准读取并显示版本，不补填新检查，不能参与 v3 双人共识；需要重新评审时导出新目录，保留原评分。分组标签隐藏不能消除画面本身的线索。

朝向与机位约束从分镜的结构化字段编译到正常提示词、安全重试提示词、提交前检查和自动语义验收。只有明确支持的朝向表达进入该有界投影，不把自由动作描述混入重试。二维参考目前仅接受与分镜一致、朝向彼此的侧面骨架；颜色按合约角色绑定，不依赖骨架数组顺序。修复的是输入与验收的一致性，不能据此声称最终视频已改善，也无需为此扩展三维建模系统。

汇总回查原始 Take、分组、SHA-256 与真实帧数，拒绝重复映射和越界帧号。分别报告计划、已评审、通过数；未评审不计为成功。评审者诚实性和被同时篡改的整个工作区无法靠本地哈希证明。当前报告不估算费用；需用供应商账单与实际人工工时对账，小样本结果仅为试验观察。

### 5. 离线反例校准与双人复核

本轮只深化评审可靠性，暂不认定生成模型失败的根因。对输出根目录下 90 个工作区的只读账本检查发现 28 条 Take、15 条 rejected，Take 级结构化拒绝证据为 0；这不足以把历史失败归因于缺少建模或某个模型。缺失证据保持未知。

先用六段合成教学动画熟悉评分：正常接触、提前反应、没有接触、没有反应、角色互换、接触阶段不可见。每段沿用同一个四秒二维骨架样例，只改变指定因素；构造答案不经过评分函数反推。它们没有真实人物或模型生成结果，不进入生产工作区，也不能当作合法预演包导入生成流程。

```bash
python -m tools.motion_calibration build --output output/contact-review-calibration
python -m tools.motion_calibration score --bundle output/contact-review-calibration
```

只把 `reviewer/` 交给评审者；`calibration_manifest.json` 及 score 输出包含答案，保留给协调者。空白表单报告 answered=0，不伪造评分。用相同模板的独立副本分别填写不同的匿名 reviewer_id；完成后才能核对答案。正例的接触/反应教学基准为第 48/55 帧，容差为两帧，须由独立人员逐帧复核标签和教学可读性。六例只覆盖已知失败模式，不能据此宣称评审者可靠或模型效果提升，另需保留未用于练习的真实验证样本。

```bash
# a.json 与 b.json 是两位评审者各自完成的 ratings.json 副本
python -m tools.motion_calibration score --bundle output/contact-review-calibration \
  --ratings-file a.json --ratings-file b.json

# 对真实实验的相同片段进行双人复核
python -m tools.motion_experiment report \
  --experiment output/contact-experiment-real --review-dir output/contact-review \
  --ratings-file a.json --ratings-file b.json
```

练习与真实实验分别使用各自包的评分表，不能混用。报告给出判定一致率、每个事件帧差、无法判断和分歧项。两人结论不一致，或都能判断但事件帧号相差超过两帧/一方认为不存在时，列为 disputed，不自动记为通过。两帧是当前版本固定的复核触发阈值，不是统计置信界限。一方未评时保持未评，两人均无法判断时保持无法判断；不同编号不能证明评审者相互独立。一致率在没有共同已评条目时为 null。

旧评审仍可读取并明确标为 `legacy-v1`，不改写历史结论，不能加入新版双人复核或校准。现有实验包、预算与 Canonical State 均不因人工报告而改变。这里的假设是“显式保留未知并引入反例，能够暴露原先被汇总隐藏的分歧”；验收是错误接受可见、缺证据不升级为成功、分歧可定位，尚不是生成收益结论。

源码研究依据：[ATI 的轨迹归一化与时间采样](https://github.com/bytedance/ATI/blob/main/wan/utils/motion.py)说明控制输入具有模型约定；[VBench 的运动平滑度](https://github.com/Vchitect/VBench/blob/master/vbench/motion_smoothness.py)使用插帧重建差异，不能替代接触语义；[CoTracker](https://github.com/facebookresearch/co-tracker)提供轨迹/可见性测量，但不能单独证明接触与因果，且主要代码为非商业许可。本轮只借鉴测量与证据分离的思路，复用已有 FFmpeg/Pillow；没有复制这些实现、下载权重或新增依赖。

## 开源研究记录

核对日期：2026-10-27。以下 SHA 是读取文件的 Git blob 标识，不是声称这些仓库有相同版本或已经在本机运行。源码没有复制到本项目，模型也没有下载。

| 来源与文件 | 核对的 blob | 本次采用或不采用的判断 |
| --- | --- | --- |
| [huashu-motion / cli/lib/render.js](https://github.com/alchaincyf/huashu-motion/blob/main/cli/lib/render.js) | `5d9ff44cbd688bed9acbe49204b6f22b87d8ed9d` | 参考显式帧率、渲染限制和产物检查；不引入完整浏览器编辑器 |
| [huashu-art-motion / scripts/qa.py](https://github.com/alchaincyf/huashu-art-motion/blob/main/scripts/qa.py) | `98758fc3c7071b584fa40cb09dd76d40332ee344` | 参考数字诊断与关键帧一起交付；运动面积指标不能代替接触正确性 |
| [lemo-opuscar / TECHNIQUE.md](https://github.com/lemomo-ai/lemo-opuscar/blob/main/TECHNIQUE.md) | `97326879ff86ea0daa782b73e7faa90719061f11` | 参考共享时间轴和确定性逐帧生成；本切片不扩展配乐与风格库 |
| [taishang-laojun / README.md](https://github.com/liuhaisi2026212-maker/taishang-laojun/blob/main/README.md) | `a19ee0653374435ee9c0e96a446b4dea7ef4e0e9` | 内容策划不直接解决动作控制；没有复制其技能文本。其声明许可为 CC BY-NC 4.0 |
| [ByteDance ATI / wan/utils/motion.py](https://github.com/bytedance/ATI/blob/main/wan/utils/motion.py) | `d9f36f6de2f57e4fc29e7ce305d2507735cc1f67` | 轨迹有坐标归一化与模型特定时间采样；因此本例保留坐标系、帧率和逐帧数据，不伪装成 ATI 格式 |
| [Wan2.2 / wan/animate.py](https://github.com/Wan-Video/Wan2.2/blob/main/wan/animate.py) | `6fa4af4660469115fba7b0e8af090acb82108ff6` | 源码分别读取姿态视频、面部视频与角色图，说明需要专门模型与输入链；本次不引入 GPU 模型运行时 |

## 离线回归

```bash
python -m pytest -q tests/test_motion_previs.py tests/test_motion_reference.py tests/test_reference_video_url.py tests/test_motion_experiment.py tests/test_motion_upload.py tests/test_motion_assessment.py tests/test_motion_calibration.py
# 仓库现有默认值测试要求 Mini；仅对本条测试命令固定环境，不修改用户配置
SEEDANCE_MODEL=doubao-seedance-2-0-mini-260615 python -m pytest -q
```

测试覆盖时间轴端点、固定骨长、角度跨越 ±180°、横竖画幅、源绑定失效、错误作用对象、缺失接触、提前反应、无反应、出画、无效输入、重复输出、并发写入、编码超时、部分失败清理及真实 FFmpeg 编码。真实生成、盲评、不同平台编码器与 Python 3.11/3.12 环境仍须单独验证。

# MMPG 语义判停独立验证

这是一个与主程序分开的本机验证工具：浏览器采集音频并通过 RTC 发布，Python 连接火山 MMPG，页面展示豆包 ASR 转写和话轮事件。它用于观察面试官句中停顿是否被误判为提问结束，**不调用你的 LLM，也不合成或播放 AI 回答**。

本工具尚未完成你账号下的真实云端验证。页面能够启动、依赖能够安装，均不能证明云端鉴权、音频传输或语义判停已经有效；需要完成本文的现场测试。

## 1. 手动开通与准备凭证

请在自己的浏览器中操作控制台，真实密钥只填写到本机 `config.local.json`，不要发到聊天或提交到版本库。

### 第一步：实名认证并开通 RTC

1. 登录[火山引擎控制台](https://console.volcengine.com/)，完成实名认证。
2. 打开[实时音视频 RTC](https://console.volcengine.com/rtc?from=doc)，按页面提示开通服务。
3. 切换到 **AI 音视频互动方案**，再开通该方案。
4. 打开 [AI 音视频互动方案 → 应用管理](https://console.volcengine.com/conversational-ai/devTools/appIdManage)，记录该方案应用的 **AppId**。不要拿普通 RTC 应用、豆包语音应用的 AppId 相互替代。

官方建议测试先使用方案自动创建的默认应用；首次赠送的方案 Tokens 仅对默认应用生效，当前文档标注有效期一个月，实际额度以控制台为准。[开通服务说明](https://docs.volcengine.com/docs/real-time-communication/rtc-activate-service?lang=zh)、[应用与临时 Token 说明](https://docs.volcengine.com/docs/real-time-communication/integrate-software-application?lang=zh)

### 第二步：生成本次测试的 RTC 临时 Token

在上述 AI 方案应用的管理页面生成临时 Token，使用以下测试标识，或者自行替换后保持所有位置一致：

| 项目 | 建议值 | 必须对应的位置 |
|---|---|---|
| AppId | 上一步的 AI 方案 AppId | `rtc_app_id` |
| RoomId | `mmpg-probe` | `rtc_room_id` |
| 真人 UserId | `tester` | `rtc_user_id`，也是 MMPG 监听的目标用户 |
| 机器人 UserId | `mmpg-agent` | `rtc_bot_id`，必须与真人 UserId 不同 |
| 临时 Token | 控制台生成的值 | `rtc_token` |

临时 Token 必须属于同一 AppId，RoomId 和真人 UserId 必须与本机配置完全一致，包括大小写。记录有效期；到期后在控制台重新生成并更新配置。机器人 ID 不要填进真人用户的 Token 配置。这个临时 Token 只用于独立测试，正式产品需要服务端签发机制。

### 第三步：准备 MMPG 签名所需的 AK/SK

在火山引擎 **访问控制 → 密钥管理** 中获取自己的 Access Key ID 和 Secret Access Key，分别用于 `volc_access_key` 和 `volc_secret_key`。如果使用子账户，官方 MMPG 接入说明要求该账户具有 `RTCFullAccess` 权限；可在[访问控制 → 用户](https://console.volcengine.com/iam/identitymanage/user)检查所属账户的授权。

这对密钥供本机 Python 对 MMPG WebSocket 请求签名。官方接入地址为 `wss://interactai.volcvideo.com/api/v2/mmpg`，签名地域为 `cn-north-1`，服务名为 `rtc`。它们不是豆包语音 API Key，也不是 RTC AppKey。[MMPG 接入与权限说明](https://docs.volcengine.com/docs/real-time-communication/quick-integrate-custom-agent-via-gateway?lang=zh)

### 第四步：开通自己的豆包 ASR 2.0 小时版

1. 打开[豆包语音控制台](https://console.volcengine.com/speech/)，确认当前项目，例如 `default`。
2. 在 **开通管理/服务管理** 中找到 **豆包流式语音识别模型 2.0**，查看试用或按用量计费入口。
3. 本工具使用小时版资源 ID：`volc.seedasr.sauc.duration`。选择这一模型及资源类型；本测试不需要端到端实时语音模型、TTS、声音复刻或纯并发包月版。
4. 在同一项目的 **API 管理/API Key 管理** 中创建或获取 API Key，填到 `doubao_api_key`。[默认项目 API Key 入口](https://console.volcengine.com/speech/new/setting/apikeys?projectName=default)
5. 确认 API Key 所在项目已开通该 ASR 服务，并有可用试用额度、资源包或按量付费余额。不同项目的资源和 API Key 相互隔离。

本工具使用自有 ASR Key。官方说明公共 ASR 资源不支持热词及替换词表，业务依赖这些能力时应使用自行开通的 ASR；不要仅凭接口存在热词字段，就认定公共资源也支持。此原型不包含热词效果验证。[豆包控制台指引](https://docs.volcengine.com/docs/doubao-voice/console-guidance?lang=zh)、[ASR 配置说明](https://docs.volcengine.com/docs/real-time-communication/configure-asr?lang=en)

### 两套服务凭证的区别

| 服务 | 凭证 | 作用 |
|---|---|---|
| RTC / AI 音视频互动 / MMPG | AI 方案 AppId、RTC 临时 Token、房间与用户 ID | 浏览器进房、发布本次测试音频，MMPG 订阅指定用户 |
| RTC / AI 音视频互动 / MMPG | 火山账号 AK/SK | 本机 Python 连接 MMPG 时签名鉴权 |
| 豆包语音 ASR | 自己开通的 ASR API Key、资源 ID | 调用指定项目的豆包流式识别 2.0 小时版 |

## 2. 填写本机配置

将工具目录里的 `config.example.json` 复制为同目录的 `config.local.json`，再用文本编辑器填写。不要修改示例文件来保存真实密钥。JSON 不支持注释，布尔值和 `null` 不要加引号。

| 字段 | 填写内容 / 默认值 |
|---|---|
| `volc_access_key` | 火山账号 Access Key ID |
| `volc_secret_key` | 与该 AK 配对的 Secret Access Key |
| `rtc_app_id` | AI 音视频互动方案应用 AppId |
| `rtc_room_id` | `mmpg-probe`；必须匹配 RTC Token |
| `rtc_user_id` | `tester`；必须匹配 RTC Token |
| `rtc_bot_id` | `mmpg-agent`；不得与真人 ID 相同 |
| `rtc_token` | 本次测试的 RTC 临时 Token |
| `doubao_api_key` | 已开通 ASR 2.0 的同项目 API Key |
| `doubao_resource_id` | `volc.seedasr.sauc.duration` |
| `silence_ms` | `800`；启动语义判停检查的静音阈值，不是固定的话轮结束时间 |
| `aivad` | `true`；启用语义判停 |
| `twopass` | `true`；开启二遍识别 |
| `enable_prefetch` | `true`；允许接收句中识别片段，用于观察，不触发 LLM |
| `expire_time_ms` | `null`；本工具不传 `ExpireTime`，见下方 2 秒兜底说明 |
| `force_end` | `false`；不要求静音达到 `silence_ms` 后直接强制结束 |
| `max_session_seconds` | `1800`；工具单次测试会话上限，单位秒 |

关闭 AIVAD 做对照时，工具也会关闭本次会话的 `EnablePrefetch`，避免厂商示例中“预取隐式开启 AIVAD”的行为干扰对照。有效参数会写入本机日志。再次打开 AIVAD 后，预取恢复为配置文件中的值。

**2 秒兜底需要单独验证：**官方 StartVoiceChat 参数说明写明，语义未完整但静音超过 2 秒也可能结束；MMPG 引用了同名 VAD 参数说明，但未单独给出修改此上限的字段。默认 `expire_time_ms=null` 只是本工具不传另一个 `ExpireTime` 强制结束参数，不代表关闭了云端的 2 秒规则，也不承诺无限等待。厂商参数页和发布说明对 `ExpireTime` 的计时起点表述不完全一致，不能把它当成语义等待上限。先保持 `null` 完成基线；如需调整，再单独测试，不要混合统计。修改判停参数后需要重新建立会话。

## 3. 启动独立工具

准备可用的 Python 和 Node.js/npm。首次启动需要访问依赖下载源，浏览器需要访问火山 RTC 服务。

1. 在 `tools/mmpg-probe` 中双击 **启动验证.bat**，由它运行该目录的 `run.ps1`。
2. 首次运行会在工具目录安装隔离的 Python/npm 依赖并通过 esbuild 构建前端，不修改主项目依赖。
3. 保留启动窗口，打开 [http://localhost:18765](http://localhost:18765)。不要直接双击前端 HTML 文件。
4. 默认先用麦克风：允许浏览器麦克风权限，选择正确输入设备，在页面开始会话。
5. 确认浏览器 RTC 连接及 MMPG 会话都建立成功，然后开始说测试文本。仅看到网页或 WebSocket 已连接，不等于已收到可识别的音频。
6. 测试后在页面停止会话，确认连接结束，再关闭浏览器及启动窗口，避免遗留在房会话。

本地 Python 只监听本机端口；它仍需向公网连接 MMPG。该验证不要求你部署一个供火山回调访问的公网 HTTP 服务。它不接入主程序，也不使用主程序的 LLM 配置。

### 可选：浏览器分享音频

麦克风链路跑通后，可选择页面的浏览器分享音频输入。按浏览器提示选择带声音的标签页或系统共享来源，并勾选“分享音频”；可用来源取决于浏览器与系统，不能保证每个共享窗口都有音轨。

如果页面没有收到音轨或没有识别文字，先检查共享对话框是否选中了音频。不要同时把同一段声音经麦克风和共享音轨送入，以免重复识别。分享音频是独立的输入测试，不代表已经验证 Windows 主程序的系统音频采集。

WAV 回放属于后续扩展，本轮以麦克风和浏览器可用的分享音频为准；不要把文件选择或播放成功当作已经接入 RTC。

## 4. 如何看事件与时间

- `user.conversation.created`：检测到开始说话。
- `user.audio.transcription.started`：识别到首个有效文字。
- `user.audio.transcription.segment`：句中片段；不能用它判断面试官已经问完。
- `user.audio.transcription.done`：服务端给出的本轮最终转写。测试要核对内容是否完整、有没有抢在后半句之前到达。
- 页面手动结束标记：由测试者在**确实说完整段提问后**点击，作为本机时间参照。它是测量标记，不是要求云端立即提交 ASR 的指令。

`done` 的本机接收时间与手动标记之差，包含识别/判停/传输以及手动点击误差，不等于模型推理耗时。标记只关联当时尚未结束的同一云端话轮；如果 `done` 已先到，随后点击的标记只留在时间线，不会错误关联下一轮。判断抢答仍要看事件与音频顺序：若后半句尚未说完就收到了 `done`，才是明确的提前切分证据；不要仅凭手动点击稍慢就认定误判。

## 5. 测试清单

先用默认参数、安静环境及固定音量，每个案例重复至少三次。方括号是操作说明，不要朗读出来。重点是正常问完后尽快得到完整文字，同时保留句中短暂停顿。

| 测试 | 朗读或操作 | 观察重点 |
|---|---|---|
| 连续提问 | “请介绍一下你在上一个项目里负责的模块，以及你是怎么定位性能瓶颈的？” | 最终转写完整；问完后出现 `done` |
| 语义未完的短暂停顿 | “如果缓存突然失效，[停 0.5 秒]你会如何避免请求直接压垮数据库？” | 停顿处不要先生成独立 `done` |
| 句中思考 | 同一句分别停 1.0、1.5、1.9 秒 | 记录每种停顿的提前切分次数，特别核对不到两秒的情况 |
| 两秒边界 | 同一句分别停约 2.0、2.5 秒，再说后半句 | 记录实际行为，不预设一定等待；与兜底配置关联 |
| 多个分句 | “先讲一下方案背景。然后说明你做了哪些权衡。最后说说如果重做，你会改什么？” | 句号或 ASR 分句不一定代表整段提问结束 |
| 明确问完 | “这个接口为什么需要幂等？”说完立即点击手动结束标记并保持安静 | 对比 `done` 接收时刻和标记时刻 |
| 自我纠正 | “你使用的是 Redis，[短暂停顿]不对，我想问的是 MySQL 的隔离级别。” | 后半句修正是否保留，是否提前切分 |
| 中英技术词 | “请比较 CAS 和 synchronized，并解释 ABA 问题。” | 名词转写与轮次切分分别评价 |
| 连续两个问题 | 问完一题、等 `done`，再问第二题 | 不丢后续问题、不重复累计旧文字 |
| 分享音频 | 同样文本通过浏览器可分享音频播放 | 与麦克风输入分开记录，确认不是空音轨 |
| 会话释放 | 点击停止，等待连接结束 | 机器人与真人退出，后续不再收到转写 |

每次记录：输入来源、配置组合、测试文本、停顿长度、预期段数、实际 `done` 段数、完整转写、是否提前切分、手动结束时刻、`done` 接收时刻。保留原始事件中的会话/轮次标识以区分重复通知和新的话轮，报告中不要附上密钥。

判断效果时先统计“完整问题被提前切开的比例”，再比较正常问完后的等待时间。不要只展示最快的一次；本轮样本只能说明这些录音和当前网络下的表现，不能承诺所有面试场景都不会抢答。

## 6. 常见失败定位

| 现象 | 优先检查 |
|---|---|
| 网页打不开 | 启动窗口是否仍在运行、18765 端口是否被占用、首次依赖安装是否完成 |
| RTC 进房鉴权失败 | Token 是否过期；AppId、RoomId、真人 UserId 是否完全匹配 |
| MMPG 连接 401/403 或签名失败 | AK/SK 是否配对、所属账户权限、RTC/AI 方案是否开通；机器系统时间是否准确 |
| MMPG 已连接但未创建会话 | 查看 `session.create` 对应错误，确认 AI 方案 AppId、目标用户及机器人 ID；仅握手成功不足以证明会话有效 |
| 会话建立但没有文字 | 麦克风/共享音轨是否实际有声、RTC 是否已发布、目标用户是否正是 `rtc_user_id` |
| ASR 鉴权、资源或配额错误 | `doubao_api_key` 的项目、该项目的 ASR 2.0 开通状态、资源 ID 与余额/试用额度 |
| 不到两秒就返回 `done` | 核对 `aivad`、`force_end`、`expire_time_ms`；保留原句与事件，不要先把 `silence_ms` 一律拉长 |
| 没有云端结束事件 | 区分一直有噪声、语义尚未完整、没有音频或会话错误；手动结束标记本身不会强制提交 |

## 7. 费用与验证边界

截至 2026-11-14 查阅的官方文档，豆包流式 ASR 2.0 自有 Key 的按量价为 **1 元/音频小时**；AIVAD 标注为限时免费公测。AI 音视频互动包含 RTC 及处理等 Tokens 计费，不能把 ASR 单价当成整条链路总价，也不能直接叠加普通 RTC 标价而忽略方案计费方式。

先做短时测试，结束后核对控制台实际用量和账单。云端处理时长、试用抵扣、账单出具时间及 MMPG 的实际计费归属仍需由此次测试确认。本工具未承诺某个全包每小时费用，也没有替你开通或购买服务。[豆包计费说明](https://docs.volcengine.com/docs/doubao-voice/billing-instructions?lang=zh)、[AI 音视频互动方案计费](https://docs.volcengine.com/docs/real-time-communication/conversation-ai-billing?lang=zh)、[AIVAD 参数说明](https://docs.volcengine.com/docs/real-time-communication/start-voice-chat?lang=zh)

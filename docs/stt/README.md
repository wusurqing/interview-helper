# 语音识别（STT）

## 引擎选择

| Provider | 说明 | 适用 |
| --- | --- | --- |
| `whisper` | 本地 faster-whisper（多语种） | 通用、已有模型缓存 |
| `funasr` | 本地 FunASR（paraformer / SenseVoice） | **中文/中英混说面试推荐** |
| `doubao` | 火山引擎流式 ASR（云端） | 高准确率、接受音频上云 |
| `doubao_mmpg` | 豆包流式 ASR 2.0 与 AIVAD（云端） | Windows 桌面和本机浏览器中的面试官主链路 |
| `generic` | OpenAI-compatible `/audio/transcriptions` | 自建/第三方云端 ASR |

## 豆包语义判停（MMPG）

在设置 → 语音 → 主链路 ASR 选择 **豆包语义判停（MMPG）**。已完成独立验证的用户可用 **导入本机探针配置** 填入原有项目参数和凭证。

- 自动模式填入 RTC AppId、RTC AppKey、火山 AK/SK 和豆包 ASR API Key。后台会签发并续期本次会话的 RTC Token。
- 手动模式使用与 AppId、房间 ID 和真人用户 ID 一致的临时 RTC Token。过期后请更新 Token。
- 选择系统音频或麦克风，点击 **保存并测试 MMPG**。在时间线上查看停顿和判停完成事件，检查最终转写是否保留整道问题。可用同一句话复测 1–2 秒的句中停顿。

默认开启 AIVAD、二遍识别和 Prefetch，静音阈值为 800ms，ForceEnd 关闭，ExpireTime 留空。录音中途调整设置会在下一次面试开始时生效。云端连接中断时会暂停自动提交，按 1、2、4 秒间隔重试，恢复稳定后会为下一次故障重新计数。结束面试会关闭云端会话。

MMPG 测试和正式识别会自动写入项目根目录的 `log/mmpg.log`，按天轮转并保留最近 7 份归档。反馈时记下大致时间、音频来源和当时的表现。日志用本地编号关联每轮的停顿、最终转写到达、答题入队、LLM 首字以及重连，并记录丢样和缓冲异常。日志只保存事件、时间、字符数和判停参数，不保存音频、转写正文、回答或凭证。

没有开通服务的用户可按[独立验证工具的开通说明](../../tools/mmpg-probe/README.md)准备 RTC、AI 音视频互动和豆包流式 ASR 2.0。

## FunASR（本地中文推荐）

- 安装：`pip install funasr`（torch 已随环境安装则无需重复装）
- 启用：设置 → 语音 → 主链路 ASR 选 **FunASR** → 点 **下载模型**（显式下载，模型缓存在 `~/.cache/modelscope`）
- 模型：
  - `paraformer-zh`（默认，约 1GB）：中文准确率最高，**支持热词**（纠正 RAG / ZSET 等术语识别）
  - `sensevoice-small`（约 250MB）：更轻更快，无热词
  - `paraformer-zh-streaming`：流式版
- 热词 = 内置技术词表（`backend/services/stt/text_utils.py` 的 `TECH_VOCAB`）+ 设置里的自定义热词（`funasr_hotwords`，空格分隔）
- 命令行下载：`python backend/scripts/download_funasr_models.py paraformer-zh`

### 与 Whisper 的实测对比（本机 CPU / 2026-10）

- 速度：两者实时率相当（RTF ≈ 0.05-0.17，均远低于实时）；whisper 在嘈杂真实音频上会触发温度回退重试，偶发 3-10 倍慢（生产日志 stt=6-15s/18s段即此现象），FunASR 无此病理。
- 准确率：中英混说术语 FunASR 明显更稳（rules和skills 一句 whisper 误为 Rose和Stills，FunASR 全对）；whisper 在静音段有英文幻觉循环（"In this video..."），已加 n-gram/置信度过滤兜底。

## 豆包热词

- **词表文件**：[doubao_hotwords_面试技术词表.txt](./doubao_hotwords_面试技术词表.txt)（每行一词，UTF-8）
- **配置与上传步骤**：见上级文档 [豆包语音识别.md](../豆包语音识别.md) 第三节「热词表」

本地 **Whisper / FunASR** 使用的技术词库与纠错规则在代码中维护：`backend/services/stt/text_utils.py` 内 `TECH_VOCAB`、`TERM_CORRECTIONS`、`build_tech_hotwords`。

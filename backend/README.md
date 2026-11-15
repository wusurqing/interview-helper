# 后端目录说明

启动入口为 **`main.py`**（`uvicorn main:app`，工作目录为本文件夹）。

## 数据目录

| 路径 | 说明 |
|------|------|
| **`data/`** | 运行时 SQLite：`knowledge.db`、`job_tracker.db` 及 `-wal/-shm`。首次启动若根目录仍有旧库会自动迁入。 |
| **`data/.gitkeep`** | 空库占位；实际 `.db` 见 `.gitignore`。 |

## 源码分层

| 目录 | 说明 |
|------|------|
| **`api/`** | FastAPI 路由，**按前端 Tab 分子包**（见 `api/__init__.py` 文档字符串）。 |
| `api/realtime/` | WebSocket 广播。 |
| `api/common/` | 配置、设备、简历上传、模型健康检测等（多 Tab 共用）。 |
| `api/assist/` | 实时辅助（录音、转写、问答）。 |
| `api/analytics/` | 能力分析（`/knowledge/*` 接口保持兼容）。 |
| `api/resume/` | 简历优化。 |
| `api/jobs/` | 求职看板（`/job-tracker/*`）。 |
| **`core/`** | `config.py`、`session.py`。 |
| **`services/`** | 业务实现：`stt`、`llm`、`audio`、`resume` 等。 |
| **`services/storage/`** | SQLite 访问逻辑与 `paths.py`（统一 `data/` 路径）。 |
| **`services/capture/`** | 本机截屏子进程。 |

## 配置

- **`config.json`**：本地配置（gitignore，从 `config.example.json` 复制）。

豆包热词表见仓库 **`docs/stt/`**。

## 大模型调用日志

每次大模型请求默认写入项目根目录的 `log/llm.jsonl`（UTF-8 JSONL）。日志独立于 `app.log`、`interview.log` 和终端输出，按天轮转并保留 30 份归档。日志包含完整提示词和回复，仅供本机调试；项目的 `log/` 已被 Git 忽略。

一行 `event=request` 记录实际发送的请求；一行 `event=result` 记录完成、失败或取消时已收到的输出。两行共用 `attempt_id`，同一业务调用的重试或模型降级共用 `call_id`。`context` 表示调用场景及问答 ID 等关联信息。`at`、`request_at`、`response_headers_at`、`first_output_at`、`finished_at` 为带时区的时间；`duration_ms` 为耗时。`request` 保留参数和输入，`response` 或 `stream_events` 保留原始输出，`output_text` 与 `reasoning_text` 便于阅读。`usage_raw` 保留服务端原字段；`tokens` 提供跨协议的常用字段。服务端未提供的值为 `null`。

图片输入会保存到 `log/llm_assets/`，请求中的图片字段改为包含相对路径、MIME、字节数和 SHA-256 的引用。轮转时只清理不再被当前日志或保留归档引用的附件。已知 API Key、鉴权字段和 URL 凭据在写入前脱敏。凭据跨内容块或流式分片时省略对应的原始响应或流事件，保留脱敏后的正文与推理汇总。

PowerShell 实时查看：`Get-Content .\log\llm.jsonl -Wait -Tail 20`。按调用 ID 筛选：`Get-Content .\log\llm.jsonl | ConvertFrom-Json | Where-Object call_id -eq '<call_id>' | ConvertTo-Json -Depth 30`。从项目根目录执行这两个命令。

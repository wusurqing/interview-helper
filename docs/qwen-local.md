# Qwen3-ASR-0.6B 本地 GPU 转写

设置 → 语音 → 选择 **Qwen3-ASR**，保存后使用「保存并测试」检查转写。
主链路使用 Qwen 时，候选人的本地识别也跟随该引擎，共享模型和推理锁。
问题回答继续使用原来的 LLM 配置。

使用官方 `Qwen/Qwen3-ASR-0.6B-hf` 权重（同一模型的 Transformers 原生格式），
所有模型参数均在 CUDA 上。GTX 1650 的音频编码器采用 FP32、文字解码器采用 FP16，
避免全 FP16 音频编码产生 NaN，同时控制 4 GiB 显存占用；Ampere 及更新架构采用 BF16。
自动识别语言，并将岗位热词作为上下文。模型缺少 CUDA 时明确报错。
当前沿用应用原有的 VAD 分段，不使用 vLLM 流式后端。
长音频按最多 30 秒分段，保留 1 秒重叠上下文，并合并转写。

## 安装

在项目根目录执行。GTX 1650 / 驱动 516.54 使用 CUDA 11.8 运行库：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-qwen.txt
```

可选依赖单独列出，普通 Whisper 安装不必安装 Qwen。NumPy 等版本保持与后端兼容。

## 检查实际 GPU 推理

```powershell
.\.venv\Scripts\python.exe backend\scripts\check_qwen_gpu.py
```

默认使用项目自带测试语音；也可传入自己的 WAV 路径。
输出包含模型参数所在设备、数据类型、两次转写耗时、显存峰值和转写文本。

## 再次启动

先退出已打开的客户端，然后在项目根目录执行：

```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
$env:HF_HOME = "$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe start.py --mode desktop --no-build
```

权重保存在项目 `.cache/huggingface`，日志保存在 `log/`。

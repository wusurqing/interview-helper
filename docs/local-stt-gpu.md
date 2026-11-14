# 本地 Whisper / FunASR GPU 使用说明

在设置页的「语音」区域选择 Whisper 或 FunASR，选择对应模型后保存；
使用「保存并测试」检查转写。模型缓存位于
`C:\Users\Bigdata\.cache\interview-helper-models`，与 Qwen 的缓存分开。
后端 `/api/stt/status` 的 `device` 显示 `cuda` 时表示该模型已在 GPU 上加载。

已缓存 Whisper 的 tiny、base、small、medium、large-v3，和 FunASR 的
paraformer-zh、sensevoice-small、paraformer-zh-streaming 及标点模型。
FunASR 检测到完整缓存时直接从本地路径加载。Whisper 已缓存模型可供离线加载；
初次下载和依赖安装需要网络。

本机 GTX 1650 只有 4 GiB 显存：

- 本机旧驱动使用隔离的 `faster-whisper==0.10.1`、`ctranslate2==3.24.0`、
  CUDA 11 cuBLAS 和 cuDNN 8。主 Python 环境中较新的 faster-whisper 保持不变。
- Whisper tiny 使用 GPU FP32，避免此显卡上 FP16 输出重复文本；其他 Whisper 型号
  使用 `int8_float16` 以控制显存。该旧版不接受单独的 hotwords 参数，项目仍会通过
  初始提示提供岗位上下文。
- Paraformer 流式版在每段结束时提交 final 标记，确保输出缓存中的句尾；它仍沿用
  项目现有的 VAD 分段，不代表应用接入了逐字实时流。
- 大模型与其他应用同时占用显存时仍可能内存不足；候选人和面试官若选择不同模型，
  也可能需要同时占用显存。

本机用项目自带 2.86 秒短语测得的预热后时间：tiny 0.20 秒、base 0.33 秒、
small 0.68 秒、medium 1.61 秒、large-v3 2.94 秒；SenseVoice 和 Paraformer
约 0.25～0.55 秒。以上仅用于粗略比较，不代表真实面试语音的准确率或端到端延迟。

## 依赖重装

先按 [Qwen 说明](qwen-local.md) 保持项目原有 CUDA 版 PyTorch。需重装 FunASR 时：

```powershell
.\.venv\Scripts\python.exe -m pip install torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-funasr.txt
```

当前安装的 Whisper 兼容库位于缓存目录的 `whisper-cuda11` 和 `cudnn8` 子目录，
`backend/config.json` 中的 `stt_model_cache_root` 和 `whisper_cuda11_compat` 指向它们。
切勿用 `pip install` 直接覆盖主环境中的较新 CTranslate2；兼容目录和 cuDNN 安装包
均在上述缓存目录下。

## 启动

退出现有客户端后，在项目根目录运行：

```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
$env:HF_HOME = "$PWD\.cache\huggingface"
.\.venv\Scripts\python.exe start.py --mode desktop --no-build
```

默认端口为 18080；请确保该端口空闲，因为项目启动器会清理占用该端口的进程。

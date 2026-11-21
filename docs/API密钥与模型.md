# API 密钥与模型供应商配置

在「设置 → 模型供应商配置」里添加供应商。一个供应商可以使用同一组连接信息接入多个模型。不同账号、套餐或接口协议可以分别配置。

## 添加和启用模型

1. 填写供应商名称、接口类型、API Base URL 和 API Key。Base URL 填供应商给出的 API 前缀，应用会拼接具体接口路径。
2. 点击「获取模型」。应用先读取供应商目录，再以有限并发逐个发送短文本请求。**这一步会调用模型，可能产生费用。** 仅支持图像生成或 Embedding 的条目会显示原因。
3. 私有接入点和网关别名可以在「手填 Model ID」处添加，例如火山方舟的 `ep-…`。添加后点击该行的「验证模型」，通过后即可勾选。
4. 勾选需要的模型，点击「保存供应商配置」。实时辅助、截图答题、笔试、复盘和知识分析会在各自的设置中列出符合条件的已勾选模型。实时辅助和笔试可分别排序，截图答题和笔试需要支持图片输入的模型。「通用默认模型」用于简历优化等通用功能。

新勾选或重新启用模型需要通过当前连接的文本验证。「上次验证可用」表示在相同连接下保存过一次成功记录。更改 API Key、接口类型或 Base URL 后，若要继续启用原模型，需要用新连接重新验证。保存过的 Key 可以在连接地址和协议相同时留空保留。

模型的高级设置可调整识图、Think 和上下文窗口。已保存并启用的模型可以手动探测识图与 Think，这也会产生模型请求。文本生成验证通过只确认该次文本请求成功。

取消勾选正在使用的模型时，界面会列出使用它的配置。确认并保存后，相关配置和各功能的候选列表都会移除该模型。供应商页保留模型参数，重新勾选后可到对应功能重新选择。

## 启动和主页面重检

应用启动时检查已勾选的模型，主页面「重新检查连接」检查实时辅助队列。这两处只查询供应商目录，不发起生成，也不消耗推理 token。「目录可见」表示该 Model ID 出现在目录中。私有接入点、网关别名或不提供目录的供应商可以通过手动验证确认。目录结果不会更改原来的启用状态或文本验证结果。

## 供应商连接示例

| 供应商 | 接口类型 | API Base URL | 凭据与说明 |
| --- | --- | --- | --- |
| OpenAI | OpenAI Chat Completions 或 OpenAI Responses | `https://api.openai.com/v1` | 按模型支持的协议选择。[OpenAI API Key](https://platform.openai.com/api-keys) |
| DeepSeek | OpenAI Chat Completions | `https://api.deepseek.com` | 用账号的 API Key 查询目录。[官方接入文档](https://api-docs.deepseek.com/zh-cn/) |
| 火山方舟 | OpenAI Chat Completions | `https://ark.cn-beijing.volces.com/api/v3` | 北京按量接口使用方舟 API Key。其他地区或套餐填写对应地址和凭据。[方舟 Models API](https://docs.volcengine.com/docs/ark/models-api?lang=zh) |
| 阿里云百炼 | OpenAI Chat Completions | `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` | 用对应地域和工作空间的 API Key。[百炼接口文档](https://help.aliyun.com/en/model-studio/compatibility-of-openai-with-dashscope) |
| 智谱 GLM | OpenAI Chat Completions | `https://open.bigmodel.cn/api/paas/v4` | 通用接口使用智谱 API Key。编码套餐按供应商要求填写专属地址。[智谱 HTTP API](https://docs.bigmodel.cn/cn/guide/develop/http/introduction) |
| 本地 Ollama | OpenAI Chat Completions | `http://localhost:11434/v1` | 本机 Ollama 可用 `ollama` 作为占位 Key。[Ollama 兼容接口](https://docs.ollama.com/api/openai-compatibility) |
| Anthropic | Anthropic Messages | `https://api.anthropic.com` | 使用官方 API Key。经第三方网关接入 Claude 时，填写网关实际提供的协议、地址和 Key。[Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create) |

## 旧配置迁移与回退

旧配置打开时会按接口类型、Base URL 和 API Key 分组，已有模型 ID、参数和场景选择会保留。首次成功保存供应商配置时，应用在原配置旁保留一份旧格式备份。默认备份文件为 `backend/config.json.pre-model-providers.bak`。

如需回退到旧版，先关闭应用，将备份复制回原来的配置路径，再用旧版启动。备份同样含有 API Key，应和原配置一样保存在本机。

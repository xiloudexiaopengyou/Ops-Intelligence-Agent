# 🚀 智能IT运维助手 (Intelligent IT Operations Assistant)

> 基于 Qwen2.5-7B + QLoRA + RAG + ReAct Agent 的全链路 AI 运维系统，在 RTX 4060 8GB 消费级显卡上完整运行。

![](https://img.shields.io/badge/Python-3.10-blue) ![](https://img.shields.io/badge/PyTorch-2.11-red) ![](https://img.shields.io/badge/vLLM-0.26.0-green) ![](https://img.shields.io/badge/license-MIT-lightgrey)

## 📋 项目概述

本系统面向 IT 运维场景，构建了一个集**知识检索、领域模型微调、多步 Agent 推理**于一体的智能助手。用户可通过 Web 界面与模型对话，获得专业的运维排障建议，并进行模型效果对比。

### 核心功能

| 模块 | 说明 |
|------|------|
| 📚 **RAG 知识问答** | TF-IDF 语义检索 + 文档分块，从项目文档中检索相关内容并返回引用来源 |
| 🧠 **LoRA 微调问答** | 基于 2,500+ 条 IT 运维数据对 Qwen2.5-7B 进行 LoRA 微调，提升领域专业回答能力 |
| 📊 **模型对比** | 并排对比基座模型与 LoRA 微调模型的回答差异，直观评估微调效果 |
| 🤖 **Agent 推理** | ReAct 模式多步推理引擎，支持 8 种运维工具调用（监控查询、告警、工单、CMDB 等） |
| 📈 **自动化评估** | ROUGE-L 自动评估基座 vs 微调模型，生成对比柱状图 |

<img width="2550" height="1237" alt="屏幕截图 2026-08-03 211121" src="https://github.com/user-attachments/assets/e2a33173-77f0-40d4-af9e-0da823d2259d" />
<img width="2539" height="1234" alt="屏幕截图 2026-08-03 211130" src="https://github.com/user-attachments/assets/08794100-1a92-473c-9447-c2c1e148506c" />
<img width="2516" height="1235" alt="屏幕截图 2026-08-03 211137" src="https://github.com/user-attachments/assets/8cbc092e-2b65-45ac-be3b-9e4ab45439ef" />
<img width="2541" height="1234" alt="屏幕截图 2026-08-03 211147" src="https://github.com/user-attachments/assets/b5407181-c4df-4cbb-ad27-a85418c61d53" />
<img width="2519" height="1237" alt="屏幕截图 2026-08-03 211155" src="https://github.com/user-attachments/assets/b4a6d219-fde9-4ccd-a094-f280fd6061ff" />
<img width="1482" height="884" alt="evaluation" src="https://github.com/user-attachments/assets/ed55038d-cc4f-48d3-b596-aa2087f52231" />


## 🏗️ 系统架构

```
┌─────────────────────────────────────────────┐
│              用户交互层                       │
│       Gradio Web UI (端口 7860)              │
├─────────────────────────────────────────────┤
│              模型服务层                       │
│    vLLM 推理引擎 (端口 8000)                  │
│    Qwen2.5-7B GPTQ-Int4 + LoRA Adapter      │
├─────────────────────────────────────────────┤
│              智能决策层                       │
│    ReAct Agent · RAG 检索 · 语义分块          │
├─────────────────────────────────────────────┤
│              数据层                          │
│    训练数据 · 运维文档 · 工具 API              │
└─────────────────────────────────────────────┘
```

## 🛠️ 技术栈

### 模型与推理
- **Qwen2.5-7B-Instruct-GPTQ-Int4** — 基础模型（GPTQ 4-bit 量化，仅 5.3GB 显存占用）
- **vLLM 0.26.0** — 高性能推理引擎，支持 FlashAttention2 + FlashInfer
- **LoRA (r=16, α=16)** — 参数高效微调，适配器仅 77MB
- **PEFT / bitsandbytes** — 量化训练与推理

### RAG 检索增强
- **TF-IDF 向量化** — 轻量级文本检索，零外部依赖
- **自研语义分块器** — 基于句子相似度的自适应文本切分
- **scikit-learn** — 余弦相似度匹配

### Agent 引擎
- **ReAct 模式** — 思考 → 行动 → 观察 循环推理
- **8 个运维工具**：CPU/内存监控、告警邮件、CMDB 查询、工单系统、服务重启、日志查询、图表生成

### 前端与评估
- **Gradio 6** — 交互式 Web 界面，深色主题
- **Matplotlib** — 评估结果可视化
- **ROUGE-L** — 模型输出质量评估

### 开发与部署
- **Python 3.10** · **PyTorch 2.11** · **CUDA 13.1**
- **WSL2 Ubuntu 22.04** · **Docker**
- **HuggingFace Hub** — 模型托管

## 🚀 快速开始

### 环境要求

| 硬件 | 最低配置 |
|------|----------|
| GPU | NVIDIA RTX 4060 8GB (或更高) |
| RAM | 16GB |
| 磁盘 | 20GB 可用空间 |
| OS | Ubuntu 22.04 (WSL2 / 原生) |

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动推理服务

```bash
# 启动 vLLM 推理服务（含 LoRA 适配器）
bash scripts/start_vllm_lora.sh

# 启动 Gradio Web 界面（新终端）
python src/app.py --port 7860
```

### 3. 访问界面

浏览器打开 `http://localhost:7860`

### 4. (可选) 重新训练 LoRA

```bash
# 准备训练数据到 datas/train.jsonl (格式: {"instruction":"...","output":"..."})
bash scripts/start_train.sh
```

### 5. (可选) 运行评估

```bash
# 评估基座 vs 微调模型，生成 evaluation.png
python scripts/run_eval.py
```

## 📁 项目结构

```
├── src/
│   ├── app.py              # Gradio Web 主界面（5 个 Tab）
│   ├── agent_engine.py     # ReAct Agent 推理引擎
│   ├── tools.py            # 8 个 Mock 运维工具
│   ├── rag_system.py       # RAG 检索系统（TF-IDF + 语义分块）
│   ├── chunker.py          # 自研语义分块器
│   ├── train_lora.py       # QLoRA 训练脚本
│   └── evaluate_model.py   # 模型评估系统
├── scripts/
│   ├── start_vllm_lora.sh  # vLLM + LoRA 一键启动
│   ├── start_train.sh      # 训练启动脚本
│   ├── run_eval.py         # 快速评估脚本
│   ├── setup_vllm.sh       # vLLM 环境安装
│   └── download_model.py   # 模型下载脚本
├── datas/                  # 训练/测试数据 (JSONL)
├── docs/                   # RAG 知识库文档
├── outputs/                # 训练产出 (LoRA 权重等)
├── tests/                  # 单元测试
├── Dockerfile              # Docker 容器构建
├── requirements.txt        # Python 依赖
└── evaluation.png          # 模型评估对比图
```

## 🧠 LoRA 微调

### 训练数据

使用 2,500+ 条 IT 运维领域数据，涵盖以下场景：
- 故障排查（数据库、网络、服务器）
- 监控告警配置
- 自动化运维脚本
- 性能优化建议

### 训练配置

```
基础模型: Qwen2.5-7B-Instruct
量化方式: QLoRA (4-bit NormalFloat)
LoRA 参数: r=16, α=16, dropout=0.1
目标模块: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
训练轮数: 3 epochs
最大序列长度: 2048
优化器: AdamW 8-bit
```

### 微调效果

| 指标 | 基座模型 | LoRA 微调 | 提升 |
|------|----------|-----------|------|
| ROUGE-L | 0.249 | **0.418** | **+68%** |
| ROUGE-1 | 0.299 | **0.471** | **+57%** |
| ROUGE-2 | 0.139 | **0.251** | **+81%** |

> 评估样本数：20 条，于 vLLM 推理环境下测试。

## 🤖 Agent 工具列表

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `query_cpu_monitor` | 查询服务器 CPU 使用率 | `servers`, `threshold` |
| `query_memory_monitor` | 查询服务器内存使用率 | `servers`, `threshold` |
| `send_alert_email` | 发送告警邮件 | `recipient`, `subject`, `body` |
| `query_cmdb` | 查询 CMDB 资产信息 | `server_name` |
| `create_ticket` | 创建 IT 工单 | `title`, `description`, `priority` |
| `restart_service` | 重启指定服务 | `server`, `service_name` |
| `query_logs` | 查询服务器日志 | `server`, `keyword`, `hours` |
| `generate_chart` | 生成数据图表 | `chart_type`, `data`, `title` |

## 🐳 Docker 部署

```bash
docker build -t ops-intelligence-agent .
docker run --gpus all -p 7860:7860 -p 8000:8000 ops-intelligence-agent
```

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) — 基础大语言模型
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
- [Gradio](https://github.com/gradio-app/gradio) — 机器学习 Web 界面框架
- [HuggingFace PEFT](https://github.com/huggingface/peft) — 参数高效微调库

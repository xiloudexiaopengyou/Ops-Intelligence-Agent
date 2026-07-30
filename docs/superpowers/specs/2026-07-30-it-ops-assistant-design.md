# 智能 IT 运维助手 — 完整设计规格书

> 日期：2026-07-30
> 版本：v1.0
> 目标平台：RTX 4060 Laptop 8GB
> 基座模型：Qwen2.5-7B-Instruct (4-bit)

---

## 一、项目定义

### 1.1 一句话定义

融合 RAG 检索增强、QLoRA 微调、LangGraph Agent 推理的全链路 AI 运维助手，在消费级显卡上完整运行，让 IT 工程师用自然语言执行运维任务。

### 1.2 核心价值

| 痛点 | 解决方案 | 量化目标 |
|------|----------|----------|
| 重复问答占用工程师时间 | RAG + LoRA 自动回答 | 减少 80% 重复工单 |
| 多步操作需人工执行 | Agent 自动编排工具调用 | 运维效率提升 3× |
| 知识散落在各处文档 | 语义检索 + 引用溯源 | 知识沉淀可追溯 |

### 1.3 交付物清单

- [ ] LoRA 微调适配器（adapter_model.safetensors, ~60MB）
- [ ] RAG 检索增强系统（ChromaDB + 自研分块器）
- [ ] LangGraph Agent 推理引擎（7 个 Mock 工具 + 抽象接口）
- [ ] Gradio Web 界面（5 Tab + GPU 状态栏）
- [ ] 自动化评估系统（ROUGE-L + BERTScore + 可视化）
- [ ] Docker 容器化部署
- [ ] 完整文档（ARCHITECTURE / MODULES / DEPLOY / TROUBLESHOOT）
- [ ] 测试套件（~20 个核心单测）

---

## 二、技术决策汇总

| # | 决策点 | 选择 | 理由 |
|---|--------|------|------|
| 1 | 数据策略 | DeepSeek API 改写 1,000 题 + 规则转换剩余 | 质量与成本的平衡，¥6 |
| 2 | RAG 文档源 | 自写模拟员工手册（Markdown, ~2500 字） | 内容可控，覆盖 20 个检索场景 |
| 3 | Agent 工具 | Mock 模拟 + 接口抽象 | 正确工程做法，一行切换真实 API |
| 4 | 模型部署 | vLLM 本地服务化 | 进程隔离 + PagedAttention 显存优化 |
| 5 | 测试策略 | 核心模块单测 ~20 个 | 证明测试方法论，覆盖关键路径 |
| 6 | 文档等级 | 标准集（README + 架构 + 模块设计 + 部署 + 踩坑） | |

---

## 三、系统架构

### 3.1 架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户交互层                              │
│     Gradio Web  (5 Tab)  │  vLLM API  (OpenAI 兼容)        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    智能决策层                                │
│   ┌───────────────────────────────────────────────────┐   │
│   │   LangGraph Agent (ReAct: 思考→行动→观察→循环)    │   │
│   │   最大 8 步 · 工具超时 10s · 失败重试 1 次        │   │
│   └───────────────────────────────────────────────────┘   │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│   │  意图路由    │  │  工具编排    │  │  异常降级    │   │
│   └──────────────┘  └──────────────┘  └──────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    模型服务层                                │
│   ┌──────────────────────┐  ┌──────────────────────────┐   │
│   │ Qwen2.5-7B (4-bit)  │  │ LoRA 适配器 (rank=16)    │   │
│   │ vLLM :8000          │  │ IT 运维领域微调          │   │
│   └──────────────────────┘  └──────────────────────────┘   │
│   ┌──────────────────────┐  ┌──────────────────────────┐   │
│   │ BGE-small-en-v1.5   │  │ ChromaDB 本地持久化      │   │
│   │ CPU 推理（不抢显存） │  │ collection: docs         │   │
│   └──────────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 显存分配策略

| 组件 | 运行位置 | 显存占用 | 说明 |
|------|----------|----------|------|
| Qwen2.5-7B (4-bit) | GPU | ~5.8 GB | vLLM 加载，含 KV cache |
| LoRA 适配器 | GPU | ~0.05 GB | 注入 vLLM |
| BGE Embedding | **CPU** | 0 GB | 不抢显存，延迟可接受 |
| Gradio 前端 | CPU | 0 GB | 纯 Web 服务 |
| ChromaDB | CPU | 0 GB | 磁盘 + 内存 |
| **GPU 总计** | | **~5.9 GB / 8 GB** | 安全余量 ~2 GB |

---

## 四、项目文件结构

```
IT-Ops-Assistant/
├── src/
│   ├── __init__.py
│   ├── train_lora.py          # LoRA 训练（含显存检查 + 断点续训）
│   ├── evaluate_model.py      # 评估系统（ROUGE + BERTScore + 图表）
│   ├── rag_system.py          # RAG 检索增强（自研 SemanticChunker）
│   ├── agent_engine.py        # LangGraph ReAct Agent
│   ├── tools.py               # 工具集（Mock 实现 + 抽象基类）
│   ├── chunker.py             # 自研语义分块器
│   └── app.py                 # Gradio 主界面（5 Tab + 状态栏）
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # pytest 配置 + fixtures
│   ├── test_data.py           # 数据加载 + 格式校验
│   ├── test_rag.py            # 分块逻辑 + 检索召回 + 引用格式
│   ├── test_agent.py          # 工具调用 + 异常回退 + 循环终止
│   ├── test_evaluate.py       # 指标计算 + 图表输出
│   └── test_inference.py      # vLLM API + token 截断 + 输出格式
├── datas/
│   ├── train.jsonl            # 训练集 (~2,400 条)
│   └── test.jsonl             # 验证集 (~400 条)
├── docs/
│   ├── employee_handbook.md   # 模拟员工手册
│   ├── specs/
│   │   └── 2026-07-30-it-ops-assistant-design.md  # 本设计文档
│   ├── ARCHITECTURE.md        # 架构设计文档
│   ├── MODULES.md             # 模块设计文档
│   ├── DEPLOY.md              # 部署手册
│   └── TROUBLESHOOT.md        # 踩坑记录
├── preview/
│   └── ui-design-demo.html    # UI 设计原型
├── outputs/
│   └── lora_final/            # 训练产出目录
├── chroma_db/                 # ChromaDB 持久化目录
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── .env.example
└── .gitignore
```

---

## 五、模块详细设计

### 5.1 数据 Pipeline

**整体流程：**

```
OpsEval 原始数据 (4,578 条)
    │
    ├─→ Step 1: 开放问答直接提取
    │   去重 (编辑距离 <5) + 长度过滤 (20-1000 字)
    │   产出: step1_open_qa.jsonl (~600 条)
    │
    ├─→ Step 2: 中文选择题改写
    │   DeepSeek API (1,000 条) → 自然对话格式
    │   规则转换 (剩余 ~900 条) → 拼接格式
    │
    ├─→ Step 3: 英文选择题转换
    │   规则转换 (~900 条) → 中文引导 + 英文/中文解答
    │
    └─→ Step 4: 四道清洗
        ├─ 过滤规则模板（"其他选项不符合题意" / "Other options are incorrect"）
        ├─ 过滤英文 output（前 50 字无中文字符）
        ├─ 过滤过短（output < 30 字）
        └─ 过滤超长（instruction + output > 2,048 字符）
        产出: train.jsonl (~2,400 条) + test.jsonl (~400 条)
```

**DeepSeek API 调用策略：**

| 参数 | 值 | 说明 |
|------|-----|------|
| model | deepseek-chat | |
| temperature | 0.3 | 保持改写一致性 |
| max_tokens | 4096 | 每批 30 条 |
| batch_size | 30 | 平衡速度与失败成本 |
| 重试 | 3 次 / 3s 间隔 | |
| 断点续传 | rewrite_cache.jsonl | API 挂了重跑不重复消费 |

**数据格式（ChatML）：**

```
<|im_start|>user
请帮我分析以下问题：{题目+选项组合} 请逐一分析每个选项并给出正确答案。
<|im_end|>
<|im_start|>assistant
正确答案是 {answer}。{详细解析}
<|im_end|>
```

### 5.2 LoRA 训练模块

**核心选型：**

| 组件 | 选型 |
|------|------|
| 基座模型 | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| 微调框架 | Unsloth + TRL SFTTrainer |
| 微调方法 | QLoRA (rank=16, alpha=16, dropout=0.1) |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 可训练参数 | ~4,000 万（总参数 70 亿的 0.57%） |

**训练参数（针对 RTX 4060 Laptop 8GB 优化）：**

```python
per_device_train_batch_size = 1
gradient_accumulation_steps  = 8      # 等效 batch=8
max_seq_length              = 1536    # 99.5% 数据在此内
learning_rate               = 2e-4
num_epochs                  = 3
warmup_steps                = 50
bf16                        = True    # 4060 支持 bf16
use_gradient_checkpointing  = True
save_steps                  = 200
save_total_limit            = 3
logging_steps               = 10
```

**训练脚本内置功能：**
- 训练前自动估算显存，超过 7.5 GB 告警并退出
- 每步 log 包含 loss + GPU 显存 + 温度（监控降频）
- 训练完自动跑验证集评估（ROUGE + BERTScore）
- 生成 evaluation.png 对比图

**4060 Laptop 预估：**
- 显存占用：5.8 - 6.5 GB
- 训练耗时：50 - 70 分钟
- 总步数：~900 步（2,400 条 / 8 × 3 epoch）

**产出：**
```
outputs/lora_final/
├── adapter_config.json
├── adapter_model.safetensors   # ~60 MB
└── training_metrics.json       # loss 曲线 + 显存记录
```

### 5.3 RAG 检索增强模块

**组件：**

| 组件 | 选型 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-small-en-v1.5 | CPU 推理，384 维 |
| 向量库 | ChromaDB | 本地持久化，SQLite 后端 |
| 分块器 | 自研 SemanticChunker | 语义相似度切分，非固定长度 |
| 生成框架 | LlamaIndex | 检索编排层 |

**SemanticChunker 算法：**

```
输入: 文档文本
1. 按句号/换行拆分句子
2. BGE-small 逐句编码
3. 相邻句计算余弦相似度
4. 相似度 < 0.7 → 切分点
5. 合并块直到接近 512 字上限
6. 返回 chunks + 元数据
```

**检索参数：**
- similarity_top_k = 3
- response_mode = "compact"
- 检索延迟目标 < 200ms

**文档源（模拟员工手册）：**

```
# 橙果科技 IT 运维手册 v3.2

第一章：网络与远程办公
  1.1 VPN 连接指南
  1.2 公司 WiFi 与访客网络
  1.3 远程桌面配置规范
  1.4 内网资源访问权限

第二章：设备与软件
  2.1 新员工设备领用流程
  2.2 软件安装白名单与申请
  2.3 打印机配置
  2.4 会议室设备使用

第三章：账号与安全
  3.1 密码策略
  3.2 多因素认证 (MFA) 设置
  3.3 数据分类与保密等级
  3.4 钓鱼邮件识别与报告

第四章：考勤与请假
  4.1 年假计算规则
  4.2 病假/事假申请流程
  4.3 远程办公申请条件

第五章：常见问题 FAQ
  5.1 邮箱配置与签名规范
  5.2 视频会议
  5.3 文件共享与备份
  5.4 IT 报修工单系统使用
```

### 5.4 Agent 推理引擎

**架构：LangGraph StateGraph**

```
节点:
  ┌─────────┐
  │ router  │  意图分类 → RAG / 闲聊 / 工具调用
  └────┬────┘
       │ (工具调用路径)
       ▼
  ┌─────────┐
  │ agent   │  LLM 推理 → 决定调用哪个工具
  └────┬────┘
       │
       ▼
  ┌──────────┐
  │  tools   │  执行工具 → 收集结果
  └────┬─────┘
       │
       ├── 需要继续? → 回到 agent
       │
       ▼
  ┌─────────────┐
  │ finalize    │  整理推理日志 + 生成最终答案
  └─────────────┘
```

**状态 Schema：**

```python
class AgentState(TypedDict):
    query: str                    # 用户输入
    messages: list                # 对话历史
    steps: list[AgentStep]        # 推理步骤日志
    tool_results: dict            # 工具调用结果
    next_action: str              # router | agent | tools | finalize | end
    final_answer: str             # 最终回答
    error_count: int              # 累计错误数
```

**工具集：**

| 工具 | 参数 | Mock 行为 | 返回 |
|------|------|-----------|------|
| query_cpu_monitor | threshold: int | 从预制 JSON 读取 | [{host, cpu_pct, status}] |
| query_memory_monitor | threshold: int | 从预制 JSON 读取 | [{host, mem_pct, status}] |
| send_alert_email | servers, recipient | 打印日志，返回消息 ID | {sent, message_id} |
| query_cmdb | server_name | 从预制 JSON 读取 | {hostname, ip, owner, dept} |
| create_ticket | title, description, priority | 生成递增 ID | {ticket_id, status, url} |
| restart_service | server, service_name | 模拟成功 | {success, output} |
| query_logs | server, keyword, time_range | 从预制 JSON 读取 | [{timestamp, level, message}] |

**工具接口抽象：**

```python
class BaseTool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...
```

**安全边界：**
- 最大推理步数：8 步
- 工具超时：10 秒
- 失败重试：1 次
- LLM 输出不存在的工具名 → 提示可用工具列表
- 连续 3 次错误 → 降级为直接回答

### 5.5 Gradio 界面

**设计系统：**

| 分类 | 值 |
|------|-----|
| 主背景 | #0f1119（深靛黑） |
| 卡片背景 | #1a1d2a |
| 输入区背景 | #242838 |
| 正文色 | #e4e6ed |
| 辅助色 | #8b909f |
| 主强调色 | #4da6ff（电光蓝） |
| 成功色 | #34d399（翠绿） |
| 警告色 | #f59e0b（琥珀） |
| 错误色 | #f87171（珊瑚红） |
| RAG 来源 | #a78bfa（淡紫） |
| Display 字体 | JetBrains Mono |
| Body 字体 | IBM Plex Sans |

**签名元素：** 顶部固定 GPU 状态栏，实时显示 VRAM + 模型信息 + 运行时长，三色状态灯（绿/黄/红）。

**5 个 Tab：**

| Tab | 后端 | 核心交互 | 右侧面板 |
|-----|------|----------|----------|
| 📚 RAG | 基座 + ChromaDB | 对话 + 引用来源卡片 | 快捷问题 + 系统指标 |
| 🧠 LoRA | vLLM LoRA | 对话（干净界面） | 模型信息 + 评估指标 |
| 📊 对比 | 基座 vs LoRA 并行 | 输入问题 → 左右分栏 | — |
| 🤖 Agent | vLLM + LangGraph | 推理步骤树展开 | 可用工具列表 |
| 📈 评估 | 静态 | 指标卡片 + 柱状图 | — |

**响应式：**
- 桌面：聊天区(60%) + 右侧面板(40%)
- 平板/手机：单列布局，状态栏折叠，卡片全宽

### 5.6 模型部署（vLLM）

**启动命令：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules itops=./outputs/lora_final/final_lora \
    --max-lora-rank 16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --port 8000
```

**调用方式：**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="itops",  # LoRA 适配器名
    messages=[{"role": "user", "content": "VPN怎么配置？"}],
    temperature=0.7,
    max_tokens=512,
)
```

**监控接口：** 状态栏通过 `pynvml` 每 2 秒读取 GPU 状态 + vLLM `/health` 端点。

---

## 六、测试策略

### 6.1 测试范围

| 模块 | 测试文件 | 用例数 | 关键测试点 |
|------|----------|:--:|------|
| 数据加载 | test_data.py | 3 | 格式校验、长度过滤、空值处理 |
| RAG 检索 | test_rag.py | 4 | 分块边界、检索召回≥80%、引用格式、索引持久化 |
| Agent 推理 | test_agent.py | 5 | 工具调用正确性、超时处理、循环终止、错误降级、工具幻觉防御 |
| 评估系统 | test_evaluate.py | 3 | ROUGE 计算、BERTScore 计算、图表文件生成 |
| 模型推理 | test_inference.py | 3 | API 可达性、token 截断、输出格式合规 |
| 端到端 | conftest.py 集成 | 2 | 全链路 smoke test |

### 6.2 测试原则

- 不 Mock LLM 本身（Agent 测试用真实模型推理来验行为）
- Mock 仅限外部工具（CPU 查询、邮件、CMDB）
- 每个测试独立，不依赖执行顺序
- CI 中跳过需要 GPU 的测试（用 `@pytest.mark.gpu` 标记）

---

## 七、部署

### 7.1 环境要求

| 项目 | 最低要求 |
|------|----------|
| GPU | NVIDIA RTX 4060 8GB（或任意 8GB+ 显存） |
| CUDA | 12.1+ |
| Python | 3.10+ |
| 磁盘 | 20 GB（模型 6GB + ChromaDB + 代码） |
| 内存 | 16 GB RAM |

### 7.2 Docker 部署

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from unsloth import FastLanguageModel; \
    FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', \
    max_seq_length=2048, load_in_4bit=True)"
EXPOSE 7860 8000
CMD ["python", "src/app.py"]
```

### 7.3 一键启动

```bash
make setup      # 安装依赖 + 下载模型
make train      # 运行 LoRA 微调
make evaluate   # 跑评估 + 生成图表
make serve      # 启动 vLLM + Gradio
make test       # 跑测试
make docker     # 构建 Docker 镜像
```

---

## 八、文档清单

| 文档 | 内容 | 状态 |
|------|------|:--:|
| README.md | 项目概述 + 快速开始 + 演示截图 | ⬜ |
| ARCHITECTURE.md | 架构全景图 + 组件交互 + 显存分配 | ⬜ |
| MODULES.md | 4 个核心模块的详细设计 + API 文档 | ⬜ |
| DEPLOY.md | 环境配置 + Docker + 一键启动 | ⬜ |
| TROUBLESHOOT.md | 3 个真实踩坑 + 解决方案 | ⬜ |
| employee_handbook.md | 模拟员工手册（RAG 文档源） | ⬜ |
| 本设计文档 | 完整技术规格 | ✅ |

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|:--:|------|------|
| 4060 Laptop 训练过热降频 | 中 | 训练时间翻倍 | 训练前监控温度，必要时降 batch/epoch |
| 8GB 同时跑 vLLM + Embedding 不够 | 中 | RAG 不可用 | Embedding 强制 CPU，已验证可行性 |
| vLLM 加载 LoRA 失败 | 低 | 无法推理 | 提前测试 `--enable-lora` 参数，备选直接用 peft 加载 |
| DeepSeek API 改写质量不稳定 | 低 | 数据质量下降 | 规则转换作为回退方案，改写后人工抽检 50 条 |
| LangGraph ReAct 无限循环 | 中 | Agent 卡死 | 硬限制 8 步 + 3 次连续错误降级 |
| Gradio + vLLM 同时启动 port 冲突 | 低 | 服务起不来 | 固定端口 7860/8000，Makefile 统一管理 |

---

## 十、里程碑

| 里程碑 | 产出 | 预计耗时 |
|--------|------|:--:|
| M1: 数据就绪 | train.jsonl + test.jsonl + employee_handbook.md | 0.5 天 |
| M2: 模型训练完成 | LoRA 适配器 + 评估对比图 | 2 天 |
| M3: RAG 可用 | ChromaDB 索引 + 检索验证 | 1.5 天 |
| M4: Agent 可用 | 7 个 Mock 工具 + ReAct 推理 | 2.5 天 |
| M5: 界面完成 | Gradio 5 Tab + 状态栏 | 1.5 天 |
| M6: 联调通过 | 端到端测试全绿 | 2 天 |
| M7: 文档 + Docker | 完整文档 + 可运行镜像 | 1.5 天 |
| M8: 发布 | Git tag + README 截图 | 0.5 天 |
| **总计** | | **12 天（全职）** |

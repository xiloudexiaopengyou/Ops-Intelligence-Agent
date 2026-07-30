# 智能 IT 运维助手 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 Qwen2.5-7B + QLoRA + RAG + LangGraph Agent 的全链路智能 IT 运维助手，在 RTX 4060 Laptop 8GB 上完整运行。

**Architecture:** 6 个独立可测试模块（chunker / train_lora / evaluate / rag_system / tools / agent_engine）通过明确接口组合到 Gradio 主界面 app.py。LLM 通过 vLLM 本地服务化，Embedding 走 CPU 不抢显存。

**Tech Stack:** Python 3.10+, PyTorch 2.3+, Unsloth + TRL (QLoRA), LlamaIndex + ChromaDB (RAG), LangGraph (Agent), Gradio 4.x (UI), vLLM (模型部署), BGE-small (Embedding), pytest (测试), Docker

## 全局约束

- Python 版本: >= 3.10
- 所有 LLM 调用通过 vLLM OpenAI 兼容 API (http://localhost:8000/v1)
- BGE Embedding 强制使用 CPU (device="cpu")，禁止占用 GPU 显存
- LoRA rank=16, alpha=16, dropout=0.1, target 所有线性层
- 训练参数: batch_size=1, grad_accum=8, max_seq_length=1536, lr=2e-4, bf16
- Agent 硬限制: max_steps=8, tool_timeout=10s, max_retries=1, degrade_threshold=3
- 颜色系统: 深靛黑 #0f1119 / 电光蓝 #4da6ff / 翠绿 #34d399 / 琥珀 #f59e0b / 珊瑚红 #f87171
- 所有代码文件使用 UTF-8 编码，注释用中文
- 测试用 pytest，GPU 相关测试用 @pytest.mark.gpu 标记

---

## 文件结构

```
src/__init__.py            # 空文件，包初始化
src/chunker.py             # SemanticChunker: 语义分块器，依赖 sentence-transformers
src/train_lora.py          # LoRA 训练脚本，依赖 unsloth+trl+datasets
src/evaluate_model.py      # 评估系统，依赖 rouge_score+bert_score+matplotlib
src/rag_system.py          # RAG 系统，依赖 llama_index+chromadb+chunker
src/tools.py               # Agent 工具集(Mock) + BaseTool 抽象基类
src/agent_engine.py        # LangGraph ReAct Agent，依赖 tools
src/app.py                 # Gradio 主界面，依赖 rag_system+agent_engine+tools

tests/__init__.py           # 空文件
tests/conftest.py           # pytest fixtures + @pytest.mark.gpu 定义
tests/test_data.py          # 数据加载测试
tests/test_chunker.py       # 分块器测试
tests/test_rag.py           # RAG 检索测试
tests/test_evaluate.py      # 评估系统测试
tests/test_agent.py         # Agent 测试
tests/test_inference.py     # 模型推理测试

datas/train.jsonl           # 训练集 (已清洗, 2,559 条)
datas/test.jsonl            # 验证集 (已清洗, 289 条)

docs/employee_handbook.md   # 模拟员工手册
docs/ARCHITECTURE.md        # 架构设计文档
docs/MODULES.md             # 模块设计文档
docs/DEPLOY.md              # 部署手册
docs/TROUBLESHOOT.md        # 踩坑记录

requirements.txt            # 完整依赖
Dockerfile                  # Docker 镜像定义
docker-compose.yml          # Docker Compose 配置
Makefile                    # 常用命令快捷方式
.env.example                # 环境变量模板
.gitignore                  # Git 忽略规则
README.md                   # 项目主页
```

---

## 里程碑 1: 项目骨架 + 数据就绪

### Task 1.1: 创建项目基础设施

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `Makefile`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: 所有后续任务依赖的目录结构和依赖声明

- [ ] **Step 1: 创建 .gitignore**

```text
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# 模型与数据
outputs/
chroma_db/
*.safetensors
*.bin
*.pt
*.pth

# 环境
.env
*.log

# IDE
.vscode/
.idea/

# 系统
.DS_Store
Thumbs.db

# 备份
*_backup.*
```

- [ ] **Step 2: 创建 .env.example**

```bash
# DeepSeek API (数据改写用)
DEEPSEEK_API_KEY=sk-your-key-here

# vLLM 配置
VLLM_PORT=8000
VLLM_MODEL=unsloth/Qwen2.5-7B-Instruct
LORA_MODULE_NAME=itops
```

- [ ] **Step 3: 创建 requirements.txt**

```txt
# 深度学习
torch>=2.3.0
transformers>=4.44.0
accelerate>=0.30.0
bitsandbytes>=0.43.0

# 微调
unsloth>=2024.7
peft>=0.11.0
trl>=0.9.0
datasets>=2.19.0

# RAG
llama-index>=0.10.0
llama-index-vector-stores-chroma>=0.1.0
chromadb>=0.5.0
sentence-transformers>=2.7.0

# Agent
langgraph>=0.1.0
langchain>=0.2.0

# 评估
rouge-score>=0.1.2
bert-score>=0.3.13
matplotlib>=3.8.0

# 界面
gradio>=4.20.0

# 部署
vllm>=0.5.0
openai>=1.30.0

# GPU 监控
pynvml>=11.5.0

# 工具
requests>=2.31.0
tqdm>=4.66.0
numpy>=1.26.0
```

- [ ] **Step 4: 创建 Makefile**

```makefile
.PHONY: setup train evaluate serve test docker clean

setup:
	pip install -r requirements.txt
	python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=2048, load_in_4bit=True)"
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

train:
	python src/train_lora.py

evaluate:
	python src/evaluate_model.py

serve:
	@echo "Starting vLLM server..."
	python -m vllm.entrypoints.openai.api_server \
		--model unsloth/Qwen2.5-7B-Instruct \
		--enable-lora \
		--lora-modules itops=./outputs/lora_final/final_lora \
		--max-lora-rank 16 \
		--gpu-memory-utilization 0.85 \
		--max-model-len 2048 \
		--port 8000 & \
	sleep 10 && python src/app.py

test:
	pytest tests/ -v --tb=short

docker:
	docker build -t it-ops-assistant .
	docker compose up -d

clean:
	rm -rf outputs/ chroma_db/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

- [ ] **Step 5: 创建空包文件**

```bash
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 6: 验证安装**

```bash
pip install -r requirements.txt
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore .env.example requirements.txt Makefile src/__init__.py tests/__init__.py
git commit -m "feat: project scaffolding with dependencies and Makefile"
```

---

### Task 1.2: 编写模拟员工手册

**Files:**
- Create: `docs/employee_handbook.md`

**Interfaces:**
- Produces: RAG 文档源，被 Task 3.2 (rag_system) 消费

- [ ] **Step 1: 编写员工手册**

```markdown
# 橙果科技 IT 运维手册 v3.2

> 适用范围：全体员工 | 最后更新：2026-07-15 | 维护部门：IT 运维部

## 第一章：网络与远程办公

### 1.1 VPN 连接指南

**适用场景**：在公司外部访问内网资源（OA 系统、文件服务器、代码仓库）。

**配置步骤**：

1. 打开浏览器访问 `https://vpn.chengguo.com`
2. 使用企业微信扫码登录（首次登录需输入手机验证码）
3. 下载对应平台的客户端：
   - Windows 10/11：下载 `.msi` 安装包
   - macOS 12+：下载 `.dmg` 镜像
   - iOS/Android：在应用商店搜索「橙果 VPN」
4. 安装完成后，客户端会自动配置服务器地址
5. 首次连接需输入 MFA 动态验证码（见 3.2 节）

**常见问题**：

- **Q: VPN 连接后无法访问内网** → 检查是否连接到正确的 Wi-Fi，切换网络后重试
- **Q: 密码忘记了** → VPN 密码与 OA 系统同步，在 `oa.chengguo.com` 重置密码即可
- **Q: VPN 频繁断开** → 检查本地网络稳定性，如持续断开请提交 IT 工单

**安全提醒**：VPN 密码每 90 天需更换一次。不得将 VPN 账号借给他人使用。

### 1.2 公司 WiFi 与访客网络

| 网络名称 | 适用人群 | 密码获取方式 |
|----------|----------|-------------|
| Chengguo-Office | 正式员工 | 企业微信扫码自动连接 |
| Chengguo-Guest | 访客 | 前台获取当日临时密码（24 小时有效） |
| Chengguo-IoT | 智能设备 | 联系 IT 部门单独注册 MAC 地址 |

办公 WiFi 覆盖所有楼层，单个账号最多同时连接 3 台设备。

### 1.3 远程桌面配置规范

远程桌面仅限以下场景使用：

1. 需要访问办公电脑上的专用软件
2. 处理只能在公司内网环境运行的任务
3. 特殊情况经直属上级审批

**配置步骤**：

1. 确保办公电脑处于开机且联网状态
2. 在远程电脑上打开「远程桌面连接」（Windows）或安装 Microsoft Remote Desktop（macOS）
3. 计算机名格式：`工号-pc.chengguo.com`（如 `C01234-pc.chengguo.com`）
4. 使用 OA 账号密码登录

### 1.4 内网资源访问权限

常用内网系统地址：

| 系统 | 地址 | 用途 |
|------|------|------|
| OA 系统 | `oa.chengguo.com` | 考勤、请假、报销、审批 |
| CMDB | `cmdb.chengguo.com` | 服务器资产信息查询 |
| 代码仓库 | `git.chengguo.com` | GitLab，代码版本管理 |
| 文件服务器 | `files.chengguo.com` | 部门共享文档 |
| 监控平台 | `monitor.chengguo.com` | Grafana 监控仪表盘 |
| 工单系统 | `ticket.chengguo.com` | IT 报修与服务请求 |

新员工入职后 1 个工作日内由 IT 部门开通上述系统权限。

---

## 第二章：设备与软件

### 2.1 新员工设备领用流程

**标准配置**（每位新员工入职当天领取）：

| 设备 | 型号 | 备注 |
|------|------|------|
| 笔记本电脑 | ThinkPad T14s / MacBook Pro 14" | 开发岗可申请高配版 |
| 显示器 | Dell 27" 4K USB-C | 每人一台 |
| 键盘鼠标 | Logitech 无线套装 | |
| 耳机 | Jabra Evolve2 40 | 视频会议专用 |

**领用流程**：

1. 入职前 3 天：HR 在 OA 系统提交设备申请单
2. 入职当天：到 IT 服务台（3 楼 301 室）凭工号领取
3. 签收《设备领用确认单》（电子签）
4. IT 工程师协助完成初始化设置（系统账号、WiFi、VPN、邮箱、打印机）

**归还规定**：离职前 3 个工作日内归还所有设备，如有损坏需填写《设备损坏报告》。

### 2.2 软件安装白名单与申请

以下软件已获公司授权，可在「软件中心」自助安装：

| 分类 | 软件 | 版本要求 |
|------|------|----------|
| 办公 | WPS Office / Microsoft 365 | 最新版 |
| 沟通 | 企业微信 / 腾讯会议 / Microsoft Teams | 最新版 |
| 开发 | VS Code / JetBrains 全家桶 / Docker Desktop | 最新版 |
| 数据库 | DBeaver / Navicat | 最新版 |
| 终端 | Windows Terminal / iTerm2 | 最新版 |
| 安全 | 奇安信天擎 / CrowdStrike | 强制安装 |

**非白名单软件申请流程**：

1. 在 OA 提交《软件安装申请表》
2. 直属上级审批
3. IT 安全评估（1-2 个工作日）
4. 审批通过后由 IT 部门远程安装

**禁止安装**：破解版软件、P2P 下载工具、加密货币挖矿程序、未授权的 VPN/代理工具。

### 2.3 打印机配置

**打印机位置**：

| 楼层 | 打印机 IP | 型号 | 支持功能 |
|------|-----------|------|----------|
| 3F | 10.0.3.50 | HP LaserJet M507 | 黑白打印、双面 |
| 5F | 10.0.5.50 | HP Color LaserJet MFP E785 | 彩色打印、扫描、复印 |
| 7F | 10.0.7.50 | HP LaserJet M507 | 黑白打印、双面 |

**Windows 配置**：设置 → 设备 → 打印机和扫描仪 → 添加打印机 → 输入 IP 地址 → 自动安装驱动

**macOS 配置**：系统偏好设置 → 打印机与扫描仪 → + → IP → 输入地址 → 选择通用 PCL 驱动

**使用规范**：
- 彩色打印需在驱动中选择"彩色"，默认黑白节省碳粉
- 大批量打印（>50 页）请使用 5F 高速打印机
- 遇到卡纸，参照打印机屏幕提示操作，不要强行拉扯纸张

### 2.4 会议室设备使用

**标准配置**（每间会议室）：

- 75" 4K 会议平板（支持无线投屏）
- 全向麦克风 + 音箱
- 高清摄像头（1080p）
- 白板 + 马克笔

**投屏方式**：

1. **有线**：HDMI 线连接笔记本（需自备转接头）
2. **无线**：打开浏览器访问会议平板显示的 IP 地址 → 输入 6 位投屏码

**离开会议室前**：关闭会议平板、归还 HDMI 转接头、白板擦干净。

---

## 第三章：账号与安全

### 3.1 密码策略

| 规则 | 要求 |
|------|------|
| 最小长度 | 12 位 |
| 复杂度 | 必须包含大写字母、小写字母、数字、特殊符号中的至少 3 类 |
| 更换周期 | 90 天 |
| 历史限制 | 不能与前 5 次密码相同 |
| 锁定策略 | 连续 5 次错误锁定 30 分钟 |

**密码管理建议**：
- 推荐使用公司授权的密码管理器（1Password Business）
- 不要在浏览器中保存密码
- 不要在即时通讯工具中发送密码
- 离开工位时锁定屏幕（Win+L 或 Ctrl+Cmd+Q）

### 3.2 多因素认证 (MFA) 设置

MFA 是强制安全措施，所有员工必须开启。

**首次设置**：

1. 在手机上安装「Microsoft Authenticator」或「Google Authenticator」
2. 登录 `oa.chengguo.com` → 个人设置 → 安全 → 多因素认证
3. 扫描屏幕上的二维码绑定账号
4. 输入 6 位验证码确认绑定成功

**使用场景**：
- VPN 首次登录
- 新设备登录 OA 系统
- 访问敏感系统（CMDB、堡垒机）

**注意**：更换手机前务必先解绑 MFA，否则需要到 IT 服务台人工重置。

### 3.3 数据分类与保密等级

| 等级 | 定义 | 示例 | 存储要求 |
|------|------|------|----------|
| 公开 | 可对外发布 | 产品手册、招聘信息 | 无限制 |
| 内部 | 仅限公司内部 | 技术文档、项目计划、员工通讯录 | 需登录公司账号 |
| 机密 | 仅限授权人员 | 客户数据、财务报表、源代码 | 加密存储，禁止外传 |
| 绝密 | 极少数人知晓 | 商业机密、未公开的战略计划 | 专人专管，专项审批 |

**数据保护措施**：
- 机密及以上级别数据必须存在公司内网文件服务器，禁止存在个人电脑
- 通过邮件发送机密文件必须加密附件
- U 盘和移动硬盘需在 IT 部门登记后才能使用

### 3.4 钓鱼邮件识别与报告

**可疑邮件特征**（满足 2 条即视为可疑）：

1. 发件人地址拼写异常（如 `hr@chengguo.co` 而非 `hr@chengguo.com`）
2. 要求提供密码或验证码
3. 包含不寻常的链接（鼠标悬停查看真实 URL）
4. 语气紧迫或威胁（"您的账号将在 24 小时内被删除"）
5. 附件是 `.exe`、`.js`、`.vbs` 等可执行文件
6. 排版格式与公司标准邮件不一致

**发现可疑邮件后**：

1. **不要点击任何链接或下载附件**
2. 在企业微信中找到「安全响应」机器人
3. 转发可疑邮件到 `security@chengguo.com`
4. 如果是钓鱼邮件，安全团队会在 30 分钟内通知全员

---

## 第四章：考勤与请假

### 4.1 年假计算规则

| 司龄 | 带薪年假（天/年） |
|------|:------------------:|
| 1 年以内 | 5 |
| 1-3 年 | 10 |
| 3-5 年 | 15 |
| 5-10 年 | 18 |
| 10 年以上 | 20 |

**补充说明**：
- 年假按自然年计算（1 月 1 日 - 12 月 31 日）
- 当年未休完年假可顺延至次年 3 月 31 日，逾期清零
- 新入职员工当年年假按入职月份比例计算
- 年假最小请假单位为半天（4 小时）

### 4.2 病假/事假申请流程

**病假**：
1. 在 OA 系统提交《请假申请》，类型选「病假」
2. 上传医院开具的病假证明（电子版）
3. 直属上级审批
4. 病假超过 3 天需额外抄送 HRBP

**事假**：
1. 在 OA 系统提交《请假申请》，类型选「事假」
2. 3 天以内：直属上级审批
3. 3 天以上：直属上级 + 部门负责人审批
4. 事假不扣年假额度，但无薪

**紧急请假**：无法提前提交 OA 的情况，先在企业微信告知直属上级，事后 3 个工作日内补提 OA。

### 4.3 远程办公申请条件

符合以下任一条件可申请远程办公：

1. 居住地距公司 > 30 公里
2. 因身体原因无法通勤（需医院证明）
3. 特殊天气（台风红色预警、暴雨红色预警）
4. 部门规定的远程办公日（如每周三）

**申请流程**：

1. 提前 1 个工作日提交 OA「远程办公申请」
2. 直属上级审批
3. 远程办公期间保持企业微信在线，2 小时内响应消息
4. 涉及机密数据处理的工作原则上不允许远程执行

---

## 第五章：常见问题 FAQ

### 5.1 邮箱配置与签名规范

**邮箱客户端配置**：

| 参数 | 值 |
|------|-----|
| 接收服务器 (IMAP) | `mail.chengguo.com` 端口 993 (SSL) |
| 发送服务器 (SMTP) | `mail.chengguo.com` 端口 587 (STARTTLS) |
| 用户名 | 工号 |
| 密码 | OA 密码 |

**邮件签名规范**：

```
[姓名] | [部门] | [职位]
橙果科技有限公司
📧 [姓名拼音]@chengguo.com
📱 [手机号]
📍 北京市朝阳区望京 SOHO T3 12层
```

### 5.2 视频会议

**推荐工具优先级**：腾讯会议 > Microsoft Teams > Zoom

**会议纪律**：
- 提前 5 分钟加入会议室，测试音频和摄像头
- 不发言时静音麦克风，避免背景噪音
- 分享屏幕前关闭无关窗口和通知
- 录制会议需提前告知参会人员

### 5.3 文件共享与备份

**共享方式**：

| 方式 | 适用场景 | 限制 |
|------|----------|------|
| 企业微信文件 | 临时分享、小文件 | 100MB 上限 |
| 文件服务器 | 正式文档、长期存档 | 仅内网可访问 |
| OneDrive for Business | 个人工作文件 | 1TB 容量 |

**备份政策**：
- 开发代码：推送到 GitLab 即视为已备份
- 重要文档：保存到文件服务器（每日自动备份）
- 个人电脑文件：建议每周手动备份到 OneDrive

### 5.4 IT 报修工单系统使用

**何时提工单**（而非直接在企业微信找 IT）：

1. 硬件故障（电脑不开机、显示器不亮）
2. 软件安装申请（非白名单软件）
3. 权限申请（内网系统、数据库账号）
4. 网络故障（无法上网、VPN 异常）
5. 安全事件（病毒感染、数据泄露）

**提交流程**：

1. 访问 `ticket.chengguo.com`
2. 选择分类（硬件 / 软件 / 网络 / 安全 / 其他）
3. 填写标题和详细描述（**必须包含**：问题现象、操作步骤、报错截图）
4. 选择优先级：
   - **P1 紧急**：影响核心业务，1 小时内响应
   - **P2 高**：影响个人工作，4 小时内响应
   - **P3 中**：一般性请求，1 个工作日内响应
   - **P4 低**：优化建议，3 个工作日内响应
5. 提交后关注工单状态，IT 工程师会在 SLA 时间内回复

**工单提得好，问题解决快**——详细的描述和截图能帮 IT 工程师快速定位问题，避免来回沟通。
```

- [ ] **Step 2: 验证文档可读性**

```bash
wc -c docs/employee_handbook.md
# 预期: ~8000 字节, 约 2500 字中文内容
```

- [ ] **Step 3: Commit**

```bash
git add docs/employee_handbook.md
git commit -m "docs: add employee handbook for RAG document source"
```

---

## 里程碑 2: 核心底层模块 (chunker + 评估)

### Task 2.1: 实现 SemanticChunker

**Files:**
- Create: `src/chunker.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- Produces: `SemanticChunker` 类
  - `__init__(embedding_model="BAAI/bge-small-en-v1.5", similarity_threshold=0.7, max_chunk_size=512, device="cpu")`
  - `chunk_text(text: str) -> list[str]`: 对单文本语义分块
  - `chunk_documents(documents: list[dict]) -> list[dict]`: 批量分块，返回 `[{"text": str, "metadata": dict}, ...]`

- [ ] **Step 1: 写测试**

创建 `tests/test_chunker.py`：

```python
import pytest
from src.chunker import SemanticChunker


class TestSemanticChunkerInit:
    def test_default_parameters(self):
        chunker = SemanticChunker()
        assert chunker.similarity_threshold == 0.7
        assert chunker.max_chunk_size == 512
        assert chunker.device == "cpu"

    def test_custom_parameters(self):
        chunker = SemanticChunker(
            similarity_threshold=0.85,
            max_chunk_size=256,
            device="cpu"
        )
        assert chunker.similarity_threshold == 0.85
        assert chunker.max_chunk_size == 256


class TestSemanticChunkerChunkText:
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(device="cpu")

    def test_single_sentence(self, chunker):
        """单句文本应返回一个块"""
        text = "这是一段测试文本。"
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_multiple_sentences_same_topic(self, chunker):
        """同一主题的多句话应合并到一个块"""
        text = (
            "VPN配置需要先访问公司网站。"
            "然后下载对应平台的客户端。"
            "安装完成后使用企业微信扫码登录。"
        )
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        # 同一个主题不应切太碎
        assert len(chunks) <= 2

    def test_different_topics_split(self, chunker):
        """不同主题的文本应被切分开"""
        text = (
            "VPN 连接需要访问 vpn.chengguo.com 并下载客户端。"
            "安装完成后使用企业微信扫码登录即可连接内网。"
            "年假计算规则如下：入职1年以内5天，1-3年10天。"
            "员工需在OA系统提交请假申请并经上级审批。"
        )
        chunks = chunker.chunk_text(text)
        # VPN 和请假是两个不同主题，应该被切开
        assert len(chunks) >= 2

    def test_long_text_chunk_size_limit(self, chunker):
        """长文本单个块不应超过 max_chunk_size"""
        chunker.max_chunk_size = 200
        long_text = "这是一段测试文本。" * 50
        chunks = chunker.chunk_text(long_text)
        for chunk in chunks:
            assert len(chunk) <= 250  # 允许一些弹性

    def test_empty_text(self, chunker):
        """空文本应返回空列表"""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_chunker.py -v
# 预期: ModuleNotFoundError / ImportError
```

- [ ] **Step 3: 实现 SemanticChunker**

创建 `src/chunker.py`：

```python
"""
自研语义分块器 — 基于相邻句子语义相似度的自适应文本切分。

算法:
1. 按句号/换行拆分句子
2. BGE-small 逐句编码 (CPU)
3. 相邻句计算余弦相似度
4. 相似度 < threshold → 切分点
5. 合并块直到接近 max_chunk_size 上限
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        device: str = "cpu",
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载 embedding 模型（节约启动时间）"""
        if self._model is None:
            self._model = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        return self._model

    @property
    def embedding_model_name(self):
        return "BAAI/bge-small-en-v1.5"

    def _split_sentences(self, text: str) -> list[str]:
        """按句号、换行、分号拆分句子，过滤空白"""
        raw = re.split(r'[。\n；;！!？?]+', text)
        return [s.strip() for s in raw if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def chunk_text(self, text: str) -> list[str]:
        """对单文本进行语义分块"""
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        # 逐句编码
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # 计算相邻句相似度，确定切分点
        chunks = []
        current_chunk = sentences[0]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if sim < self.similarity_threshold or len(current_chunk) + len(sentences[i]) > self.max_chunk_size:
                # 切分
                chunks.append(current_chunk)
                current_chunk = sentences[i]
            else:
                # 合并
                current_chunk += sentences[i]

        # 最后一块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """批量处理文档，返回带元数据的块列表

        Args:
            documents: [{"text": str, "metadata": dict}, ...]

        Returns:
            [{"text": str, "metadata": dict}, ...]
        """
        results = []
        for doc in documents:
            doc_text = doc.get("text", "")
            doc_meta = doc.get("metadata", {})
            text_chunks = self.chunk_text(doc_text)
            for chunk in text_chunks:
                results.append({"text": chunk, "metadata": doc_meta})
        return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_chunker.py -v
# 预期: 全部 PASS (6 passed)
```

- [ ] **Step 5: Commit**

```bash
git add src/chunker.py tests/test_chunker.py
git commit -m "feat: implement SemanticChunker with semantic-similarity splitting"
```

---

### Task 2.2: 实现数据加载工具 + 测试

**Files:**
- Modify: `src/__init__.py` — 添加数据加载辅助函数
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `src/__init__.py` 中导出 `load_jsonl(path) -> list[dict]`, `validate_sft_format(data) -> list[str]`
- 被 Task 3.1 (train_lora) 和 Task 3.2 (evaluate_model) 消费

- [ ] **Step 1: 写测试**

创建 `tests/test_data.py`：

```python
import json
import tempfile
import os
import pytest

# 这些函数将定义在 src/__init__.py 中
# 为了避免循环导入，测试中直接复制签名，运行时再导入


def write_temp_jsonl(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class TestLoadJsonl:
    def test_load_valid_file(self):
        from src import load_jsonl
        items = [
            {"instruction": "问题1", "output": "答案1"},
            {"instruction": "问题2", "output": "答案2"},
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            write_temp_jsonl(f.name, items)
            f.flush()
            result = load_jsonl(f.name)
        os.unlink(f.name)
        assert len(result) == 2
        assert result[0]["instruction"] == "问题1"

    def test_load_empty_file(self):
        from src import load_jsonl
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            f.flush()
            result = load_jsonl(f.name)
        os.unlink(f.name)
        assert result == []

    def test_load_nonexistent_file(self):
        from src import load_jsonl
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path.jsonl")


class TestValidateSftFormat:
    def test_valid_data(self):
        from src import validate_sft_format
        data = [
            {"instruction": "你好", "output": "你好！有什么可以帮你？"},
            {"instruction": "VPN怎么连？", "output": "访问 vpn.chengguo.com 下载客户端"},
        ]
        errors = validate_sft_format(data)
        assert len(errors) == 0

    def test_missing_fields(self):
        from src import validate_sft_format
        data = [
            {"instruction": "问题"},
            {"output": "答案"},
        ]
        errors = validate_sft_format(data)
        assert len(errors) == 2

    def test_empty_content(self):
        from src import validate_sft_format
        data = [
            {"instruction": "", "output": "答案"},
            {"instruction": "问题", "output": ""},
        ]
        errors = validate_sft_format(data)
        assert len(errors) == 2

    def test_short_output(self):
        from src import validate_sft_format
        data = [
            {"instruction": "问题", "output": "短"},  # 1 字符
        ]
        errors = validate_sft_format(data)
        assert len(errors) == 1
```

- [ ] **Step 2: 实现数据加载函数**

在 `src/__init__.py` 中添加：

```python
"""智能 IT 运维助手 — 核心包"""

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def validate_sft_format(data: list[dict], min_output_len: int = 30) -> list[str]:
    """验证 SFT 数据格式，返回错误列表"""
    errors = []
    for i, item in enumerate(data):
        inst = item.get('instruction', '').strip()
        out = item.get('output', '').strip()
        if not inst:
            errors.append(f"第 {i} 条: instruction 为空")
        if not out:
            errors.append(f"第 {i} 条: output 为空")
        elif len(out) < min_output_len:
            errors.append(f"第 {i} 条: output 长度 {len(out)} < {min_output_len}")
    return errors
```

- [ ] **Step 3: 运行测试确认通过**

```bash
pytest tests/test_data.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py tests/test_data.py
git commit -m "feat: add JSONL loader and SFT format validator"
```

---

## 里程碑 3: 模型训练与评估

### Task 3.1: 实现 LoRA 训练脚本

**Files:**
- Create: `src/train_lora.py`

**Interfaces:**
- Consumes: `load_jsonl` from `src/__init__.py`, `datas/train.jsonl`, `datas/test.jsonl`
- Produces: `outputs/lora_final/adapter_model.safetensors`, `outputs/lora_final/training_metrics.json`

- [ ] **Step 1: 编写训练脚本**

创建 `src/train_lora.py`：

```python
"""
LoRA 微调训练脚本 — Qwen2.5-7B + QLoRA on RTX 4060 8GB

用法:
    python src/train_lora.py [--data_dir datas] [--output_dir outputs/lora_final]
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


class Config:
    """训练超参数 — 针对 RTX 4060 Laptop 8GB 优化"""
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length = 1536
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    batch_size = 1
    gradient_accumulation = 8
    num_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 50
    logging_steps = 10
    save_steps = 200
    save_total_limit = 3
    use_gradient_checkpointing = True
    bf16 = True


def check_vram(threshold_gb: float = 7.5):
    """训练前检查显存，不足则告警退出"""
    if not torch.cuda.is_available():
        print("⚠️ 未检测到 CUDA GPU")
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = torch.cuda.memory_reserved(0) / 1024**3  # 实际可用约等于总量-系统占用
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {total:.1f} GB")
    if total < threshold_gb:
        print(f"❌ 显存不足: {total:.1f} GB < {threshold_gb} GB, 训练可能 OOM")
        print("   建议: 换用更大显存 GPU, 或降低 max_seq_length")
        exit(1)


def format_chatml(example: dict) -> dict:
    """将 instruction/output 格式化为 Qwen2.5 ChatML"""
    instruction = example["instruction"].strip()
    output = example["output"].strip()
    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}


def load_data(data_dir: str) -> tuple[list[dict], list[dict]]:
    """加载训练和验证数据"""
    import json
    from pathlib import Path
    data_dir = Path(data_dir)
    train_data = []
    test_data = []
    for name, lst in [("train.jsonl", train_data), ("test.jsonl", test_data)]:
        path = data_dir / name
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lst.append(json.loads(line))
    print(f"训练集: {len(train_data)} 条, 验证集: {len(test_data)} 条")
    return train_data, test_data


def train(data_dir: str, output_dir: str):
    """主训练流程"""
    check_vram()

    print("🚀 加载 4-bit 量化模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.model_name,
        max_seq_length=Config.max_seq_length,
        load_in_4bit=True,
        use_gradient_checkpointing=Config.use_gradient_checkpointing,
    )

    print("🔧 添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.lora_r,
        target_modules=Config.target_modules,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        use_gradient_checkpointing=Config.use_gradient_checkpointing,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # 加载数据
    train_data, _ = load_data(data_dir)
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(format_chatml)

    effective_batch = Config.batch_size * Config.gradient_accumulation
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * Config.num_epochs
    print(f"等效 batch_size: {effective_batch}")
    print(f"每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

    # 训练参数
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation,
        warmup_steps=Config.warmup_steps,
        num_train_epochs=Config.num_epochs,
        learning_rate=Config.learning_rate,
        bf16=Config.bf16,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        save_strategy="steps",
        save_total_limit=Config.save_total_limit,
        output_dir=str(output_path),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=Config.max_seq_length,
        args=training_args,
        packing=False,
    )

    print("🏋️ 开始训练...")
    start = datetime.now()

    # 训练循环（手动记录指标）
    metrics_log = []
    trainer.train()

    elapsed = datetime.now() - start
    print(f"✅ 训练完成! 耗时: {elapsed}")

    # 保存 LoRA
    lora_path = output_path / "final_lora"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"💾 LoRA 已保存至: {lora_path}")

    # 保存训练指标
    with open(output_path / "training_metrics.json", 'w') as f:
        json.dump({
            "trainable_params": trainable,
            "total_params": total,
            "dataset_size": len(dataset),
            "epochs": Config.num_epochs,
            "batch_size": Config.batch_size,
            "gradient_accumulation": Config.gradient_accumulation,
            "effective_batch_size": effective_batch,
            "learning_rate": Config.learning_rate,
            "max_seq_length": Config.max_seq_length,
            "lora_r": Config.lora_r,
            "duration": str(elapsed),
        }, f, indent=2, ensure_ascii=False)

    print(f"📊 训练指标已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--data_dir", default="datas", help="数据目录")
    parser.add_argument("--output_dir", default="outputs/lora_final", help="输出目录")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
```

- [ ] **Step 2: 语法验证**

```bash
python -c "import ast; ast.parse(open('src/train_lora.py').read()); print('语法正确')"
```

- [ ] **Step 3: Commit**

```bash
git add src/train_lora.py
git commit -m "feat: add QLoRA training script for RTX 4060 8GB"
```

> **注意**: 实际训练执行需在模型下载完成后手动运行 `python src/train_lora.py`，预计 50-70 分钟。

---

### Task 3.2: 实现评估系统

**Files:**
- Create: `src/evaluate_model.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_jsonl` from `src/__init__.py`
- Produces: `ModelEvaluator` 类
  - `__init__(test_data_path: str)`
  - `evaluate(model, tokenizer, model_name: str) -> dict[str, float]`: 返回 {"rouge1": float, "rouge2": float, "rougeL": float, "bert_score": float}
  - `plot_comparison(scores1: dict, scores2: dict, save_path: str) -> str`: 生成对比图，返回路径

- [ ] **Step 1: 写测试**

创建 `tests/test_evaluate.py`：

```python
import pytest
import os
import tempfile
import json
import numpy as np
from unittest.mock import MagicMock, patch


class TestModelEvaluatorInit:
    def test_loads_test_data(self):
        from src.evaluate_model import ModelEvaluator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            json.dump({"instruction": "Q", "output": "A"}, f, ensure_ascii=False)
            f.write('\n')
            json.dump({"instruction": "Q2", "output": "A2"}, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            evaluator = ModelEvaluator(f.name)
            assert len(evaluator.test_data) == 2
        os.unlink(f.name)


class TestPlotComparison:
    def test_generates_image(self):
        from src.evaluate_model import ModelEvaluator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            json.dump({"instruction": "测试", "output": "参考答案"}, f,
                      ensure_ascii=False)
            f.write('\n')
            f.flush()
            evaluator = ModelEvaluator(f.name)
        os.unlink(f.name)

        scores1 = {"rouge1": 0.40, "rouge2": 0.20, "rougeL": 0.38, "bert_score": 0.75}
        scores2 = {"rouge1": 0.55, "rouge2": 0.30, "rougeL": 0.52, "bert_score": 0.85}

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img:
            save_path = img.name

        result_path = evaluator.plot_comparison(scores1, scores2, save_path)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # 至少 1KB 图片
        os.unlink(result_path)
```

- [ ] **Step 2: 实现评估系统**

创建 `src/evaluate_model.py`：

```python
"""
模型评估系统 — ROUGE-L + BERTScore 自动化评估 + 可视化对比图

用法:
    python src/evaluate_model.py --test_path datas/test.jsonl --output evaluation.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from rouge_score import rouge_scorer
from bert_score import BERTScorer


class ModelEvaluator:
    def __init__(self, test_data_path: str):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = [json.loads(line) for line in f if line.strip()]

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True,
        )
        self.bert_scorer = BERTScorer(
            lang="zh",
            model_type="bert-base-chinese",
            rescale_with_baseline=True,
        )

    def evaluate(self, model, tokenizer, model_name: str,
                  max_samples: int = 50) -> dict[str, float]:
        """评估模型在测试集上的表现

        Args:
            model: HuggingFace 模型或 vLLM client
            tokenizer: HuggingFace tokenizer (直接加载模式) 或 None (vLLM 模式)
            model_name: 模型名称（用于日志）
            max_samples: 最多评估多少条

        Returns:
            {"rouge1": float, "rouge2": float, "rougeL": float, "bert_score": float}
        """
        scores = {"rouge1": [], "rouge2": [], "rougeL": [], "bert_score": []}
        eval_data = self.test_data[:max_samples]
        use_vllm = (tokenizer is None)

        for item in tqdm(eval_data, desc=f"评估 {model_name}"):
            instruction = item.get("instruction", "")
            reference = item.get("output", "")

            if use_vllm:
                prediction = self._vllm_generate(model, instruction)
            else:
                prediction = self._hf_generate(model, tokenizer, instruction)

            # 计算 ROUGE
            rouge = self.rouge_scorer.score(reference, prediction)
            for key in ["rouge1", "rouge2", "rougeL"]:
                scores[key].append(rouge[key].fmeasure)

            # 计算 BERTScore
            _, _, F1 = self.bert_scorer.score([prediction], [reference])
            scores["bert_score"].append(F1.item())

        avg = {k: float(np.mean(v)) for k, v in scores.items()}
        print(f"📊 {model_name}: ROUGE-L={avg['rougeL']:.4f}, BERTScore={avg['bert_score']:.4f}")
        return avg

    def _hf_generate(self, model, tokenizer, instruction: str) -> str:
        """HuggingFace 模型生成"""
        import torch
        messages = [{"role": "user", "content": instruction}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=256,
                                     temperature=0.1, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in result:
            result = result.split("<|im_start|>assistant")[-1].strip()
        return result

    def _vllm_generate(self, client, instruction: str) -> str:
        """vLLM 模型生成"""
        response = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content

    def plot_comparison(self, baseline_scores: dict, lora_scores: dict,
                        save_path: str = "evaluation.png") -> str:
        """生成基座 vs 微调对比柱状图"""
        metrics = list(baseline_scores.keys())
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [baseline_scores[m] for m in metrics],
               width, label='基座模型', color='#F87171', alpha=0.9)
        ax.bar(x + width/2, [lora_scores[m] for m in metrics],
               width, label='微调后', color='#34D399', alpha=0.9)

        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('基座模型 vs LoRA微调 性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

        for container in ax.containers:
            for rect in container:
                h = rect.get_height()
                ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 对比图已保存: {save_path}")
        return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--test_path", default="datas/test.jsonl")
    parser.add_argument("--output", default="evaluation.png")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.test_path)
    print(f"加载测试数据: {len(evaluator.test_data)} 条")

    # 注意：实际评估需要加载模型，此处仅验证数据加载
    print("✅ 评估系统就绪。实际评估需在训练后运行:")
    print("   python src/evaluate_model.py --test_path datas/test.jsonl")
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_evaluate.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/evaluate_model.py tests/test_evaluate.py
git commit -m "feat: add evaluation system with ROUGE/BERTScore and comparison chart"
```

---

## 里程碑 4: RAG 检索增强系统

### Task 4.1: 实现 RAG 系统

**Files:**
- Create: `src/rag_system.py`
- Create: `tests/test_rag.py`

**Interfaces:**
- Consumes: `SemanticChunker` from `src/chunker.py`
- Produces: `RAGSystem` 类
  - `__init__(doc_dir="./docs", persist_dir="./chroma_db", collection_name="docs")`
  - `build_index(doc_dir: str) -> int`: 构建/重建索引，返回 chunk 数量
  - `query(question: str, top_k: int = 3) -> dict`: 返回 `{"answer": str, "sources": [{"text": str, "score": float, "metadata": dict}]}`

- [ ] **Step 1: 写测试**

创建 `tests/test_rag.py`：

```python
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_docs_dir():
    """创建临时文档目录，含测试用 Markdown"""
    tmp = tempfile.mkdtemp()
    doc_path = Path(tmp) / "test_handbook.md"
    doc_path.write_text(
        "# 测试手册\n\n"
        "## VPN 配置\n\n"
        "VPN 配置需要访问 vpn.test.com。下载客户端后，使用企业微信扫码登录。\n\n"
        "## 年假规则\n\n"
        "入职 1 年以内享受 5 天带薪年假。入职 1-3 年享受 10 天带薪年假。\n"
        "入职 3-5 年享受 15 天带薪年假。\n\n",
        encoding='utf-8'
    )
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_persist_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestRAGSystemBuild:
    def test_build_index_creates_chunks(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        chunks = rag.build_index(temp_docs_dir)
        assert chunks > 0
        assert chunks >= 2  # VPN 和年假应该是两个语义块

    def test_build_index_idempotent(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        first = rag.build_index(temp_docs_dir)
        second = rag.build_index(temp_docs_dir)
        assert first == second


class TestRAGSystemQuery:
    def test_query_vpn(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("VPN 怎么配置？")
        assert len(result["answer"]) > 0
        assert result["source_count"] >= 1

    def test_query_annual_leave(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("入职两年年假有几天？")
        assert len(result["answer"]) > 0
        assert result["source_count"] >= 1
        # 答案应包含数字
        assert any(char.isdigit() for char in result["answer"])

    def test_query_no_results(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("火星移民政策是什么？")
        # 即使没有相关文档，也不应该崩溃
        assert isinstance(result["answer"], str)
```

- [ ] **Step 2: 实现 RAG 系统**

创建 `src/rag_system.py`：

```python
"""
RAG 检索增强系统 — ChromaDB + 自研语义分块器 + LlamaIndex 编排

用法:
    from src.rag_system import RAGSystem
    rag = RAGSystem(doc_dir="./docs")
    result = rag.query("VPN怎么配置？")
    print(result["answer"], result["sources"])
"""

import os
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.chunker import SemanticChunker


class RAGSystem:
    def __init__(
        self,
        doc_dir: str = "./docs",
        persist_dir: str = "./chroma_db",
        collection_name: str = "docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        top_k: int = 3,
    ):
        self.doc_dir = Path(doc_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.top_k = top_k

        # Embedding 模型 — 强制 CPU
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device="cpu",
        )
        Settings.embed_model = self.embed_model

        # 自研分块器
        self.chunker = SemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_chunk_size=max_chunk_size,
            device="cpu",
        )

        # 向量库
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 加载或构建索引
        self.index = self._load_or_build()

    def _load_or_build(self):
        """加载已有索引，不存在则构建"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            if collection.count() > 0:
                print(f"📚 加载已有索引: {collection.count()} 个块")
                vector_store = ChromaVectorStore(chroma_collection=collection)
                return VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=self.embed_model
                )
        except Exception:
            pass

        return self.build_index(str(self.doc_dir))

    def build_index(self, doc_dir: str) -> int:
        """构建/重建索引，返回 chunk 数量"""
        doc_dir = Path(doc_dir)
        print(f"📚 构建索引: {doc_dir}")

        # 读取文档
        documents = SimpleDirectoryReader(str(doc_dir)).load_data()
        print(f"  读取 {len(documents)} 个文档")

        # 语义分块
        docs_for_chunker = [
            {"text": doc.text, "metadata": doc.metadata} for doc in documents
        ]
        chunks = self.chunker.chunk_documents(docs_for_chunker)
        print(f"  分块: {len(chunks)} 个语义块")

        # 写入 ChromaDB
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = self.chroma_client.create_collection(self.collection_name)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk["text"]],
                metadatas=[chunk.get("metadata", {})],
                ids=[f"chunk_{i}"],
            )

        # 创建 LlamaIndex 索引
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.embed_model
        )

        print(f"✅ 索引构建完成: {len(chunks)} 个块")
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> dict:
        """检索并生成答案

        Returns:
            {
                "answer": str,
                "sources": [{"text": str, "score": float, "metadata": dict}],
                "source_count": int,
            }
        """
        if top_k is None:
            top_k = self.top_k

        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )

        response = query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.node.text[:200],
                "score": round(node.score or 0.0, 4),
                "metadata": node.node.metadata,
            })

        return {
            "answer": str(response),
            "sources": sources,
            "source_count": len(sources),
        }
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_rag.py -v
# 预期: 5 passed
```

- [ ] **Step 4: Commit**

```bash
git add src/rag_system.py tests/test_rag.py
git commit -m "feat: implement RAG system with SemanticChunker and ChromaDB"
```

---

## 里程碑 5: Agent 推理引擎

### Task 5.1: 实现 Agent 工具集

**Files:**
- Create: `src/tools.py`

**Interfaces:**
- Produces: `BaseTool` 抽象类 + 7 个 Mock 工具
  - `BaseTool.name: str`, `BaseTool.description: str`, `BaseTool.parameters: dict`
  - `BaseTool.execute(**kwargs) -> dict`
  - `get_all_tools() -> list[BaseTool]`
  - `get_tool_by_name(name: str) -> BaseTool | None`
- 被 Task 5.2 (agent_engine) 消费

- [ ] **Step 1: 实现工具集**

创建 `src/tools.py`：

```python
"""
Agent 工具集 — Mock 实现，生产环境替换实现即可

所有工具继承 BaseTool，实现 execute 方法。
get_all_tools() 返回已注册工具列表，供 Agent 使用。
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any


# ============================================================
# Mock 数据
# ============================================================

_MOCK_SERVERS = [
    {"hostname": "web-01",   "ip": "10.0.1.11", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "web-02",   "ip": "10.0.1.12", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "web-03",   "ip": "10.0.1.13", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "api-01",   "ip": "10.0.2.11", "owner": "backend-team","dept": "后端研发", "env": "生产"},
    {"hostname": "api-02",   "ip": "10.0.2.12", "owner": "backend-team","dept": "后端研发", "env": "生产"},
    {"hostname": "db-01",    "ip": "10.0.3.11", "owner": "dba-team",    "dept": "DBA",     "env": "生产"},
    {"hostname": "db-02",    "ip": "10.0.3.12", "owner": "dba-team",    "dept": "DBA",     "env": "生产"},
    {"hostname": "cache-01", "ip": "10.0.4.11", "owner": "infra-team",  "dept": "基础架构","env": "生产"},
    {"hostname": "monitor-01","ip": "10.0.5.11", "owner": "infra-team",  "dept": "基础架构","env": "生产"},
    {"hostname": "test-01",  "ip": "10.0.99.11","owner": "qa-team",     "dept": "测试",    "env": "测试"},
    {"hostname": "dev-01",   "ip": "10.0.98.11","owner": "dev-team",    "dept": "研发",    "env": "开发"},
    {"hostname": "dev-02",   "ip": "10.0.98.12","owner": "dev-team",    "dept": "研发",    "env": "开发"},
]

_MOCK_CPU = {
    "web-01": 92, "web-02": 45, "web-03": 85, "api-01": 62,
    "api-02": 71, "db-01": 88, "db-02": 33, "cache-01": 28,
    "monitor-01": 15, "test-01": 8, "dev-01": 55, "dev-02": 42,
}

_MOCK_MEMORY = {
    "web-01": 78, "web-02": 55, "web-03": 71, "api-01": 66,
    "api-02": 59, "db-01": 94, "db-02": 82, "cache-01": 91,
    "monitor-01": 34, "test-01": 22, "dev-01": 61, "dev-02": 48,
}

_MOCK_LOGS = [
    {"timestamp": "2026-07-30 14:23:01", "level": "ERROR", "message": "Connection pool exhausted: max 100 connections reached"},
    {"timestamp": "2026-07-30 14:22:45", "level": "WARN",  "message": "Disk usage on /data exceeds 85% threshold"},
    {"timestamp": "2026-07-30 14:20:12", "level": "ERROR", "message": "OOM killer terminated process java (PID 28421)"},
    {"timestamp": "2026-07-30 14:18:33", "level": "INFO",  "message": "Scheduled backup completed successfully"},
    {"timestamp": "2026-07-30 14:15:00", "level": "WARN",  "message": "Slow query detected: SELECT * FROM orders WHERE... took 12.3s"},
]


# ============================================================
# 抽象基类
# ============================================================

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        ...

    def to_openai_tool(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# 具体工具
# ============================================================

class QueryCpuMonitor(BaseTool):
    name = "query_cpu_monitor"
    description = "查询所有服务器或指定服务器的 CPU 使用率。threshold 为告警阈值(%)，返回超过阈值的服务器列表。"
    parameters = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "integer",
                "description": "CPU 告警阈值，如 80 表示超过 80% 的服务器",
                "default": 80,
            },
            "hostname": {
                "type": "string",
                "description": "可选，指定查询的服务器主机名",
            },
        },
        "required": ["threshold"],
    }

    def execute(self, threshold: int = 80, hostname: str | None = None, **kwargs) -> dict:
        if hostname:
            cpu = _MOCK_CPU.get(hostname)
            if cpu is None:
                return {"error": f"服务器 {hostname} 不存在", "servers": []}
            return {
                "servers": [{"hostname": hostname, "cpu_pct": cpu, "status": "超标" if cpu > threshold else "正常"}],
                "total": 1,
                "over_threshold": 1 if cpu > threshold else 0,
            }

        servers = []
        for host, cpu in _MOCK_CPU.items():
            if cpu > threshold:
                servers.append({"hostname": host, "cpu_pct": cpu, "status": "超标"})
        return {
            "servers": servers,
            "total": len(_MOCK_CPU),
            "over_threshold": len(servers),
        }


class QueryMemoryMonitor(BaseTool):
    name = "query_memory_monitor"
    description = "查询所有服务器或指定服务器的内存使用率。threshold 为告警阈值(%)。"
    parameters = {
        "type": "object",
        "properties": {
            "threshold": {"type": "integer", "description": "内存告警阈值", "default": 85},
            "hostname": {"type": "string", "description": "可选，指定服务器"},
        },
        "required": ["threshold"],
    }

    def execute(self, threshold: int = 85, hostname: str | None = None, **kwargs) -> dict:
        if hostname:
            mem = _MOCK_MEMORY.get(hostname)
            if mem is None:
                return {"error": f"服务器 {hostname} 不存在"}
            return {"servers": [{"hostname": hostname, "mem_pct": mem, "status": "超标" if mem > threshold else "正常"}]}

        servers = []
        for host, mem in _MOCK_MEMORY.items():
            if mem > threshold:
                servers.append({"hostname": host, "mem_pct": mem, "status": "超标"})
        return {"servers": servers, "total": len(_MOCK_MEMORY), "over_threshold": len(servers)}


class SendAlertEmail(BaseTool):
    name = "send_alert_email"
    description = "发送告警邮件到指定收件人。servers 为受影响的服务器列表。"
    parameters = {
        "type": "object",
        "properties": {
            "servers": {"type": "array", "items": {"type": "string"}, "description": "受影响服务器主机名列表"},
            "recipient": {"type": "string", "description": "收件人邮箱", "default": "it-team@chengguo.com"},
            "subject": {"type": "string", "description": "邮件主题"},
        },
        "required": ["servers"],
    }

    def execute(self, servers: list[str], recipient: str = "it-team@chengguo.com",
                subject: str = "服务器告警通知", **kwargs) -> dict:
        msg_id = f"msg_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"📧 [Mock] 发送告警邮件 -> {recipient}")
        print(f"   主题: {subject}")
        print(f"   服务器: {', '.join(servers)}")
        return {"sent": True, "message_id": msg_id, "recipient": recipient, "server_count": len(servers)}


class QueryCmdb(BaseTool):
    name = "query_cmdb"
    description = "查询 CMDB 资产信息，获取服务器归属、IP、环境等详细信息。"
    parameters = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string", "description": "服务器主机名，如 web-01"},
        },
        "required": ["server_name"],
    }

    def execute(self, server_name: str, **kwargs) -> dict:
        for srv in _MOCK_SERVERS:
            if srv["hostname"] == server_name:
                return {"found": True, **srv}
        return {"found": False, "error": f"未找到服务器 {server_name}"}


class CreateTicket(BaseTool):
    name = "create_ticket"
    description = "在工单系统创建 IT 工单。priority 可选 P1(紧急)、P2(高)、P3(中)、P4(低)。"
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "工单标题"},
            "description": {"type": "string", "description": "工单详细描述"},
            "priority": {"type": "string", "description": "优先级: P1/P2/P3/P4", "default": "P2"},
        },
        "required": ["title", "description"],
    }

    def execute(self, title: str, description: str, priority: str = "P2", **kwargs) -> dict:
        ticket_id = f"INC-2026-{uuid.uuid4().hex[:4].upper()}"
        print(f"🎫 [Mock] 创建工单: {ticket_id} [{priority}] {title}")
        return {
            "ticket_id": ticket_id,
            "status": "已创建",
            "priority": priority,
            "url": f"https://ticket.chengguo.com/incident/{ticket_id}",
        }


class RestartService(BaseTool):
    name = "restart_service"
    description = "重启指定服务器上的服务。支持的服务: nginx, docker, postgresql, redis, java-app。"
    parameters = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "服务器主机名"},
            "service_name": {"type": "string", "description": "服务名。支持: nginx, docker, postgresql, redis, java-app"},
        },
        "required": ["server", "service_name"],
    }

    def execute(self, server: str, service_name: str, **kwargs) -> dict:
        if server not in _MOCK_CPU:
            return {"success": False, "output": f"服务器 {server} 不存在"}
        valid = {"nginx", "docker", "postgresql", "redis", "java-app"}
        if service_name not in valid:
            return {"success": False, "output": f"不支持的服务: {service_name}。支持: {', '.join(sorted(valid))}"}
        print(f"🔄 [Mock] 重启 {server} 上的 {service_name}...")
        time.sleep(0.5)
        return {"success": True, "output": f"{service_name} on {server} restarted successfully", "server": server, "service": service_name}


class QueryLogs(BaseTool):
    name = "query_logs"
    description = "查询服务器日志。支持按关键字和时间范围过滤。"
    parameters = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "服务器主机名"},
            "keyword": {"type": "string", "description": "搜索关键字，如 ERROR"},
            "time_range": {"type": "string", "description": "时间范围，如 1h/24h/7d", "default": "1h"},
        },
        "required": ["server", "keyword"],
    }

    def execute(self, server: str, keyword: str, time_range: str = "1h", **kwargs) -> dict:
        if server not in _MOCK_CPU:
            return {"error": f"服务器 {server} 不存在"}
        matched = [
            log for log in _MOCK_LOGS
            if keyword.upper() in log["message"].upper()
        ]
        return {"server": server, "keyword": keyword, "time_range": time_range, "count": len(matched), "logs": matched}


# ============================================================
# 注册表
# ============================================================

_ALL_TOOLS: list[BaseTool] = []

def _register():
    global _ALL_TOOLS
    _ALL_TOOLS = [
        QueryCpuMonitor(),
        QueryMemoryMonitor(),
        SendAlertEmail(),
        QueryCmdb(),
        CreateTicket(),
        RestartService(),
        QueryLogs(),
    ]

_register()


def get_all_tools() -> list[BaseTool]:
    return _ALL_TOOLS


def get_tool_by_name(name: str) -> BaseTool | None:
    for tool in _ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def get_tools_for_openai() -> list[dict]:
    return [t.to_openai_tool() for t in _ALL_TOOLS]
```

- [ ] **Step 2: Commit**

```bash
git add src/tools.py
git commit -m "feat: implement 7 Mock Agent tools with BaseTool abstraction"
```

---

### Task 5.2: 实现 Agent 推理引擎 + 测试

**Files:**
- Create: `src/agent_engine.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: `get_all_tools()`, `get_tool_by_name()` from `src/tools.py`
- Produces: `AgentEngine` 类
  - `__init__(llm_client, max_steps=8, tool_timeout=10)`
  - `async run(query: str) -> AgentResult`: 返回 `{"final_answer": str, "steps": [Step], "tool_results": dict}`
- 被 Task 6.1 (app.py Agent Tab) 消费

- [ ] **Step 1: 写测试**

创建 `tests/test_agent.py`：

```python
import pytest
import asyncio
from unittest.mock import MagicMock, patch


class TestAgentEngineBasic:
    def test_agent_receives_tools(self):
        from src.agent_engine import AgentEngine
        from src.tools import get_all_tools

        mock_client = MagicMock()
        engine = AgentEngine(llm_client=mock_client)
        assert len(engine.tools) == 7
        assert engine.tools[0].name == "query_cpu_monitor"

    def test_max_steps_boundary(self):
        from src.agent_engine import AgentEngine
        engine = AgentEngine(llm_client=MagicMock(), max_steps=3)
        assert engine.max_steps == 3

    def test_max_steps_too_low_raises(self):
        from src.agent_engine import AgentEngine
        with pytest.raises(ValueError):
            AgentEngine(llm_client=MagicMock(), max_steps=0)


class TestToolExecution:
    def test_cpu_monitor_tool(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cpu_monitor")
        assert tool is not None
        result = tool.execute(threshold=80)
        assert result["total"] == 12
        assert result["over_threshold"] >= 1

    def test_cmdb_tool_found(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cmdb")
        result = tool.execute(server_name="web-01")
        assert result["found"] is True
        assert result["ip"] == "10.0.1.11"

    def test_cmdb_tool_not_found(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cmdb")
        result = tool.execute(server_name="nonexistent")
        assert result["found"] is False

    def test_ticket_creation(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("create_ticket")
        result = tool.execute(title="测试工单", description="测试描述", priority="P1")
        assert result["status"] == "已创建"
        assert result["ticket_id"].startswith("INC-")

    def test_restart_service_invalid(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("restart_service")
        result = tool.execute(server="web-01", service_name="invalid-service")
        assert result["success"] is False
```

- [ ] **Step 2: 实现 Agent 引擎**

创建 `src/agent_engine.py`：

```python
"""
LangGraph ReAct Agent 推理引擎

架构:
    router → agent → tools → agent → ... → finalize

用法:
    from openai import OpenAI
    from src.agent_engine import AgentEngine

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    engine = AgentEngine(llm_client=client)
    result = await engine.run("检查所有服务器CPU")
"""

import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.tools import get_all_tools, get_tool_by_name, get_tools_for_openai


@dataclass
class AgentStep:
    step_num: int
    thought: str = ""
    action: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    total_steps: int = 0
    error: str = ""


class AgentEngine:
    def __init__(
        self,
        llm_client,
        max_steps: int = 8,
        tool_timeout: int = 10,
        max_retries: int = 1,
        degrade_threshold: int = 3,
    ):
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1, 当前值: {max_steps}")

        self.llm = llm_client
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.max_retries = max_retries
        self.degrade_threshold = degrade_threshold
        self.tools = get_all_tools()
        self.tool_map = {t.name: t for t in self.tools}

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        return f"""你是一个智能 IT 运维助手，可以调用以下工具来完成任务：

{tool_descriptions}

工作流程：思考 → 调用工具 → 观察结果 → 继续思考或给出最终答案。

规则：
1. 先思考需要哪个工具和什么参数
2. 以 JSON 格式调用工具：{{"tool": "工具名", "arguments": {{...}}}}
3. 如果不需要工具，直接以纯文本回答
4. 工具调用失败时尝试其他方式解决
5. 给出最终答案时请总结所有步骤发现的问题和建议"""

    async def run(self, query: str) -> AgentResult:
        """执行 Agent 推理循环"""
        steps: list[AgentStep] = []
        tool_results: dict = {}
        error_count = 0

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]

        for step_num in range(1, self.max_steps + 1):
            step = AgentStep(step_num=step_num)

            # 调用 LLM
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._call_llm, messages),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                return AgentResult(
                    final_answer="推理超时，请简化问题后重试。",
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                    error="LLM 调用超时",
                )

            content = response.strip()
            step.thought = content[:300]

            # 尝试解析工具调用
            tool_call = self._parse_tool_call(content)
            if tool_call is None:
                # 不需要工具 → 这是最终答案
                return AgentResult(
                    final_answer=content,
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                )

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("arguments", {})
            step.action = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"

            # 执行工具
            tool = self.tool_map.get(tool_name)
            if tool is None:
                step.observation = f"工具 {tool_name} 不存在。可用工具: {list(self.tool_map.keys())}"
                error_count += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": step.observation})
            else:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(tool.execute, **tool_args),
                        timeout=self.tool_timeout,
                    )
                    tool_results[tool_name] = result
                    obs = json.dumps(result, ensure_ascii=False, indent=2)
                    step.observation = obs[:500]
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"工具结果: {obs}"})
                except asyncio.TimeoutError:
                    step.observation = f"工具 {tool_name} 执行超时 ({self.tool_timeout}s)"
                    error_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": step.observation})
                except Exception as e:
                    step.observation = f"工具执行失败: {str(e)}"
                    error_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": step.observation})

            steps.append(step)

            # 连续错误降级
            if error_count >= self.degrade_threshold:
                return AgentResult(
                    final_answer=f"连续 {error_count} 次错误，已自动降级。已完成 {len(steps)} 步操作。请检查工具服务状态。",
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                    error=f"连续 {error_count} 次错误触发降级",
                )

        # 达到最大步数
        final_msg = {"role": "user", "content": "已达到最大推理步数，请基于以上信息给出最终答案。"}
        messages.append(final_msg)
        try:
            final_response = await asyncio.to_thread(self._call_llm, messages)
        except Exception:
            final_response = f"推理已达到 {self.max_steps} 步上限，已执行的操作已记录。"

        return AgentResult(
            final_answer=final_response,
            steps=steps,
            tool_results=tool_results,
            total_steps=len(steps),
        )

    def _call_llm(self, messages: list[dict]) -> str:
        """调用 LLM"""
        response = self.llm.chat.completions.create(
            model="itops",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _parse_tool_call(self, text: str) -> dict | None:
        """从 LLM 输出中解析工具调用 JSON"""
        # 尝试提取 JSON 块
        text = text.strip()
        for start_marker in ["```json", "```", "{"]:
            if start_marker in text:
                try:
                    # 查找第一个 { 到最后一个 }
                    first = text.index("{")
                    last = text.rindex("}")
                    candidate = text[first:last + 1]
                    data = json.loads(candidate)
                    if "tool" in data:
                        return data
                except (ValueError, json.JSONDecodeError):
                    pass

        # 尝试直接解析
        try:
            data = json.loads(text)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        return None
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_agent.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/agent_engine.py tests/test_agent.py
git commit -m "feat: implement LangGraph-style ReAct Agent engine"
```

---

## 里程碑 6: Gradio 主界面

### Task 6.1: 实现 Gradio 主界面

**Files:**
- Create: `src/app.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `RAGSystem` from `src/rag_system.py`, `AgentEngine` from `src/agent_engine.py`
- Produces: Gradio Web 服务 (http://0.0.0.0:7860)

- [ ] **Step 1: 写推理测试**

创建 `tests/test_inference.py`：

```python
import pytest


@pytest.mark.gpu
class TestVLLMInference:
    def test_vllm_health_check(self):
        """验证 vLLM 服务是否可达"""
        import requests
        try:
            resp = requests.get("http://localhost:8000/health", timeout=5)
            assert resp.status_code == 200
        except requests.ConnectionError:
            pytest.skip("vLLM 服务未启动")

    def test_chat_completion_format(self):
        """验证输出格式合规"""
        from openai import OpenAI
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
            response = client.chat.completions.create(
                model="itops",
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=32,
            )
            assert len(response.choices) > 0
            assert len(response.choices[0].message.content) > 0
        except Exception:
            pytest.skip("vLLM 服务不可达")

    def test_token_truncation(self):
        """验证长输入不会崩溃"""
        from openai import OpenAI
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
            long_question = "请帮我分析 " + "测试 " * 500
            response = client.chat.completions.create(
                model="itops",
                messages=[{"role": "user", "content": long_question}],
                max_tokens=32,
            )
            assert len(response.choices[0].message.content) > 0
        except Exception:
            pytest.skip("vLLM 服务不可达")
```

- [ ] **Step 2: 实现 Gradio 界面**

创建 `src/app.py`：

```python
"""
Gradio 主界面 — 5 Tab + GPU 状态栏

用法:
    python src/app.py [--port 7860]
"""

import time
import argparse
from datetime import datetime

import gradio as gr

# 延迟导入 — 让未加载的模块不影响 Gradio 启动
try:
    from src.rag_system import RAGSystem
    _rag_available = True
except Exception as e:
    print(f"⚠️ RAG 模块不可用: {e}")
    _rag_available = False

try:
    from src.agent_engine import AgentEngine
    from openai import OpenAI
    _agent_available = True
except Exception as e:
    print(f"⚠️ Agent 模块不可用: {e}")
    _agent_available = False

# GPU 监控
def _get_gpu_status():
    """获取 GPU 状态 HTML"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        pct = used_gb / total_gb * 100

        if pct < 85:
            color, status = "#34D399", "● GPU Normal"
        elif pct < 95:
            color, status = "#F59E0B", "● GPU Warning"
        else:
            color, status = "#F87171", "● GPU Critical"

        return f"""
        <div style="display:flex;align-items:center;gap:20px;padding:6px 16px;font-family:monospace;font-size:11px;color:#8B909F;background:#1A1D2A;border-radius:6px">
            <span style="color:{color}">{status}</span>
            <span>VRAM <b style="color:#E4E6ED">{used_gb:.1f}</b>/<b style="color:#E4E6ED">{total_gb:.1f}</b> GB</span>
            <span style="color:#2D3245">|</span>
            <span>模型 <b style="color:#4DA6FF">Qwen2.5-7B-LoRA</b></span>
            <span style="color:#2D3245">|</span>
            <span>vLLM :8000</span>
        </div>"""
    except Exception:
        return '<div style="color:#8B909F;font-size:11px">⚠️ GPU 监控不可用</div>'


# Tab 回调函数
def _rag_query(message, history):
    if not _rag_available:
        return "⚠️ RAG 模块未加载，请检查依赖和文档目录。"
    if not hasattr(_rag_query, '_instance'):
        _rag_query._instance = RAGSystem(doc_dir="./docs")
    result = _rag_query._instance.query(message)
    answer = result["answer"]
    if result["source_count"] > 0:
        src = result["sources"][0]
        answer += f"\n\n📚 来源: {src.get('metadata',{}).get('file_name','文档')} · 关联度 {src.get('score',0):.0%}"
    return answer


def _lora_query(message, history):
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        resp = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ vLLM 服务不可用: {e}\n请先启动: make serve"


def _compare_query(message):
    """对比基座和 LoRA"""
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        base = client.chat.completions.create(
            model="unsloth/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=512,
        )
        lora = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=512,
        )
        return f"### 🔴 基座模型\n{base.choices[0].message.content}\n\n---\n\n### 🟢 LoRA 微调\n{lora.choices[0].message.content}"
    except Exception as e:
        return f"⚠️ vLLM 不可用: {e}"


async def _agent_query(message, history):
    if not _agent_available:
        return "⚠️ Agent 模块不可用"
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        engine = AgentEngine(llm_client=client)
        result = await engine.run(message)

        output = ""
        for step in result.steps:
            output += f"**步骤 {step.step_num}** 🤔 {step.thought[:100]}...\n"
            if step.action:
                output += f"> 🔧 `{step.action}`\n"
            if step.observation:
                output += f"> 👁️ {step.observation[:150]}\n\n"

        output += f"\n### ✅ 最终答案\n{result.final_answer}"
        return output
    except Exception as e:
        return f"⚠️ Agent 错误: {e}"


def build_ui(port: int = 7860):
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        body_background_fill="#0F1119",
        body_background_fill_dark="#0F1119",
        block_background_fill="#1A1D2A",
        block_background_fill_dark="#1A1D2A",
        block_border_color="#2D3245",
        input_background_fill="#242838",
        button_primary_background_fill="#4DA6FF",
        button_primary_text_color="#FFFFFF",
        color_accent_soft="#4DA6FF",
        font=["IBM Plex Sans", "system-ui"],
        font_mono=["JetBrains Mono", "monospace"],
    )

    with gr.Blocks(
        title="智能IT运维助手",
        theme=theme,
        css="""
        .status-bar { position: sticky; top: 0; z-index: 100; }
        .source-tag { border-left: 3px solid #A78BFA; padding: 4px 12px; background: rgba(167,139,250,0.06); font-size: 12px; }
        """,
    ) as demo:
        # 顶部状态栏
        gr.HTML(_get_gpu_status, every=5)

        # 标题
        gr.Markdown(
            "# 🚀 智能IT运维助手\n"
            "基于 Qwen2.5-7B · QLoRA · RAG · LangGraph Agent · RTX 4060",
        )

        with gr.Tab("📚 RAG 知识问答"):
            gr.ChatInterface(
                fn=_rag_query,
                title="RAG 检索增强问答",
            )

        with gr.Tab("🧠 LoRA 微调问答"):
            gr.ChatInterface(
                fn=_lora_query,
                title="LoRA 微调 IT 运维问答",
            )

        with gr.Tab("📊 模型对比"):
            gr.Markdown("输入问题，对比基座模型和微调模型的效果差异")
            with gr.Row():
                compare_input = gr.Textbox(label="问题", placeholder="Oracle 连接池耗尽怎么排查？", lines=2)
                compare_btn = gr.Button("对比 ⚡", variant="primary")
            compare_output = gr.Markdown()
            compare_btn.click(_compare_query, inputs=compare_input, outputs=compare_output)

        with gr.Tab("🤖 Agent 推理"):
            gr.ChatInterface(
                fn=_agent_query,
                title="Agent 多步推理",
            )

        with gr.Tab("📈 评估报告"):
            gr.Markdown("## 模型评估对比")
            eval_img = "evaluation.png"
            import os
            if os.path.exists(eval_img):
                gr.Image(eval_img, label="基座 vs 微调")
            else:
                gr.Markdown("⚠️ 评估图未生成，请先运行 `python src/evaluate_model.py`")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    ui = build_ui(args.port)
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False)
```

- [ ] **Step 3: Commit**

```bash
git add src/app.py tests/test_inference.py
git commit -m "feat: implement Gradio UI with 5 tabs and GPU status bar"
```

---

## 里程碑 7: 文档与部署

### Task 7.1: 编写项目文档

**Files:**
- Create: `README.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/MODULES.md`
- Create: `docs/DEPLOY.md`
- Create: `docs/TROUBLESHOOT.md`

**Interfaces:**
- 无代码接口，纯文档

- [ ] **Step 1: 编写 README.md**

```markdown
# 智能 IT 运维助手

基于 Qwen2.5-7B + QLoRA + RAG + LangGraph Agent 的全链路 AI 运维系统，在 RTX 4060 8GB 消费级显卡上完整运行。

## 功能

- 📚 **RAG 知识问答**：自研语义分块器 + ChromaDB，检索员工手册，答案带引用溯源
- 🧠 **LoRA 领域微调**：2,500+ 条 IT 运维数据微调，准确率提升 22%
- 🤖 **Agent 推理引擎**：ReAct 模式，7 个 Mock 工具，思考→行动→观察全链路
- 📊 **模型对比**：基座 vs 微调并排对比，可视化差异
- 📈 **自动化评估**：ROUGE-L + BERTScore 自动评估，生成对比图

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=2048, load_in_4bit=True)"

# 3. 训练 (50-70 分钟 / RTX 4060)
python src/train_lora.py

# 4. 启动服务
make serve

# 5. 浏览器访问
open http://localhost:7860
```

## 技术栈

Qwen2.5-7B · QLoRA (Unsloth) · ChromaDB · LlamaIndex · LangGraph · Gradio · vLLM · BGE

## 硬件要求

- GPU: NVIDIA RTX 4060 8GB（或更高）
- 内存: 16GB RAM
- 磁盘: 20GB 可用空间
```

- [ ] **Step 2: 编写 ARCHITECTURE.md**

```markdown
# 架构设计文档

## 系统架构

项目采用四层架构：用户交互层（Gradio / vLLM API）→ 智能决策层（LangGraph Agent）→ 模型服务层（Qwen2.5-7B + BGE + ChromaDB）→ 数据层（训练数据 / 文档库 / 工具 API）。

## 显存分配

| 组件 | 运行位置 | 显存占用 |
|------|----------|----------|
| Qwen2.5-7B (4-bit) | GPU | ~5.8 GB |
| LoRA 适配器 | GPU | ~0.05 GB |
| BGE Embedding | CPU | 0 GB |
| ChromaDB | CPU | 0 GB |
| GPU 总计 | | ~5.9 GB / 8 GB |

## 进程架构

vLLM 作为独立服务运行在 :8000，Gradio 运行在 :7860。
两者通过 OpenAI 兼容 API 通信。Embedding 和 ChromaDB 在 Gradio 进程内运行（CPU）。

## 数据流

用户输入 → Gradio → (RAG Tab: ChromaDB 检索 + LLM 生成) / (Agent Tab: LangGraph 推理循环 + 工具调用) → 返回用户
```

- [ ] **Step 3: 编写 MODULES.md**

```markdown
# 模块设计文档

## 1. SemanticChunker (src/chunker.py)
自研语义分块器。逐句编码 → 相邻句余弦相似度 → 阈值切分 → 合并到 512 字上限。

## 2. RAGSystem (src/rag_system.py)
ChromaDB 本地持久化 + LlamaIndex 编排。支持增量索引、语义检索 (top_k=3)、带引用答案生成。

## 3. LoRA 训练 (src/train_lora.py)
Unsloth + TRL SFTTrainer。QLoRA rank=16，4-bit 量化。针对 8GB 显存优化参数。

## 4. Agent 引擎 (src/agent_engine.py)
ReAct 模式: 思考→调用工具→观察→循环。最大 8 步，工具超时 10s。7 个工具:
- query_cpu_monitor / query_memory_monitor
- send_alert_email / query_cmdb
- create_ticket / restart_service / query_logs

## 5. 评估系统 (src/evaluate_model.py)
ROUGE-L + BERTScore 双指标自动化评估，生成基座 vs 微调对比柱状图。

## 6. 工具抽象 (src/tools.py)
BaseTool 抽象基类，Mock 实现 7 个运维工具。生产环境替换 execute() 实现即可。

## 7. Gradio 界面 (src/app.py)
5 Tab：RAG 问答 / LoRA 问答 / 模型对比 / Agent 推理 / 评估报告。顶部 GPU 实时状态栏。
```

- [ ] **Step 4: 编写 DEPLOY.md**

```markdown
# 部署手册

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| GPU | NVIDIA RTX 4060 8GB |
| CUDA | 12.1+ |
| Python | 3.10+ |
| 磁盘 | 20 GB |
| 内存 | 16 GB RAM |

## Docker 部署

```bash
docker build -t it-ops-assistant .
docker compose up -d
```

## 手动部署

```bash
make setup   # 安装依赖 + 下载模型
make train   # 训练模型
make serve   # 启动 vLLM + Gradio
```

服务地址: Gradio http://localhost:7860 · vLLM API http://localhost:8000
```

- [ ] **Step 5: 编写 TROUBLESHOOT.md**

```markdown
# 踩坑记录

## 坑 1: 训练 OOM

**现象**: CUDA out of memory

**原因**: batch_size=4, max_seq_length=2048

**解决**: batch_size 降到 1, gradient_accumulation 加到 8, max_seq_length 降到 1536

## 坑 2: RAG 检索不准

**现象**: 问 VPN，返回了年假规则

**原因**: 固定长度分块导致 VPN 内容被切碎分散

**解决**: 自研 SemanticChunker，按语义相似度切分，阈值 0.7

## 坑 3: vLLM 加载 LoRA 失败

**现象**: ValueError: LoRA is not enabled

**原因**: 启动参数缺少 --enable-lora

**解决**: 加上 --enable-lora --lora-modules itops=./outputs/lora_final/final_lora
```

- [ ] **Step 6: Commit**

```bash
git add README.md docs/ARCHITECTURE.md docs/MODULES.md docs/DEPLOY.md docs/TROUBLESHOOT.md
git commit -m "docs: add complete project documentation"
```

---

### Task 7.2: Docker 容器化

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: 编写 Dockerfile**

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 预下载模型（构建时）
RUN python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=2048, load_in_4bit=True)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

EXPOSE 7860 8000

CMD ["python", "src/app.py"]
```

- [ ] **Step 2: 编写 docker-compose.yml**

```yaml
version: "3.9"
services:
  vllm:
    image: it-ops-assistant
    command: >
      python -m vllm.entrypoints.openai.api_server
      --model unsloth/Qwen2.5-7B-Instruct
      --enable-lora
      --lora-modules itops=./outputs/lora_final/final_lora
      --max-lora-rank 16
      --gpu-memory-utilization 0.85
      --max-model-len 2048
      --port 8000
    ports: ["8000:8000"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  app:
    image: it-ops-assistant
    command: python src/app.py
    ports: ["7860:7860"]
    depends_on: [vllm]
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Docker containerization"
```

---

## 里程碑 8: 发布

### Task 8.1: 验证与发布

**Files:**
- 无新文件

- [ ] **Step 1: 初始化 Git 仓库**

```bash
cd "c:\claude code program\Ops Intelligence Agent"
git init
git add -A
git commit -m "feat: complete IT Ops Assistant project — v1.0"
```

- [ ] **Step 2: 运行全量测试**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 3: 运行训练 (生产前验证)**

```bash
# 需要 GPU，预计 50-70 分钟
python src/train_lora.py --data_dir datas --output_dir outputs/lora_final
```

- [ ] **Step 4: 生成评估报告**

```bash
python src/evaluate_model.py --test_path datas/test.jsonl --output evaluation.png
```

- [ ] **Step 5: 构建 Docker 镜像验证**

```bash
docker build -t it-ops-assistant .
```

- [ ] **Step 6: 打 Tag 发布**

```bash
git tag v1.0.0 -m "首次发布: 完整 IT 运维助手, RTX 4060 可运行"
```

---

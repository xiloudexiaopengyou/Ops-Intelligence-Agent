# 🚀 智能 IT 运维助手

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

## 项目结构

```
├── src/                  # 源代码
│   ├── chunker.py        # 自研语义分块器
│   ├── train_lora.py     # LoRA 训练脚本
│   ├── evaluate_model.py # 评估系统
│   ├── rag_system.py     # RAG 检索增强
│   ├── tools.py          # Agent 工具集
│   ├── agent_engine.py   # Agent 推理引擎
│   └── app.py            # Gradio 主界面
├── tests/                # 测试
├── datas/                # 训练/验证数据
├── docs/                 # 文档 + 员工手册
└── outputs/              # 训练产出
```

## License

MIT

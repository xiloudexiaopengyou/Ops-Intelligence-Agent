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

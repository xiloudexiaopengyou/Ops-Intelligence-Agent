# 架构设计文档

## 系统架构

四层架构：用户交互层（Gradio / vLLM API）→ 智能决策层（LangGraph Agent）→ 模型服务层（Qwen2.5-7B + BGE + ChromaDB）→ 数据层（训练数据 / 文档库 / 工具 API）。

## 显存分配

| 组件 | 运行位置 | 显存占用 |
|------|----------|----------|
| Qwen2.5-7B (4-bit) | GPU | ~5.8 GB |
| LoRA 适配器 | GPU | ~0.05 GB |
| BGE Embedding | CPU | 0 GB |
| ChromaDB | CPU | 0 GB |
| **GPU 总计** | | **~5.9 GB / 8 GB** |

## 进程架构

vLLM 独立服务 :8000 → Gradio :7860。OpenAI 兼容 API 通信。Embedding + ChromaDB 在 Gradio 进程内（CPU）。

## 数据流

用户输入 → Gradio → (RAG: ChromaDB 检索 + LLM 生成) / (Agent: LangGraph 推理循环 + 工具调用) → 返回用户

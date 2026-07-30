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

#!/bin/bash
# vLLM 推理部署脚本
# 用法: bash scripts/setup_vllm.sh
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_XET=0

echo "============================================"
echo "  vLLM 推理环境部署"
echo "  $(date)"
echo "============================================"

# ── 1. 安装 vLLM ──
echo ""
echo "📦 1. 安装 vLLM..."
pip3 install vllm 2>&1 | tail -3

# ── 2. 安装 auto-gptq（模型量化工具）──
echo ""
echo "📦 2. 安装 auto-gptq..."
pip3 install auto-gptq 2>&1 | tail -3

# ── 3. 下载 GPTQ 量化模型（vLLM 兼容，~4.5 GB）──
echo ""
echo "📥 3. 下载 Qwen2.5-7B-Instruct-GPTQ-Int4..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
    cache_dir='/root/.cache/huggingface/hub',
    resume_download=True,
)
print('GPTQ 模型下载完成')
"

echo ""
echo "============================================"
echo "  vLLM 环境部署完成!"
echo "  $(date)"
echo "============================================"

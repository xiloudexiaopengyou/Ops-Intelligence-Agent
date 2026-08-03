#!/bin/bash
# vLLM + LoRA 启动脚本 (适配 8GB RTX 4060)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/cu13
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_XET=0

echo "============================================"
echo "  vLLM + LoRA 推理服务启动"
echo "  模型: Qwen2.5-7B-Instruct-GPTQ-Int4"
echo "  LoRA: outputs/lora_final/final_lora"
echo "  端口: 8000"
echo "  显存利用率: 0.82 | max-model-len: 512"
echo "============================================"

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --enable-lora \
  --lora-modules "itops=outputs/lora_final/final_lora" \
  --max-lora-rank 16 \
  --port 8000 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 512 \
  --dtype float16 \
  --quantization gptq \
  --enforce-eager

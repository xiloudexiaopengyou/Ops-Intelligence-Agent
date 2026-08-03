#!/bin/bash
# 训练启动脚本 - 由 start_train.sh 调用
# 在 WSL2 内部运行

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_XET=0

echo "============================================"
echo "  智能 IT 运维助手 — LoRA 训练"
echo "  开始时间: $(date)"
echo "  HF_ENDPOINT: $HF_ENDPOINT"
echo "  HF_HUB_ENABLE_HF_XET: $HF_HUB_ENABLE_HF_XET"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

python3 src/train_lora.py \
    --data_dir datas \
    --output_dir outputs/lora_final

echo ""
echo "============================================"
echo "  训练结束: $(date)"
echo "  退出码: $?"
echo "============================================"

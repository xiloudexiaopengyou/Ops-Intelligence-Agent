#!/bin/bash
# WSL2 训练启动脚本 — 自动处理 HF 镜像和 Xet 问题
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_XET=0

echo "=========================================="
echo "  智能 IT 运维助手 — LoRA 训练"
echo "  开始时间: $(date)"
echo "  镜像: $HF_ENDPOINT"
echo "  Xet: $HF_HUB_ENABLE_HF_XET"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

python3 src/train_lora.py \
    --data_dir datas \
    --output_dir outputs/lora_final \
    2>&1 | tee /tmp/train.log

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=========================================="
echo "  训练结束: $(date)"
echo "  退出码: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

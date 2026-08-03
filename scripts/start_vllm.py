"""
vLLM 推理启动脚本 — 加载 GPTQ 模型 + LoRA 适配器

用法:
    # 先确保 setup_vllm.sh 已完成（vLLM 安装 + GPTQ 模型下载）
    python3 scripts/start_vllm.py [--lora_dir outputs/lora_final/final_lora] [--port 8000]
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")

# ── 配置 ──
GPTQ_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEFAULT_LORA = "outputs/lora_final/final_lora"
DEFAULT_PORT = 8000


def check_lora(lora_dir: str) -> bool:
    """检查 LoRA 适配器是否存在"""
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        print(f"❌ LoRA 目录不存在: {lora_dir}")
        return False
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (lora_path / f).exists()]
    if missing:
        print(f"❌ LoRA 文件缺失: {missing}")
        return False
    return True


def check_gptq_model() -> bool:
    """检查 GPTQ 模型是否已下载"""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    # Snapshot download 产生的目录
    for d in cache.glob("models--Qwen--Qwen2.5-7B-Instruct-GPTQ-Int4"):
        if (d / "snapshots").exists():
            return True
    return False


def start_vllm(lora_dir: str, port: int):
    """启动 vLLM 推理服务"""
    print("=" * 50)
    print("  vLLM 推理服务启动")
    print(f"  GPTQ 基础模型: {GPTQ_MODEL}")
    print(f"  LoRA 适配器:   {lora_dir}")
    print(f"  端口:           {port}")
    print("=" * 50)

    # vLLM v1 engine with GPTQ + LoRA
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", GPTQ_MODEL,
        "--enable-lora",
        "--lora-modules", f"itops={lora_dir}",
        "--max-lora-rank", "16",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048",
        "--dtype", "float16",
        "--quantization", "gptq",
    ]

    print(f"\n🚀 启动命令:")
    print("  " + " ".join(cmd))
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动 vLLM 推理服务")
    parser.add_argument("--lora_dir", default=DEFAULT_LORA, help="LoRA 适配器目录")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="服务端口")
    args = parser.parse_args()

    # 检查 LoRA
    if not check_lora(args.lora_dir):
        print("\n⚠️ LoRA 适配器未就绪，将仅使用 GPTQ 基础模型（不含微调权重）")
        lora_ready = False
    else:
        lora_ready = True

    # 检查 GPTQ 模型
    if not check_gptq_model():
        print(f"\n❌ GPTQ 模型未下载，请先运行:")
        print(f"   bash scripts/setup_vllm.sh")
        sys.exit(1)

    if lora_ready:
        start_vllm(args.lora_dir, args.port)
    else:
        # 无 LoRA 模式
        import subprocess
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", GPTQ_MODEL,
            "--port", str(args.port),
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "2048",
            "--dtype", "float16",
            "--quantization", "gptq",
        ]
        print(f"\n🚀 无 LoRA 模式启动:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd)

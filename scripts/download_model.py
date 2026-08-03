"""
从 ModelScope 下载 Qwen2.5-7B-Instruct 模型到本地
用法: python3 scripts/download_model.py
"""
import os
from pathlib import Path

# 使用 ModelScope（Qwen 官方源，国内直连）
from modelscope import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"

print(f"📥 从 ModelScope 下载 {MODEL_NAME}")
print(f"   目标目录: {LOCAL_DIR}")
print(f"   大小约: 14 GB")

# 下载到 huggingface cache 目录，后续训练脚本可直接加载
snapshot_download(
    model_id=MODEL_NAME,
    cache_dir=Path.home() / ".cache" / "huggingface" / "hub",
    revision="master",
)

print("✅ 模型下载完成！")
print(f"   路径: {LOCAL_DIR}")

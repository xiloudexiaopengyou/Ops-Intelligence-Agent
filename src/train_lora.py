"""
LoRA 微调训练脚本 — Qwen2.5-7B + QLoRA on RTX 4060 8GB
使用标准 HuggingFace + bitsandbytes（不依赖 unsloth）

用法:
    python src/train_lora.py [--data_dir datas] [--output_dir outputs/lora_final]
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# ── 网络配置（必须在导入 transformers/huggingface_hub 之前设置）──
os.environ["HF_HUB_ENABLE_HF_XET"] = "0"                  # 强制禁用 Xet
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"       # 国内镜像
# 以上环境变量必须设为 Python 字符串（非布尔），且严格早于任何 huggingface_hub 导入

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer


class Config:
    """训练超参数 — 针对 RTX 4060 Laptop 8GB 优化"""
    # 基础模型（不要改 — unsloth 的 bnb-4bit 预处理版不可用，直接用原版+量化）
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length = 1280       # 降低 256 以留出 VRAM 余量
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    batch_size = 1
    gradient_accumulation = 8
    num_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 50
    logging_steps = 10
    save_steps = 200
    save_total_limit = 3
    use_gradient_checkpointing = True
    bf16 = True

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def check_vram(threshold_gb: float = 7.5):
    """训练前检查显存，不足则告警退出"""
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA GPU，训练无法进行")
        exit(1)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # 用 memory_allocated 估算系统占用
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total - reserved
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {total:.1f} GB, 系统已占用: {reserved:.2f} GB, 可用约: {free:.2f} GB")
    if total < threshold_gb:
        print(f"❌ 显存不足: {total:.1f} GB < {threshold_gb} GB, 训练可能 OOM")
        print("   建议: 换用更大显存 GPU, 或降低 max_seq_length")
        exit(1)


def format_chatml(example: dict) -> dict:
    """将 instruction/output 格式化为 Qwen2.5 ChatML"""
    instruction = example["instruction"].strip()
    output = example["output"].strip()
    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text}


def load_data(data_dir: str) -> tuple[list[dict], list[dict]]:
    """加载训练和验证数据"""
    data_dir = Path(data_dir)
    train_data = []
    test_data = []
    for name, lst in [("train.jsonl", train_data), ("test.jsonl", test_data)]:
        path = data_dir / name
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lst.append(json.loads(line))
    print(f"训练集: {len(train_data)} 条, 验证集: {len(test_data)} 条")
    return train_data, test_data


def train(data_dir: str, output_dir: str):
    """主训练流程"""
    check_vram()

    # ── 二次保险：直接覆盖 huggingface_hub 的 Xet 开关 ──
    import huggingface_hub.file_download as hf_dl
    hf_dl.HF_HUB_ENABLE_HF_XET = False
    print(f"🔧 Xet 状态: HF_HUB_ENABLE_HF_XET = {hf_dl.HF_HUB_ENABLE_HF_XET}")
    print(f"   HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '未设置')}")

    print(f"🚀 加载 4-bit 量化模型: {Config.model_name}")
    print("   首次运行会从 HuggingFace 下载模型 (~14 GB)，请耐心等待...")

    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        trust_remote_code=True,
    )
    # Qwen tokenizer 没有默认 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL 1.9+ 通过 tokenizer.model_max_length 控制序列长度（不再通过 SFTTrainer 参数）
    tokenizer.model_max_length = Config.max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=Config.bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # SDPA 替代 Flash Attention（无需额外安装）
    )

    # 为 k-bit 训练做准备
    model = prepare_model_for_kbit_training(model)

    # 启用 gradient checkpointing
    if Config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA 配置
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=Config.target_modules,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("🔧 添加 LoRA 适配器...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # 加载数据
    train_data, test_data = load_data(data_dir)
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(format_chatml)

    if test_data:
        test_dataset = Dataset.from_list(test_data)
        test_dataset = test_dataset.map(format_chatml)
    else:
        test_dataset = None

    effective_batch = Config.batch_size * Config.gradient_accumulation
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * Config.num_epochs
    print(f"等效 batch_size: {effective_batch}")
    print(f"每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

    # 输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation,
        warmup_steps=Config.warmup_steps,
        num_train_epochs=Config.num_epochs,
        learning_rate=Config.learning_rate,
        bf16=Config.bf16,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        save_strategy="steps",
        save_total_limit=Config.save_total_limit,
        output_dir=str(output_path),
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        formatting_func=lambda x: x["text"],
        args=training_args,
    )

    print(f"\n🏋️ 开始训练...")
    print(f"   数据: {len(dataset)} 条 / seq_len={Config.max_seq_length} / batch={effective_batch}")
    print(f"   设备: {model.device}")
    start = datetime.now()

    trainer.train()

    elapsed = datetime.now() - start
    print(f"\n✅ 训练完成! 耗时: {elapsed}")

    # 保存 LoRA 适配器
    lora_path = output_path / "final_lora"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"💾 LoRA 适配器已保存至: {lora_path}")

    # 保存训练指标
    with open(output_path / "training_metrics.json", 'w', encoding='utf-8') as f:
        json.dump({
            "trainable_params": trainable,
            "total_params": total,
            "dataset_size": len(dataset),
            "test_dataset_size": len(test_data),
            "epochs": Config.num_epochs,
            "batch_size": Config.batch_size,
            "gradient_accumulation": Config.gradient_accumulation,
            "effective_batch_size": effective_batch,
            "learning_rate": Config.learning_rate,
            "max_seq_length": Config.max_seq_length,
            "lora_r": Config.lora_r,
            "duration": str(elapsed),
            "model_name": Config.model_name,
        }, f, indent=2, ensure_ascii=False)

    print(f"📊 训练指标已保存至: {output_path / 'training_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--data_dir", default="datas", help="数据目录")
    parser.add_argument("--output_dir", default="outputs/lora_final", help="输出目录")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)

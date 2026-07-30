"""
LoRA 微调训练脚本 — Qwen2.5-7B + QLoRA on RTX 4060 8GB

用法:
    python src/train_lora.py [--data_dir datas] [--output_dir outputs/lora_final]
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


class Config:
    """训练超参数 — 针对 RTX 4060 Laptop 8GB 优化"""
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length = 1536
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


def check_vram(threshold_gb: float = 7.5):
    """训练前检查显存，不足则告警退出"""
    if not torch.cuda.is_available():
        print("⚠️ 未检测到 CUDA GPU")
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = torch.cuda.memory_reserved(0) / 1024**3  # 实际可用约等于总量-系统占用
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {total:.1f} GB")
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
    import json
    from pathlib import Path
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

    print("🚀 加载 4-bit 量化模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.model_name,
        max_seq_length=Config.max_seq_length,
        load_in_4bit=True,
        use_gradient_checkpointing=Config.use_gradient_checkpointing,
    )

    print("🔧 添加 LoRA 适配器...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.lora_r,
        target_modules=Config.target_modules,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        use_gradient_checkpointing=Config.use_gradient_checkpointing,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # 加载数据
    train_data, _ = load_data(data_dir)
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(format_chatml)

    effective_batch = Config.batch_size * Config.gradient_accumulation
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * Config.num_epochs
    print(f"等效 batch_size: {effective_batch}")
    print(f"每 epoch 步数: {steps_per_epoch}, 总步数: {total_steps}")

    # 训练参数
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=Config.max_seq_length,
        args=training_args,
        packing=False,
    )

    print("🏋️ 开始训练...")
    start = datetime.now()

    # 训练循环（手动记录指标）
    metrics_log = []
    trainer.train()

    elapsed = datetime.now() - start
    print(f"✅ 训练完成! 耗时: {elapsed}")

    # 保存 LoRA
    lora_path = output_path / "final_lora"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"💾 LoRA 已保存至: {lora_path}")

    # 保存训练指标
    with open(output_path / "training_metrics.json", 'w') as f:
        json.dump({
            "trainable_params": trainable,
            "total_params": total,
            "dataset_size": len(dataset),
            "epochs": Config.num_epochs,
            "batch_size": Config.batch_size,
            "gradient_accumulation": Config.gradient_accumulation,
            "effective_batch_size": effective_batch,
            "learning_rate": Config.learning_rate,
            "max_seq_length": Config.max_seq_length,
            "lora_r": Config.lora_r,
            "duration": str(elapsed),
        }, f, indent=2, ensure_ascii=False)

    print(f"📊 训练指标已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--data_dir", default="datas", help="数据目录")
    parser.add_argument("--output_dir", default="outputs/lora_final", help="输出目录")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)

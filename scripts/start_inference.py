"""
备用推理方案 — transformers + FastAPI（不依赖 vLLM）
直接使用 bitsandbytes 4-bit 模型 + LoRA 适配器

用法:
    python3 scripts/start_inference.py [--lora_dir outputs/lora_final/final_lora] [--port 8000]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── 配置 ──
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LORA = "outputs/lora_final/final_lora"
DEFAULT_PORT = 8000

# ── FastAPI ──
app = FastAPI(title="IT-Ops Agent Inference", version="1.0")
model = None
tokenizer = None
lora_loaded = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "itops"
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    id: str = "chatcmpl-1"
    object: str = "chat.completion"
    created: int = 0
    model: str = "itops"
    choices: list[dict]


def load_model(lora_dir: str | None = None):
    """加载模型 + LoRA"""
    global model, tokenizer, lora_loaded

    print(f"🚀 加载基础模型: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if lora_dir and Path(lora_dir).exists():
        print(f"🔧 加载 LoRA 适配器: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        lora_loaded = True
        print("✅ LoRA 已合并")
    else:
        print("⚠️ 未加载 LoRA（使用基础模型）")

    model.eval()
    print(f"✅ 模型就绪 (设备: {model.device})")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global model, tokenizer

    if model is None:
        raise HTTPException(503, "模型未加载")

    # 构建 ChatML message
    text_parts = []
    for msg in req.messages:
        text_parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    text_parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(text_parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(
        created=int(datetime.now().timestamp()),
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    )


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "itops", "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "lora_loaded": lora_loaded}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动推理服务")
    parser.add_argument("--lora_dir", default=DEFAULT_LORA, help="LoRA 适配器目录")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="服务端口")
    parser.add_argument("--no_lora", action="store_true", help="不使用 LoRA")
    args = parser.parse_args()

    lora = None if args.no_lora else args.lora_dir
    load_model(lora)

    print(f"\n🌐 启动 API 服务: http://0.0.0.0:{args.port}")
    print(f"   OpenAI 兼容端点: http://localhost:{args.port}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

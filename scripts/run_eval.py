"""快速评估：vLLM 基座 vs LoRA，仅用 ROUGE（无需下载模型）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm

# ── 加载测试数据 ──
test_path = "datas/test.jsonl"
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f if line.strip()]
print(f"测试数据: {len(test_data)} 条")

# ── vLLM 客户端 ──
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_model(model_name, label, max_samples=20):
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for item in tqdm(test_data[:max_samples], desc=label):
        instruction = item.get("instruction", "")
        reference = item.get("output", "")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.1, max_tokens=256,
        )
        prediction = resp.choices[0].message.content
        r = scorer.score(reference, prediction)
        for k in scores:
            scores[k].append(r[k].fmeasure)
    avg = {k: float(np.mean(v)) for k, v in scores.items()}
    print(f"  {label}: ROUGE-L={avg['rougeL']:.4f}, ROUGE-1={avg['rouge1']:.4f}, ROUGE-2={avg['rouge2']:.4f}")
    return avg

# ── 评估 ──
print("\n📊 评估基座模型...")
base = evaluate_model("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "基座-GPTQ", max_samples=20)

print("\n📊 评估 LoRA 模型...")
lora = evaluate_model("itops", "LoRA-itops", max_samples=20)

# ── 画图 ──
metrics = list(base.keys())
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, [base[m] for m in metrics], width,
       label='Base GPTQ', color='#E06C6C', alpha=0.9)
ax.bar(x + width/2, [lora[m] for m in metrics], width,
       label='LoRA Fine-tuned', color='#7BC67E', alpha=0.9)

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Base Model vs LoRA Fine-tuned (ROUGE)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
ax.legend(['Base GPTQ', 'LoRA Fine-tuned'], fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

for container in ax.containers:
    for rect in container:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("evaluation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ evaluation.png 已生成！刷新 UI 即可查看。")

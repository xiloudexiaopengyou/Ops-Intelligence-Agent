"""
模型评估系统 — ROUGE-L + BERTScore 自动化评估 + 可视化对比图

用法:
    python src/evaluate_model.py --test_path datas/test.jsonl --output evaluation.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, test_data_path: str):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = [json.loads(line) for line in f if line.strip()]

        # 延迟初始化 — 评估时按需加载 rouge_score / bert_score
        self._rouge_scorer = None
        self._bert_scorer = None

    def _init_scorers(self):
        """延迟加载 rouge_score 和 bert_score（避免未安装时影响 __init__）"""
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer as _rouge_scorer
            self._rouge_scorer = _rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True,
            )
        if self._bert_scorer is None:
            from bert_score import BERTScorer as _BERTScorer
            self._bert_scorer = _BERTScorer(
                lang="zh",
                model_type="bert-base-chinese",
                rescale_with_baseline=True,
            )

    def evaluate(self, model, tokenizer, model_name: str,
                  max_samples: int = 50) -> dict[str, float]:
        """评估模型在测试集上的表现

        Args:
            model: HuggingFace 模型或 vLLM client
            tokenizer: HuggingFace tokenizer (直接加载模式) 或 None (vLLM 模式)
            model_name: 模型名称（用于日志）
            max_samples: 最多评估多少条

        Returns:
            {"rouge1": float, "rouge2": float, "rougeL": float, "bert_score": float}
        """
        self._init_scorers()
        scores = {"rouge1": [], "rouge2": [], "rougeL": [], "bert_score": []}
        eval_data = self.test_data[:max_samples]
        use_vllm = (tokenizer is None)

        for item in tqdm(eval_data, desc=f"评估 {model_name}"):
            instruction = item.get("instruction", "")
            reference = item.get("output", "")

            if use_vllm:
                prediction = self._vllm_generate(model, instruction)
            else:
                prediction = self._hf_generate(model, tokenizer, instruction)

            # 计算 ROUGE
            rouge = self._rouge_scorer.score(reference, prediction)
            for key in ["rouge1", "rouge2", "rougeL"]:
                scores[key].append(rouge[key].fmeasure)

            # 计算 BERTScore
            _, _, F1 = self._bert_scorer.score([prediction], [reference])
            scores["bert_score"].append(F1.item())

        avg = {k: float(np.mean(v)) for k, v in scores.items()}
        print(f"📊 {model_name}: ROUGE-L={avg['rougeL']:.4f}, BERTScore={avg['bert_score']:.4f}")
        return avg

    def _hf_generate(self, model, tokenizer, instruction: str) -> str:
        """HuggingFace 模型生成"""
        import torch
        messages = [{"role": "user", "content": instruction}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=256,
                                     temperature=0.1, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in result:
            result = result.split("<|im_start|>assistant")[-1].strip()
        return result

    def _vllm_generate(self, client, instruction: str) -> str:
        """vLLM 模型生成"""
        response = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": instruction}],
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content

    def plot_comparison(self, baseline_scores: dict, lora_scores: dict,
                        save_path: str = "evaluation.png") -> str:
        """生成基座 vs 微调对比柱状图"""
        metrics = list(baseline_scores.keys())
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [baseline_scores[m] for m in metrics],
               width, label='基座模型', color='#F87171', alpha=0.9)
        ax.bar(x + width/2, [lora_scores[m] for m in metrics],
               width, label='微调后', color='#34D399', alpha=0.9)

        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('基座模型 vs LoRA微调 性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

        for container in ax.containers:
            for rect in container:
                h = rect.get_height()
                ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 对比图已保存: {save_path}")
        return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--test_path", default="datas/test.jsonl")
    parser.add_argument("--output", default="evaluation.png")
    args = parser.parse_args()

    evaluator = ModelEvaluator(args.test_path)
    print(f"加载测试数据: {len(evaluator.test_data)} 条")

    # 注意：实际评估需要加载模型，此处仅验证数据加载
    print("✅ 评估系统就绪。实际评估需在训练后运行:")
    print("   python src/evaluate_model.py --test_path datas/test.jsonl")

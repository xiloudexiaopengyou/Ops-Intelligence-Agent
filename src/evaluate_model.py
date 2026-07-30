"""
模型评估系统 — 对比基座模型与 LoRA 微调模型的生成质量

用法:
    python src/evaluate_model.py --test_path datas/test.jsonl --output evaluation.png
    python src/evaluate_model.py --test_path datas/test.jsonl --output evaluation.png --backend vllm
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


# ============================================================
# 中文字体设置
# ============================================================
def _setup_chinese_font():
    """尝试配置中文字体，失败则回退到英文"""
    chinese_fonts = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    for font in chinese_fonts:
        try:
            rcParams["font.sans-serif"] = [font]
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    # 回退: 不设中文字体，标签使用英文
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


# ============================================================
# 评估指标计算
# ============================================================
def _compute_rouge_l(reference: str, candidate: str) -> float:
    """计算 ROUGE-L F1 分数（基于最长公共子序列）"""
    def _lcs_len(a: list, b: list) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    ref_chars = list(reference)
    cand_chars = list(candidate)
    lcs = _lcs_len(ref_chars, cand_chars)

    if len(ref_chars) == 0 or len(cand_chars) == 0:
        return 0.0

    precision = lcs / len(cand_chars) if len(cand_chars) > 0 else 0.0
    recall = lcs / len(ref_chars) if len(ref_chars) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_exact_match(reference: str, candidate: str) -> bool:
    """精确匹配（去除首尾空白后比较）"""
    return reference.strip() == candidate.strip()


def _compute_token_f1(reference: str, candidate: str) -> float:
    """基于字的 Token F1 分数"""
    ref_chars = set(reference)
    cand_chars = set(candidate)
    if not ref_chars or not cand_chars:
        return 0.0
    intersection = ref_chars & cand_chars
    precision = len(intersection) / len(cand_chars)
    recall = len(intersection) / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# ModelEvaluator
# ============================================================
class ModelEvaluator:
    """模型评估器 — 对比基座与微调模型的推理质量

    参数:
        test_path: JSONL 测试集路径，每行含 "instruction" 和 "output"
        base_model: 基座模型名称（HF model id 或 vLLM model name）
        lora_model: LoRA 微调模型名称（HF model id 或 vLLM model name）
        vllm_url: vLLM API 地址（backend="vllm" 时使用）
        max_samples: 最大评估样本数（None = 全量）
        device: HF 推理设备（"cuda" / "cpu"）
    """

    def __init__(
        self,
        test_path: str = "datas/test.jsonl",
        base_model: str = "unsloth/Qwen2.5-7B-Instruct",
        lora_model: str = "itops",
        vllm_url: str = "http://localhost:8000/v1",
        max_samples: Optional[int] = None,
        device: str = "cuda",
    ):
        self.test_path = Path(test_path)
        self.base_model = base_model
        self.lora_model = lora_model
        self.vllm_url = vllm_url
        self.max_samples = max_samples
        self.device = device

        # 延迟导入，避免未安装依赖时直接报错
        self._hf_available = None
        self._vllm_available = None

    # ---- 依赖检查 ----

    def _check_hf(self) -> bool:
        """检查 HuggingFace 依赖是否可用"""
        if self._hf_available is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._hf_available = True
            except ImportError:
                self._hf_available = False
        return self._hf_available

    def _check_vllm(self) -> bool:
        """检查 vLLM API 是否可达"""
        if self._vllm_available is None:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=self.vllm_url, api_key="test")
                client.models.list()
                self._vllm_available = True
            except Exception:
                self._vllm_available = False
        return self._vllm_available

    # ---- 数据加载 ----

    def _load_test_data(self) -> list[dict]:
        """加载 JSONL 测试数据"""
        samples = []
        with open(self.test_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # 统一字段名
                samples.append({
                    "instruction": data.get("instruction", ""),
                    "reference": data.get("output", ""),
                })

        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]

        print(f"已加载 {len(samples)} 条测试样本")
        return samples

    # ---- 推理 ----

    def _hf_generate(self, prompt: str, model_name: str) -> tuple[str, float]:
        """使用 HuggingFace 本地推理，返回 (生成文本, 耗时秒)"""
        if not self._check_hf():
            raise RuntimeError("HuggingFace 依赖不可用，请安装: pip install torch transformers")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 缓存模型
        cache_key = f"_hf_{model_name}"
        if not hasattr(self, cache_key):
            print(f"  加载 HF 模型: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                model = model.to("cpu")
            setattr(self, f"{cache_key}_model", model)
            setattr(self, f"{cache_key}_tokenizer", tokenizer)

        model = getattr(self, f"{cache_key}_model")
        tokenizer = getattr(self, f"{cache_key}_tokenizer")

        # 构造 ChatML 格式
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        # 解码（仅保留生成部分）
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True)

        return result.strip(), elapsed

    def _vllm_generate(self, prompt: str, model_name: str) -> tuple[str, float]:
        """使用 vLLM API 推理，返回 (生成文本, 耗时秒)"""
        if not self._check_vllm():
            raise RuntimeError(f"vLLM 服务不可达: {self.vllm_url}")

        from openai import OpenAI

        client = OpenAI(base_url=self.vllm_url, api_key="not-needed")

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        elapsed = time.perf_counter() - start

        return response.choices[0].message.content.strip(), elapsed

    # ---- 评估主流程 ----

    def evaluate(self, backend: str = "hf") -> dict:
        """对基座模型和 LoRA 模型分别评估

        Args:
            backend: "hf" (HuggingFace 本地) 或 "vllm" (vLLM API)

        Returns:
            {
                "base":  {"rouge_l": [], "token_f1": [], "exact_match": 0, "avg_time": 0.0, "responses": [...]},
                "lora":  {...},
                "samples": [{"instruction": ..., "reference": ..., "base": ..., "lora": ...}, ...],
                "total_samples": int,
            }
        """
        samples = self._load_test_data()

        generate_fn = self._vllm_generate if backend == "vllm" else self._hf_generate
        print(f"使用后端: {backend}")

        results = {
            "base": {"rouge_l": [], "token_f1": [], "exact_match": 0, "avg_time": 0.0, "responses": []},
            "lora": {"rouge_l": [], "token_f1": [], "exact_match": 0, "avg_time": 0.0, "responses": []},
            "samples": [],
            "total_samples": len(samples),
            "backend": backend,
        }

        for idx, sample in enumerate(samples):
            instruction = sample["instruction"]
            reference = sample["reference"]
            print(f"[{idx + 1}/{len(samples)}] {instruction[:50]}...")

            sample_result = {
                "instruction": instruction,
                "reference": reference,
                "base_response": "",
                "lora_response": "",
                "base_time": 0.0,
                "lora_time": 0.0,
            }

            # 基座模型推理
            try:
                base_resp, base_time = generate_fn(instruction, self.base_model)
                sample_result["base_response"] = base_resp
                sample_result["base_time"] = base_time
                results["base"]["rouge_l"].append(_compute_rouge_l(reference, base_resp))
                results["base"]["token_f1"].append(_compute_token_f1(reference, base_resp))
                if _compute_exact_match(reference, base_resp):
                    results["base"]["exact_match"] += 1
                results["base"]["responses"].append(base_resp)
            except Exception as e:
                print(f"  ⚠️ 基座模型推理失败: {e}")
                results["base"]["rouge_l"].append(0.0)
                results["base"]["token_f1"].append(0.0)
                results["base"]["responses"].append("")
                sample_result["base_response"] = f"[ERROR] {e}"

            # LoRA 模型推理
            try:
                lora_resp, lora_time = generate_fn(instruction, self.lora_model)
                sample_result["lora_response"] = lora_resp
                sample_result["lora_time"] = lora_time
                results["lora"]["rouge_l"].append(_compute_rouge_l(reference, lora_resp))
                results["lora"]["token_f1"].append(_compute_token_f1(reference, lora_resp))
                if _compute_exact_match(reference, lora_resp):
                    results["lora"]["exact_match"] += 1
                results["lora"]["responses"].append(lora_resp)
            except Exception as e:
                print(f"  ⚠️ LoRA 模型推理失败: {e}")
                results["lora"]["rouge_l"].append(0.0)
                results["lora"]["token_f1"].append(0.0)
                results["lora"]["responses"].append("")
                sample_result["lora_response"] = f"[ERROR] {e}"

            results["samples"].append(sample_result)

        # 汇总统计
        for model_key in ("base", "lora"):
            scores = results[model_key]["rouge_l"]
            times = [
                s[f"{model_key}_time"]
                for s in results["samples"]
                if s[f"{model_key}_time"] > 0
            ]
            results[model_key]["avg_rouge_l"] = np.mean(scores) if scores else 0.0
            results[model_key]["avg_token_f1"] = np.mean(results[model_key]["token_f1"]) if results[model_key]["token_f1"] else 0.0
            results[model_key]["std_rouge_l"] = np.std(scores) if scores else 0.0
            results[model_key]["avg_time"] = np.mean(times) if times else 0.0
            results[model_key]["exact_match_pct"] = (
                results[model_key]["exact_match"] / len(samples) * 100
                if samples else 0.0
            )

        # 打印汇总
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """打印评估结果摘要"""
        n = results["total_samples"]
        print(f"\n{'='*60}")
        print(f"评估完成 — 共 {n} 条样本 (backend={results['backend']})")
        print(f"{'='*60}")
        print(f"{'指标':<20} {'基座模型':>18} {'LoRA微调':>18}")
        print(f"{'-'*20} {'-'*18} {'-'*18}")
        print(f"{'ROUGE-L (avg)':<20} {results['base']['avg_rouge_l']:>18.4f} {results['lora']['avg_rouge_l']:>18.4f}")
        print(f"{'ROUGE-L (std)':<20} {results['base']['std_rouge_l']:>18.4f} {results['lora']['std_rouge_l']:>18.4f}")
        print(f"{'Token-F1 (avg)':<20} {results['base']['avg_token_f1']:>18.4f} {results['lora']['avg_token_f1']:>18.4f}")
        print(f"{'Exact Match %':<20} {results['base']['exact_match_pct']:>17.1f}% {results['lora']['exact_match_pct']:>17.1f}%")
        print(f"{'Avg Time (s)':<20} {results['base']['avg_time']:>18.2f} {results['lora']['avg_time']:>18.2f}")

        # 对比结论
        delta_rouge = results["lora"]["avg_rouge_l"] - results["base"]["avg_rouge_l"]
        if delta_rouge > 0.01:
            print(f"\n✅ LoRA 微调后 ROUGE-L 提升: +{delta_rouge:.4f}")
        elif delta_rouge < -0.01:
            print(f"\n⚠️ LoRA 微调后 ROUGE-L 下降: {delta_rouge:.4f}")
        else:
            print(f"\n➡️ LoRA 微调前后 ROUGE-L 无显著变化")

    # ---- 可视化 ----

    def plot_comparison(self, results: dict, output_path: str = "evaluation.png"):
        """生成基座 vs LoRA 对比图并保存

        Args:
            results: evaluate() 返回的结果字典
            output_path: 输出图片路径
        """
        base_rouge = results["base"]["rouge_l"]
        lora_rouge = results["lora"]["rouge_l"]
        n = len(base_rouge)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("基座模型 vs LoRA 微调 — 评估对比", fontsize=16, fontweight="bold", y=0.98)

        # ---- 子图 1: 每条样本 ROUGE-L 散点对比 ----
        ax1 = axes[0, 0]
        x = range(n)
        ax1.scatter(x, base_rouge, s=30, alpha=0.6, c="#E74C3C", label="基座模型", edgecolors="none")
        ax1.scatter(x, lora_rouge, s=30, alpha=0.6, c="#3498DB", label="LoRA 微调", edgecolors="none")
        ax1.set_xlabel("样本编号")
        ax1.set_ylabel("ROUGE-L 分数")
        ax1.set_title("逐样本 ROUGE-L 对比")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        # ---- 子图 2: 汇总指标柱状图 ----
        ax2 = axes[0, 1]
        metrics = ["ROUGE-L", "Token-F1", "Exact Match %"]
        base_vals = [results["base"]["avg_rouge_l"], results["base"]["avg_token_f1"], results["base"]["exact_match_pct"] / 100]
        lora_vals = [results["lora"]["avg_rouge_l"], results["lora"]["avg_token_f1"], results["lora"]["exact_match_pct"] / 100]

        x_pos = np.arange(len(metrics))
        width = 0.35
        bars1 = ax2.bar(x_pos - width / 2, base_vals, width, color="#E74C3C", alpha=0.85, label="基座模型")
        bars2 = ax2.bar(x_pos + width / 2, lora_vals, width, color="#3498DB", alpha=0.85, label="LoRA 微调")

        ax2.set_ylabel("分数")
        ax2.set_title("汇总指标对比")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        # 数值标签
        for bar, val in zip(bars1, base_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=8)
        for bar, val in zip(bars2, lora_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=8)

        # ---- 子图 3: ROUGE-L 分布直方图 ----
        ax3 = axes[1, 0]
        if n > 0:
            bins = max(10, min(30, n // 2))
            ax3.hist(base_rouge, bins=bins, alpha=0.5, color="#E74C3C", label="基座模型", edgecolor="white")
            ax3.hist(lora_rouge, bins=bins, alpha=0.5, color="#3498DB", label="LoRA 微调", edgecolor="white")
            ax3.axvline(results["base"]["avg_rouge_l"], color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.8)
            ax3.axvline(results["lora"]["avg_rouge_l"], color="#3498DB", linestyle="--", linewidth=1.5, alpha=0.8)
        ax3.set_xlabel("ROUGE-L 分数")
        ax3.set_ylabel("样本数")
        ax3.set_title("ROUGE-L 分布直方图")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # ---- 子图 4: 推理时间对比 ----
        ax4 = axes[1, 1]
        base_times = [s["base_time"] for s in results["samples"] if s["base_time"] > 0]
        lora_times = [s["lora_time"] for s in results["samples"] if s["lora_time"] > 0]
        time_data = [
            base_times if base_times else [0],
            lora_times if lora_times else [0],
        ]
        bp = ax4.boxplot(time_data, tick_labels=["基座模型", "LoRA 微调"], patch_artist=True, widths=0.4)
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][1].set_facecolor("#3498DB")
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax4.set_ylabel("推理耗时 (秒)")
        ax4.set_title("单样本推理时间分布")
        ax4.grid(True, alpha=0.3, axis="y")

        # 添加平均时间注释
        avg_base = results["base"]["avg_time"]
        avg_lora = results["lora"]["avg_time"]
        ax4.annotate(f"均值 {avg_base:.2f}s", xy=(1, avg_base), fontsize=8,
                      ha="center", color="#E74C3C",
                      xytext=(1.3, avg_base * 1.1), arrowprops=dict(arrowstyle="->", color="#888888"))
        ax4.annotate(f"均值 {avg_lora:.2f}s", xy=(2, avg_lora), fontsize=8,
                      ha="center", color="#3498DB",
                      xytext=(2.3, avg_lora * 1.1), arrowprops=dict(arrowstyle="->", color="#888888"))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"📊 评估图已保存至: {output_path}")
        return output_path


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估: 基座 vs LoRA")
    parser.add_argument("--test_path", default="datas/test.jsonl", help="JSONL 测试集路径")
    parser.add_argument("--output", default="evaluation.png", help="输出图片路径")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"], help="推理后端")
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-7B-Instruct", help="基座模型名称")
    parser.add_argument("--lora_model", default="itops", help="LoRA 模型名称")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1", help="vLLM API 地址")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="HF 推理设备")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        test_path=args.test_path,
        base_model=args.base_model,
        lora_model=args.lora_model,
        vllm_url=args.vllm_url,
        max_samples=args.max_samples,
        device=args.device,
    )

    results = evaluator.evaluate(backend=args.backend)
    evaluator.plot_comparison(results, output_path=args.output)

"""测试模型评估系统"""

import json
import os
import tempfile
import pytest
from pathlib import Path

import numpy as np


# ============================================================
# 测试数据构造辅助
# ============================================================
def _make_test_jsonl(num_samples: int = 10) -> str:
    """创建临时 JSONL 测试数据，返回文件路径"""
    samples = []
    for i in range(num_samples):
        samples.append({
            "instruction": f"测试问题 {i}: 请解释什么是VLAN?",
            "output": f"VLAN是虚拟局域网，用于逻辑分割网络。这是样本{i}的标准答案。",
        })
    return samples


def _write_temp_jsonl(samples: list[dict]) -> str:
    """将样本写入临时 JSONL 文件，返回路径"""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="test_eval_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return path


# ============================================================
# TestModelEvaluatorInit
# ============================================================
class TestModelEvaluatorInit:
    """验证 ModelEvaluator 初始化参数"""

    def test_default_parameters(self):
        """默认参数初始化"""
        from src.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator()
        assert evaluator.test_path == Path("datas/test.jsonl")
        assert evaluator.base_model == "unsloth/Qwen2.5-7B-Instruct"
        assert evaluator.lora_model == "itops"
        assert evaluator.vllm_url == "http://localhost:8000/v1"
        assert evaluator.max_samples is None
        assert evaluator.device == "cuda"

    def test_custom_test_path(self):
        """自定义测试集路径"""
        from src.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator(test_path="/tmp/custom_test.jsonl")
        assert evaluator.test_path == Path("/tmp/custom_test.jsonl")

    def test_custom_models(self):
        """自定义模型名称"""
        from src.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator(
            base_model="meta-llama/Llama-2-7b",
            lora_model="my-lora-adapter",
        )
        assert evaluator.base_model == "meta-llama/Llama-2-7b"
        assert evaluator.lora_model == "my-lora-adapter"

    def test_max_samples_limit(self):
        """max_samples 参数限制样本数"""
        from src.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator(max_samples=20)
        assert evaluator.max_samples == 20

    def test_device_selection(self):
        """设备选择参数"""
        from src.evaluate_model import ModelEvaluator
        cpu_eval = ModelEvaluator(device="cpu")
        assert cpu_eval.device == "cpu"

        cuda_eval = ModelEvaluator(device="cuda")
        assert cuda_eval.device == "cuda"

    def test_vllm_url_config(self):
        """vLLM URL 配置"""
        from src.evaluate_model import ModelEvaluator
        evaluator = ModelEvaluator(vllm_url="http://192.168.1.100:8080/v1")
        assert evaluator.vllm_url == "http://192.168.1.100:8080/v1"


# ============================================================
# TestPlotComparison
# ============================================================
class TestPlotComparison:
    """验证 plot_comparison 图表生成"""

    def _make_mock_results(self, n: int = 20) -> dict:
        """构造模拟评估结果"""
        np.random.seed(42)
        base_rouge = np.random.uniform(0.3, 0.7, n).tolist()
        # LoRA 略优
        lora_rouge = [min(1.0, v + np.random.uniform(-0.1, 0.15)) for v in base_rouge]

        samples = []
        for i in range(n):
            samples.append({
                "instruction": f"问题 {i}",
                "reference": f"参考答案 {i}",
                "base_response": f"基座回答 {i}",
                "lora_response": f"LoRA回答 {i}",
                "base_time": np.random.uniform(0.5, 2.0),
                "lora_time": np.random.uniform(0.5, 2.0),
            })

        return {
            "base": {
                "rouge_l": base_rouge,
                "token_f1": [np.random.uniform(0.4, 0.8) for _ in range(n)],
                "exact_match": 1,
                "avg_rouge_l": float(np.mean(base_rouge)),
                "avg_token_f1": float(np.mean([np.random.uniform(0.4, 0.8) for _ in range(n)])),
                "std_rouge_l": float(np.std(base_rouge)),
                "avg_time": float(np.mean([s["base_time"] for s in samples])),
                "exact_match_pct": 5.0,
                "responses": [s["base_response"] for s in samples],
            },
            "lora": {
                "rouge_l": lora_rouge,
                "token_f1": [np.random.uniform(0.4, 0.85) for _ in range(n)],
                "exact_match": 2,
                "avg_rouge_l": float(np.mean(lora_rouge)),
                "avg_token_f1": float(np.mean([np.random.uniform(0.4, 0.85) for _ in range(n)])),
                "std_rouge_l": float(np.std(lora_rouge)),
                "avg_time": float(np.mean([s["lora_time"] for s in samples])),
                "exact_match_pct": 10.0,
                "responses": [s["lora_response"] for s in samples],
            },
            "samples": samples,
            "total_samples": n,
            "backend": "mock",
        }

    def test_plot_generation(self):
        """验证图表文件生成"""
        from src.evaluate_model import ModelEvaluator

        evaluator = ModelEvaluator()
        results = self._make_mock_results(n=15)

        with tempfile.NamedTemporaryFile(suffix=".png", prefix="test_eval_plot_", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = evaluator.plot_comparison(results, output_path=output_path)
            assert result_path == output_path
            assert os.path.exists(output_path)
            # 验证文件非空
            assert os.path.getsize(output_path) > 1000  # 至少 1KB
        finally:
            _cleanup_file(output_path)

    def test_plot_with_single_sample(self):
        """单样本评估图生成"""
        from src.evaluate_model import ModelEvaluator

        evaluator = ModelEvaluator()
        results = self._make_mock_results(n=1)

        with tempfile.NamedTemporaryFile(suffix=".png", prefix="test_eval_plot_", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = evaluator.plot_comparison(results, output_path=output_path)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 1000
        finally:
            _cleanup_file(output_path)

    def test_plot_with_empty_results(self):
        """空结果图生成（边界情况）"""
        from src.evaluate_model import ModelEvaluator

        evaluator = ModelEvaluator()
        results = self._make_mock_results(n=0)

        with tempfile.NamedTemporaryFile(suffix=".png", prefix="test_eval_plot_", delete=False) as tmp:
            output_path = tmp.name

        try:
            result_path = evaluator.plot_comparison(results, output_path=output_path)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 1000
        finally:
            _cleanup_file(output_path)


# ============================================================
# TestDataLoading (tempfile)
# ============================================================
class TestDataLoading:
    """验证 JSONL 数据加载（使用 tempfile）"""

    def test_load_test_data_basic(self):
        """基本数据加载"""
        from src.evaluate_model import ModelEvaluator

        samples = _make_test_jsonl(5)
        test_path = _write_temp_jsonl(samples)

        try:
            evaluator = ModelEvaluator(test_path=test_path)
            data = evaluator._load_test_data()
            assert len(data) == 5
            assert "instruction" in data[0]
            assert "reference" in data[0]
            assert data[0]["instruction"] == "测试问题 0: 请解释什么是VLAN?"
        finally:
            _cleanup_file(test_path)

    def test_load_test_data_with_max_samples(self):
        """max_samples 限制生效"""
        from src.evaluate_model import ModelEvaluator

        samples = _make_test_jsonl(20)
        test_path = _write_temp_jsonl(samples)

        try:
            evaluator = ModelEvaluator(test_path=test_path, max_samples=5)
            data = evaluator._load_test_data()
            assert len(data) == 5
        finally:
            _cleanup_file(test_path)

    def test_load_empty_file(self):
        """空文件加载"""
        from src.evaluate_model import ModelEvaluator

        test_path = _write_temp_jsonl([])

        try:
            evaluator = ModelEvaluator(test_path=test_path)
            data = evaluator._load_test_data()
            assert len(data) == 0
        finally:
            _cleanup_file(test_path)


# ============================================================
# TestMetrics
# ============================================================
class TestMetrics:
    """验证评估指标计算函数"""

    def test_exact_match_true(self):
        """精确匹配 — 相同"""
        from src.evaluate_model import _compute_exact_match
        assert _compute_exact_match("VLAN是虚拟局域网", "VLAN是虚拟局域网") is True

    def test_exact_match_false(self):
        """精确匹配 — 不同"""
        from src.evaluate_model import _compute_exact_match
        assert _compute_exact_match("VLAN是虚拟局域网", "VLAN不是虚拟局域网") is False

    def test_exact_match_whitespace(self):
        """精确匹配 — 忽略首尾空白"""
        from src.evaluate_model import _compute_exact_match
        assert _compute_exact_match("  VLAN是虚拟局域网  ", "VLAN是虚拟局域网") is True

    def test_rouge_l_identical(self):
        """ROUGE-L — 完全相同"""
        from src.evaluate_model import _compute_rouge_l
        score = _compute_rouge_l("VLAN是虚拟局域网", "VLAN是虚拟局域网")
        assert score == 1.0

    def test_rouge_l_completely_different(self):
        """ROUGE-L — 完全不同"""
        from src.evaluate_model import _compute_rouge_l
        score = _compute_rouge_l("VLAN是虚拟局域网", "TCP是传输控制协议")
        assert score < 0.3

    def test_rouge_l_empty_candidate(self):
        """ROUGE-L — 空候选"""
        from src.evaluate_model import _compute_rouge_l
        score = _compute_rouge_l("VLAN是虚拟局域网", "")
        assert score == 0.0

    def test_rouge_l_empty_reference(self):
        """ROUGE-L — 空参考"""
        from src.evaluate_model import _compute_rouge_l
        score = _compute_rouge_l("", "VLAN是虚拟局域网")
        assert score == 0.0

    def test_token_f1_identical(self):
        """Token-F1 — 完全相同"""
        from src.evaluate_model import _compute_token_f1
        score = _compute_token_f1("VLAN是虚拟局域网", "VLAN是虚拟局域网")
        assert score == 1.0

    def test_token_f1_partial(self):
        """Token-F1 — 部分重叠"""
        from src.evaluate_model import _compute_token_f1
        score = _compute_token_f1("VLAN是虚拟局域网", "VLAN用于网络分割")
        assert 0.0 < score < 1.0


# ============================================================
# 辅助函数
# ============================================================
def _cleanup_file(path: str):
    """安全删除临时文件"""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass

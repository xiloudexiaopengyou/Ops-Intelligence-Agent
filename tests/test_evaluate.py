import pytest
import os
import tempfile
import json
import numpy as np
from unittest.mock import MagicMock, patch


class TestModelEvaluatorInit:
    def test_loads_test_data(self):
        from src.evaluate_model import ModelEvaluator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            json.dump({"instruction": "Q", "output": "A"}, f, ensure_ascii=False)
            f.write('\n')
            json.dump({"instruction": "Q2", "output": "A2"}, f, ensure_ascii=False)
            f.write('\n')
            f.flush()
            evaluator = ModelEvaluator(f.name)
            assert len(evaluator.test_data) == 2
        os.unlink(f.name)


class TestPlotComparison:
    def test_generates_image(self):
        from src.evaluate_model import ModelEvaluator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            json.dump({"instruction": "测试", "output": "参考答案"}, f,
                      ensure_ascii=False)
            f.write('\n')
            f.flush()
            evaluator = ModelEvaluator(f.name)
        os.unlink(f.name)

        scores1 = {"rouge1": 0.40, "rouge2": 0.20, "rougeL": 0.38, "bert_score": 0.75}
        scores2 = {"rouge1": 0.55, "rouge2": 0.30, "rougeL": 0.52, "bert_score": 0.85}

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img:
            save_path = img.name

        result_path = evaluator.plot_comparison(scores1, scores2, save_path)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1000  # 至少 1KB 图片
        os.unlink(result_path)

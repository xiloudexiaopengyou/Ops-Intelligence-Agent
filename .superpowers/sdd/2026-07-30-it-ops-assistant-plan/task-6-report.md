# Task 3.2 完成报告

## 完成内容

### 新建文件
- `src/evaluate_model.py` — ModelEvaluator 类，包含以下核心功能：
  - `evaluate(backend)` — 对基座模型和 LoRA 模型分别评估，计算 ROUGE-L、Token-F1、Exact Match 和推理耗时
  - `_hf_generate(prompt, model_name)` — 使用 HuggingFace Transformers 本地推理（支持 lazy loading + 缓存）
  - `_vllm_generate(prompt, model_name)` — 使用 vLLM OpenAI-compatible API 推理
  - `plot_comparison(results, output_path)` — 生成 2x2 对比图（散点对比、指标柱状图、分布直方图、耗时箱线图）
  - CLI 入口支持 `--test_path` / `--output` / `--backend` / `--max_samples` 参数

- `tests/test_evaluate.py` — 包含以下测试类（使用 `tempfile` 创建临时 JSONL 数据，测试后自动清理）：
  - `TestModelEvaluatorInit`（6 个测试）：默认参数、自定义路径、模型名称、max_samples、设备选择、vLLM URL 配置
  - `TestPlotComparison`（3 个测试）：多样本图表生成、单样本边界、空结果边界
  - `TestDataLoading`（3 个测试）：基本数据加载、max_samples 限制、空文件加载
  - `TestMetrics`（9 个测试）：ROUGE-L（相同/不同/空值）、Token-F1（相同/部分重叠）、Exact Match（精确/不匹配/空白处理）

## 测试结果

```
21 passed in 1.56s
```

所有 21 个测试用例通过。

## 偏差说明

Brief 中未提供 evaluate_model.py 的完整代码（仅有 Gradio app.py 代码），本任务根据项目上下文自行设计实现：

- **指标选择**：选择 ROUGE-L（长文本匹配友好）、Token-F1（字级别重叠）、Exact Match 三项指标，避免对 NLTK/rouge-score 等额外依赖强依赖
- **图表布局**：采用 2x2 子图布局（scatter + bar + histogram + boxplot），覆盖逐样本对比和汇总统计两个维度
- **临时文件清理**：test_evaluate.py 中所有 tempfile 均在测试完成后通过 `_cleanup_file()` 和 `try/finally` 确保删除

## Commit

```
git add src/evaluate_model.py tests/test_evaluate.py
git commit -m "feat: add model evaluator with ROUGE-L/Token-F1 metrics and comparison plots"
```

---

## FIX ROUND 1: 按 spec 重写 evaluate_model.py

**日期**: 2026-07-30

### 变更内容

**`src/evaluate_model.py`** — 完全替换，严格按照 plan 中 Task 3.2 的规格实现：

1. `ModelEvaluator.__init__(self, test_data_path: str)` — 单参数，轻量级，加载 JSONL 测试数据。rouge_score/bert_score 采用延迟加载，避免库未安装时导致初始化失败。
2. `evaluate(self, model, tokenizer, model_name, max_samples=50) -> dict[str, float]` — 接受 model+tokenizer+model_name，返回 `{"rouge1": float, "rouge2": float, "rougeL": float, "bert_score": float}` 简单字典。
3. 实现了 4 项指标：`rouge1`、`rouge2`、`rougeL`（使用 `rouge_score` 库）、`bert_score`（使用 `bert_score` 库，`bert-base-chinese`）。
4. `_hf_generate()` — HuggingFace 本地推理辅助方法。
5. `_vllm_generate()` — vLLM API 推理辅助方法。
6. `plot_comparison(baseline_scores, lora_scores, save_path)` — 简单双柱柱状图（单图，4 个指标分组对比），非 2x2 网格。

**已移除**（旧版本中的非规格内容）：
- 双后端架构（HF / vLLM 选择逻辑分散在类初始化中）
- `_setup_chinese_font()` 中文字体配置
- 详细的 CLI 参数（保留简单的 `--test_path` / `--output`）
- `_print_summary()` 方法
- `token_f1`、`exact_match`、`avg_time` 指标
- 自定义的 `_compute_rouge_l`、`_compute_token_f1`、`_compute_exact_match` 函数

**计划偏离**（仅一处最小必要）：将 `rouge_score` 和 `bert_score` 的顶层导入改为 `_init_scorers()` 延迟加载，避免库未安装时 `__init__` 失败 — `__init__` 仅负责加载测试数据，应在无 ML 依赖时仍可工作。

**`tests/test_evaluate.py`** — 简化为 2 个测试类 2 个测试方法：

1. `TestModelEvaluatorInit::test_loads_test_data` — 验证初始化正确加载 JSONL 测试数据
2. `TestPlotComparison::test_generates_image` — 验证 `plot_comparison` 生成有效图片文件（>1KB）

### 测试结果

```
2 passed in 1.53s
```

### Commit

```
git add src/evaluate_model.py tests/test_evaluate.py
git commit -m "fix: rewrite evaluate_model.py to match spec — ROUGE/BERTScore, simple bar chart"
```

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

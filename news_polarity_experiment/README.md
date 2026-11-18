# UltraBERT-150M 新闻政治倾向分类实验

本实验预测数据集中 AllSides 的 `left`、`center`、`right` 标签，而非声称识别新闻的客观政治立场。标注定义见 [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)。

模型输入只包含清洗后的 `headline + text`。数据准备阶段会移除 URL、独立的媒体名/作者署名行、常见模板化页脚，并在模型侧排除 publisher、URL、author、byline、ID、topic、date、story_id 等元数据。

## 运行顺序

1. 生成防泄漏 split（按 `story_id` 与规范化全文重复内容共同分组）。
2. 在验证集上运行 TF-IDF + logistic regression 基线。
3. 在验证集上调 UltraBERT-150M，锁定超参后才评测测试集。

```bash
python3 prepare_splits.py \
  --input-dir /Users/lianshuang/Desktop/allsides-dataset/splits_full_articles_70_15_15 \
  --output-dir artifacts/splits_ultrabert_clean_70_15_15

python3 run_baseline.py \
  --data-dir artifacts/splits_ultrabert_clean_70_15_15 \
  --output artifacts/baseline_clean_validation.json

.venv_ultrabert311/bin/python run_ultrabert.py \
  --data-dir artifacts/splits_ultrabert_clean_70_15_15 \
  --output-dir artifacts/ultrabert_dev_seed42_len512 \
  --max-length 512 --seed 42
```

`run_ultrabert.py` 默认只评估验证集，底层实现位于 `run_modernbert.py`，默认模型为 `dmis-lab/UltraBERT-150M`，并启用 `trust_remote_code=True`。确定配置后，以新的输出目录加上 `--evaluate-test` 运行最终五个随机种子；不要根据测试集结果挑选配置。

当前项目内兼容环境为 `.venv_ultrabert311`（Python 3.11 + Transformers 5.16.1）。首次运行可将 `--model-id` 指向本地已下载的 UltraBERT 权重目录；若使用 Hugging Face ID，需要确保模型权重可访问。

模型输入固定为清洗后的 `headline + text`，不会传入 publisher、URL、author、byline、ID、topic、date 或 story_id。

数据已在 `/Users/lianshuang/Desktop/allsides-dataset/splits_full_articles_70_15_15` 找到；每条记录至少应包含 `headline`、`text`、`label`，并尽量提供 `publisher`、`story_id`、日期和 URL 以便审计。当前执行状态、基线结果和 UltraBERT 依赖阻塞见 [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)。

## 大模型 few-shot 基线

`run_fewshot_llm.py` 使用给定的政治倾向提示词，通过 OpenAI-compatible Chat Completions 接口进行 few-shot 分类。每个类别的示例只从 `train.jsonl` 抽取；默认先评估 `validation.jsonl`，读取测试集必须显式加入 `--split test --evaluate-test`。模型输入仍只包含清洗后的 `headline + text`。

先生成提示词预览（不调用模型）：

```bash
python3 run_fewshot_llm.py \
  --output-dir artifacts/fewshot_llm_preview \
  --preview --shots-per-label 2
```

配置 API key 后运行验证集：

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
python3 run_fewshot_llm.py \
  --data-dir artifacts/splits_ultrabert_clean_70_15_15 \
  --output-dir artifacts/fewshot_llm/gpt4o_mini_validation_seed42 \
  --model gpt-4o-mini \
  --shots-per-label 2 --seed 42
```

输出包括 `config.json`、`fewshot_examples.jsonl`、逐条 `predictions.jsonl` 和 `summary.json`。固定模型与 few-shot 配置后，才运行测试集：

```bash
python3 run_fewshot_llm.py \
  --data-dir artifacts/splits_ultrabert_clean_70_15_15 \
  --output-dir artifacts/fewshot_llm/gpt4o_mini_test_seed42 \
  --model gpt-4o-mini \
  --shots-per-label 2 --seed 42 \
  --split test --evaluate-test
```

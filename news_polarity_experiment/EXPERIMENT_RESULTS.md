# 新闻政治倾向分类实验汇总

更新日期：2026-11-18（Asia/Shanghai）

## 任务与数据

- 任务：预测数据集提供的 AllSides `left` / `center` / `right` 标签。结果应表述为“预测 AllSides 来源标签”，不应表述为识别新闻的客观政治立场。
- 输入：清洗后的 `headline + text`。不输入 publisher、URL、作者、byline、ID、topic、date 或 story_id。
- 数据：2,897 篇文章；train / validation / test = 2,028 / 434 / 435。
- 切分：按 `story_id` 与规范化正文哈希的连通组件做分组分层切分。三个 split 之间的 story_id 与规范化正文交集均为 0。
- 主指标：macro-F1；验证集用于模型开发。测试集只评测过一次，见“测试集结果与边界”。

## 验证集结果

| 系统 / 设置 | 训练方式 | Accuracy | Macro-F1 | 备注 |
|---|---|---:|---:|---|
| TF-IDF + Logistic Regression | C=10 | 0.8387 | 0.8293 | 清洗文本基线 |
| ModernBERT-base, 512 tokens, seed 42 | 3 epochs | 0.8341 | 0.8274 | |
| ModernBERT-base, 512 tokens, seed 13 | 3 epochs | 0.8479 | 0.8433 | |
| ModernBERT-base, 512 tokens, seed 101 | 3 epochs | 0.8134 | 0.8028 | |
| ModernBERT-base, 512 tokens | 三个完成 seeds 的均值 ± 样本标准差 | 0.8318 | **0.8245 ± 0.0204** | 低于基线，方差较大 |
| ModernBERT-base, 1,024 tokens, seed 42 | 从头训练 1 epoch | 0.7488 | 0.7462 | 不足训练 |
| ModernBERT-base, 1,024 tokens, seed 42 | 从头训练 2 epochs | 0.8341 | 0.8287 | 接近基线 |
| ModernBERT-base, 1,024 tokens, seed 42 | 从上述 2-epoch checkpoint 继续 1 epoch | 0.8525 | 0.8458 | 训练总量约 3 epochs；优化器调度在继续阶段重启 |
| ModernBERT-base, 2,048 tokens, seed 42 | 从 1,024-token checkpoint 继续 1 epoch；gradient checkpointing | **0.8825** | **0.8770** | 后续探索性验证结果 |

## 唯一一次测试集结果

测试的 checkpoint 是 **1,024-token、2-epoch checkpoint 后继续 1 epoch** 的模型；测试集共有 435 篇文章。

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.8437 |
| Macro-F1 | 0.8422 |
| Left F1 | 0.8319 |
| Center F1 | 0.8065 |
| Right F1 | 0.8882 |

测试集混淆矩阵（行是真实标签，列是预测标签；顺序为 left / center / right）：

```text
[[99, 17,  2],
 [15,125, 13],
 [ 6, 15,143]]
```

## 解释与实验边界

1. 2,048-token 的验证 macro-F1 为 0.8770，是目前最高的验证结果；但它是在 1,024-token checkpoint 基础上继续训练所得，且发生在测试集已被评测之后。
2. 因此，0.8422 的测试 macro-F1 仅适用于前述 1,024-token checkpoint，**不能**被报告为 2,048-token 模型的最终测试成绩。
3. 若需要严谨地报告 2,048-token 的最终泛化结果，应重新建立一个从未见过的测试集（例如新的按 story_id 和正文去重分组切分），并在固定 2,048-token 配置和种子方案后只评测一次。
4. 512-token 的三 seed 结果均值低于 TF-IDF 基线，说明小数据条件下微调的随机性不可忽略；目前 1,024/2,048 结果均只有 seed 42，尚不能证明提升具有跨种子稳定性。
5. 训练曾出现 MPS 长时运行停滞：512-token 的 seed 202 和 777 未完成，不纳入任何统计。2,048-token 成功依赖 batch size 1 与 gradient checkpointing，速度明显较慢。
6. 即使移除了显式 publisher 字段，媒体写作风格仍可能是强预测信号。若目标是跨媒体政治倾向泛化，应补充 publisher-held-out 评测。

## 关键产物

- 数据审计与 split：`artifacts/splits_ultrabert_clean_70_15_15/manifest.json`
- 基线：`artifacts/baseline_clean_validation.json`
- 唯一测试评测：`artifacts/modernbert_base_seed42_len1024_continue_epoch1_test_metrics.json`
- 最高验证结果：`artifacts/modernbert_base_dev_seed42_len2048_continue_epoch1/summary.json`

# 实验执行状态

更新时间：2026-11-18（Asia/Shanghai）

## 已完成

1. 标注规范已写入 `ANNOTATION_GUIDE.md`。当前数据是 AllSides 风格的 `left/center/right` 标签，未提供双人标注字段，因此不能计算 Cohen's κ 或 Krippendorff's α。
2. 已对 2,897 篇文章完成数据检查：15 个 publisher，日期范围为 2025-01-24 至 2026-05-15；缺失 1 个 headline，其余核心字段完整。
3. 已重建清洗后的 split：`artifacts/splits_ultrabert_clean_70_15_15/`。清洗会移除 URL、独立媒体名/作者署名行和常见模板页脚；模型输入不包含 publisher、URL、author、byline、ID、topic、date、story_id。
   清洗审计：2,897 篇中正文残留 URL 为 0，空正文为 0；有 1 篇缺失 headline，需在正式训练前决定删除或补齐。
4. 已完成连接组件分组切分：train/validation/test = 2028/434/435；story_id 和规范化正文在三组之间的重叠均为 0。
5. 已运行 TF-IDF + Logistic Regression 基线。验证集 accuracy = 0.8387，macro-F1 = 0.8293，最佳 `C=10`。

## 暂停项

- 试标一致性：数据没有标注者和仲裁字段，不能按预先约定的门槛判定标签可靠性。若直接把 AllSides 标签当作金标准，需要在论文中明确这是来源标签/弱监督，而非本实验重新标注的金标准。
- UltraBERT 训练：原系统环境为 `transformers==4.57.6`，无法加载 UltraBERT 远程代码；已在项目内建立兼容的 Python 3.11/Transformers 5.16.1 环境。

## 下一步

1. 用多个随机种子复现 1,024-token 增益，再决定是否继续 2,048/4,096-token 消融。
2. 新长度配置锁定前不再读取封存测试集；应优先使用新外部测试集比较泛化。

## 用户确认后的执行记录

- 用户已确认直接使用 AllSides `label` 作为金标准。
- 已建立 `.venv_ultrabert311`：Python 3.11 + Transformers 5.16.1 + PyTorch 2.13.0，并成功加载 UltraBERT-150M（约 151M 参数）。
- 128-token 全量一轮已完成：验证集 accuracy = 0.6705，macro-F1 = 0.6717；结果见 `artifacts/ultrabert_dev_seed42_len128/summary.json`。该结果不是最终测试结果。
- 8 条训练/8 条验证的 128-token smoke test 已通过，确认前向、反向、保存和重新加载流程正常。
- 512-token 全量训练已启动，但在本机 MPS 首次长序列计算阶段吞吐过低，运行约 8 分钟后安全停止，未产生测试集结果。测试集仍未读取。
- 随后已重新完成 512-token 全量 1 epoch：验证集 accuracy = 0.8641，macro-F1 = 0.8597；结果见 `artifacts/ultrabert_dev_seed42_len512/summary.json`。测试集仍未读取。
- 已从检查点继续完成第 2、3 epoch；第 3 epoch 验证集 accuracy = 0.9263，macro-F1 = 0.9222。最终第 3 epoch 结果见 `artifacts/ultrabert_dev_seed42_len512_epoch3/summary.json`。三 epoch 训练期间测试集始终未读取。
- 已对封存测试集进行一次最终评估（435 篇）：accuracy = 0.8828，macro-F1 = 0.8811；结果见 `artifacts/ultrabert_test_evaluation_epoch3.json`。
- 1,024-token 长上下文扩展已完成：相同的 512-token epoch-3 权重仅在验证时扩展到 1,024 tokens，accuracy = 0.9401，macro-F1 = 0.9363；再续训 1 个 1,024-token epoch 后，accuracy = 0.9424，macro-F1 = 0.9392，训练损失 = 0.2386。结果分别见 `artifacts/ultrabert_validation_epoch3_inference_len1024.json` 和 `artifacts/ultrabert_dev_seed42_len1024_from_epoch3/summary.json`。该扩展实验未读取测试集。
- 已继续完成 2 个 1,024-token epoch（epoch 5–6）。Epoch 5：accuracy = 0.9401，macro-F1 = 0.9392，train loss = 0.1770；Epoch 6：accuracy = 0.9493，macro-F1 = 0.9486，train loss = 0.0520。Epoch 6 为当前验证集最佳 checkpoint，见 `artifacts/ultrabert_dev_seed42_len1024_epochs5_6/best_model/`。这两轮使用同一进程内连续的优化器/调度器状态，且未读取测试集。
- 2,048-token 扩展已完成。使用 Epoch 6 权重仅将验证输入扩展到 2,048 tokens 时，accuracy = 0.9631，macro-F1 = 0.9618，为当前最佳验证配置；继续微调 1 个 2,048-token epoch 后，accuracy 降至 0.9516，macro-F1 降至 0.9494，train loss = 0.0953。因此不采用 Epoch 7 checkpoint，当前推荐为 `ultrabert_dev_seed42_len1024_epochs5_6/best_model` 权重搭配 `max_length=2048` 推理。对照与微调结果分别见 `artifacts/ultrabert_validation_epoch6_inference_len2048.json` 和 `artifacts/ultrabert_dev_seed42_len2048_epoch7/summary.json`。该实验未读取测试集。


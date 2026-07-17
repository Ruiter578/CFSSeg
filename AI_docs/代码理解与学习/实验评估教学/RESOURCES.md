# Resources

- [datasets/voc.py](datasets/voc.py): `image_set=test` 使用完整 `val.txt`，其他 split 使用任务过滤后的列表。
- [datasets/init_dataset.py](datasets/init_dataset.py): 同时创建 train、val、test 三个 loader。
- [utils/tasks.py](utils/tasks.py): step1 sequential 的样本筛选规则。
- [metrics/stream_metrics.py](metrics/stream_metrics.py): IoU、Class Acc 与 `EPS` 的实际计算。
- [当前 E1.1 验证汇总](Codex_Plans/20260712_e1_1_air_feature_search_trs_summaries/final_validation_summary.tsv): 这次需要重新解释的实际结果。

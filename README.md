# SegACIL / CFSSeg

基于冻结特征、随机隐层RHL与闭式递归岭回归的类增量语义分割。二维主任务为PASCAL VOC 2012；`CFSSeg-code3D/`保留三维旁线。

## 模型与统一入口

DeepLabV3与DeepLabV3+均可通过超参数选择，默认仍为DeepLabV3–ResNet101。

| 参数 | 默认 | 用途 |
| --- | --- | --- |
| `MODEL` / `--model` | `deeplabv3_resnet101` | 设为`deeplabv3plus_resnet101`即可使用V3+；两者也支持ResNet50 |
| `AIR_FEATURE_SOURCE` / `--air_feature_source` | `auto` | V3自动用decoder；V3+自动用aspp_up |
| `SUBPATH` | 当日日期与模型名 | 本次输出目录；正式实验应显式设置唯一名称 |
| `BASE_SUBPATH` | 同SUBPATH | step1读取哪组step0；复用已有模型时显式指定，且架构须与MODEL一致 |
| `DATA_ROOT` | 仓库内`data_root/VOC2012` | VOC数据集路径，可覆盖 |
| `PYTHON` | `python3` | 当前环境的解释器，可指定绝对路径 |
| `BUFFER` / `GAMMA` | 8192 / 1 | 随机特征维度与岭回归正则；复现某组实验时按其配置显式设置 |

从step0开始训练并进行一次增量：

```bash
MODEL=deeplabv3_resnet101 SUBPATH=v3_new_run \
START_STEP=0 END_STEP=1 bash run.sh

MODEL=deeplabv3plus_resnet101 SUBPATH=v3plus_new_run \
START_STEP=0 END_STEP=1 bash run.sh
```

复用已有V3+ step0时：

```bash
MODEL=deeplabv3plus_resnet101 AIR_FEATURE_SOURCE=auto \
SUBPATH=v3plus_incremental_new_run BASE_SUBPATH=your_existing_v3plus_step0 \
START_STEP=1 END_STEP=1 bash run.sh
```

`run_trs.sh`使用同一入口，默认GPU2；`run_rhl_norm.sh`在同一入口上设置归一化配方。两者都支持MODEL、clean holdout与其他主入口参数，不再维护独立训练命令。旧脚本见[历史脚本](AI_docs/早期旧分支文档/main历史脚本/README.md)。

`15-5`为step0–1；`15-1`为step0–5；`10-1`为step0–10。`PRETRAINED_BACKBONE=""`可关闭ImageNet预训练加载，默认启用。完整模型checkpoint包含序列化模型对象，加载项目自行保存或确认来源的权重。

## 环境与数据

推荐Python 3.10及以上，PyTorch≥2.0与匹配的torchvision；依赖numpy、Pillow、scikit-learn、tqdm、matplotlib，TensorBoard为可选日志依赖。GPU训练需要匹配的CUDA环境。

VOC训练池使用JPEGImages、SegmentationClassAug和项目任务列表。数据目录与权重不随Git归档，启动实验前检查实际文件。clean validation需同时指定`VALIDATION_LIST`和覆盖全部验证图片的`TRAIN_EXCLUDE_LIST`，并在step0起使用同一划分。

## 测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python -m unittest discover -s tests -p 'test*.py' -v

PYTHON=/bin/echo MODEL=deeplabv3plus_resnet101 \
SUBPATH=dry_run START_STEP=0 END_STEP=1 bash run.sh
```

CPU测试覆盖双模型前向/反向、解析拟合、checkpoint、训练恢复、clean划分及启动参数。CPU smoke不产生新的分割精度结论。

## 项目导航

- [当前项目状态与下一轮任务](AI_docs/课题Home.md)
- [Git整合、修复与验收记录](AI_docs/代码改动报告/2026-11-19_Git分支整合与项目整理报告.md)
- [完整方法与实验回顾](AI_docs/2026-10-24_项目重启交接/03_方法与实验完整回顾_修订版.md)
- [早期旧分支文档与恢复标签](AI_docs/早期旧分支文档/README.md)
- [10月重启交接资料](AI_docs/2026-10-24_项目重启交接/00_阅读入口.md)

活动分支保留main与feature/rhl-next；伪标签、V3+ control和早期P0/P1通过归档标签追溯。feature/rhl-next的合并或归档留待下一轮决定。

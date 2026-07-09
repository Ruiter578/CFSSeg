# CFSSeg-code3D 本地化审计与集成报告

时间：2026-07-07  
目录：`SegACIL/CFSSeg-code3D`

## 结论

`CFSSeg-code3D` 是原 CFSSeg 3D 点云分割分支，代码结构完整覆盖 S3DIS/ScanNet 的 DGCNN、训练、评估、预处理和脚本调度。它可以作为 SegACIL 的重要子目录纳入仓库，但数据集、训练日志、checkpoint、pycache、本地 wheel 和编辑器目录不应进入 Git。

本次已做的本地化重点是路径和仓库边界，不改模型、损失、训练循环和评估指标的算法逻辑。3D 实验输出已统一映射到仓库根目录下的 `checkpoints_3d/`，避免继续散落在 `CFSSeg-code3D/log_s3dis/` 和 `CFSSeg-code3D/log_scannet/`。

## 特殊文件处理建议

| 文件 | 作用 | 本服务器是否有用 | 处理 |
| --- | --- | --- | --- |
| `CFSSeg-code3D/.vscode/` | VS Code 工作区设置；当前只包含关闭 Git ignore limit warning 和推荐 Copilot 插件 | 对运行无影响；你的远程容器 VS Code 可以读到，但不是项目必要文件 | 继续不提交。根 `.gitignore` 已忽略 `.vscode` |
| `CFSSeg-code3D/.cursorignore` | Cursor IDE 索引忽略配置，忽略 datasets/logs | 对 VS Code/Codex 无运行作用；如果以后用 Cursor 有帮助 | 可保留并随 3D 代码提交，也可不管；不是敏感文件 |
| `CFSSeg-code3D/.gitignore` | 原作者 3D 子项目忽略规则，忽略 `log_s3dis/`、`log_scannet/`、`datasets/`、`nohup*`、`slurm*`、pycache | 有参考价值，但根仓库仍需要自己的规则 | 保留。已结合它更新 `SegACIL/.gitignore` |
| `torch_cluster-1.6.3+pt25cu118-cp312-cp312-linux_x86_64.whl` | `torch_cluster` 本地安装包，面向 Python 3.12、PyTorch 2.5、CUDA 11.8 | 当前环境是 Python cp311、torch `2.5.1+cu121`、CUDA `12.1`，不匹配；且当前未安装 `torch_cluster` | 不提交。已在根 `.gitignore` 忽略；除非另建 cp312/cu118 环境，否则可以本地删除 |
| `results.png` | 结果图或示例图 | 运行无关 | 根 `.gitignore` 的 `*.png` 已忽略 |
| `test.sh` | 原来是运行 `test.py` 的临时脚本 | 原文件包含明文代理凭据，风险高 | 已移除凭据，只继承 shell 环境代理 |
| `test.py` | MNIST 示例，不属于 3D 训练主链路 | 对 CFSSeg 3D 复现实验无用 | 暂未删除，避免误删原作者附带文件；报告中标记为非主链路 |
| `zhanka.py` | 占用 GPU 显存的工具脚本 | 只适合临时调试，不应误运行 | 暂未删除；运行前需人工确认 |

## 代码结构

主入口是 `main.py`，根据 `--phase` 分发到 `runs/`：

- `joint_train.py`：联合训练上界。
- `freeze_and_add.py`、`finetuning.py`：直接适配基线。
- `train_ewc.py`、`train_lwF.py`：遗忘抑制基线。
- `train_ours.py`、`train_ACL.py`：3DPC-CISS / ACL 相关训练。
- `eval*.py`：base、joint、incremental、多步增量评估。

数据相关模块：

- `dataloaders/s3dis.py`、`dataloaders/scannet.py`：类别顺序、base/increment class split。
- `dataloaders/loader.py`、`joint_loader.py`：块数据读取和测试集读取。
- `preprocess/`：S3DIS/ScanNet 原始点云到 blocks 的预处理工具。

模型与工具：

- `models/dgcnn*.py`：DGCNN backbone 和分割分类器。
- `utils/AnalyticLinear.py`、`utils/Buffer.py`：解析线性层和随机 buffer。
- `utils/checkpoint_util.py`：checkpoint 读写。
- `utils/logger.py`：日志目录创建和文本日志。

## 本地化修改

1. `SegACIL/.gitignore`
   - 新增 `CFSSeg-code3D/datasets/`、`log_s3dis/`、`log_scannet/`、`nohup*`、`slurm*`、pycache、`.pytest_cache`、`.mypy_cache`、`*.whl`。
   - 新增 `data_root_3d/`。
   - 新增 `checkpoints_3d/**` 忽略规则，只保留 `checkpoints_3d/.gitkeep`。

2. `CFSSeg-code3D/main.py`
   - 新增本地路径规范化。
   - 默认把旧输出前缀 `./log_s3dis/...` 映射到 `SegACIL/checkpoints_3d/s3dis/...`。
   - 默认把旧输出前缀 `./log_scannet/...` 映射到 `SegACIL/checkpoints_3d/scannet/...`。
   - 支持 `CFSSeg3D_OUTPUT_ROOT` 覆盖输出根目录。
   - 支持 `CFSSeg3D_DATA_ROOT` 覆盖数据根目录，例如把 `./datasets/ScanNet/blocks_bs1_s1/` 映射到 `$CFSSeg3D_DATA_ROOT/ScanNet/blocks_bs1_s1`。

3. ScanNet/S3DIS 路径推导
   - 将 `args.data_path[:-14]` 这类固定长度截断替换为 `os.path.dirname(args.data_path.rstrip('/'))`。
   - 将写死的 `./datasets/ScanNet/` 验证列表路径替换为从 `args.data_path` 推导。
   - 影响文件包括 `dataloaders/s3dis.py`、`dataloaders/scannet.py` 和多个 `runs/*.py`。

4. `CFSSeg-code3D/scripts/bash_train_tasks.sh`
   - 原脚本调用不存在的 `train_joint_segmentor.sh` 等旧文件名。
   - 已改为当前目录实际存在的 `train3.sh`、`train6.sh`、`train5.sh`、`train4.sh`、`train2.sh`、`train1.sh`、`train.sh`。

5. `CFSSeg-code3D/test.sh`
   - 移除明文代理账号信息。

## 输出目录判断

应该单独建立 `SegACIL/checkpoints_3d/`。

原因：

- 2D SegACIL 已经使用 `checkpoints/`，其中有现有 JSON 结果保留策略。
- 3D 代码会产出 `.tar` checkpoint、TensorBoard events、文本日志和不同数据集子目录，和 2D 的 step 结构不同。
- 单独目录能避免 2D/3D checkpoint 命名和清理策略互相污染。

当前规则：

- 训练写入：`SegACIL/checkpoints_3d/{s3dis,scannet}/log_*`
- 旧脚本仍可传 `./log_s3dis/...` 或 `./log_scannet/...`，入口会自动映射。
- Git 只保留 `checkpoints_3d/.gitkeep`，不保留实际训练输出。

## 运行前提

当前服务器环境检查结果：

- Python：cp311
- torch：`2.5.1+cu121`
- CUDA runtime：`12.1`
- 已有：`torch`、`torchvision`、`sklearn`、`matplotlib`、`tensorboard`
- 缺失：`torch_cluster`、`plyfile`、`transforms3d`

因此当前环境还不能直接跑完整 3D 训练。随源码来的 `torch_cluster` wheel 是 `cp312 + cu118`，当前环境不能直接用。

数据也尚未准备好：当前 `SegACIL/datasets` 和 `SegACIL/data_root` 只看到 VOC 相关目录，没有 S3DIS/ScanNet blocks。3D 数据建议放在 `CFSSeg-code3D/datasets/` 或 `SegACIL/data_root_3d/`，两者都不进 Git。

## Code Review

CodeRabbit CLI 检查：

- CLI 版本：`0.6.4`
- 认证：已登录
- 远端审查命令：`coderabbit review --agent -t uncommitted`
- 结果：对包含 `CFSSeg-code3D` 的未提交改动返回 60 条 findings，其中包含 critical 和 major 级别问题。

已确认并局部修正的问题：

- `dataloaders/joint_loader.py`、`dataloaders/loader.py` 的 `super(MyDataset).__init__()` 写法已改为 `super().__init__()`。
- `dataloaders/s3dis.py`、`dataloaders/scannet.py` 中只构造 `NotImplementedError` 但没有 `raise` 的分支已改为显式抛错。
- S3DIS/ScanNet class name 文件读取已改为 context-managed `with open(...)`。

仍需后续单独处理的问题：

- `runs/test.py` 存在 `num_tasks` 未定义、`WRITER` 未初始化等 critical findings。
- 多个 `runs/*.py` / `eval*.py` 存在 CPU/CUDA device handling、`drop_last=True`、IoU 除零、hardcoded class count 等 major findings。
- `models/dgcnn_seg.py`、`utils/AnalyticLinear.py`、`utils/checkpoint_util.py` 还有设备兼容、数值稳定性和 checkpoint 目录创建问题。

本地审查结论：

- Critical：未发现仍存在的明文代理凭据；原 `test.sh` 明文凭据已移除。
- Warning：当前环境缺 `torch_cluster`、`plyfile`、`transforms3d`，且本地 wheel 不匹配当前环境，3D 训练运行前必须补齐依赖。
- Warning：S3DIS/ScanNet blocks 与 meta 文件当前未在服务器发现，必须先准备数据，否则 loader 会失败。
- Warning：`CFSSeg-code3D` 可以作为外部 3D 分支导入，但不应在未完成上述 review 修复前被视为已验收主线训练代码。
- Info：`test.py` 和 `zhanka.py` 不是主训练链路，后续如果要清理原作者临时文件，建议单独确认后再删。

## 验证记录

已执行并通过：

- `python -m compileall -q CFSSeg-code3D`
- `find CFSSeg-code3D/scripts -name '*.sh' -exec sh -n {} +`
- `sh -n CFSSeg-code3D/test.sh`
- 旧输出路径映射断言：`./log_s3dis/...` 和 `./log_scannet/...` 到 `checkpoints_3d/...`
- 环境变量映射断言：`CFSSeg3D_DATA_ROOT`、`CFSSeg3D_OUTPUT_ROOT`
- 敏感信息扫描：未命中旧代理账号或带账号密码的 URL
- `.gitignore` 命中检查：wheel、3D datasets、3D logs、`checkpoints_3d` 输出被忽略，`.gitkeep` 被保留
- `git diff --check -- .gitignore`

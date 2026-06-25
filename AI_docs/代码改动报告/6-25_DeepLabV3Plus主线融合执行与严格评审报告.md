# 6-25 DeepLabV3+ 主线融合执行与严格评审报告

> 工作目录：`/root/2TStorage/lyc/SegACIL_deeplabv3plus`  
> 集成分支：`feature/integrate-deeplabv3plus`  
> 基线分支：`main`，起点 commit `10ce0df`  
> 目标：把 DeepLabV3+ 干净融合为主线 `MODEL` 可选项，并保留原 DeepLabV3 行为、RHL 默认和后续方法实现入口。

## 1. 中断续接判断

本轮任务不是从零开始，而是续接上一次 Codex 额度/上下文中断后的收尾阶段。

现场复核结果：

| 检查项 | 结果 |
|---|---|
| 当前 feature worktree | `/root/2TStorage/lyc/SegACIL_deeplabv3plus` |
| 当前分支 | `feature/integrate-deeplabv3plus` |
| 主目录分支 | `/root/2TStorage/lyc/SegACIL` 仍为 `main` |
| 中断点 | 代码主体已实现，剩余未完成项为最后测试提交、CodeRabbit review、报告、main 合并 |
| 后台训练 PID | 上轮记录的 PID 已结束 |
| golden replay 输出 | 已生成 `final.pth` 和 `test_results_20260624_153910.json` |
| 中断前遗留改动 | 仅 `tests/test_air_feature_integration.py` 有 1 个未提交测试 |

因此判断：上个任务确实停在收尾阶段，表现符合 Codex 额度/上下文中断后的恢复状态；没有发现代码实现中途损坏或实验半成品需要重跑。

## 2. 前置结论

先把 V3+ 体系完整、自然地融合进 `main`，再在此基础上推进 RHL 五方法和自适应伪标签阈值，是可行且推荐的路线。

原因：

1. V3+ 当前已经不是一次性实验脚本，而是可以作为 `MODEL=deeplabv3plus_resnet101` 进入统一训练入口。
2. `AIR_FEATURE_SOURCE=auto` 可以按模型自动解析：V3 使用历史 `decoder`，V3+ 使用已验证最佳的 `aspp_up`。
3. 原 DeepLabV3 行为被测试锁定，不会因为 V3+ 接入而静默改变 baseline。
4. RHL 和伪标签方案后续只需要基于统一的模型/特征接口实现，不需要再维护一套 V3+ 特殊旁路。

需要保持的实验表述边界：

- `aspp_up` 不是新增可训练模块，不引入额外数据、监督或 loss。
- 它属于 DeepLabV3+ decoder 路径中的内部 feature tap：`ASPP -> bilinear upsample` 后、low-level 拼接和 decoder conv 前。
- 因此最佳体系应表述为 `DeepLabV3+ + AIR feature source aspp_up`，而不是“只把模型名从 V3 改成 V3+”。

## 3. Git 操作记录

本轮采用干净集成分支，而不是直接把历史 `feature/deeplabv3plus-control` 整体合入。

已形成的 commit 序列：

```text
35e8dad docs: finalize DeepLabV3+ integration plan
14fc116 feat: integrate DeepLabV3+ as a first-class AIR model
85dc571 fix: validate AIR resume metadata and harden manifests
5d56d8c feat: record complete experiment manifests
63a5e10 test: lock DeepLabV3 decoder feature behavior
```

关键原则：

- `feature/integrate-deeplabv3plus` 从 `main@10ce0df` 新建。
- 只选择性移植 V3+ 主线化需要的代码和文档。
- 没有整批带入旧 feature 分支中的历史 runner、临时实验代码或 class-cap 负实验逻辑。
- 主目录 `/root/2TStorage/lyc/SegACIL` 保持 `main`，V3+ worktree 保持集成分支，直到本地 merge。

本轮已完成本地合并：

```text
main merge commit: ddb9acc merge: integrate DeepLabV3+ mainline model
main status after merge: ahead origin/main by 7 commits
```

合并后已在 `/root/2TStorage/lyc/SegACIL` 的 `main` worktree 重新执行静态检查和 15 个单元测试，结果通过。

远端同步注意：

- 之前本机已有 `git push` 失败记录，原因是当前非交互环境缺少 GitHub HTTPS/SSH 凭据。
- 本地合并可以完成；远端 push 需要用户在终端配置 GitHub 登录后执行。

建议最终远端同步命令：

```bash
cd /root/2TStorage/lyc/SegACIL
git push origin main
```

如果需要保留集成分支到远端：

```bash
cd /root/2TStorage/lyc/SegACIL_deeplabv3plus
git push -u origin feature/integrate-deeplabv3plus
```

## 4. 代码实现范围

### 4.1 模型接入

涉及文件：

- `network/_deeplab.py`
- `network/modeling.py`
- `network/utils.py`

实现内容：

1. 增加 `DeepLabHeadV3Plus` 的正式 forward、feature extraction 和 AIR feature selection。
2. 在 `network.modeling` 中把 `deeplabv3plus_resnet50`、`deeplabv3plus_resnet101`、`deeplabv3plus_mobilenet` 接入统一 model map。
3. 在 `_SimpleSegmentationModel` 暴露：
   - `resolve_air_feature_source(source="auto")`
   - `forward_air_features(x, source="auto")`
4. DeepLabV3 默认 AIR source 保持 `decoder`。
5. DeepLabV3+ 默认 AIR source 为 `aspp_up`，同时支持显式消融：
   - `decoder`
   - `decoder_stride8`
   - `aspp`
   - `aspp_up`

### 4.2 Trainer 接入

涉及文件：

- `trainer/trainer.py`
- `utils/parser.py`
- `utils/run_manifest.py`

实现内容：

1. 新增 CLI 参数：

```bash
--air_feature_source {auto,decoder,decoder_stride8,aspp,aspp_up}
```

2. step1 不再通过 `self.model.classifier.head = nn.Identity()` 这种脆弱方式抽特征，而是调用模型自己的 `forward_air_features()`。
3. step0 dataloader 配置使用拷贝，不再临时修改 live `self.opts.curr_step`。
4. resume 后续 step 时校验 AIR checkpoint 中的 `feature_source`：
   - checkpoint 未记录时按旧模型兼容为 `decoder`；
   - 显式指定 source 与 checkpoint 不一致时直接报错；
   - `auto` 时沿用 checkpoint source。
5. 新增 `run_manifest.json`，记录模型、AIR source、base checkpoint hash、RHL 配置和关键训练配置。

### 4.3 Runner 接入

涉及文件：

- `run.sh`

实现内容：

1. `MODEL` 成为脚本级环境变量，默认仍为 `deeplabv3_resnet101`。
2. `AIR_FEATURE_SOURCE` 成为脚本级环境变量，默认 `auto`。
3. 保留 `BASE_SUBPATH` 与 `SUBPATH` 分离，避免读取 checkpoint 和写输出目录混淆。
4. 保留 RHL 默认：
   - `RHL_NORM=none`
   - `RHL_SEED=-1`
   - `RHL_STATS=0`
5. 明确提示：默认 `BASE_SUBPATH=20260606` 属于原 V3；跑 V3+ 时必须显式设置 V3+ step0 的 `BASE_SUBPATH`。

V3+ 推荐启动方式：

```bash
cd /root/2TStorage/lyc/SegACIL
PYTHON=/home/linyichen/miniconda3/envs/segacil/bin/python \
MODEL=deeplabv3plus_resnet101 \
AIR_FEATURE_SOURCE=auto \
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16 \
SUBPATH=20260625_v3plus_seq15-5 \
START_STEP=1 END_STEP=1 DEFAULT_BATCH_SIZE=16 \
BUFFER=8196 GAMMA=1 RANDOM_SEED=1 \
RHL_NORM=none RHL_SEED=-1 RHL_STATS=0 \
bash run.sh
```

## 5. 不纳入本次合并的内容

以下内容没有进入主线：

1. 旧 `feature/deeplabv3plus-control` 的历史专用 runner，例如临时的 `run_v3plus.sh` 逻辑。
2. class-cap / pixel-balance 一类此前未成为最终方案的实验分支。
3. 与 V3+ 主线化无关的 RHL 新方法实现。
4. 自适应伪标签阈值代码。

这样做的目的不是保守，而是保持主线干净：当前合并只解决“V3+ 作为一等 MODEL 和 AIR feature source 原生接入”。

## 6. 严格 code review 结果

### 6.1 CodeRabbit

本轮最终 review 命令：

```bash
coderabbit review --agent --base main --dir /root/2TStorage/lyc/SegACIL_deeplabv3plus
```

最终结果：

```text
review_completed
findings: 0
```

中间 review 曾指出的有效问题已经修复：

| 问题 | 处理 |
|---|---|
| `run_manifest.py` 在无 git metadata 环境下可能异常 | 已改为 fallback 到 `unknown` |
| 后续 step resume 未校验 AIR feature source | 已增加 mismatch 显式报错 |

中间 review 中关于 BgA `details["feature"]` 的建议未采纳。原因是 BgA 原有语义依赖空间 feature，强行改成 flattened logits 会破坏已有路径；本次只要求新增 AIR feature API，不改 BgA 历史返回约定。

### 6.2 人工 review

人工审查重点：

| 审查点 | 结论 |
|---|---|
| 原 V3 baseline 是否被改动 | 已用测试锁定旧 decoder/head 数学路径 |
| V3+ 是否作为一等 MODEL 接入 | 是，进入 `network.modeling` model map |
| `aspp_up` 是否隐藏硬编码 | 否，作为模型默认 source，可显式覆盖 |
| RHL 默认是否被改变 | 否，默认仍为 `none/-1/0` |
| step1 是否仍用完整 step0 teacher | 是，teacher checkpoint 正常加载，AIR 只更换 feature tap |
| checkpoint/manifest 是否可追溯 | 是，记录 model、source、base checkpoint hash、RHL 配置 |
| 旧 checkpoint 是否兼容 | 是，缺失 `feature_source` 时按 legacy `decoder` 处理 |

## 7. 验证结果

### 7.1 静态与单元测试

使用环境：

```bash
/home/linyichen/miniconda3/envs/segacil/bin/python
```

执行命令与结果：

| 命令 | 结果 |
|---|---|
| `bash -n run.sh run_rhl_norm.sh run_trs.sh` | 通过 |
| `python -m py_compile ...` | 通过 |
| `grep -n '[“”‘’]' ...` | 无非法弯引号 |
| `git diff --check` | 通过 |
| `python -m unittest discover -s tests -p 'test*.py' -v` | 15 tests, OK |

新增测试覆盖：

1. V3+ standard forward 与多种 AIR source shape。
2. `auto` 在 V3 下解析为 `decoder`。
3. `auto` 在 V3+ 下解析为 `aspp_up`。
4. V3 不支持 `aspp_up` 时显式失败，不做伪别名。
5. 原 V3 forward 与旧 decoder/head 数学路径完全一致。
6. AIR 构造时保存 source 且保留 RHL 参数。
7. resume checkpoint source mismatch 显式报错。
8. legacy AIR checkpoint 缺失 source 时兼容为 `decoder`。
9. manifest 写入 model、source、checkpoint hash、RHL 和关键训练配置。

### 7.2 模型 smoke

已完成的真实模型 smoke：

| 模型 | 输入 | 输出 | AIR source | AIR feature |
|---|---|---|---|---|
| `deeplabv3_resnet50` | `1x3x65x65` | `1x16x9x9` | `decoder` | `1x256x9x9` |
| `deeplabv3plus_resnet50` | `1x3x65x65` | `1x16x17x17` | `aspp_up` | `1x256x17x17` |

真实 checkpoint smoke：

| checkpoint | 结论 |
|---|---|
| V3 step0 `/root/2TStorage/lyc/SegACIL/checkpoints/20260606/.../deeplabv3_resnet101_...pth` | `auto -> decoder`，feature 与 decoder 完全一致 |
| V3+ step0 `checkpoints/20260614_v3plus_voc15-5_seq_bs32-16/.../deeplabv3plus_resnet101_...pth` | `auto -> aspp_up`，513 输入下 feature 为 `1x256x129x129` |

### 7.3 Golden replay

复现命令核心配置：

```bash
MODEL=deeplabv3plus_resnet101
AIR_FEATURE_SOURCE=auto
SUBPATH=20260624_v3plus_integration_golden_replay_14fc116
BASE_SUBPATH=20260614_v3plus_voc15-5_seq_bs32-16
START_STEP=1
END_STEP=1
DEFAULT_BATCH_SIZE=16
BUFFER=8196
GAMMA=1
RANDOM_SEED=1
RHL_NORM=none
RHL_SEED=-1
RHL_STATS=0
```

输出路径：

```text
checkpoints/20260624_v3plus_integration_golden_replay_14fc116/voc/15-5/sequential/step1/test_results_20260624_153910.json
logs/integration/20260624_v3plus_integration_golden_replay_14fc116.log
```

结果对照：

| 指标 | 既有目标值 | 本次 replay | 差值 |
|---|---:|---:|---:|
| `0 to 15 mIoU` | 0.7792763236 | 0.7792763236 | 0 |
| `16 to 20 mIoU` | 0.4612870137 | 0.4612870137 | 0 |
| `Mean IoU` | 0.7035645831 | 0.7035645831 | 0 |

结论：当前主线化实现可以复现 V3+ 最佳体系的 `0.7036`，不是只做到“代码能跑”。

说明：这次 replay 记录的 manifest commit 为 `14fc116`，因为 replay 在后续 manifest/review/test commit 前启动。后续 commit 只涉及 manifest 完整性、resume 校验和测试/报告，不改变 V3+ forward 或 AIR feature 数学路径。

## 8. 对后续 RHL 与自适应阈值的影响

当前 V3+ 体系可以作为后续 RHL 五方法和自适应伪标签阈值方案的 base。

推荐约束：

1. 新方法默认使用统一入口，不再新建 V3+ 专用脚本。
2. V3+ 实验使用：

```bash
MODEL=deeplabv3plus_resnet101
AIR_FEATURE_SOURCE=auto
BASE_SUBPATH=<V3+ step0 checkpoint subpath>
```

3. 需要和原 V3 做公平对照时，同时报告：
   - `deeplabv3_resnet101 + auto(decoder)`
   - `deeplabv3plus_resnet101 + auto(aspp_up)`
4. RHL 或伪标签自身的新增变量必须独立开关，不能混入 V3+ 集成代码。
5. 后续论文/报告应把 V3+ 最佳 pipeline 写为 `DeepLabV3+ with aspp_up AIR feature`，避免声称唯一变量只是 backbone/head 名称。

## 9. 当前风险与处理

| 风险 | 当前状态 | 处理 |
|---|---|---|
| 远端 push 认证 | 本机非交互环境可能缺少 GitHub 凭据 | 本地 merge 后需用户终端执行 push |
| V3+ step1 显存 | bs16 约 56GB，bs32 不适合当前单卡 | 默认继续 bs16，已验证 batch size 不影响精度结论 |
| V3+ 与 V3 公平性表述 | `aspp_up` 是 feature tap 改动 | 报告和 manifest 显式记录 source |
| 历史 runner 混淆 | 旧 `run_v3plus.sh` 曾误导 source | 主线只推荐统一 `run.sh` + `MODEL/AIR_FEATURE_SOURCE` |

## 10. 结论

DeepLabV3+ 已达到可以进入主线的条件：

1. 作为 `MODEL` 选项自然接入，不是畸形旁路。
2. `aspp_up` 作为模型感知的 AIR source 默认，且可显式消融。
3. 原 DeepLabV3 路径有测试保护。
4. RHL 默认和后续方法入口未被污染。
5. CodeRabbit 最终 review 无 findings。
6. golden replay 精确复现 `Mean IoU=0.7035645831`。

本地 `main` 已完成合并并通过验证。剩余动作是远端同步：若当前非交互环境仍缺少 GitHub 凭据，需要用户在终端完成登录后执行 `git push origin main`。

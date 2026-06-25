# DeepLabV3+ 原生主线集成方案与执行工作流

> 日期：2026-06-24  
> 当前 main：`10ce0df`，与 `origin/main` 同步，工作区 clean  
> 当前 feature：`b62f73c`，与 `origin/feature/deeplabv3plus-control` 同步，工作区 clean  
> 目标：把已验证的 DeepLabV3+ 能力自然集成到最新 main，使其成为正式 `MODEL` 选项，并保留 DeepLabV3 的现有默认行为。

## 1. 先给结论

1. 当前 main **不是只实现了** `deeplabv3_resnet101`。`network/modeling.py` 已注册 ResNet50/101、MobileNet、DeepLabV3、DeepLabV3+ 和 BgA 等多个名称。
2. 但当前 main 的正式 `run.sh` 把 `MODEL` 固定为 `deeplabv3_resnet101`，main 中旧 `DeepLabHeadV3Plus` 也不满足当前 step0 `(logits, details)` 与 step1 AIR feature API 的完整契约。因此 V3+ 目前只是“模型工厂里有名字”，并非“主线端到端一等选项”。
3. 6-23 文档的“在最新 main 上建立 integration 分支、选择性移植”方向正确。需要微调的是：以模型 head 自描述 feature 能力，用统一 `auto` resolver 解析 source；不要在脚本或 Trainer 中硬编码模型名判断。
4. 推荐配置界面：`MODEL` 和 `AIR_FEATURE_SOURCE` 都是独立参数；`AIR_FEATURE_SOURCE=auto` 时，V3 自动取 `decoder`，V3+ 自动取 `aspp_up`。用户仍可显式指定 `aspp`、`decoder` 等做消融。
5. `aspp_up` 不是新增的可训练模块。它是 V3+ 内部已有的 `ASPP -> bilinear upsample` 中间张量，位于 low-level 拼接和 decoder 之前；项目只是在 step1 把它选作 AIR 输入。
6. 当前 feature 上的 `0.7036` 已由原始 JSON 确认，但在 integration 分支实际完成 golden replay 前，不能承诺“合并后必然复现”。应把 replay 作为工程合并硬验收。
7. golden replay 通过后，V3+ 可以作为 main 的一等可选 stronger base。multi-seed 和方法迁移实验决定它能否进一步成为 preferred/canonical base，不应阻塞代码能力进入 main。

## 2. 当前代码事实

### 2.1 为什么不能简单回答“MODEL 只有 V3”

main 的 `network/modeling.py` 已注册：

```text
deeplabv3_resnet50
deeplabv3plus_resnet50
deeplabv3_resnet101
deeplabv3plus_resnet101
deeplabv3_mobilenet
deeplabv3plus_mobilenet
deeplabv3bga_resnet101
```

因此从 model factory 层看，答案是“否”。

但 main 的 `run.sh` 当前写死：

```bash
MODEL="deeplabv3_resnet101"
```

而 main 的旧 `DeepLabHeadV3Plus.forward()` 只返回 logits，通用 `_SimpleSegmentationModel.forward()` 却要求 classifier 返回 `(logits, details)`。step1 又需要明确、稳定的 256-channel dense feature。故从正式训练链路看，目前只有 V3 路径是项目实际长期使用的有效默认；V3+ 名称不等于完整可用能力。

### 2.2 feature 分支已经补齐什么

feature 分支已验证以下通用能力：

- V3+ 使用共享 decoder 和 pixel-wise Linear classifier，满足 step0 logits 契约；
- V3/V3+ head 暴露 `extract_features()` 与 `select_air_feature()`；
- segmentation model 暴露 `forward_air_features()`；
- AIR 显式接收 `feature_source`；
- V3+ 支持 `decoder`、`decoder_stride8`、`aspp`、`aspp_up`；
- 真实 checkpoint source sweep 和单元测试均已完成。

这些改动是在模型边界定义统一 feature API，不是为 V3+ 在 Trainer 中开一条私有旁路，方向是泛用的。

### 2.3 当前 feature 的一个重要脚本事实

`run_v3plus.sh` 当前没有传 `--air_feature_source`，而 feature parser 默认仍为 `decoder`。所以它是历史 V3+ control runner，不是 `0.7036` 的复现入口。

当前最佳结果由 `run_v3plus_air.sh` 显式传递以下配置得到：

```text
MODEL=deeplabv3plus_resnet101
AIR_FEATURE_SOURCE=aspp_up
AIR_PIXEL_BALANCE=none
BUFFER=8196
GAMMA=1
RANDOM_SEED=1
```

主线集成后，通用 `run.sh` 的 `auto` 应消除“只换 MODEL 却静默落到 decoder”的风险。

## 3. `aspp_up` 到底是什么

### 3.1 step0 中的位置

V3+ 的 step0 正常分割路径是：

```text
image
  -> ResNet101 backbone
  -> high-level out
  -> ASPP                         [256 channels, stride 8]
  -> bilinear upsample            [aspp_up, 对齐 low-level 空间尺寸]
  -> concat(project(low_level))
  -> V3+ decoder                  [256 channels, stride 4]
  -> Linear classifier
  -> segmentation logits
```

`aspp_up` 是这条正常 V3+ decoder 路径中的已有中间结果。它没有新增参数、监督、数据或 loss。

### 3.2 step1 中的位置

step1 会加载并冻结完整 step0 segmentation model，再把选定 dense feature 送入：

```text
selected dense feature
  -> RandomBuffer
  -> RecursiveLinear
  -> step1 logits
```

选择 `aspp_up` 后，step1 使用高层 ASPP 语义特征，同时保留较密的 stride-4 标签对齐；它绕过 low-level 拼接和 V3+ decoder。low-level project/decoder 虽不直接参与这次 AIR forward，但它们在 step0 联合监督训练时影响了整个模型的优化结果。

### 3.3 应如何命名和归属

准确表述是：

> `aspp_up` 是 V3+ 的内部 feature tap，也是 V3+ 到 CFSSeg/AIR 的无参数空间适配策略。

它可以作为“V3+ 在本项目中的推荐配套接口”一起使用，但不能声称它是标准 DeepLabV3+ 架构新增模块，也不能把 `0.7036` 全部归因于只改了模型名。

公平报告至少保留：

| 配置 | 0-15 | 16-20 | All | 解释 |
|---|---:|---:|---:|---|
| V3+ decoder | 0.7815 | 0.3959 | 0.6897 | 原 V3+ decoder 接口 |
| V3+ aspp | 0.7771 | 0.4510 | 0.6995 | 高层 feature-tap 消融 |
| V3+ aspp_up | 0.7793 | 0.4613 | 0.7036 | 最佳 CFSSeg-compatible pipeline |

## 4. 三种集成方式比较

### 方案 A：选择 V3+ 就在 Trainer 中强制 `aspp_up`

做法：根据模型名写死 `deeplabv3plus_* -> aspp_up`，不提供 source 参数。

优点：调用简单。

问题：策略不可见，无法方便复现 `decoder/aspp` 消融；新增模型要继续修改 Trainer；日志若不记录 source，实验不可追溯。这是畸形特判，不推荐。

### 方案 B：MODEL 与 source 完全独立，默认始终为 `decoder`

做法：只增加 `--air_feature_source`，用户自己保证组合正确。

优点：最灵活，代码简单。

问题：只切换 `MODEL=deeplabv3plus_resnet101` 时会静默得到已知较差的 `decoder=0.6897`，极易误判 V3+。当前 feature 正是这个风险。可用于底层 API，不适合作为用户默认体验。

### 方案 C：独立开关 + 模型自描述 `auto`，推荐

做法：

```text
MODEL=deeplabv3_resnet101
AIR_FEATURE_SOURCE=auto
```

每个 classifier/head 声明：

```python
default_air_feature_source = "decoder"  # V3
supported_air_feature_sources = ("decoder", "aspp")

default_air_feature_source = "aspp_up"  # V3+
supported_air_feature_sources = (
    "decoder", "decoder_stride8", "aspp", "aspp_up"
)
```

统一 resolver 的规则：

```text
requested=auto     -> 使用当前模型声明的 default
requested=显式值   -> 校验是否在 supported 中，通过后原样使用
不支持的组合       -> 启动时显式失败
```

这是推荐方案，因为：

- 默认行为安全：V3 保持历史 `decoder`，V3+ 得到已验证 `aspp_up`；
- 消融仍可显式覆盖；
- Trainer 不感知具体模型名；
- 新模型通过声明能力接入，不新增散落条件分支；
- requested/resolved 值可同时记录，复现实验时没有隐式状态。

## 5. 能否复现 `0.7036`

### 5.1 已确认的部分

原始结果文件：

```text
checkpoints/20260622_v3plus_air_aspp_up/voc/15-5/sequential/step1/
test_results_20260622_204635.json
```

精确结果：

```text
0-15 mIoU  = 0.7792763235836693
16-20 mIoU = 0.4612870136816847
All mIoU   = 0.7035645831308158
```

实际加载的 step0 checkpoint：

```text
checkpoints/20260614_v3plus_voc15-5_seq_bs32-16/voc/15-5/sequential/step0/
deeplabv3plus_resnet101_voc_15-5_step_0_sequential.pth
sha256=4bd0b63ed535a2f1c5871f073b7e45e7bdfcda703b5d148ab42b27fc0d6928b7
```

运行日志确认 `batch_size=16`、`buffer=8196`、`gamma=1`、`random_seed=1`、`air_feature_source=aspp_up`。

### 5.2 不能提前承诺的部分

尚未存在的 integration commit 没有跑过该实验。因此现在只能确认“设计能够保持同一数学路径”，不能把未来结果当成已验证事实。

会破坏复现的典型因素包括：

- main 的 `BUFFER=8192` 覆盖 feature 实验的 `8196`；
- `auto` 未解析或 resolved source 未传入 AIR；
- 合并时覆盖 main/feature 的 RHL 随机初始化逻辑；
- 加载了 `final.pth` 或错误 subpath，而不是上述命名 checkpoint；
- 修改 seed、数据 list、transform、PyTorch/CUDA 环境；
- 把 class-cap 或 RHL normalization 意外带入。

### 5.3 必须执行的确认方法

在 integration 分支复用上述命名 checkpoint，只重跑 step1：

```text
MODEL=deeplabv3plus_resnet101
AIR_FEATURE_SOURCE=auto
resolved_source=aspp_up
BUFFER=8196
GAMMA=1
RHL_NORM=none
AIR_PIXEL_BALANCE=none
RANDOM_SEED=1
```

验收：

- 首选：三个 mIoU 与原 JSON 的绝对差均不超过 `1e-6`；
- 日志必须打印 requested=`auto`、resolved=`aspp_up` 和 checkpoint SHA256；
- 若环境造成非确定性，应先定位来源，再设有依据的容差；
- replay 未通过前，不合并 main，也不声称已复现。

## 6. 最终代码设计

### 6.1 配置面

```text
--model
--air_feature_source {auto,decoder,decoder_stride8,aspp,aspp_up}
```

默认：

```text
model=deeplabv3_resnet101
air_feature_source=auto
```

V3 的 resolved source 仍为 `decoder`，因此默认数学路径不变。

### 6.2 模型层

修改 `network/_deeplab.py`：

1. 选择性移植 V3+ shared decoder + Linear head contract。
2. V3/V3+ 实现 `extract_features()` 和 `select_air_feature()`。
3. head 声明 `default_air_feature_source` 与 `supported_air_feature_sources`。
4. V3 的 source 不应把 `aspp_up` 静默别名成 `aspp`；不支持就显式失败。

修改 `network/utils.py`：

1. 保留标准 `forward()` 契约。
2. 增加 `resolve_air_feature_source(requested)`。
3. 增加 `forward_air_features(x, resolved_source)`。
4. classifier 不实现接口时明确抛错。

### 6.3 Trainer/AIR 层

修改 `trainer/trainer.py`：

1. step1 加载完整 step0 model 后调用 resolver。
2. AIR 只接收 resolved source，不解析模型名。
3. 同时打印 requested/resolved source。
4. 保留 main 的 RHL norm/seed/stats，不用 feature 文件覆盖 main 文件。
5. 使用 `step0_opts = copy.deepcopy(self.opts)` 初始化 step0 dataloader，避免改写共享 `self.opts.curr_step`。
6. class-cap 不进入本次主线核心提交；其负结果代码留在实验分支或后续独立工具提交。

### 6.4 Runner 层

修改通用 `run.sh`：

```bash
MODEL="${MODEL:-deeplabv3_resnet101}"
AIR_FEATURE_SOURCE="${AIR_FEATURE_SOURCE:-auto}"
```

并显式传入：

```bash
--model "$MODEL"
--air_feature_source "$AIR_FEATURE_SOURCE"
```

规则：

- `BASE_SUBPATH` 与 `SUBPATH` 分离；
- V3/V3+ 必须使用各自 checkpoint；
- `BUFFER`、`GAMMA`、RHL 配置都可覆盖并打印；
- 不在脚本中解析 `auto`，真实 resolved source 由已加载模型决定；
- `run_v3plus_air.sh` 保留为历史结果复现脚本，不再承担正式通用入口职责。

### 6.5 可复现记录

当前 checkpoint 只保存 model/optimizer/score，不能完整恢复实验配置。集成时至少新增每次 run 的 manifest JSON，记录：

```text
git_commit
model
requested_air_feature_source
resolved_air_feature_source
dataset/task/setting/curr_step
base_subpath/base_checkpoint_path/base_checkpoint_sha256
buffer/gamma/random_seed
rhl_norm/rhl_seed
batch_size/output_stride
pixel_balance
```

不要求为本次任务重构整个 checkpoint 格式，但 manifest 必须与输出目录一一对应。

## 7. 供下一位 Codex 直接执行的工作流

### Phase 0：建立隔离集成工作区

```bash
cd /root/2TStorage/lyc/SegACIL
git status --short --branch
git fetch origin
git pull --ff-only origin main
git worktree add -b feature/integrate-deeplabv3plus \
  ../SegACIL_integrate_v3plus main
cd ../SegACIL_integrate_v3plus
```

不要直接 merge 整个 `feature/deeplabv3plus-control`。以 main 为底，按下列提交边界选择性实现。

### Phase 1：先写接口和回归测试

新增或移植测试，先让测试失败：

1. V3/V3+ 标准 forward 返回 `(logits, details)`，shape 正确。
2. V3 `auto -> decoder`。
3. V3+ `auto -> aspp_up`。
4. V3+ 四个显式 source shape 正确。
5. V3 对不支持 source 显式失败，不允许别名偷换。
6. AIR 使用 resolved source。
7. V3 历史默认 feature 逐元素等价。

### Phase 2：实现模型自描述 feature API

按顺序修改：

```text
network/_deeplab.py
network/utils.py
```

先完成模型级测试，不触碰 Trainer 的 RHL 逻辑。

### Phase 3：接入 main Trainer 与 parser

按 main 当前代码人工合并：

```text
utils/parser.py
trainer/trainer.py
```

要求：

- parser 增加 `auto`；
- Trainer 只做一次 requested -> resolved；
- main 的 RHL norm/seed/stats 全部保留；
- source 和 RHL 参数分别负责不同机制；
- 修复 step0 dataloader 对共享 opts 的原地改写；
- 不顺带移植 class-cap。

### Phase 4：通用 runner 与 manifest

修改 `run.sh`，新增轻量 manifest 写入。运行前打印所有关键配置和输出目录。保留现有 V3 默认、TRS runner 与历史专用 runner，不覆盖用户实验脚本。

### Phase 5：静态和 CPU smoke

```bash
python -m py_compile network/_deeplab.py network/utils.py \
  trainer/trainer.py utils/parser.py
bash -n run.sh
python -m unittest tests.test_air_feature_sources
grep -n '[“”‘’]' network/_deeplab.py network/utils.py \
  trainer/trainer.py utils/parser.py run.sh
```

再跑 V3/V3+ CPU forward，检查 logits 和各 source 的 channel/stride。

### Phase 6：GPU smoke

运行前：

```bash
nvidia-smi
export CUDA_MPS_PIPE_DIRECTORY=/home/linyichen/.mps_bypass
export TMPDIR=/root/2TStorage/tmp
```

执行：

1. V3 step0 one-batch forward/validation。
2. V3 step1 `auto -> decoder` one-batch AIR fit。
3. V3+ step0 one-batch forward/validation。
4. V3+ step1 `auto -> aspp_up` one-batch AIR fit。
5. 错误 model/source/checkpoint 组合启动时失败。

### Phase 7：golden replay

复用第 5 节的命名 step0 checkpoint，使用独立 `SUBPATH`，只跑 step1。禁止复用已有正式结果目录。

完成后核对：

- 输出 JSON 存在；
- old/new/all 与 golden 指标比较；
- manifest 中 checkpoint SHA256 和 requested/resolved source 正确；
- 日志没有启用 class-cap、RHL norm 或错误 buffer。

### Phase 8：工程合并判定

以下全部满足即可提交 integration 分支并合并 main：

```text
V3 默认路径无回归
V3+ golden replay 复现 0.7036
run.sh 可切换 MODEL
auto 和显式 source 均可追溯
main RHL 功能保留
静态检查、单测、CPU/GPU smoke 通过
工作区 clean
```

建议提交拆分：

```text
1. test: define model-aware AIR feature contract
2. feat: integrate DeepLabV3+ and AIR source resolver
3. feat: expose model/source in generic runner and manifest
4. docs: record regression and golden replay evidence
```

### Phase 9：科学晋级实验，不阻塞工程合并

工程合并后再做：

1. V3 与 V3+ `aspp_up` paired seeds 1/2/3，固定 checkpoint 生成规则、buffer、gamma 和 batch protocol。
2. 选一个优先级最高的方法，做 `V3/V3+ x method off/on` 2x2 pilot。
3. 若 V3+ 的均值优势稳定且方法收益不反转，再把它设为后续 preferred base。
4. 原论文公平对比表仍保留 V3 canonical baseline，不静默替换历史结果。

## 8. 最终决策

采用方案 C：

```text
通用 MODEL 选项
  + 独立 AIR_FEATURE_SOURCE 开关
  + 模型 head 自描述默认与支持集合
  + auto 在 V3 下解析为 decoder
  + auto 在 V3+ 下解析为 aspp_up
  + 显式 source 保留消融能力
```

这能把 V3+ 自然融合为主线一等模型，而不为它建立畸形旁路。`aspp_up` 与 V3+ 作为推荐 pipeline 配套，但在代码和论文语义上仍保持独立、可见、可覆盖。主线集成完成与否以 golden replay 是否复现 `0.7036` 为准，而不是以代码能否启动为准。

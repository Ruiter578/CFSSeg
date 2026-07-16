# CFSSeg 3D 数据集准备与检查指南

日期：2026-07-15
目标：把 S3DIS 和 ScanNet v2 准备到 `SegACIL/CFSSeg-code3D` 可直接复现 CFSSeg 3D 实验的状态。

## 目录约定

本文默认路径如下。换服务器时只需要改 `LINWEI_ROOT`。

```bash
export LINWEI_ROOT=/TRS-SAS/linwei
export SEGACIL_ROOT=$LINWEI_ROOT/SegACIL
export CODE3D_ROOT=$SEGACIL_ROOT/CFSSeg-code3D
export RAW_ROOT=$LINWEI_ROOT/datasets_raw
```

约定：

- 原始下载和解压放在 `$RAW_ROOT`。
- CFSSeg 3D 实际读取的数据入口放在 `$CODE3D_ROOT/datasets`。
- 训练输出由 `main.py` 统一映射到 `$SEGACIL_ROOT/checkpoints_3d/{s3dis,scannet}`。
- `CFSSeg-code3D/datasets` 和 `checkpoints_3d` 实际内容不提交 Git。

## 环境检查

在 `segacil` conda 环境中执行：

```bash
conda activate segacil
cd "$CODE3D_ROOT"

python - <<'PY'
import numpy as np, torch, torch_cluster, plyfile, transforms3d
from torch_cluster import fps
from torch.utils.tensorboard import SummaryWriter
print('numpy', np.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('imports OK')
PY
```

当前服务器已验证版本：

- `numpy 1.26.4`
- `torch 2.1.2+cu118`
- `torch_cluster 1.6.3+pt21cu118`
- `plyfile 1.1.3`
- `transforms3d 0.4.2`
- `tensorboard 2.21.0`

如果缺包，按当前环境补装：

```bash
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "torch_cluster==1.6.3+pt21cu118" \
  -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
python -m pip install --no-cache-dir plyfile transforms3d tensorboard
```

注意：随源码带入的 `torch_cluster-1.6.3+pt25cu118-cp312-cp312-linux_x86_64.whl` 不适用于当前 `python 3.10 + torch 2.1.2` 环境。

## S3DIS 准备流程

推荐从 ModelScope 的 S3DIS LFS 仓库获取原始压缩包。Git LFS 是 Git Large File Storage，用来管理大文件；安装在系统或用户级都可以，不会改当前 feature 分支代码。

```bash
mkdir -p "$RAW_ROOT"
cd "$RAW_ROOT"

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/datasets/OpenDataLab/S3DIS.git S3DIS_modelscope
cd S3DIS_modelscope
git lfs pull --include="raw/S3DIS.tar.gz"
```

解压并定位 Stanford 原始目录：

```bash
mkdir -p "$RAW_ROOT/S3DIS_extracted"
tar -xzf "$RAW_ROOT/S3DIS_modelscope/raw/S3DIS.tar.gz" -C "$RAW_ROOT/S3DIS_extracted"

find "$RAW_ROOT/S3DIS_extracted" -type f -name '*.zip' | sort
```

如果 `find ... -name Stanford3dDataset_v1.2_Aligned_Version` 没有输出，通常说明外层 tar 解开后还有内层 zip 没解。继续执行：

```bash
cd "$RAW_ROOT/S3DIS_extracted"
unzip -q Stanford3dDataset_v1.2_Aligned_Version.zip

export S3DIS_RAW=$(find "$RAW_ROOT/S3DIS_extracted" -type d -name 'Stanford3dDataset_v1.2_Aligned_Version' | head -1)
test -d "$S3DIS_RAW"
```

建立 CFSSeg 3D 数据入口和 class names：

```bash
mkdir -p "$CODE3D_ROOT/datasets/S3DIS/meta"
ln -sfn "$S3DIS_RAW" "$CODE3D_ROOT/datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version"

cat > "$CODE3D_ROOT/datasets/S3DIS/meta/s3dis_classnames.txt" <<'EOF'
ceiling
floor
wall
beam
column
window
door
table
chair
sofa
bookcase
board
clutter
EOF
```

当前 S3DIS 原始数据中发现过一处脏行：`Area_5/hallway_6/Annotations/ceiling_1.txt` 第 180389 行把两个 RGB 数字粘成了 `185187`。如果检查命中，先备份再修：

```bash
bad_file="$S3DIS_RAW/Area_5/hallway_6/Annotations/ceiling_1.txt"
sed -n '180389p' "$bad_file"
cp -n "$bad_file" "$bad_file.bak"
sed -i '180389s/185187/185 187/' "$bad_file"
sed -n '180389p' "$bad_file"
```

预处理：

```bash
cd "$CODE3D_ROOT"

python -u preprocess/collect_s3dis_data.py \
  --data_path datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version \
  2>&1 | tee datasets/S3DIS/preprocess_collect_s3dis.log

python -u preprocess/room2blocks.py \
  --dataset s3dis \
  --data_path datasets/S3DIS/scenes \
  --block_size 1 \
  --stride 1 \
  --min_npts 1000 \
  2>&1 | tee datasets/S3DIS/preprocess_room2blocks_s3dis.log

ln -sfn blocks_bs1.0_s1.0 datasets/S3DIS/blocks_bs1_s1
```

S3DIS 可用性检查：

```bash
cd "$CODE3D_ROOT"
printf 'S3DIS scenes: '; find datasets/S3DIS/scenes -maxdepth 1 -type f -name '*.npy' | wc -l
printf 'S3DIS blocks: '; find -L datasets/S3DIS/blocks_bs1_s1/data -maxdepth 1 -type f -name '*.npy' | wc -l

python - <<'PY'
from dataloaders.s3dis import S3DISDataset
from dataloaders.loader import MyDataset
D=S3DISDataset(0, '12-1', 'datasets/S3DIS/blocks_bs1_s1')
train=MyDataset('datasets/S3DIS/blocks_bs1_s1', D.base_classes + D.incre_classes,
                D.base_classes, D.class2scans, 0, mode='train', valid_set='Area_5',
                num_point=2048, pc_attribs='xyzrgbXYZ', pc_augm=False,
                pc_augm_config={'scale':0,'rot':0,'mirror_prob':0,'jitter':0})
pt,label=train[0]
print('dataset_init_ok')
print('classes', D.classes, len(D.base_classes), len(D.incre_classes))
print('train_len', len(train))
print('sample', tuple(pt.shape), tuple(label.shape), label.unique().tolist())
PY
```

当前服务器通过结果：

- `S3DIS scenes: 272`
- `S3DIS blocks: 7547`
- `train_len 5315`
- sample shape `(9, 2048)`, label shape `(2048,)`

## ScanNet v2 准备流程

结论：使用 ScanNet v2。CFSSeg 3D 不需要完整 ScanNet 多 TB 数据，也不需要官方脚本默认下载的 `.sens` 或 2D zip。当前已验证只需要每个 scan 的三个文件：

- `${sid}_vh_clean_2.ply`
- `${sid}_vh_clean_2.0.010000.segs.json`
- `${sid}.aggregation.json`

还需要：

- `scannetv2-labels.combined.tsv`
- `scannetv2_val.txt`

官方脚本如果卡在 `_2d-instance.zip`，可以停止；那是全量下载路径，不是 CFSSeg 3D 最小复现路径。

创建最小下载脚本：

```bash
cat > "$RAW_ROOT/download_scannet_v2_cfssseg_needed.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${RAW_ROOT:-/TRS-SAS/linwei/datasets_raw/ScanNet_v2_cfssseg_needed}
CODE3D_ROOT=${CODE3D_ROOT:-/TRS-SAS/linwei/SegACIL/CFSSeg-code3D}
BASE_URL=${BASE_URL:-http://kaldir.vc.in.tum.de/scannet}
SCAN_LIST="$RAW_ROOT/scannetv2_trainval_scan_ids.txt"

mkdir -p "$RAW_ROOT/scans" "$CODE3D_ROOT/datasets/ScanNet/meta"

fetch() {
  local url=$1
  local out=$2
  local part="$out.part"
  local done_marker="$out.done"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" && -f "$done_marker" ]]; then
    return 0
  fi
  rm -f "$out" "$done_marker"
  local attempt
  for attempt in $(seq 1 50); do
    echo "[$(date '+%F %T')] fetch attempt $attempt: $url"
    if wget -c --timeout=60 --tries=5 --retry-connrefused --waitretry=10 --read-timeout=60 -O "$part" "$url"; then
      mv "$part" "$out"
      touch "$done_marker"
      return 0
    fi
    sleep 20
  done
  echo "ERROR: failed after repeated attempts: $url" >&2
  return 1
}

fetch "$BASE_URL/v2/scans.txt" "$SCAN_LIST"
fetch "$BASE_URL/v2/tasks/scannetv2-labels.combined.tsv" "$RAW_ROOT/scannetv2-labels.combined.tsv"
fetch "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt" "$RAW_ROOT/scannetv2_val.txt"

ln -sfn "$RAW_ROOT/scans" "$CODE3D_ROOT/datasets/ScanNet/scans"
cp -f "$RAW_ROOT/scannetv2-labels.combined.tsv" "$CODE3D_ROOT/datasets/ScanNet/meta/scannetv2-labels.combined.tsv"
cp -f "$RAW_ROOT/scannetv2_val.txt" "$CODE3D_ROOT/datasets/ScanNet/scannetv2_val.txt"

suffixes=(
  "_vh_clean_2.ply"
  "_vh_clean_2.0.010000.segs.json"
  ".aggregation.json"
)

total=$(wc -l < "$SCAN_LIST")
i=0
while IFS= read -r sid; do
  [[ -z "$sid" ]] && continue
  i=$((i + 1))
  echo "[$(date '+%F %T')] [$i/$total] $sid"
  mkdir -p "$RAW_ROOT/scans/$sid"
  for suffix in "${suffixes[@]}"; do
    fetch "$BASE_URL/v2/scans/$sid/$sid$suffix" "$RAW_ROOT/scans/$sid/$sid$suffix"
  done
done < "$SCAN_LIST"

echo "[$(date '+%F %T')] ScanNet v2 CFSSeg-needed download complete"
EOF
chmod +x "$RAW_ROOT/download_scannet_v2_cfssseg_needed.sh"
```

下载：

```bash
mkdir -p "$RAW_ROOT/ScanNet_v2_cfssseg_needed"
RAW_ROOT="$RAW_ROOT/ScanNet_v2_cfssseg_needed" \
CODE3D_ROOT="$CODE3D_ROOT" \
  "$RAW_ROOT/download_scannet_v2_cfssseg_needed.sh" \
  2>&1 | tee "$RAW_ROOT/scannet_v2_cfssseg_needed_download.log"
```

下载完整性检查：

```bash
raw_scannet="$RAW_ROOT/ScanNet_v2_cfssseg_needed"
printf 'scan ids: '; wc -l < "$raw_scannet/scannetv2_trainval_scan_ids.txt"
printf 'scene dirs: '; find "$raw_scannet/scans" -mindepth 1 -maxdepth 1 -type d | wc -l
printf 'complete triples: '
python - <<'PY'
from pathlib import Path
root = Path('/TRS-SAS/linwei/datasets_raw/ScanNet_v2_cfssseg_needed')
ids = [x.strip() for x in (root/'scannetv2_trainval_scan_ids.txt').read_text().splitlines() if x.strip()]
suffixes = ['_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json']
ok = 0
missing = []
for sid in ids:
    files = [root/'scans'/sid/f'{sid}{s}' for s in suffixes]
    if all(p.is_file() and p.stat().st_size > 0 for p in files):
        ok += 1
    else:
        missing.append(sid)
print(ok)
if missing:
    print('missing', missing[:20])
PY
printf 'part files: '; find "$raw_scannet" -name '*.part' | wc -l
du -sh "$raw_scannet"
```

当前服务器通过结果：

- `scan ids: 1513`
- `scene dirs: 1513`
- `complete triples: 1513`
- `part files: 0`
- raw size 约 `9.9G`

建立 ScanNet class names：

```bash
mkdir -p "$CODE3D_ROOT/datasets/ScanNet/meta"
cat > "$CODE3D_ROOT/datasets/ScanNet/meta/scannet_classnames.txt" <<'EOF'
unannotated
wall
floor
chair
table
desk
bed
bookshelf
sofa
sink
bathtub
toilet
curtain
counter
door
window
shower curtain
refrigerator
picture
cabinet
otherfurniture
EOF
```

ScanNet 预处理：

```bash
cd "$CODE3D_ROOT"

python -u preprocess/collect_scannet_data.py \
  --data_path datasets/ScanNet/scans \
  2>&1 | tee datasets/ScanNet/preprocess_collect_scannet.log

python -u preprocess/room2blocks.py \
  --dataset scannet \
  --data_path datasets/ScanNet/scenes \
  --block_size 1 \
  --stride 1 \
  --min_npts 1000 \
  2>&1 | tee datasets/ScanNet/preprocess_room2blocks_scannet.log

ln -sfn blocks_bs1.0_s1.0 datasets/ScanNet/blocks_bs1_s1
```

ScanNet 可用性检查：

```bash
cd "$CODE3D_ROOT"
printf 'ScanNet scenes: '; find datasets/ScanNet/scenes -maxdepth 1 -type f -name '*.npy' | wc -l
printf 'ScanNet blocks: '; find -L datasets/ScanNet/blocks_bs1_s1/data -maxdepth 1 -type f -name '*.npy' | wc -l

python - <<'PY'
from dataloaders.scannet import ScanNetDataset
from dataloaders.loader import MyDataset
valid_set = [x.strip() for x in open('datasets/ScanNet/scannetv2_val.txt') if x.strip()]
D=ScanNetDataset(0, '19-1', 'datasets/ScanNet/blocks_bs1_s1')
train=MyDataset('datasets/ScanNet/blocks_bs1_s1', D.base_classes + D.incre_classes,
                D.base_classes, D.class2scans, 0, mode='train', valid_set=valid_set,
                num_point=2048, pc_attribs='xyzrgbXYZ', pc_augm=False,
                pc_augm_config={'scale':0,'rot':0,'mirror_prob':0,'jitter':0})
valid=MyDataset('datasets/ScanNet/blocks_bs1_s1', D.base_classes + D.incre_classes,
                D.base_classes, D.class2scans, 0, mode='test', valid_set=valid_set,
                num_point=2048, pc_attribs='xyzrgbXYZ', pc_augm=False,
                pc_augm_config={'scale':0,'rot':0,'mirror_prob':0,'jitter':0})
pt,label=train[0]
print('dataset_init_ok')
print('classes', D.classes, len(D.base_classes), len(D.incre_classes))
print('train_len', len(train), 'valid_len', len(valid))
print('sample', tuple(pt.shape), tuple(label.shape), label.unique().tolist())
PY
```

当前服务器通过结果：

- `ScanNet scenes: 1513`
- `ScanNet blocks: 36350`
- `train_len 28305`
- `valid_len 7760`
- sample shape `(9, 2048)`, label shape `(2048,)`

## 结果保存路径

原作者脚本仍然传：

- `--save_path ./log_s3dis/`
- `--save_path ./log_scannet/`

当前本地化后的 `main.py` 会自动映射到：

- `$SEGACIL_ROOT/checkpoints_3d/s3dis/...`
- `$SEGACIL_ROOT/checkpoints_3d/scannet/...`

如需换输出根目录：

```bash
export CFSSeg3D_OUTPUT_ROOT=/path/to/checkpoints_3d
```

如需把 `datasets` 放到其他位置：

```bash
export CFSSeg3D_DATA_ROOT=/path/to/CFSSeg-code3D/datasets
```

## 一键最终检查

```bash
cd "$CODE3D_ROOT"

python - <<'PY'
import numpy as np, torch, torch_cluster, plyfile, transforms3d
print('env_ok', np.__version__, torch.__version__, torch.version.cuda)
PY

printf 'S3DIS scenes: '; find datasets/S3DIS/scenes -maxdepth 1 -type f -name '*.npy' | wc -l
printf 'S3DIS blocks: '; find -L datasets/S3DIS/blocks_bs1_s1/data -maxdepth 1 -type f -name '*.npy' | wc -l
printf 'ScanNet scenes: '; find datasets/ScanNet/scenes -maxdepth 1 -type f -name '*.npy' | wc -l
printf 'ScanNet blocks: '; find -L datasets/ScanNet/blocks_bs1_s1/data -maxdepth 1 -type f -name '*.npy' | wc -l

test -f datasets/S3DIS/meta/s3dis_classnames.txt
test -f datasets/ScanNet/meta/scannet_classnames.txt
test -f datasets/ScanNet/meta/scannetv2-labels.combined.tsv
test -f datasets/ScanNet/scannetv2_val.txt
test -L datasets/S3DIS/blocks_bs1_s1
test -L datasets/ScanNet/blocks_bs1_s1
```

期望核心计数：

- S3DIS：`272 scenes`，`7547 blocks`
- ScanNet v2：`1513 scenes`，`36350 blocks`

## 常见问题

- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`：降级到 `numpy==1.26.4`。
- `torch_cluster ... undefined symbol`：`torch_cluster` wheel 与 torch 版本不匹配，安装 `torch_cluster==1.6.3+pt21cu118` 并匹配 `torch==2.1.2+cu118`。
- S3DIS 找不到 `Stanford3dDataset_v1.2_Aligned_Version`：外层 tar 解开后还有内层 zip 未解。
- S3DIS 预处理在某个 annotation txt 报列数错误：检查并修复 `Area_5/hallway_6/Annotations/ceiling_1.txt` 第 180389 行。
- ScanNet 官方脚本卡在 `_2d-instance.zip`：停止全量下载，改用本文的 CFSSeg 最小下载脚本。
- `blocks_bs1_s1` 不存在：`room2blocks.py` 默认生成 `blocks_bs1.0_s1.0`，需要建立兼容软链接。
- dataloader 生成 `class2scans.pkl` 输出很多日志：这是正常缓存生成行为。

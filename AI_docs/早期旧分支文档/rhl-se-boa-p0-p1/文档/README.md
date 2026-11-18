<div align="center">
<h1>SegACIL: Solving the Stability-Plasticity Dilemma in Class-Incremental Semantic Segmentation</h1>

</div>

## Preparation

### Requirements

- CUDA>=11.8
- torch>=2.0.0
- torchvision>=0.15.0
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib

### Datasets

We use the Pascal VOC 2012 and ADE20K datasets for evaluation following the previous methods. You can download the datasets from the following links:

Download Pascal VOC 2012 dataset:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Download Additional Segmentation Class Annotations:
```bash
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
```

```
data_root/
   ├── VOC2012/
       ├── Annotations/
       ├── ImageSet/
       ├── JPEGImages/
       ├── SegmentationClassAug/
       └── saliency_map/

```

## Getting Started

### Class-Incremental Segmentation Segmentation on VOC 2012

Run our scripts `run.sh` for class-incremental segmentation on VOC 2012 dataset, or follow the instructions below.



```bash
DATA_ROOT="/mnt/petrelfs/lirui/SegACIL/datasets/data/voc"
MODEL="deeplabv3_resnet101"
LR=0.01
LOSS_TYPE="bce_loss"
DATASET="voc"
TASK="15-1"
LR_POLICY="poly"
SUBPATH="1128"
METHOD="acil"
SETTING="sequential"
TRAIN_EPOCH=50
PRETRAINED_BACKBONE="--pretrained_backbone"
BUFFER=8196
OUTPUT_STRIDE=8


DEFAULT_BATCH_SIZE=64   # Batch sizes for different steps
SPECIAL_BATCH_SIZE=32   # Batch size for step=0


# Loop through steps
START_STEP=1
END_STEP=10
STEP_INCREMENT=1

for ((CURR_STEP=$START_STEP; CURR_STEP<=$END_STEP; CURR_STEP+=$STEP_INCREMENT))
do
    if [ $CURR_STEP -eq 0 ]; then
        CURR_BATCH_SIZE=$SPECIAL_BATCH_SIZE
    else
        CURR_BATCH_SIZE=$DEFAULT_BATCH_SIZE
    fi

    echo "Running training for step $CURR_STEP with batch size $CURR_BATCH_SIZE..."
    python train.py \
        --data_root $DATA_ROOT \
        --model $MODEL \
        --lr $LR \
        --batch_size $CURR_BATCH_SIZE \
        --loss_type $LOSS_TYPE \
        --dataset $DATASET \
        --task $TASK \
        --lr_policy $LR_POLICY \
        --curr_step $CURR_STEP \
        --subpath $SUBPATH \
        --method $METHOD \
        --setting $SETTING \
        $PRETRAINED_BACKBONE \
        --crop_val \
        --train_epoch $TRAIN_EPOCH \
        --gamma 1 \
        --buffer $BUFFER \
        --output_stride $OUTPUT_STRIDE
done
```


# DATA_ROOT="/mnt/petrelfs/lirui/SegACIL/datasets/data/voc"
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

DATA_ROOT="/TRS-SAS/linwei/SegACIL/data_root/VOC2012"
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


DEFAULT_BATCH_SIZE=16   # Batch sizes for different steps
SPECIAL_BATCH_SIZE=16   # Batch size for step=0


# Loop through steps
START_STEP=0
END_STEP=5
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
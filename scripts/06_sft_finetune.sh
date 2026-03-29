#!/bin/bash
# ============================================================
# Step 6: SFT 有监督微调 (Supervised Fine-Tuning)
# ============================================================

set -e

MODEL_NAME=${1:-"bigcode/starcoderbase-7b"}
DATA_PATH=${2:-"./data/sft_spider_train_text2sql.json"}
OUTPUT_DIR=${3:-"./ckpts/sft-codes-custom"}
DATASET_NAME=${4:-"spider"}
BLOCK_SIZE=${5:-4096}
BATCH_SIZE=${6:-4}
EPOCHS=${7:-4}
LR=${8:-5e-6}
TABLE_NUM=${9:-6}
COLUMN_NUM=${10:-10}

echo "=== SFT 微调配置 ==="
echo "基础模型: $MODEL_NAME"
echo "训练数据: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: $DATASET_NAME"
echo "Block Size: $BLOCK_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "学习率: $LR"
echo "Table Num: $TABLE_NUM"
echo "Column Num: $COLUMN_NUM"

mkdir -p $OUTPUT_DIR
mkdir -p "./train_logs/sft-${DATASET_NAME}"

accelerate launch train/train_causal_lm.py \
    --per_device_train_batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --seed 42 \
    --pretrained_model_name_or_path $MODEL_NAME \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_ratio 0.05 \
    --checkpointing_steps 100000 \
    --tensorboard_log_dir ./train_logs/sft-${DATASET_NAME} \
    --mode sft \
    --output_ckpt_dir $OUTPUT_DIR \
    --text2sql_data_dir $DATA_PATH \
    --table_num $TABLE_NUM \
    --column_num $COLUMN_NUM

echo ""
echo "=== SFT 微调完成 ==="
echo "模型保存在: $OUTPUT_DIR"
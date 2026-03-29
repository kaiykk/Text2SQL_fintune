#!/bin/bash
# ============================================================
# Step 5: 增量预训练 (Incremental Pre-training)
# ============================================================

set -e

MODEL_NAME=${1:-"bigcode/starcoderbase-1b"}
CORPUS_PATH=${2:-"./data/tokenized_corpus.bin"}
OUTPUT_DIR=${3:-"./ckpts/pt-codes-custom"}
BLOCK_SIZE=${4:-8192}
BATCH_SIZE=${5:-2}
EPOCHS=${6:-1}
LR=${7:-5e-5}

echo "=== 增量预训练配置 ==="
echo "基础模型: $MODEL_NAME"
echo "语料路径: $CORPUS_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "Block Size: $BLOCK_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "学习率: $LR"

mkdir -p $OUTPUT_DIR
mkdir -p "./train_logs/pt-custom"

accelerate launch train/train_causal_lm.py \
    --per_device_train_batch_size $BATCH_SIZE \
    --block_size $BLOCK_SIZE \
    --seed 42 \
    --pretrained_model_name_or_path $MODEL_NAME \
    --epochs $EPOCHS \
    --lr $LR \
    --warmup_ratio 0.0 \
    --checkpointing_steps 500 \
    --tensorboard_log_dir ./train_logs/pt-custom \
    --mode pt \
    --output_ckpt_dir $OUTPUT_DIR \
    --pt_data_dir $CORPUS_PATH

echo ""
echo "=== 预训练完成 ==="
echo "模型保存在: $OUTPUT_DIR"
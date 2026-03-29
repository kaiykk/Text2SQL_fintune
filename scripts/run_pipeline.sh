#!/bin/bash
# ============================================================
# Text2SQL 完整训练流程脚本
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  all              运行完整流程"
    echo "  data             数据准备"
    echo "  pretrain         增量预训练"
    echo "  sft              SFT 微调"
    echo "  fewshot          Few-shot 推理"
    echo "  finetuned        Fine-tuned 推理"
    echo "  evaluate         效果评估"
}

# 检查依赖
check_deps() {
    print_info "检查依赖..."
    python -c "import transformers" 2>/dev/null || { print_error "缺少 transformers"; exit 1; }
    python -c "import torch" 2>/dev/null || { print_error "缺少 torch"; exit 1; }
    print_info "依赖检查完成"
}

# 数据准备
run_data() {
    print_info "数据准备..."
    echo "请先下载数据:"
    echo "  1. data.zip -> 解压到 ./data/"
    echo "  2. sic_ckpts.zip -> 解压到 ./sic_ckpts/"
    echo "  3. test_suite_sql_eval.zip -> 解压到 ./"
    print_info "数据准备完成"
}

# 增量预训练
run_pretrain() {
    MODEL_NAME=${1:-"bigcode/starcoderbase-1b"}
    CORPUS_PATH=${2:-"./data/tokenized_corpus.bin"}
    OUTPUT_DIR=${3:-"./ckpts/pt-codes-1b"}
    
    print_info "开始增量预训练..."
    print_info "模型: $MODEL_NAME"
    print_info "语料: $CORPUS_PATH"
    print_info "输出: $OUTPUT_DIR"
    
    mkdir -p $OUTPUT_DIR
    
    accelerate launch train/train_causal_lm.py \
        --per_device_train_batch_size 2 \
        --block_size 4096 \
        --seed 42 \
        --pretrained_model_name_or_path $MODEL_NAME \
        --epochs 1 \
        --lr 5e-5 \
        --warmup_ratio 0.0 \
        --checkpointing_steps 500 \
        --tensorboard_log_dir ./train_logs/pt \
        --mode pt \
        --output_ckpt_dir $OUTPUT_DIR \
        --pt_data_dir $CORPUS_PATH
    
    print_info "预训练完成"
}

# SFT 微调
run_sft() {
    MODEL_NAME=${1:-"bigcode/starcoderbase-7b"}
    DATA_PATH=${2:-"./data/sft_spider_train_text2sql.json"}
    OUTPUT_DIR=${3:-"./ckpts/sft-codes-7b-spider"}
    
    print_info "开始 SFT 微调..."
    print_info "模型: $MODEL_NAME"
    print_info "数据: $DATA_PATH"
    print_info "输出: $OUTPUT_DIR"
    
    mkdir -p $OUTPUT_DIR
    
    accelerate launch train/train_causal_lm.py \
        --per_device_train_batch_size 4 \
        --block_size 4096 \
        --seed 42 \
        --pretrained_model_name_or_path $MODEL_NAME \
        --epochs 4 \
        --lr 5e-6 \
        --warmup_ratio 0.05 \
        --checkpointing_steps 100000 \
        --tensorboard_log_dir ./train_logs/sft \
        --mode sft \
        --output_ckpt_dir $OUTPUT_DIR \
        --text2sql_data_dir $DATA_PATH \
        --table_num 6 \
        --column_num 10
    
    print_info "SFT 完成"
}

# Few-shot 推理
run_fewshot() {
    MODEL_NAME=${1:-"seeklhy/codes-7b"}
    EVAL_DATA=${2:-"./data/sft_spider_dev_text2sql.json"}
    DEMO_DATA=${3:-"./data/sft_spider_train_text2sql.json"}
    
    print_info "开始 Few-shot 推理..."
    python inference/few_shot_inference.py \
        --model_path $MODEL_NAME \
        --dataset_path $EVAL_DATA \
        --demonstration_set_path $DEMO_DATA \
        --num_of_demonstrations 4 \
        --output_path pred_sqls.txt
    
    print_info "推理完成,结果保存到 pred_sqls.txt"
}

# Fine-tuned 推理
run_finetuned() {
    MODEL_NAME=${1:-"./ckpts/sft-codes-7b-spider"}
    EVAL_DATA=${2:-"./data/sft_spider_dev_text2sql.json"}
    
    print_info "开始 Fine-tuned 推理..."
    python inference/finetuned_inference.py \
        --model_path $MODEL_NAME \
        --dataset_path $EVAL_DATA \
        --output_path pred_sqls.txt
    
    print_info "推理完成,结果保存到 pred_sqls.txt"
}

# 评估
run_evaluate() {
    BENCHMARK=${1:-"spider"}
    
    print_info "开始评估 (benchmark: $BENCHMARK)..."
    python evaluate/evaluate.py \
        --benchmark $BENCHMARK \
        --pred_file pred_sqls.txt
    
    print_info "评估完成"
}

# 主入口
COMMAND=${1:-help}
shift || true

case $COMMAND in
    all)
        check_deps
        run_data
        run_sft
        run_finetuned
        run_evaluate spider
        ;;
    data) check_deps; run_data ;;
    pretrain) check_deps; run_pretrain "$@";;
    sft) check_deps; run_sft "$@";;
    fewshot) check_deps; run_fewshot "$@";;
    finetuned) check_deps; run_finetuned "$@";;
    evaluate) check_deps; run_evaluate "$@";;
    help|--help|-h) show_help ;;
    *) print_error "未知命令: $COMMAND"; show_help; exit 1 ;;
esac
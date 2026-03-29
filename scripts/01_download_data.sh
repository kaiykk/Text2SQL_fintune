#!/bin/bash
# ============================================================
# Step 1: 下载所需数据集和模型权重
# ============================================================

set -e

echo "=== Step 1: 下载数据集和依赖 ==="

# 创建数据目录
mkdir -p data/sft_data_collections
mkdir -p sic_ckpts
mkdir -p test_suite_sql_eval

echo "请手动下载以下文件并解压到项目根目录:"
echo "  - data.zip -> 解压后得到 data/ 目录"
echo "  - sic_ckpts.zip -> 解压后得到 sic_ckpts/ 目录"  
echo "  - test_suite_sql_eval.zip -> 解压后得到 test_suite_sql_eval/ 目录"

echo ""
echo "可选: 下载 Spider 原始数据"
echo "  https://yale-lily.github.io/spider"

echo ""
echo "可选: 下载 BIRD 原始数据"
echo "  https://bird-bench.github.io"

echo ""
echo "=== 数据下载完成 ==="
exit 0
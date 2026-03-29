#!/usr/bin/env python3
"""
效果评估脚本
在 Spider/Bird 基准上评估模型性能

评估指标:
- Spider: EX (Execution Accuracy), TS (Test Suite Accuracy)
- BIRD: EX (Execution Accuracy), VES (Valid Efficiency Score)
"""

import argparse
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


def parse_option():
    parser = argparse.ArgumentParser(description="Text2SQL 效果评估")
    parser.add_argument('--benchmark', type=str, required=True, choices=['spider', 'bird', 'custom'], help='评估基准')
    parser.add_argument('--pred_file', type=str, required=True, help='预测结果文件')
    parser.add_argument('--gold_file', type=str, default=None, help='标准答案文件')
    parser.add_argument('--db_dir', type=str, default=None, help='数据库目录')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='输出目录')
    return parser.parse_args()


def evaluate_spider(pred_file, gold_file, db_dir):
    """评估 Spider 数据集"""
    print("=" * 50)
    print("评估 Spider 数据集")
    print("=" * 50)
    
    if os.path.exists("./test_suite_sql_eval/evaluation.py"):
        print("\n执行准确率 (Execution Accuracy):")
        os.system(f'python -u test_suite_sql_eval/evaluation.py --gold {gold_file} --pred {pred_file} --db {db_dir} --etype exec')
        
        print("\n测试套件准确率 (Test Suite Accuracy):")
        os.system(f'python -u test_suite_sql_eval/evaluation.py --gold {gold_file} --pred {pred_file} --db test_suite_sql_eval/test_suite_database --etype exec')
    else:
        print("错误: 找不到 test_suite_sql_eval 评估脚本")
        print("请下载: https://drive.google.com/file/d/1HIKBL7pP_hzWH1ryRNsjPO-N__UluOlK/view")


def evaluate_bird(pred_file, db_dir):
    """评估 BIRD 数据集"""
    print("=" * 50)
    print("评估 BIRD 数据集")
    print("=" * 50)
    
    if os.path.exists("./bird_evaluation/evaluation.py"):
        os.system(f'python bird_evaluation/evaluation.py --pred_file {pred_file} --db_dir {db_dir}')
    else:
        print("错误: 找不到 bird_evaluation 评估脚本")


def evaluate_custom(pred_file, db_dir):
    """评估自定义数据集"""
    print("=" * 50)
    print("评估自定义数据集")
    print("=" * 50)
    
    from utils.db_utils import check_sql_executability
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f.readlines()]
    
    total = len(predictions)
    empty_count = sum(1 for p in predictions if not p or p == "SQL placeholder")
    
    print(f"\n基本统计:")
    print(f"  总样本数: {total}")
    print(f"  空预测数: {empty_count}")
    print(f"  有效预测数: {total - empty_count}")
    
    if db_dir and os.path.exists(db_dir):
        print(f"\n检查 SQL 可执行性...")
        exec_count = 0
        for pred in predictions:
            if pred and pred != "SQL placeholder":
                db_files = [f for f in os.path.listdir(db_dir) if f.endswith('.sqlite')]
                if db_files:
                    db_path = os.path.join(db_dir, db_files[0])
                    error = check_sql_executability(pred, db_path)
                    if error is None:
                        exec_count += 1
        valid_count = total - empty_count
        if valid_count > 0:
            exec_rate = 100 * exec_count / valid_count
            print(f"  可执行 SQL: {exec_count}/{valid_count} ({exec_rate:.2f}%)")


def main():
    opt = parse_option()
    os.makedirs(opt.output_dir, exist_ok=True)
    
    if opt.benchmark == 'spider':
        gold_file = opt.gold_file or "./data/sft_data_collections/spider/dev_gold.sql"
        db_dir = opt.db_dir or "./data/sft_data_collections/spider/database"
        evaluate_spider(opt.pred_file, gold_file, db_dir)
    elif opt.benchmark == 'bird':
        db_dir = opt.db_dir or "./data/sft_data_collections/bird/dev/dev_databases"
        evaluate_bird(opt.pred_file, db_dir)
    else:
        evaluate_custom(opt.pred_file, opt.db_dir)
    
    print("\n" + "=" * 50)
    print("评估完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
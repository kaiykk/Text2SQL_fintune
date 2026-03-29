#!/usr/bin/env python3
"""
Step 7b: Fine-tuned 模型推理
使用微调后的模型进行 Text2SQL 生成
"""

import argparse
import os
import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char


def parse_option():
    parser = argparse.ArgumentParser(description="Fine-tuned Text2SQL 推理")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sic_path', type=str, default=None)
    parser.add_argument('--table_num', type=int, default=6)
    parser.add_argument('--column_num', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='pred_sqls.txt')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()


def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`" + column_name + "`")
    sql = sql.replace(" order ", " `order` ")
    return sql


def text2sql_func(model, inputs, tokenizer, max_new_tokens):
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            num_return_sequences=4
        )
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated_sqls


def main():
    opt = parse_option()
    print(opt)

    tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_dataset = json.load(open(opt.dataset_path))
    eval_set = SFTSQLGenerationDataset(
        opt.dataset_path,
        tokenizer,
        opt.max_tokens - opt.max_new_tokens,
        "eval",
        opt.table_num,
        opt.column_num,
        opt.sic_path
    )
    dataloader = DataLoader(eval_set, batch_size=opt.batch_size)

    print(f"加载模型: {opt.model_path}")
    if opt.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True)
    elif opt.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    predicted_sqls = []
    for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        generated_sqls = text2sql_func(model, batch_data, tokenizer, opt.max_new_tokens)
        generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in generated_sqls]

        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, raw_data["db_path"])
            if execution_error is None:
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            final_generated_sql = generated_sqls[0] if generated_sqls[0].strip() else "SQL placeholder"
        predicted_sqls.append(final_generated_sql)

    with open(opt.output_path, "w", encoding='utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print(f"结果已保存到: {opt.output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Few-shot 推理脚本
使用 ICL (In-Context Learning) 进行 Text2SQL 生成

核心组件:
1. Demonstration Retriever: 相似问题检索
2. Schema Item Classifier: 相关 schema 过滤
3. Execution-guided Decoding: 执行导向解码
"""

import argparse
import os
import json
import torch
import nltk
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from simcse import SimCSE
from transformers.trainer_utils import set_seed

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils.db_utils import check_sql_executability, get_db_schema_sequence, get_matched_content_sequence, detect_special_char
from schema.schema_item_filter import SchemaItemClassifierInference, filter_schema


def parse_option():
    parser = argparse.ArgumentParser(description="Few-shot Text2SQL 推理")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径或 HuggingFace 名称')
    parser.add_argument('--sic_path', type=str, default=None, help='Schema Item Classifier 路径')
    parser.add_argument('--table_num', type=int, default=5)
    parser.add_argument('--column_num', type=int, default=6)
    parser.add_argument('--dataset_path', type=str, required=True, help='评估集路径')
    parser.add_argument('--demonstration_set_path', type=str, required=True, help='Demonstration 池路径')
    parser.add_argument('--num_of_demonstrations', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--output_path', type=str, default='pred_sqls.txt')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    return parser.parse_args()


def post_process(sql, schema_items):
    """后处理生成的 SQL"""
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`" + column_name + "`")
    while "``" in sql:
        sql = sql.replace("``", "`")
    sql = sql.replace(" order ", " `order` ")
    return sql


def extract_skeleton(text):
    """提取文本骨架,用于更鲁棒的相似度匹配"""
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))
    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'CD', 'SYM', 'FW', 'IN']:
            output_tokens.append("_")
        elif token in ["'", "''", '(', ')', ',', '--', '.', ':']:
            pass
        else:
            output_tokens.append(token)
    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")
    while("_ _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ _", "_")
    while("_ , _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ , _", "_")
    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]
    return text_skeleton


def prepare_input_ids_and_attention_mask(tokenizer, input_seq, max_input_length, device):
    """准备输入的 input_ids 和 attention_mask"""
    input_ids = tokenizer(input_seq, truncation=False)["input_ids"]
    if len(input_ids) <= max_input_length:
        attention_mask = [1] * len(input_ids)
    else:
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_input_length - 1):]
        attention_mask = [1] * max_input_length
    return {
        "input_ids": torch.tensor([input_ids]).to(device),
        "attention_mask": torch.tensor([attention_mask]).to(device)
    }


def prepare_cross_domain_input_seq(eval_data, demonstration_set, similarity, num_demonstrations):
    """准备 Few-shot 输入序列"""
    top_k_indices = sorted(range(len(similarity)), key=lambda x: similarity[x], reverse=True)[:num_demonstrations]
    input_seq = ""
    for idx in top_k_indices:
        demonstration_sql = demonstration_set[idx]["sql"]
        if demonstration_sql.endswith(";"):
            demonstration_sql = demonstration_sql[:-1].strip() + " ;"
        else:
            demonstration_sql = demonstration_sql.strip() + " ;"
        input_seq += demonstration_set[idx]["schema_sequence"] + "\n" + \
                     demonstration_set[idx]["content_sequence"] + "\n" + \
                     demonstration_set[idx]["text"] + "\n" + \
                     demonstration_sql + "\n\n"
    input_seq += eval_data["schema_sequence"] + "\n" + \
                 eval_data["content_sequence"] + "\n" + \
                 eval_data["text"] + "\n"
    return input_seq


def text2sql_func(model, text2sql_input_seq, tokenizer, max_tokens, max_new_tokens, device):
    """Text2SQL 生成函数"""
    inputs = prepare_input_ids_and_attention_mask(tokenizer, text2sql_input_seq, max_tokens - max_new_tokens, device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            num_return_sequences=4,
            use_cache=True
        )
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated_sqls


def main():
    set_seed(42)
    opt = parse_option()
    print(opt)

    # 加载数据
    eval_set = json.load(open(opt.dataset_path))
    eval_set_questions = [data["question"] for data in eval_set]
    eval_set_question_skeletons = [extract_skeleton(question) for question in eval_set_questions]
    print(f"评估集大小: {len(eval_set)}")

    demonstration_set = json.load(open(opt.demonstration_set_path))
    demonstration_set_questions = [data["question"] for data in demonstration_set]
    demonstration_set_question_skeletons = [extract_skeleton(question) for question in demonstration_set_questions]
    print(f"Demonstration 池大小: {len(demonstration_set)}")

    # Schema Filtering
    if opt.sic_path is not None:
        demonstration_set = filter_schema(demonstration_set, "train", None, opt.table_num, opt.column_num)
        sic = SchemaItemClassifierInference(opt.sic_path)
        eval_set = filter_schema(eval_set, "eval", sic, opt.table_num, opt.column_num)
        del sic

    # 准备序列
    for demonstration_sample in demonstration_set:
        demonstration_sample["schema_sequence"] = get_db_schema_sequence(demonstration_sample["schema"])
        demonstration_sample["content_sequence"] = get_matched_content_sequence(demonstration_sample["matched_contents"])
    for eval_sample in eval_set:
        eval_sample["schema_sequence"] = get_db_schema_sequence(eval_sample["schema"])
        eval_sample["content_sequence"] = get_matched_content_sequence(eval_sample["matched_contents"])

    # 计算相似度
    print("计算相似度...")
    simsce_model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
    question_similarities = simsce_model.similarity(eval_set_questions, demonstration_set_questions)
    question_skeleton_similarities = simsce_model.similarity(eval_set_question_skeletons, demonstration_set_question_skeletons)
    similarities = np.maximum(question_similarities, question_skeleton_similarities)
    del simsce_model

    # 加载模型
    print(f"加载模型: {opt.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(opt.model_path)
    if opt.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True)
    elif opt.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(opt.model_path, device_map="auto", torch_dtype=torch.float16)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # 更新 EOS token
    token_ids_of_example_sql = tokenizer("SELECT * FROM table ;")["input_ids"]
    if token_ids_of_example_sql[-1] == tokenizer.eos_token_id:
        new_eos_token_id = token_ids_of_example_sql[-2]
    else:
        new_eos_token_id = token_ids_of_example_sql[-1]
    model.config.eos_token_id = new_eos_token_id
    tokenizer.eos_token_id = new_eos_token_id

    # 推理
    predicted_sqls = []
    for eval_data_idx, eval_data in tqdm(enumerate(eval_set)):
        input_seq = prepare_cross_domain_input_seq(eval_data, demonstration_set, similarities[eval_data_idx], opt.num_of_demonstrations)
        generated_sqls = text2sql_func(model, input_seq, tokenizer, opt.max_tokens, opt.max_new_tokens, model.device)
        generated_sqls = [post_process(generated_sql, eval_data["schema"]["schema_items"]) for generated_sql in generated_sqls]

        # 执行导向解码
        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, eval_data["db_path"])
            if execution_error is None:
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            final_generated_sql = generated_sqls[0] if generated_sqls[0].strip() else "SQL placeholder"
        predicted_sqls.append(final_generated_sql)

    # 保存结果
    with open(opt.output_path, "w", encoding='utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")
    print(f"结果已保存到: {opt.output_path}")


if __name__ == "__main__":
    main()
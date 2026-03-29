#!/usr/bin/env python3
"""
Step 4: 预训练语料 Tokenize
将 SQL 相关语料转换为二进制格式用于增量预训练
"""

import os
import sys
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


def tokenize_corpus(tokenizer, corpus_dir, output_path):
    tokenizer.model_max_length = int(1e30)
    num_proc = 32
    datasets = []
    
    pure_sql_path = os.path.join(corpus_dir, "pure_sql.jsonl")
    if os.path.exists(pure_sql_path):
        pure_sql_dataset = Dataset.from_json(pure_sql_path)
        print(f"纯SQL语料: {len(pure_sql_dataset)} 条")
        datasets.append(("pure_sql", pure_sql_dataset))
    
    text2code_path = os.path.join(corpus_dir, "text2code.jsonl")
    if os.path.exists(text2code_path):
        text2code_dataset = Dataset.from_json(text2code_path)
        print(f"Text2Code语料: {len(text2code_dataset)} 条")
        datasets.append(("text2code", text2code_dataset))
    
    text2text_path = os.path.join(corpus_dir, "text2text.jsonl")
    if os.path.exists(text2text_path):
        text2text_dataset = Dataset.from_json(text2text_path)
        print(f"Text2Text语料: {len(text2text_dataset)} 条")
        datasets.append(("text2text", text2text_dataset))
    
    def tokenize_fn(sequences):
        input_ids = tokenizer(sequences, truncation=False)["input_ids"]
        input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]
        length = [len(ids) for ids in input_ids]
        return {'input_ids': input_ids, 'length': length}
    
    def process_sql(examples):
        sequences = [sql for sql in examples["sql"]]
        sequences = [seq.strip() for seq in sequences]
        return tokenize_fn(sequences)
    
    def process_text2code(examples):
        sequences = [text + "\n" + code for text, code in zip(examples["text"], examples["code"])]
        sequences = [sequence.strip() for sequence in sequences]
        return tokenize_fn(sequences)
    
    def process_text2text(examples):
        sequences = [input_text + "\n" + output_text for input_text, output_text in zip(examples["input_text"], examples["output_text"])]
        sequences = [sequence.strip() for sequence in sequences]
        return tokenize_fn(sequences)
    
    processed_datasets = []
    for name, dataset in datasets:
        if name == "pure_sql":
            processed = dataset.map(process_sql, num_proc=num_proc, desc=f"tokenizing {name}", remove_columns=["sql"], batched=True)
        elif name == "text2code":
            processed = dataset.map(process_text2code, num_proc=num_proc, desc=f"tokenizing {name}", remove_columns=["text", "code"], batched=True)
        elif name == "text2text":
            processed = dataset.map(process_text2text, num_proc=num_proc, desc=f"tokenizing {name}", remove_columns=["input_text", "output_text"], batched=True)
        processed_datasets.append(processed)
    
    if not processed_datasets:
        print("未找到任何语料文件!")
        print(f"请在 {corpus_dir} 目录下放置以下文件:")
        print("  - pure_sql.jsonl: 纯SQL语句,格式: {'sql': 'SELECT ...'}")
        print("  - text2code.jsonl: 文本到代码,格式: {'text': '...', 'code': '...'}")
        print("  - text2text.jsonl: 文本到文本,格式: {'input_text': '...', 'output_text': '...'}")
        return
    
    final_corpus = [processed_datasets[0], processed_datasets[0]]
    for i in range(1, len(processed_datasets)):
        final_corpus.append(processed_datasets[i])
    
    arr_len = sum(np.sum(tokenized_dataset['length']) for tokenized_dataset in final_corpus)
    print(f"总 token 数: {arr_len:,}")
    
    dtype = np.uint16
    arr = np.memmap(output_path, dtype=dtype, mode='w+', shape=(arr_len,))
    
    idx = 0
    total_batches = 2048
    
    for tokenized_dataset in final_corpus:
        for batch_idx in tqdm(range(total_batches), desc="Writing to memmap"):
            batch = tokenized_dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['input_ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    print(f"语料已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="./codes_pretrain_corpus", help="语料目录")
    parser.add_argument("--output_path", type=str, default="./data/tokenized_corpus.bin", help="输出文件路径")
    parser.add_argument("--model_name", type=str, default="bigcode/starcoder", help="模型名称")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Tokenize 预训练语料")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_corpus(tokenizer, args.corpus_dir, args.output_path)
    
    print("=" * 50)
    print("Tokenize 完成!")
    print("=" * 50)
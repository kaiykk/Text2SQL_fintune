"""
SFT 数据加载器
加载 Text2SQL 训练数据
"""

import json
import torch
from torch.utils.data import Dataset
from schema.schema_item_filter import SchemaItemClassifierInference, filter_schema
from utils.db_utils import get_db_schema_sequence, get_matched_content_sequence


def prepare_text2sql_prefix_sequence(data):
    """
    准备 Text2SQL 输入序列
    
    格式: schema_sequence + content_sequence + text
    """
    prefix_seq = data["schema_sequence"] + "\n" + data["content_sequence"] + "\n" + data["text"] + "\n"
    return prefix_seq


def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    """
    准备输入和标签
    
    参数:
        prefix_seq: 输入序列 (schema + content + question)
        target_seq: 目标序列 (SQL)
        tokenizer: 分词器
        max_tokens: 最大 token 数
    
    返回:
        包含 input_ids, attention_mask, labels 的字典
    """
    # 编码输入和目标
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    
    if seq_length <= max_tokens:
        # 填充
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = [1] * seq_length + [0] * pad_length
        # 只有 target 部分计算 loss
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else:
        # 截断
        print("序列超长,进行截断")
        input_ids = prefix_ids + target_ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        labels = [-100] * len(prefix_ids) + target_ids
        labels = labels[-max_tokens:]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64), 
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    """
    准备推理输入 (不包含 labels)
    """
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("输入超长,进行截断")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64)
    }


class SFTSQLGenerationDataset(Dataset):
    """
    Text2SQL SFT 数据集
    """
    
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num, sic_path):
        """
        初始化
        
        参数:
            text2sql_data_dir: 数据文件路径
            tokenizer: 分词器
            max_tokens: 最大 token 数
            mode: 模式 (train/eval)
            table_num: 最大表数量
            column_num: 每表最大列数
            sic_path: Schema Item Classifier 路径
        """
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        print("应用 schema 过滤策略...")
        if mode == "train":
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
        elif mode == "eval":
            sic = SchemaItemClassifierInference(sic_path)
            dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()

        # 准备 schema sequence 和 content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])
            data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2sql_prefix_sequence(data)

        if self.mode == "train":
            target_seq = data["sql"]
            return prepare_inputs_and_labels(prefix_seq, target_seq, self.tokenizer, self.max_tokens)
        elif self.mode == "eval":
            return prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)

    def __len__(self):
        return len(self.dataset)
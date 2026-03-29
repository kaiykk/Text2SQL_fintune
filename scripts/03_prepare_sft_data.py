#!/usr/bin/env python3
"""
Step 3: SFT 数据预处理
将 Spider/Bird 原始数据转换为训练所需的 JSON 格式
"""

import json
import os
import re
import random
import sqlparse

from nltk.tokenize import word_tokenize
from nltk import ngrams
from sql_metadata import Parser
from pyserini.search.lucene import LuceneSearcher

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils.bridge_content_encoder import get_matched_entries
from utils.db_utils import get_db_schema

random.seed(42)

DATA_DIR = "./data/sft_data_collections"
OUTPUT_DIR = "./data"


def extract_large_numbers(text):
    """从文本中提取大数字信息"""
    number_information = []
    patterns = {'thousand': 10**3, 'million': 10**6, 'billion': 10**9, 'trillion': 10**12}
    for word, multiplier in patterns.items():
        matches = re.findall(r'(\d+\.?\d*)\s*{}'.format(word), text, re.IGNORECASE)
        for match in matches:
            number = float(match) * multiplier
            number_information.append(match + " " + word + " = " + str(int(number)))
    large_number_evidence = ""
    for info in number_information:
        large_number_evidence += info + "; "
    return large_number_evidence.strip()


def remove_table_alias(s):
    """移除 SQL 中的表别名"""
    try:
        tables_aliases = Parser(s).tables_aliases
    except:
        return s
    new_tables_aliases = {}
    for i in range(1, 11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        s = s.replace("AS " + k + " ", "")
        s = s.replace(k, v)
    return s


def remove_similar_comments(names, comments):
    """移除与名称高度相似的注释"""
    new_comments = []
    for name, comment in zip(names, comments):
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    return new_comments


def str_replace_ignore_case(evidence, schema_item_name):
    """忽略大小写替换字符串"""
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)
    return evidence


def obtain_n_grams(sequence, max_n):
    """获取文本的所有 n-gram"""
    tokens = word_tokenize(sequence)
    all_grams = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    return all_grams


def preprocess_evidence(evidence, schema_items):
    """预处理 evidence"""
    if evidence.strip() == "":
        return ""
    evidence = evidence.strip()
    if not evidence.endswith(";"):
        evidence += ";"
    for table in schema_items:
        if table["table_name"] in evidence.lower():
            evidence = str_replace_ignore_case(evidence, table["table_name"])
        for column_name in table["column_names"]:
            if column_name in evidence.lower():
                evidence = str_replace_ignore_case(evidence, column_name)
    evidence = evidence.replace("< =", "<=").replace("> =", ">=")
    return evidence


def spider_style_dataset(dataset_path, db_path, db_content_index_path, source, table_json_path, use_evidence, mode):
    """处理 Spider 风格的数据集"""
    returned_dataset = []
    dataset = json.load(open(dataset_path))
    additional_db_info = json.load(open(table_json_path)) if os.path.exists(table_json_path) else []

    db_comments = dict()
    for db_info in additional_db_info:
        comment_dict = dict()
        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]]
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]]
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]]
        column_comments = remove_similar_comments(column_names, column_comments)
        table_names = [table_name.lower() for table_name in db_info["table_names_original"]]
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]
        table_comments = remove_similar_comments(table_names, table_comments)

        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {"table_comment": table_comment, "column_comments": dict()}
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][column_name] = column_comment
        db_comments[db_info["db_id"]] = comment_dict

    db_ids = set([data["db_id"] for data in dataset])
    db_id2searcher = dict()
    for db_id in db_ids:
        index_path = os.path.join(db_content_index_path, db_id)
        if os.path.exists(index_path):
            db_id2searcher[db_id] = LuceneSearcher(index_path)

    db_id2schema = dict()

    for data in dataset:
        sample = {}
        db_id = data["db_id"]
        sample["db_id"] = db_id
        sample["db_path"] = os.path.join(db_path, db_id, db_id + ".sqlite")

        if db_id in db_id2schema:
            sample["schema"] = db_id2schema[db_id]
        else:
            db_id2schema[db_id] = get_db_schema(sample["db_path"], db_comments.get(db_id), db_id)
            sample["schema"] = db_id2schema[db_id]

        if "spider-syn" in source:
            sample["question"] = data.get("SpiderSynQuestion", data["question"])
            sample["evidence"] = ""
        elif "bird" in source:
            sample["question"] = data["question"]
            evidence = preprocess_evidence(data.get("evidence", ""), sample["schema"]["schema_items"])
            sample["evidence"] = evidence
        elif "bank" in source:
            sample["question"] = data["question"]
            sample["evidence"] = extract_large_numbers(data["question"])
        else:
            sample["question"] = data["question"]
            sample["evidence"] = ""
        
        sample["question"] = sample["question"].replace("\n", " ")
        sample["evidence"] = sample["evidence"].replace("\n", " ") if sample["evidence"] else ""
        
        sample["text"] = sample["evidence"] + " " + sample["question"] if use_evidence and sample["evidence"] != "" else sample["question"]

        if mode in ["train", "dev"]:
            sql = data.get("SQL") or data.get("query", "")
            sample["sql"] = remove_table_alias(sqlparse.format(sql, keyword_case="upper", identifier_case="lower"))
        elif mode == "test":
            sample["sql"] = ""
        
        sample["table_labels"], sample["column_labels"] = [], []
        try:
            sql_tokens = [token.value for token in Parser(sample["sql"].lower()).tokens]
        except:
            sql_tokens = sample["sql"].lower().split()
        
        for table_info in sample["schema"]["schema_items"]:
            if mode in ["train", "dev"]:
                table_name = table_info["table_name"]
                sample["table_labels"].append(1 if table_name in sql_tokens else 0)
                sample["column_labels"].append([1 if column_name in sql_tokens or table_name+"."+column_name in sql_tokens else 0 for column_name in table_info["column_names"]])
            else:
                sample["table_labels"].append(0)
                sample["column_labels"].append([0 for _ in range(len(table_info["column_names"]))])

        grams = obtain_n_grams(sample["text"], 4)
        hits = []
        if db_id in db_id2searcher:
            searcher = db_id2searcher[db_id]
            for query in grams:
                hits.extend(searcher.search(query, k=10))

        coarse_matched_contents = dict()
        for i in range(len(hits)):
            matched_result = json.loads(hits[i].raw)
            tc_name = ".".join(matched_result["id"].split("-**-")[:2])
            if tc_name in coarse_matched_contents.keys():
                if matched_result["contents"] not in coarse_matched_contents[tc_name]:
                    coarse_matched_contents[tc_name].append(matched_result["contents"])
            else:
                coarse_matched_contents[tc_name] = [matched_result["contents"]]
        
        fine_matched_contents = dict()
        for tc_name, contents in coarse_matched_contents.items():
            fm_contents = get_matched_entries(sample["text"], contents) if contents else None
            if fm_contents is None:
                continue
            for _match_str, (field_value, _s_match_str, match_score, s_match_score, _match_size,) in fm_contents:
                if match_score < 0.9:
                    continue
                if tc_name in fine_matched_contents.keys():
                    if len(fine_matched_contents[tc_name]) < 25:
                        fine_matched_contents[tc_name].append(field_value.strip())
                else:
                    fine_matched_contents[tc_name] = [field_value.strip()]

        sample["matched_contents"] = fine_matched_contents
        sample["source"] = source
        returned_dataset.append(sample)

    return returned_dataset


def prepare_spider_data():
    """准备 Spider 数据集"""
    print("准备 Spider 训练集...")
    spider_train = []
    for spider_train_set in ["train_spider.json", "train_others.json"]:
        dataset_path = os.path.join(DATA_DIR, "spider", spider_train_set)
        if os.path.exists(dataset_path):
            spider_train.extend(spider_style_dataset(
                dataset_path=dataset_path, 
                db_path=os.path.join(DATA_DIR, "spider", "database"), 
                db_content_index_path=os.path.join(DATA_DIR, "spider", "db_contents_index"),
                source="spider-train",
                table_json_path=os.path.join(DATA_DIR, "spider", "tables.json"),
                use_evidence=False,
                mode="train"
            ))
    
    if spider_train:
        with open(os.path.join(OUTPUT_DIR, "sft_spider_train_text2sql.json"), "w") as f:
            f.write(json.dumps(spider_train, indent=2, ensure_ascii=False))
        print(f"Spider 训练集: {len(spider_train)} 条")
    
    print("准备 Spider dev 集...")
    spider_dev_path = os.path.join(DATA_DIR, "spider", "dev.json")
    if os.path.exists(spider_dev_path):
        spider_dev = spider_style_dataset(
            dataset_path=spider_dev_path, 
            db_path=os.path.join(DATA_DIR, "spider", "database"), 
            db_content_index_path=os.path.join(DATA_DIR, "spider", "db_contents_index"),
            source="spider-dev",
            table_json_path=os.path.join(DATA_DIR, "spider", "tables.json"),
            use_evidence=False,
            mode="dev"
        )
        with open(os.path.join(OUTPUT_DIR, "sft_spider_dev_text2sql.json"), "w") as f:
            f.write(json.dumps(spider_dev, indent=2, ensure_ascii=False))
        print(f"Spider dev 集: {len(spider_dev)} 条")


def prepare_bird_data():
    """准备 BIRD 数据集"""
    print("准备 BIRD 训练集...")
    bird_train_path = os.path.join(DATA_DIR, "bird", "train", "train.json")
    if os.path.exists(bird_train_path):
        bird_train = spider_style_dataset(
            dataset_path=bird_train_path, 
            db_path=os.path.join(DATA_DIR, "bird", "train", "train_databases"), 
            db_content_index_path=os.path.join(DATA_DIR, "bird", "train", "db_contents_index"),
            source="bird-train",
            table_json_path=os.path.join(DATA_DIR, "bird", "train", "train_tables.json"),
            use_evidence=False,
            mode="train"
        )
        with open(os.path.join(OUTPUT_DIR, "sft_bird_train_text2sql.json"), "w") as f:
            f.write(json.dumps(bird_train, indent=2, ensure_ascii=False))
        print(f"BIRD 训练集: {len(bird_train)} 条")
        
        bird_with_evidence_train = spider_style_dataset(
            dataset_path=bird_train_path, 
            db_path=os.path.join(DATA_DIR, "bird", "train", "train_databases"), 
            db_content_index_path=os.path.join(DATA_DIR, "bird", "train", "db_contents_index"),
            source="bird-train",
            table_json_path=os.path.join(DATA_DIR, "bird", "train", "train_tables.json"),
            use_evidence=True,
            mode="train"
        )
        with open(os.path.join(OUTPUT_DIR, "sft_bird_with_evidence_train_text2sql.json"), "w") as f:
            f.write(json.dumps(bird_with_evidence_train, indent=2, ensure_ascii=False))
        print(f"BIRD 训练集 (with evidence): {len(bird_with_evidence_train)} 条")
    
    print("准备 BIRD dev 集...")
    bird_dev_path = os.path.join(DATA_DIR, "bird", "dev", "dev.json")
    if os.path.exists(bird_dev_path):
        bird_dev = spider_style_dataset(
            dataset_path=bird_dev_path, 
            db_path=os.path.join(DATA_DIR, "bird", "dev", "dev_databases"), 
            db_content_index_path=os.path.join(DATA_DIR, "bird", "dev", "db_contents_index"),
            source="bird-dev",
            table_json_path=os.path.join(DATA_DIR, "bird", "dev", "dev_tables.json"),
            use_evidence=False,
            mode="dev"
        )
        with open(os.path.join(OUTPUT_DIR, "sft_bird_dev_text2sql.json"), "w") as f:
            f.write(json.dumps(bird_dev, indent=2, ensure_ascii=False))
        print(f"BIRD dev 集: {len(bird_dev)} 条")


if __name__ == "__main__":
    print("=" * 50)
    print("SFT 数据预处理")
    print("=" * 50)
    prepare_spider_data()
    prepare_bird_data()
    print("=" * 50)
    print("数据预处理完成!")
    print("=" * 50)
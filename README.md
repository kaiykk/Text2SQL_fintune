# Text2SQL 大模型训练项目

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](requirements.txt)

基于 StarCoder 的 Text2SQL 领域自适应训练框架,支持增量预训练和 SFT 有监督微调。

## Description

本项目是一个完整的 Text2SQL 大模型训练框架,复现了 CodeS 论文的训练流程。项目基于 StarCoder 进行领域自适应训练,使模型具备从自然语言生成 SQL 查询的能力。

### 核心特性

- **增量预训练 (PT)**: 在 SQL 相关语料上持续预训练,增强模型的 SQL 生成能力
- **有监督微调 (SFT)**: 在 Spider/Bird 数据集上微调,学习自然语言到 SQL 的映射
- **Few-shot 推理**: 支持 ICL (In-Context Learning) 方式,无需微调即可推理
- **Schema 过滤**: 使用分类器自动过滤不相关的表和列,减少输入长度
- **执行导向解码**: 从多个候选 SQL 中选择可执行的正确结果

### 支持的数据集

- Spider: 跨域 Text2SQL 基准
- BIRD: 大规模 Text2SQL 基准 (含外部知识)

### 评估指标

- Spider: EX (Execution Accuracy), TS (Test Suite Accuracy)
- BIRD: EX (Execution Accuracy), VES (Valid Efficiency Score)

## 项目概述

本项目实现了完整的 Text2SQL 大模型训练流程:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        完整 Text2SQL 训练流程                            │
├─────────────────────────────────────────────────────────────────────────┤
│  1. 数据准备阶段                                                          │
│     ├── 下载原始数据集 (Spider/Bird)                                     │
│     ├── 构建数据库内容索引 (Lucene)                                       │
│     └── SFT 数据预处理                                                   │
│                                                                          │
│  2. 预训练阶段 (可选)                                                     │
│     ├── 准备 SQL 语料                                                    │
│     ├── Tokenize 语料                                                    │
│     └── 增量预训练 StarCoder                                             │
│                                                                          │
│  3. SFT 微调阶段                                                         │
│     └── 在 Spider/Bird 上微调                                           │
│                                                                          │
│  4. 推理与评估                                                           │
│     ├── Few-shot 推理                                                   │
│     ├── Fine-tuned 推理                                                 │
│     └── 效果评估 (EX/TS/VES)                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 基模

本项目基于 **StarCoder** (由 BigCode 开发的大代码模型) 进行训练:

| 基模名称 | 参数规模 | HuggingFace 路径 |
|---------|---------|-----------------|
| StarCoderBase-1B | 1B | `bigcode/starcoderbase-1b` |
| StarCoderBase-3B | 3B | `bigcode/starcoderbase-3b` |
| StarCoderBase-7B | 7B | `bigcode/starcoderbase-7b` |
| StarCoder-15B | 15B | `bigcode/starcoder` |

## 训练流程

### 方式一: 使用 Shell 脚本 (推荐)

```bash
# 进入项目目录
cd text2sql

# 1. 数据准备
bash scripts/01_download_data.sh
bash scripts/02_build_index.sh
python scripts/03_prepare_sft_data.py

# 2. 增量预训练 (可选)
python scripts/04_tokenize_corpus.py --corpus_dir ./your_sql_corpus
bash scripts/05_pretrain.sh bigcode/starcoderbase-1b ./data/tokenized_corpus.bin ./ckpts/pt-codes-1b

# 3. SFT 微调
bash scripts/06_sft_finetune.sh seeklhy/codes-1b ./data/sft_spider_train_text2sql.json ./ckpts/sft-codes-1b-spider

# 4. 推理
# Few-shot 推理
python scripts/07_few_shot_inference.py --model_path seeklhy/codes-7b --dataset_path ./data/sft_spider_dev_text2sql.json --demonstration_set_path ./data/sft_spider_train_text2sql.json

# Fine-tuned 推理
python scripts/08_finetuned_inference.py --model_path ./ckpts/sft-codes-1b-spider --dataset_path ./data/sft_spider_dev_text2sql.json

# 5. 评估
python scripts/09_evaluate.py --benchmark spider --pred_file pred_sqls.txt
```

### 方式二: 一键运行

```bash
# 运行完整流程
bash scripts/run_pipeline.sh all

# 仅数据准备
bash scripts/run_pipeline.sh data

# 仅 SFT 微调
bash scripts/run_pipeline.sh sft

# 仅推理
bash scripts/run_pipeline.sh finetuned

# 仅评估
bash scripts/run_pipeline.sh evaluate
```

## 项目结构

```
text2sql/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
│
├── train/                       # 训练脚本
│   ├── train_causal_lm.py       # 主训练脚本 (支持 PT/SFT)
│   └── train.sh                 # 训练启动脚本
│
├── data/                        # 数据处理
│   ├── prepare_sft_data.py      # SFT 数据预处理
│   └── tokenize_corpus.py       # 语料 Tokenize
│
├── inference/                   # 推理脚本
│   ├── few_shot_inference.py    # Few-shot ICL 推理
│   └── finetuned_inference.py   # Fine-tuned 模型推理
│
├── evaluate/                    # 评估脚本
│   └── evaluate.py              # 效果评估
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── load_pt_dataset.py       # 预训练数据加载
│   ├── load_sft_dataset.py      # SFT 数据加载
│   ├── db_utils.py              # 数据库工具
│   ├── lr_scheduler.py          # 学习率调度
│   ├── bridge_content_encoder.py # Content Matching
│   └── classifier_model.py      # Schema 分类器模型
│
├── schema/                      # Schema 过滤
│   └── schema_item_filter.py    # Schema Item 分类器
│
└── scripts/                     # 完整流程脚本
    ├── 01_download_data.sh      # 下载数据
    ├── 02_build_index.sh        # 构建索引
    ├── 03_prepare_sft_data.py   # 预处理 SFT 数据
    ├── 04_tokenize_corpus.py    # Tokenize 语料
    ├── 05_pretrain.sh           # 增量预训练
    ├── 06_sft_finetune.sh       # SFT 微调
    ├── 07_few_shot_inference.py # Few-shot 推理
    ├── 08_finetuned_inference.py# Fine-tuned 推理
    ├── 09_evaluate.py           # 效果评估
    └── run_pipeline.sh          # 一键运行
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n text2sql python=3.8.5
conda activate text2sql

# 安装 PyTorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# 安装依赖
pip install -r requirements.txt

# 安装 SimCSE (用于 Few-shot 检索)
git clone https://github.com/lihaoyang-ruc/SimCSE.git
cd SimCSE
python setup.py install
cd ..
```

## 数据下载

需要手动下载以下文件 (Google Drive):

| 文件 | 用途 | 链接 |
|------|------|------|
| `data.zip` | Spider + Bird 数据集 | [下载](https://drive.google.com/file/d/1-tfTMpc4gEtPqje_9jv-NU4csPQQz622/view) |
| `sic_ckpts.zip` | Schema Item Classifier | [下载](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view) |
| `test_suite_sql_eval.zip` | Spider 评估脚本 | [下载](https://drive.google.com/file/d/1HIKBL7pP_hzWH1ryRNsjPO-N__UluOlK/view) |

下载后解压到项目根目录。

## 训练配置

### 增量预训练配置

```bash
# 参数说明
--per_device_train_batch_size  # 每 GPU batch size (建议: 1B=2, 3B=2, 7B=1)
--block_size                   # 序列长度 (建议: 8192)
--lr                           # 学习率 (建议: 5e-5)
--epochs                       # 训练轮数 (建议: 1-2)
--mode                         # 训练模式: pt
```

### SFT 微调配置

```bash
# 参数说明
--per_device_train_batch_size  # 每 GPU batch size (建议: 4)
--block_size                   # 序列长度 (建议: 4096)
--lr                           # 学习率 (建议: 5e-6)
--epochs                       # 训练轮数 (建议: 4)
--mode                         # 训练模式: sft
--table_num                    # 最大表数量 (建议: 6)
--column_num                   # 每表最大列数 (建议: 10)
```

## 评估指标

### Spider
- **EX (Execution Accuracy)**: 执行准确率
- **TS (Test Suite Accuracy)**: 测试套件准确率

### BIRD
- **EX (Execution Accuracy)**: 执行准确率
- **VES (Valid Efficiency Score)**: 有效效率分数

## 常见问题

### 1. 显存不足
- 减小 `block_size` (8192 → 4096)
- 使用 `--load_in_4bit` 或 `--load_in_8bit` 量化
- 启用 gradient checkpointing

### 2. 训练时间过长
- 减少 `epochs` (1-2 个即可)
- 使用多 GPU 并行 (Accelerate)
- 减小 `block_size`

### 3. 灾难性遗忘
- 混合原始语料训练
- 使用较小的学习率
- 减少训练 epoch

## 参考

- [CodeS 项目](https://github.com/RUCKBReasoning/CodeS)
- [StarCoder](https://huggingface.co/bigcode/starcoder)
- [Spider 数据集](https://yale-lily.github.io/spider)
- [BIRD 数据集](https://bird-bench.github.io)

## License

Apache-2.0 License
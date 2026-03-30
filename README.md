# Text2SQL Training Framework

A complete Text2SQL training framework based on StarCoder, supporting incremental pre-training and SFT fine-tuning.

## Features

- **Incremental Pre-training**: Continue pre-training on SQL corpora
- **SFT Fine-tuning**: Fine-tune on Spider/Bird datasets
- **Few-shot Inference**: ICL-based inference without fine-tuning
- **Schema Filtering**: Auto-filter irrelevant tables/columns
- **Execution-guided Decoding**: Select executable SQL from candidates

## Quick Start

```bash
# 1. Data Preparation
bash scripts/01_download_data.sh
bash scripts/02_build_index.sh
python scripts/03_prepare_sft_data.py

# 2. Pre-training (optional)
python scripts/04_tokenize_corpus.py --corpus_dir ./your_sql_corpus
bash scripts/05_pretrain.sh bigcode/starcoderbase-1b ./data/tokenized_corpus.bin ./ckpts/pt-codes-1b

# 3. SFT Fine-tuning
bash scripts/06_sft_finetune.sh bigcode/starcoderbase-7b ./data/sft_spider_train_text2sql.json ./ckpts/sft-codes-7b-spider

# 4. Inference
python scripts/08_finetuned_inference.py --model_path ./ckpts/sft-codes-7b-spider --dataset_path ./data/sft_spider_dev_text2sql.json

# 5. Evaluation
python scripts/09_evaluate.py --benchmark spider --pred_file pred_sqls.txt
```

## Base Model

StarCoder (bigcode/starcoderbase-1b/3b/7b/15b)

## Evaluation

- Spider: EX, TS
- BIRD: EX, VES

## Requirements

- Python 3.8+
- PyTorch 1.13+
- See requirements.txt for dependencies

## License

Apache-2.0
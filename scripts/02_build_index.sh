#!/bin/bash
# ============================================================
# Step 2: 构建数据库内容索引 (Lucene)
# ============================================================

set -e

cd "$(dirname "$0")/.."

echo "=== Step 2: 构建数据库内容索引 ==="
echo "安装 pyserini (如果需要): pip install pyserini"

python -c "
import os
import json
import sqlite3

try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index import IndexWriter
except ImportError:
    print('请先安装 pyserini: pip install pyserini')
    exit(1)

def build_index(db_path, output_path):
    index_writer = IndexWriter(output_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'\")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info({table_name})')
        columns = cursor.fetchall()
        
        for col in columns:
            col_name = col[1]
            try:
                cursor.execute(f'SELECT DISTINCT \"{col_name}\" FROM \"{table_name}\" WHERE \"{col_name}\" IS NOT NULL LIMIT 100')
                contents = cursor.fetchall()
                for content in contents:
                    if content[0] and str(content[0]).strip():
                        doc = {
                            'id': f'{table_name}.{col_name}',
                            'contents': str(content[0])
                        }
                        index_writer.add_document(json.dumps(doc))
            except:
                pass
    
    index_writer.close()
    conn.close()

db_dir = './data/sft_data_collections/spider/database'
output_dir = './data/sft_data_collections/spider/db_contents_index'
os.makedirs(output_dir, exist_ok=True)

for db_file in os.listdir(db_dir):
    if db_file.endswith('.sqlite'):
        db_id = db_file.replace('.sqlite', '')
        db_path = os.path.join(db_dir, db_file)
        index_path = os.path.join(output_dir, db_id)
        os.makedirs(index_path, exist_ok=True)
        print(f'Building index for {db_id}...')
        build_index(db_path, index_path)

print('Spider 索引构建完成!')
"

echo ""
echo "=== 索引构建完成 ==="
exit 0
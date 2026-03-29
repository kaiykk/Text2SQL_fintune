"""
Schema Item Classifier 模型
用于预测哪些表和列与问题相关
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaModel


class SchemaItemClassifier(nn.Module):
    def __init__(self, model_name_or_path, mode):
        super(SchemaItemClassifier, self).__init__()
        if mode in ["eval", "test"]:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.plm_encoder = RobertaModel(config)
        elif mode == "train":
            self.plm_encoder = RobertaModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError()

        self.plm_hidden_size = self.plm_encoder.config.hidden_size

        # column cls head
        self.column_info_cls_head_linear1 = nn.Linear(self.plm_hidden_size, 256)
        self.column_info_cls_head_linear2 = nn.Linear(256, 2)
        
        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size=self.plm_hidden_size,
            hidden_size=int(self.plm_hidden_size/2),
            num_layers=2,
            dropout=0,
            bidirectional=True
        )

        self.column_info_linear_after_pooling = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(self.plm_hidden_size, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)
        
        self.table_name_bilstm = nn.LSTM(
            input_size=self.plm_hidden_size,
            hidden_size=int(self.plm_hidden_size/2),
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        self.table_name_linear_after_pooling = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim=self.plm_hidden_size, num_heads=8)
        self.dropout = nn.Dropout(p=0.2)
    
    def table_column_cross_attention(self, table_name_embeddings_in_one_db, column_info_embeddings_in_one_db, column_number_in_each_table):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                sum(column_number_in_each_table[:table_id]) : sum(column_number_in_each_table[:table_id+1]), :]
            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding, column_info_embeddings_in_one_table, column_info_embeddings_in_one_table
            )
            table_name_embedding_attn_list.append(table_name_embedding_attn)
        
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list, dim=0)
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)
        return table_name_embeddings_in_one_db

    def table_column_cls(self, encoder_input_ids, encoder_input_attention_mask, batch_aligned_column_info_ids, batch_aligned_table_name_ids, batch_column_number_in_each_table):
        batch_size = encoder_input_ids.shape[0]
        
        encoder_output = self.plm_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            return_dict=True
        )

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :]

            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            for table_name_ids in aligned_table_name_ids:
                table_name_embeddings = sequence_embeddings[table_name_ids, :]
                output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
                table_name_embedding = hidden_state_t[-2:, :].view(1, self.plm_hidden_size)
                table_name_embedding_list.append(table_name_embedding)
            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim=0)
            table_name_embeddings_in_one_db = self.leakyrelu(self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
            
            for column_info_ids in aligned_column_info_ids:
                column_info_embeddings = sequence_embeddings[column_info_ids, :]
                output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
                column_info_embedding = hidden_state_c[-2:, :].view(1, self.plm_hidden_size)
                column_info_embedding_list.append(column_info_embedding)
            column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim=0)
            column_info_embeddings_in_one_db = self.leakyrelu(self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))

            table_name_embeddings_in_one_db = self.table_column_cross_attention(
                table_name_embeddings_in_one_db, column_info_embeddings_in_one_db, column_number_in_each_table
            )
            
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)

            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(self, encoder_input_ids, encoder_attention_mask, batch_aligned_column_info_ids, batch_aligned_table_name_ids, batch_column_number_in_each_table):  
        batch_table_name_cls_logits, batch_column_info_cls_logits = self.table_column_cls(
            encoder_input_ids, encoder_attention_mask, batch_aligned_column_info_ids, batch_aligned_table_name_ids, batch_column_number_in_each_table
        )
        return {
            "batch_table_name_cls_logits": batch_table_name_cls_logits, 
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }
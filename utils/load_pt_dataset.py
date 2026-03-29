"""
预训练数据加载器
将二进制语料加载为 PyTorch Dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集
    
    使用 memmap 加载二进制语料,支持超大语料库
    """
    
    def __init__(self, pt_data_dir, block_size):
        """
        初始化
        
        参数:
            pt_data_dir: 二进制语料文件路径
            block_size: 序列长度
        """
        super().__init__()
        # 使用 memmap 加载,避免一次性加载到内存
        self.corpus = np.memmap(pt_data_dir, dtype=np.uint16, mode='r')
        self.block_size = block_size
        # 计算可以切分的序列数量
        self.length = len(self.corpus) // self.block_size

    def __getitem__(self, index):
        """
        获取一个序列
        
        参数:
            index: 序列索引
        
        返回:
            包含 input_ids, attention_mask, labels 的字典
        """
        # 读取一个 block 的 token
        input_ids = self.corpus[index * self.block_size: (index + 1) * self.block_size]
        
        # 转换为 PyTorch tensor
        input_ids = torch.from_numpy(input_ids.astype(np.int64))
        attention_mask = torch.ones(len(input_ids))
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": input_ids
        }

    def __len__(self):
        return self.length


if __name__ == "__main__":
    # 测试
    dataset = PretrainDataset("./data/pt_corpus/starcoder_corpus.bin", 6144)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)
    for batch in dataloader:
        print("batch:", batch)
    print("数据集长度:", len(dataset))
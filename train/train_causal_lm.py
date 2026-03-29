"""
Text2SQL 训练脚本
支持增量预训练 (PT) 和有监督微调 (SFT)

基模: StarCoder (bigcode/starcoderbase-1b/3b/7b/15b)
"""

import argparse
import os
import math
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR


def parse_option():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    # 全局参数
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='每 GPU 的 batch size')
    parser.add_argument('--block_size', type=int, default=8192,
                        help='序列长度,即训练样本的最大长度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子,用于保证结果可复现')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="bigcode/starcoder",
                        help='基模路径或 HuggingFace 名称,如 bigcode/starcoderbase-1b')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率: 预训练用 5e-5,SFT 用 5e-6')
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                        help='学习率 warmup 比例')
    parser.add_argument('--checkpointing_steps', type=int, default=300,
                        help='每多少步保存一次 checkpoint')
    parser.add_argument('--tensorboard_log_dir', type=str, default="./train_logs",
                        help='TensorBoard 日志目录')
    parser.add_argument('--mode', type=str, default="pt",
                        help='训练模式: pt (预训练) 或 sft (微调)')
    parser.add_argument('--output_ckpt_dir', type=str, default="./ckpts",
                        help='模型输出目录')
    parser.add_argument('--save_all_states', action='store_true', 
                        help='是否保存 optimizer 和 lr_scheduler 状态')

    # 预训练参数
    parser.add_argument('--pt_data_dir', type=str, default="./data/corpus.bin",
                        help='预训练语料路径 (二进制格式)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--resume_tag', type=str, default=None)
    
    # SFT 参数
    parser.add_argument('--text2sql_data_dir', type=str, default="./data/sft_train_text2sql.json",
                        help='SFT 训练数据路径')
    parser.add_argument('--table_num', type=int, default=6,
                        help='每个样本保留的最大表数量')
    parser.add_argument('--column_num', type=int, default=10,
                        help='每个表保留的最大列数量')
    
    opt = parser.parse_args()
    return opt


def checkpoint_model(accelerator, model, tokenizer, output_ckpt_dir, last_global_step):    
    """保存模型 checkpoint"""
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    accelerator.print("保存模型到: {}".format(ckpt_path))
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        ckpt_path, 
        is_main_process=accelerator.is_main_process, 
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        max_shard_size="100GB"
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_path)
    return


def train(opt):
    """训练主函数"""
    # 设置随机种子
    set_seed(opt.seed)

    # 初始化 TensorBoard
    writer = SummaryWriter(opt.tensorboard_log_dir)
    
    # 初始化 Accelerator (分布式训练)
    accelerator = Accelerator()
    print("主进程:", accelerator.is_main_process)
    print("设备:", accelerator.device)

    # 计算总 batch size
    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    
    accelerator.print(opt)
    accelerator.print("每 batch token 数:", total_batch_size * opt.block_size)
    accelerator.print("每 batch 序列数:", total_batch_size)
    accelerator.print("使用模型:", opt.pretrained_model_name_or_path)

    # 加载基模 (StarCoder)
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # 启用 gradient checkpointing 节省显存 (会降低 20-30% 训练速度)
    model.gradient_checkpointing_enable()

    # 根据模式加载数据集
    if opt.mode == "pt":
        # 预训练模式: 加载二进制语料
        dataset = PretrainDataset(opt.pt_data_dir, opt.block_size)
    elif opt.mode == "pt":
        # SFT 模式: 加载 Text2SQL 数据
        dataset = SFTSQLGenerationDataset(
            opt.text2sql_data_dir, 
            tokenizer, 
            opt.block_size, 
            "train", 
            opt.table_num, 
            opt.column_num, 
            None
        )
    else:
        raise ValueError("mode 应该是 'pt' 或 'sft'")
    
    dataloader = DataLoader(dataset, batch_size=opt.per_device_train_batch_size, shuffle=True, drop_last=True)

    # 计算总 batch 数
    num_total_batches = math.ceil(opt.epochs * math.ceil(len(dataset) / total_batch_size))
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # 学习率调度器: Linear Warmup + Cosine Annealing
    num_warm_up_batches = max(int(num_total_batches * opt.warmup_ratio), 1)
    accelerator.print("warmup batch 数:", num_warm_up_batches)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer, 
        warmup_epochs=num_warm_up_batches * accelerator.num_processes,
        max_epochs=num_total_batches * accelerator.num_processes, 
        warmup_start_lr=0.0, 
        eta_min=0.1 * opt.lr
    )

    # 准备训练 (自动处理分布式)
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(optimizer, model, dataloader, lr_scheduler)

    accumulation_loss = 0
    global_completed_steps = 0
    model.train()

    st = time.time()
    for epoch in range(opt.epochs):
        accelerator.print("开始训练 epoch:", epoch + 1)
        for batch_idx, batch in enumerate(dataloader):
            # 梯度累积
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accumulation_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 同步梯度时记录
            if accelerator.sync_gradients:
                global_completed_steps += 1
                accelerator.print("GPU 0, step {}, loss {}".format(
                    global_completed_steps, 
                    accumulation_loss / accelerator.gradient_accumulation_steps
                ))
                accelerator.print("耗时:", time.time() - st)
                st = time.time()

                # 记录到 TensorBoard
                writer.add_scalar(
                    'train-loss/gpu-{}'.format(accelerator.process_index), 
                    accumulation_loss / accelerator.gradient_accumulation_steps, 
                    global_completed_steps
                )
                writer.add_scalar(
                    'learning-rate/gpu-{}'.format(accelerator.process_index), 
                    lr_scheduler.get_last_lr()[0], 
                    global_completed_steps
                )
                accumulation_loss = 0

                # 保存 checkpoint
                if global_completed_steps % opt.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)

        # 每个 epoch 结束保存 checkpoint
        accelerator.print("epoch 结束,保存 checkpoint")
        accelerator.wait_for_everyone()
        checkpoint_model(accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps)


if __name__ == "__main__":
    opt = parse_option()
    train(opt)
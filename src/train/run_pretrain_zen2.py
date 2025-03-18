"""Make pretrained dataset from .txt/.fa files.

本脚本用于从原始文本文件或预处理后的token文件构建预训练数据集，
并将训练集和验证集保存到指定的输出目录中。

执行步骤：
1. 初始化配置和参数
2. 加载tokenizer和n-gram编码器
3. 准备预训练数据
4. 初始化模型和优化器
5. 执行训练循环
6. 保存训练结果
"""
import numpy as np
import pandas as pd
import ipdb
from collections import namedtuple
import ipdb
import random  # 用于生成随机数，确保实验可复现
from typing import Literal  # 用于类型注解，限制变量取值
import time  # 用于计算加载和下载时间
import os  # 操作系统相关功能，用于文件路径处理
import datetime  # 日期时间操作，用于生成时间戳
from pathlib import Path  # 文件路径处理，提供面向对象的文件系统路径处理
import json
import logging  # 日志记录模块，用于输出调试信息
from tqdm import tqdm  # 进度条显示，提供训练过程可视化

import click  # 用于简化命令行界面开发
import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler  # 数据加载和处理工具
import transformers
from transformers import (
    AutoTokenizer,  # 加载预训练的Tokenizer
    PreTrainedTokenizer,  # 预训练Tokenizer基类
    CONFIG_NAME,  # 模型配置文件名常量
    WEIGHTS_NAME,  # 模型权重文件名常量
    TrainingArguments,  # 训练参数配置
    Trainer,  # 训练器
)

# 导入ZEN模型相关组件
from ZEN.modeling import ZenForPreTraining, ZenConfig

# 导入自定义工具和模块
from dnazen.ngram import NgramEncoder  # n-gram编码器模块，用于处理n-gram特征
# 导入优化器和学习率调度器
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids lm_label_ids is_next ngram_ids ngram_masks ngram_positions ngram_starts ngram_lengths ngram_segment_ids")


# 配置日志输出格式和级别，方便调试和记录运行信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids lm_label_ids is_next ngram_ids ngram_masks ngram_positions ngram_starts ngram_lengths ngram_segment_ids")


def convert_example_to_features(example, tokenizer, max_seq_length, max_ngram_in_sequence):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    # add ngram level information
    ngram_ids = example["ngram_ids"]
    ngram_positions = example["ngram_positions"]
    ngram_lengths = example["ngram_lengths"]
    ngram_segment_ids = example["ngram_segment_ids"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int32)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int32, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    # add ngram pads
    #ipdb.set_trace()
    #DNA某些数据会出现大量的重复序列，从而导致ngram_ids 数量超过 max_ngram_in_sequence
    ngram_id_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    if len(ngram_ids) > max_ngram_in_sequence:
        #ipdb.set_trace()
        logger.debug(f"ngram_ids 数量超过 max_ngram_in_sequence: {len(ngram_ids)} > {max_ngram_in_sequence}; 数据：example  {example}")
        logger.info(f"ngram_ids 数量超过 max_ngram_in_sequence: {len(ngram_ids)} > {max_ngram_in_sequence}; ")

        ngram_id_array[:max_ngram_in_sequence] = ngram_ids[:max_ngram_in_sequence]
    else:
        ngram_id_array[:len(ngram_ids)] = ngram_ids
    # record the masked positions

    # The matrix here take too much space either in disk or in memory, so the usage have to be lazily convert the
    # the start position and length to the matrix at training time.

    ngram_positions_matrix = np.zeros(shape=(max_seq_length, max_ngram_in_sequence), dtype=bool)
    #for i in range(len(ngram_ids)):
    #    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i]+ngram_lengths[i], i] = 1
    #使用min(len(ngram_ids), max_ngram_in_sequence)来确保不会超出ngram的最大限制，添加了边界检查，确保ngram位置和长度不会超出序列最大长度
    for i in range(min(len(ngram_ids), max_ngram_in_sequence)):
        start_pos = ngram_positions[i]
        length = ngram_lengths[i]
        if start_pos + length <= max_seq_length:
            ngram_positions_matrix[start_pos:start_pos+length, i] = 1
        else:
            logger.warning(f"ngram位置超出序列长度限制: 起始位置={start_pos}, 长度={length}, 最大序列长度={max_seq_length}")

    ngram_start_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    if len(ngram_ids) > max_ngram_in_sequence:
        ngram_start_array[:max_ngram_in_sequence] = ngram_positions[:max_ngram_in_sequence]
    else:
        ngram_start_array[:len(ngram_ids)] = ngram_positions

    ngram_length_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    if len(ngram_ids) > max_ngram_in_sequence:
        ngram_length_array[:max_ngram_in_sequence] = ngram_lengths[:max_ngram_in_sequence]
    else:
        ngram_length_array[:len(ngram_ids)] = ngram_lengths

    ngram_mask_array = np.zeros(max_ngram_in_sequence, dtype=bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    ngram_segment_array = np.zeros(max_ngram_in_sequence, dtype=bool)
    if len(ngram_ids) > max_ngram_in_sequence:
        ngram_segment_array[:max_ngram_in_sequence] = ngram_segment_ids[:max_ngram_in_sequence]
    else:
        ngram_segment_array[:len(ngram_ids)] = ngram_segment_ids
    
    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next,
                             ngram_ids=ngram_id_array,
                             ngram_masks=ngram_mask_array,
                             ngram_positions=ngram_positions_matrix,
                             ngram_starts=ngram_start_array,
                             ngram_lengths=ngram_length_array,
                             ngram_segment_ids=ngram_segment_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, fp16=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        
        # 修复路径处理问题
        data_file = os.path.join(training_path, f"epoch_{self.data_epoch}.json")
        metrics_file = os.path.join(training_path, f"epoch_{self.data_epoch}_metrics.json")
        cache_file = os.path.join(training_path, f"epoch_{self.data_epoch}_cache.pt")
        
        assert os.path.exists(data_file) and os.path.exists(metrics_file)
        
        # 读取metrics文件
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        max_ngram_in_sequence = metrics['max_ngram_in_sequence']
        self.temp_dir = None
        self.working_dir = None
        self.fp16 = fp16
        
        if reduce_memory:
            self.temp_dir = "/tmp"
            self.working_dir = Path(self.temp_dir)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
            # add ngram level features
            ngram_ids = np.memmap(filename=self.working_dir / 'ngram_ids.memmap',
                                 mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_masks = np.memmap(filename=self.working_dir / 'ngram_masks.memmap',
                                   mode='w+', dtype=np.bool, shape=(num_samples, max_ngram_in_sequence))

            ngram_positions = np.memmap(filename=self.working_dir / 'ngram_positions.memmap',
                                      mode='w+', dtype=np.bool, shape=(num_samples, seq_len, max_ngram_in_sequence))

            ngram_starts = np.memmap(filename=self.working_dir / 'ngram_starts.memmap',
                                    mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_lengths = np.memmap(filename=self.working_dir / 'ngram_lengths.memmap',
                                     mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_segment_ids = np.memmap(filename=self.working_dir / 'ngram_segment_ids.memmap',
                                         mode='w+', dtype=np.bool, shape=(num_samples, max_ngram_in_sequence))

        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
            # add ngram level features

            ngram_ids = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)
            ngram_masks = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.bool)

            ngram_positions = np.zeros(shape=(num_samples, seq_len, max_ngram_in_sequence), dtype=np.bool)
            ngram_starts = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)
            ngram_lengths = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)

            ngram_segment_ids = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.bool)

        logging.info(f"Loading training examples form {data_file} for epoch {epoch}")
        with open(data_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len, max_ngram_in_sequence)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                ngram_ids[i] = features.ngram_ids
                ngram_masks[i] = features.ngram_masks
                ngram_positions[i] = features.ngram_positions
                ngram_starts[i] = features.ngram_starts
                ngram_lengths[i] = features.ngram_lengths
                ngram_segment_ids[i] = features.ngram_segment_ids

        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        self.ngram_ids = ngram_ids
        self.ngram_masks = ngram_masks
        self.ngram_positions = ngram_positions
        self.ngram_segment_ids = ngram_segment_ids
        self.ngram_starts = ngram_starts
        self.ngram_lengths = ngram_lengths

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        position = torch.tensor(self.ngram_positions[item].astype(np.double))
        if self.fp16:
            position = position.half()
        else:
            position = position.float()
        # input_ids, input_mask, segment_ids, lm_label_ids, is_next, ngram_ids, ngram_masks, ngram_positions,  ngram_starts, ngram_lengths, ngram_segment_ids = batch
        return {
            "input_ids": torch.tensor(self.input_ids[item].astype(np.int64)),
            "attention_mask": torch.tensor(self.input_masks[item].astype(np.int64)),
            "token_type_ids": torch.tensor(self.segment_ids[item].astype(np.int64)),
            "labels": torch.tensor(self.lm_label_ids[item].astype(np.int64)),
            "next_sentence_label": torch.tensor(self.is_nexts[item].astype(np.int64)),
            "ngram_ids": torch.tensor(self.ngram_ids[item].astype(np.int64)),
            "ngram_masks": torch.tensor(self.ngram_masks[item].astype(np.int64)),
            "ngram_positions": position,
            "ngram_starts": torch.tensor(self.ngram_starts[item].astype(np.int64)),
            "ngram_lengths": torch.tensor(self.ngram_lengths[item].astype(np.int64)),
            "ngram_segment_ids": torch.tensor(self.ngram_segment_ids[item].astype(np.int64))
        }



@click.command()
@click.option("--data-source", type=click.Choice(["raw", "tokenized"]), default="raw", help="数据类型：原始数据(raw)或已分词数据(tokenized)")
@click.option("-d", "--data", "data_dir", type=str, default="/data2/peter/dnazen_pretrain_v3", help="数据文件的路径目录")
@click.option("--tok-source", "tokenizer_source", type=click.Choice(["file", "huggingface"]), default="huggingface", help="Tokenizer的加载方式：文件或Huggingface模型")
@click.option("--tok", "tokenizer_cfg", type=str, default="zhihan1996/DNABERT-2-117M", help="Tokenizer的配置，支持模型名称或路径")
@click.option("--ngram", "ngram_file", type=str, required=True, help="n-gram解码器的配置文件路径")
@click.option("--core-ngram", type=str, default=None, help="核心n-gram文件路径，可选")
@click.option("--max-ngrams", default=30, type=int, help="最大允许匹配的n-gram数量")
@click.option("--out", "output_dir", type=str, required=True, help="保存数据集输出目录的路径")
@click.option("--seed", type=int, default=42, help="随机种子，用于确保实验可重复")
@click.option("--model", "bert_model", type=str, default="zhihan1996/DNABERT-2-117M", help="预训练模型路径或名称")
@click.option("--lr", "learning_rate", type=float, default=2e-5, help="学习率，默认2e-5")
@click.option("--loss-scale", type=float, default=0, help="FP16训练的损失缩放系数，0表示动态缩放")
@click.option("--warmup", "warmup_proportion", type=float, default=0.1, help="学习率预热比例，默认0.1")
@click.option("--data-epochs", "num_data_epochs", type=int, default=1, help="数据训练轮数")
@click.option("--reduce-mem/--no-reduce-mem", "reduce_memory", default=True, help="是否启用内存优化")
@click.option("--epochs", type=int, default=10, help="训练总轮数")
@click.option("--batch-size", "train_batch_size", type=int, default=128, help="训练批次大小")
@click.option("--grad-accum", "gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
@click.option("--local-rank", type=int, default=-1, help="分布式训练的本地排名，-1表示单机训练")
@click.option("--fp16/--no-fp16", default=False, help="是否使用半精度(FP16)训练")
@click.option("--scratch/--no-scratch", default=False, help="是否从零开始训练，不使用预训练模型")
@click.option("--save-prefix", "save_name", type=str, default="dnazen_", help="保存模型的名称前缀")
def main(
    data_source: Literal["raw", "tokenized"],
    data_dir: str,
    tokenizer_source: Literal["file", "huggingface"],
    tokenizer_cfg: str,
    ngram_file: str,
    core_ngram: str | None,
    max_ngrams: int,
    output_dir: str,
    seed: int,
    bert_model: str,
    learning_rate: float,
    loss_scale: float,
    warmup_proportion: float,
    num_data_epochs: int,
    reduce_memory: bool,
    epochs: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    local_rank: int,
    fp16: bool,
    scratch: bool,
    save_name: str,
):
    """DNA序列预训练主程序
    
    本程序用于处理DNA序列数据并进行预训练，支持以下功能：
    1. 支持原始数据和已分词数据的处理
    2. 支持从Huggingface或本地加载tokenizer
    3. 支持n-gram特征的提取和使用
    4. 支持多GPU训练和混合精度训练
    5. 提供完整的训练参数配置选项

    """
    random.seed(seed)  # 设置随机种子，保证实验结果可复现

    # 获取logger对象，用于打印调试信息
    logger = logging.getLogger(__name__)

    assert output_dir is not None

    # 开始记录加载tokenizer的时间
    start_time = time.time()
    logger.info(f"加载tokenizer: {tokenizer_cfg}，tokenizer_source: {tokenizer_source}")
    if tokenizer_source == "huggingface":
        # 从Huggingface模型库中加载预训练tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg)
    else:
        # 从本地文件加载tokenizer
        tokenizer = AutoTokenizer.from_file(tokenizer_cfg)
    end_time = time.time()
    logger.info("Tokenizer加载耗时: %.2f秒", end_time - start_time)

    tokenizer.model_max_length = 256

    # 开始记录加载n-gram编码器的时间
    start_time = time.time()
    logger.info(f"加载n-gram编码器: {ngram_file}")
    ngram_encoder = NgramEncoder.from_file(ngram_file)
    ngram_encoder.set_max_ngram_match(max_ngrams=max_ngrams)
    end_time = time.time()
    logger.info("N-gram编码器加载耗时: %.2f秒", end_time - start_time)
    
    if data_source == "tokenized":
        
        prepare_pretrain_data(
            data_source=data_source,
            data_dir=data_dir,
            tokenizer_source=tokenizer_source,
            tokenizer_cfg=tokenizer_cfg,
            ngram_file=ngram_file,
            output_dir=output_dir,
            seed=seed,
            bert_model=bert_model,
            learning_rate=learning_rate,
            loss_scale=loss_scale,
            warmup_proportion=warmup_proportion,
            num_data_epochs=num_data_epochs,
            reduce_memory=reduce_memory,
            epochs=epochs,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            local_rank=local_rank,
            scratch=scratch,
            fp16=fp16,
            save_name=save_name,
            already_trained_epoch=0,
            pregenerated_data=None,
        )


def get_samples_per_epoch(pregenerated_data, num_data_epochs):
    """获取每个epoch的样本数
    
    参数:
        pregenerated_data: 预生成数据的路径
        num_data_epochs: 数据训练的轮数
        
    返回:
        samples_per_epoch: 每个epoch的样本数列表
    """
    samples_per_epoch = []
    logger.info(f"获取每个epoch的样本数: {pregenerated_data}")
    for i in range(num_data_epochs):
        epoch_file = os.path.join(pregenerated_data, f"epoch_{i}.json")
        metrics_file = os.path.join(pregenerated_data, f"epoch_{i}_metrics.json")
        logger.info(f"检查文件: {epoch_file} 和 {metrics_file}")
        if os.path.isfile(epoch_file) and os.path.isfile(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                logger.error("未找到训练数据！")
                return []
            logger.warning(f"预生成数据的epoch数({i})小于训练epoch数({num_data_epochs})")
            logger.warning("脚本将循环使用可用数据，但可能影响训练的多样性")
            break
    return samples_per_epoch

def prepare_pretrain_data(
    data_source: Literal["raw", "tokenized"],
    data_dir: str,
    tokenizer_source: Literal["file", "huggingface"],
    tokenizer_cfg: str,
    ngram_file: str,
    output_dir: str,
    seed: int,
    bert_model: str,
    learning_rate: float,
    loss_scale: float,
    warmup_proportion: float,
    num_data_epochs: int,
    reduce_memory: bool,
    epochs: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    local_rank: int,
    scratch: bool,
    fp16: bool,
    save_name: str,
    already_trained_epoch: int,
    pregenerated_data: str | None,
):
    """准备预训练数据并训练模型"""
    
    try:
        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if pregenerated_data is None:
            pregenerated_data = data_dir
            
        logger.info(f"预训练数据路径: {pregenerated_data}")
        
        # 获取每个epoch的样本数
        try:
            samples_per_epoch = get_samples_per_epoch(pregenerated_data, num_data_epochs)
            if not samples_per_epoch:
                raise ValueError(f"在路径 {pregenerated_data} 中未找到有效的预训练数据")
        except Exception as e:
            logger.error(f"获取样本数时出错: {str(e)}")
            logger.error(f"请检查数据路径 {pregenerated_data} 是否包含正确的epoch_{i}.json和metrics文件")
            raise

        # 计算总训练步数
        total_train_examples = sum(samples_per_epoch[i % len(samples_per_epoch)] for i in range(epochs))
        num_train_optimization_steps = int(total_train_examples / train_batch_size / gradient_accumulation_steps)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg)

        logger.info(f"加载预训练模型: {bert_model}")
        # 使用BertForMaskedLM替代ZenForPreTraining
        model = ZenForPreTraining.from_pretrained(bert_model)
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        logger.info(f"参数比例: {trainable_params/total_params*100:.2f}%")

        # 配置训练参数
        logger.info("配置训练参数...")
        train_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=False,  # 预训练阶段不需要评估
            save_strategy="steps",
            save_steps=1000,
            max_grad_norm=1,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            logging_steps=100,
            dataloader_num_workers=4,
            dataloader_prefetch_factor=2,
            warmup_steps=int(num_train_optimization_steps * warmup_proportion),
            save_safetensors=False,
            seed=seed,
            data_seed=seed,
            save_total_limit=10,
            load_best_model_at_end=False,  # 预训练阶段不需要加载最佳模型
            report_to="tensorboard",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_first_step=True,
            fp16=fp16,
            fp16_backend="amp",  # 使用PyTorch原生的AMP
            fp16_opt_level="O1",
            # 添加以下参数以避免某些版本的问题
            remove_unused_columns=False,
            label_names=["lm_label_ids", "is_next"],  # 修改为与模型期望的参数名匹配
        )

        # 创建日志目录
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        logger.info(f"训练日志将保存到: {os.path.join(output_dir, 'logs')}")
        
        # 创建训练数据集
        train_dataset = PregeneratedDataset(
            training_path=pregenerated_data,
            epoch=0,  # 初始epoch
            tokenizer=tokenizer,
            num_data_epochs=num_data_epochs,
            reduce_memory=reduce_memory,
            fp16=fp16
        )

        # 定义数据整理函数
        def data_collator(features):
            #ipdb.set_trace()
            batch = {
                # "input_ids": torch.stack([f["input_ids"] for f in features]),
                # "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                # "token_type_ids": torch.stack([f["token_type_ids"] for f in features]),
                # "labels": torch.stack([f["labels"] for f in features]),
                # "next_sentence_label": torch.stack([f["next_sentence_label"] for f in features]),
                # "ngram_ids": torch.stack([f["ngram_ids"] for f in features]),
                # "ngram_masks": torch.stack([f["ngram_masks"] for f in features]),
                # "ngram_positions": torch.stack([f["ngram_positions"] for f in features]),
                # "ngram_starts": torch.stack([f["ngram_starts"] for f in features]),
                # "ngram_lengths": torch.stack([f["ngram_lengths"] for f in features]),
                # "ngram_segment_ids": torch.stack([f["ngram_segment_ids"] for f in features])
                
                "input_ids": torch.stack([f["input_ids"] for f in features]),
                "token_type_ids": torch.stack([f["token_type_ids"] for f in features]),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),
                
                "input_ngram_ids": torch.stack([f["ngram_ids"] for f in features]),
                "ngram_position_matrix": torch.stack([f["ngram_positions"] for f in features]),
                #segment_ids
                "ngram_token_type_ids": torch.stack([f["ngram_segment_ids"] for f in features]),
                
                "ngram_attention_mask": torch.stack([f["ngram_masks"] for f in features]),
                
                #"is_next": torch.stack([f["next_sentence_label"] for f in features]),
                
                
        #  `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        #     with the word token indices in the vocabulary
        # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
        #     types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
        #     a `sentence B` token (see BERT paper for more details).
        # `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
        #     selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
        #     input sequence length in the current batch. It's the mask that we typically use for attention when
        #     a batch has varying length sentences.
        # `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
        # `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
        #     It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        # `input_ngram_ids`: input_ids of ngrams.
        # `ngram_token_type_ids`: token_type_ids of ngrams.
        # `ngram_attention_mask`: attention_mask of ngrams.
        # `ngram_position_matrix`: position matrix of ngrams.
            }

            return batch

        # 定义回调函数，用于记录训练进度
        class TrainingProgressCallback(transformers.TrainerCallback):
            def __init__(self, total_steps):
                self.total_steps = total_steps
                self.start_time = time.time()
                self.last_log_time = time.time()
                
            def on_step_end(self, args, state, control, **kwargs):
                current_time = time.time()
                if state.global_step % 100 == 0 or (current_time - self.last_log_time) > 60:
                    elapsed = current_time - self.start_time
                    progress = state.global_step / self.total_steps * 100
                    steps_per_second = state.global_step / elapsed
                    remaining = (self.total_steps - state.global_step) / steps_per_second if steps_per_second > 0 else 0
                    
                    logger.info(
                        f"训练进度: {progress:.2f}% ({state.global_step}/{self.total_steps}) "
                        f"速度: {steps_per_second:.2f}步/秒 "
                        f"已用时间: {elapsed/60:.2f}分钟 "
                        f"预计剩余: {remaining/60:.2f}分钟"
                    )
                    self.last_log_time = current_time
        
        # 计算总步数用于进度显示
        total_steps = num_train_optimization_steps
        logger.info(f"总训练步数: {total_steps}")
        progress_callback = TrainingProgressCallback(total_steps)

        # 创建Trainer实例
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            callbacks=[progress_callback],
            data_collator=data_collator,  # 添加数据整理函数
        )
        
        # 开始训练
        logger.info("="*80)
        logger.info("开始训练...")
        logger.info("="*80)
        
        #########################################################
        train_start_time = time.time()
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            logger.error("详细错误信息:")
            import traceback
            logger.error(traceback.format_exc())
            raise
        #########################################################
        train_time = time.time() - train_start_time
        
        # 训练完成，保存最终模型
        logger.info("="*80)
        logger.info(f"训练完成！总用时: {train_time/60:.2f}分钟")
        #########################################################
        
        
        # 保存最终模型
        final_output_dir = os.path.join(output_dir, f"{save_name}_final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        output_model_file = os.path.join(final_output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(final_output_dir, "config.json")
        
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(final_output_dir)
        
        logger.info(f"最终模型保存至: {final_output_dir}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"预训练数据准备过程出错: {str(e)}")
        logger.error("详细错误信息:")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # 当脚本作为主程序执行时，调用main函数
    main()

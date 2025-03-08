#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名: run_pretrain.py
描述: DNA序列预训练模型训练脚本，使用ZEN架构进行掩码语言模型预训练
作者: DnaZen Team
使用方法:
    python run_pretrain.py
        --train <训练数据目录>
        --dev <验证数据目录>
        --out <输出目录>
        [--其他参数]
"""

###################################################################################
# 导入所需库
###################################################################################
# 标准库
import os
import logging
import argparse
import time  # 添加time模块
from typing import Any, Dict, Tuple, Union
from argparse import ArgumentParser
import hashlib
import math
import json

# 科学计算库
import numpy as np

# 机器学习相关库
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# PyTorch相关库
import torch
import transformers
from transformers import (
    AutoTokenizer,
)
from transformers.models.bert.configuration_bert import BertConfig

# DnaZen自定义库
from dnazen.data.mlm_dataset import MlmDataset
from dnazen.model.bert_models import BertForMaskedLM
from dnazen.model.bert_config import ZenConfig
from dnazen.ngram import NgramEncoder

###################################################################################
# 日志配置
###################################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



###################################################################################
# 工具函数
###################################################################################
def set_random_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保实验可重复性

    参数:
        seed: 随机种子值，默认为42
    """
    logger.info(f"设置随机种子: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    计算评估指标，用于Huggingface Trainer

    参数:
        eval_pred: 包含预测值和标签的元组，由Trainer内部传入

    返回:
        包含多个评估指标的字典
    """
    predictions, labels = eval_pred
    # 整形预测和标签数据
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # 排除填充标记(假设-100是填充标记ID)
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    # 计算多种评估指标
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(valid_labels, valid_predictions),
        "precision": precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _) -> torch.Tensor:
    """
    在计算评估指标前预处理模型输出的logits

    参数:
        logits: 模型输出的logits，可能是张量或元组
        _: 忽略的第二个参数

    返回:
        处理后的预测张量

    注: 解决方案参考自
    https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
    """
    if isinstance(logits, tuple):  # 如果logits是元组则解包
        logits = logits[0]

    return torch.argmax(logits, dim=-1)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = ArgumentParser(description="DNA序列预训练模型训练脚本")
    
    # 数据相关参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--train", type=str, required=True, help="训练数据文件路径")
    data_group.add_argument("--dev", type=str, required=True, help="验证数据文件路径")
    data_group.add_argument("--out", type=str, required=True, help="输出目录路径")
    data_group.add_argument("--train_dir", type=str, help="训练数据目录路径")
    data_group.add_argument("--dev_dir", type=str, help="验证数据目录路径")
    data_group.add_argument("--cache-dir", type=str, help="数据缓存目录")
    
    # 模型相关参数
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument("--resume", type=str, help="从检查点恢复训练的路径")
    model_group.add_argument("--num_ngram_hidden_layer", type=int, default=6, help="N-gram隐藏层数量")
    
    # N-gram相关参数
    ngram_group = parser.add_argument_group("N-gram参数")
    ngram_group.add_argument("--ngram-encoder-path", type=str, help="N-gram编码器路径")
    ngram_group.add_argument("--train-ngram", action="store_true", help="是否训练N-gram编码器")
    ngram_group.add_argument("--min-ngram-len", type=int, default=2, help="最小N-gram长度")
    ngram_group.add_argument("--max-ngram-len", type=int, default=5, help="最大N-gram长度")
    ngram_group.add_argument("--max-ngrams-per-seq", type=int, default=30, help="每个序列最大N-gram数量")
    ngram_group.add_argument("--min-ngram-freq", type=int, default=2, help="最小N-gram频率")
    ngram_group.add_argument("--ngram-method", type=str, choices=["freq", "pmi"], default="pmi", help="N-gram选择方法")
    ngram_group.add_argument("--max-train-sequences", type=int, default=10000, help="训练N-gram时使用的最大序列数")
    
    # 训练相关参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--per-device-train-batch-size", type=int, default=32, help="每个设备的训练批量大小")
    train_group.add_argument("--grad-accumulation-steps", type=int, default=1, help="梯度累积步数")
    train_group.add_argument("--lr", type=float, default=5e-5, help="学习率")
    train_group.add_argument("--n-epoch", type=int, default=3, help="训练轮数")
    train_group.add_argument("--seed", type=int, default=42, help="随机种子")
    train_group.add_argument("--num-workers", type=int, default=4, help="数据加载器工作线程数")
    
    # 数据加载模式
    load_group = parser.add_argument_group("数据加载模式")
    load_group.add_argument("--streaming", action="store_true", help="是否使用流式数据加载")
    load_group.add_argument("--buffer-size", type=int, default=10000, help="流式加载的缓冲区大小")
    
    return parser.parse_args()


###################################################################################
# 主要训练函数
###################################################################################
def load_data_from_file(file_path, tokenizer, max_length=512, cache_dir=None):
    """
    从单个文件加载DNA序列数据，支持缓存加速

    参数:
        file_path: 数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        cache_dir: 缓存目录，如果为None则不使用缓存

    返回:
        包含已处理数据的PyTorch数据集
    """
    # 记录开始时间
    start_time = time.time()

    # 生成缓存文件路径
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"dna_dataset_{file_hash}.pt")

        # 如果缓存文件存在，直接加载
        if os.path.exists(cache_file):
            logger.info(f"从缓存加载数据: {cache_file}")
            dataset = torch.load(cache_file)
            logger.info(f"缓存加载完成，用时: {time.time() - start_time:.2f}秒")
            logger.info(f"数据集大小: {len(dataset)} 条序列")
            return dataset

    logger.info(f"从文件加载数据: {file_path}")

    # 读取文件内容 - 使用内存映射加速大文件读取
    file_read_start = time.time()
    sequences = []
    try:
        import mmap

        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for i, line in enumerate(iter(mm.readline, b"")):
                    if i % 1000000 == 0 and i > 0:
                        logger.info(f"已读取 {i} 行...")

                    line = line.decode().strip()
                    if line and not line.startswith(">"):
                        sequences.append(line)
    except (ImportError, ValueError):
        # 回退到普通文件读取
        logger.warning("无法使用内存映射，回退到标准文件读取")
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0 and i > 0:
                    logger.info(f"已读取 {i} 行...")

                line = line.strip()
                if line and not line.startswith(">"):
                    sequences.append(line)

    file_read_end = time.time()
    logger.info(f"文件读取完成，加载了 {len(sequences)} 条DNA序列")
    logger.info(f"文件读取用时: {file_read_end - file_read_start:.2f}秒")

    # 统计序列长度分布
    seq_lengths = [len(seq) for seq in sequences[:10000]]  # 取样本统计
    logger.info("序列长度统计 (基于前10000条):")
    logger.info(f"  最小长度: {min(seq_lengths)}")
    logger.info(f"  最大长度: {max(seq_lengths)}")
    logger.info(f"  平均长度: {sum(seq_lengths) / len(seq_lengths):.2f}")

    # 使用多进程加速数据处理
    logger.info("开始处理序列数据...")
    process_start = time.time()

    # 创建PyTorch数据集
    dataset = DNADataset(sequences, tokenizer, max_length)

    process_end = time.time()
    logger.info(f"数据处理完成，用时: {process_end - process_start:.2f}秒")

    # 保存到缓存
    if cache_file:
        logger.info(f"保存数据到缓存: {cache_file}")
        cache_start = time.time()
        torch.save(dataset, cache_file)
        logger.info(f"缓存保存完成，用时: {time.time() - cache_start:.2f}秒")

    total_time = time.time() - start_time
    logger.info(f"数据加载总用时: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")

    return dataset


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 使用tokenizer处理序列
        inputs = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # 准备MLM任务的输入和标签
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # 创建MLM标签（复制输入ID）
        labels = input_ids.clone()

        # 将一部分token替换为[MASK]
        mask_token_id = self.tokenizer.mask_token_id
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )

        # 不掩码特殊token
        probability_matrix = torch.full(input_ids.shape, 0.15)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 随机选择要掩码的token
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码token的损失

        # 80%的情况下用[MASK]替换
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id

        # 10%的情况下用随机token替换
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self):
        return len(self.sequences)


class StreamingDNADataset(torch.utils.data.IterableDataset):
    """流式DNA数据集，适用于超大数据集"""

    def __init__(self, file_path, tokenizer, max_length=512, buffer_size=10000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size

        # 获取文件大小用于日志
        self.file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"初始化流式数据集: {file_path} (大小: {self.file_size:.2f} MB)")
        logger.info(f"缓冲区大小: {buffer_size} 条序列")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        logger.info(f"Worker {worker_id}/{num_workers} 开始加载数据")
        start_time = time.time()
        processed_count = 0

        # 打开文件
        with open(self.file_path, "r") as f:
            # 如果有多个worker，分配文件区域
            if worker_info is not None:
                # 获取文件大小
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                # 计算每个worker的区域
                per_worker = file_size // worker_info.num_workers
                start = worker_info.id * per_worker
                end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else file_size

                logger.info(
                    f"Worker {worker_id}: 处理文件区域 {start / (1024 * 1024):.2f}MB - {end / (1024 * 1024):.2f}MB"
                )

                # 移动到起始位置
                f.seek(start)

                # 如果不在行首，读取到下一行开始
                if start > 0:
                    f.readline()

                # 读取直到结束位置
                position = f.tell()
                buffer = []

                while position < end:
                    line = f.readline().strip()
                    position = f.tell()

                    if not line or line.startswith(">"):
                        continue

                    # 处理序列并生成样本
                    buffer.append(line)

                    # 当缓冲区达到一定大小时处理
                    if len(buffer) >= self.buffer_size:
                        logger.info(f"Worker {worker_id}: 处理缓冲区 ({len(buffer)} 条序列)")
                        for seq in buffer:
                            processed_count += 1
                            if processed_count % 100000 == 0:
                                elapsed = time.time() - start_time
                                logger.info(
                                    f"Worker {worker_id}: 已处理 {processed_count} 条序列，"
                                    f"用时 {elapsed:.2f}秒 ({processed_count / elapsed:.2f} 条/秒)"
                                )
                            yield self._process_sequence(seq)
                        buffer = []

                # 处理剩余的序列
                if buffer:
                    logger.info(f"Worker {worker_id}: 处理剩余缓冲区 ({len(buffer)} 条序列)")
                    for seq in buffer:
                        processed_count += 1
                        yield self._process_sequence(seq)
            else:
                # 单worker模式
                buffer = []
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(">"):
                        continue

                    buffer.append(line)

                    if len(buffer) >= self.buffer_size:
                        logger.info(f"处理缓冲区 ({len(buffer)} 条序列)")
                        for seq in buffer:
                            processed_count += 1
                            if processed_count % 100000 == 0:
                                elapsed = time.time() - start_time
                                logger.info(
                                    f"已处理 {processed_count} 条序列，"
                                    f"用时 {elapsed:.2f}秒 ({processed_count / elapsed:.2f} 条/秒)"
                                )
                            yield self._process_sequence(seq)
                        buffer = []

                # 处理剩余的序列
                if buffer:
                    logger.info(f"处理剩余缓冲区 ({len(buffer)} 条序列)")
                    for seq in buffer:
                        processed_count += 1
                        yield self._process_sequence(seq)

        total_time = time.time() - start_time
        logger.info(
            f"Worker {worker_id}: 数据加载完成，共处理 {processed_count} 条序列，"
            f"总用时 {total_time:.2f}秒 ({processed_count / total_time:.2f} 条/秒)"
        )

    def _process_sequence(self, seq):
        # 使用tokenizer处理序列
        inputs = self.tokenizer(
            seq, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # 准备MLM任务的输入和标签
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # 创建MLM标签（复制输入ID）
        labels = input_ids.clone()

        # 将一部分token替换为[MASK]
        mask_token_id = self.tokenizer.mask_token_id
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )

        # 不掩码特殊token
        probability_matrix = torch.full(input_ids.shape, 0.15)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # 随机选择要掩码的token
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码token的损失

        # 80%的情况下用[MASK]替换
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id

        # 10%的情况下用随机token替换
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



def mlmdataset_builder(
        save_dir,
        ngram_encoder: NgramEncoder,
        data_path: str,
        data_config_path: str,
        tokenizer=None,
        max_ngrams=20,
        check_hash: bool = False,
    ):
        """从保存目录加载MLM数据集。

        Args:
            save_dir: 保存目录路径
            tokenizer: 分词器，如果为None则从保存目录加载
            max_seq_len: 最大序列长度
            max_ngrams: 每个序列最多匹配的N-gram数量
            **kwargs: 其他参数

        Returns:
            MLMDataset实例
        """
        logger.info(f"正在从目录加载MLM数据集: {save_dir}")
        

        logger.info(f"N-gram词汇表大小: {ngram_encoder.get_vocab_size()}")
        logger.info(f"N-gram长度范围: {ngram_encoder._min_ngram_len}-{ngram_encoder._max_ngram_len}")
        
        # 设置最大N-gram匹配数
        logger.info(f"设置最大N-gram匹配数: {max_ngrams}")
        ngram_encoder.set_max_ngram_match(max_ngrams)

        # 加载分词器
        if tokenizer is None:
            logger.error("使用提供的分词器")
            exit(1)

        # 加载数据
        logger.info(f"从{data_path}加载数据...")
        try:
            # 检查文件格式
            with open(data_path, 'rb') as f:
                first_bytes = f.read(10)  # 读取前10个字节来判断文件类型
            
            # 重置文件指针
            if first_bytes.startswith(b'PK\x03\x04'):
                logger.info("检测到ZIP格式文件，尝试使用torch.load加载...")
                data = torch.load(data_path)
            elif first_bytes.startswith(b'\x80\x03'):
                logger.info("检测到pickle格式文件，使用pickle加载...")
                import pickle
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                logger.info("未知文件格式，尝试使用numpy加载...")
                import numpy as np
                data = {}
                # 尝试加载.npz文件
                try:
                    npz_data = np.load(data_path)
                    for key in npz_data.files:
                        data[key] = torch.from_numpy(npz_data[key])
                    logger.info(f"成功从numpy文件加载数据，包含键: {list(data.keys())}")
                except Exception as e:
                    logger.error(f"numpy加载失败: {e}")
                    # 尝试加载文本文件
                    logger.info("尝试作为文本文件加载...")
                    with open(data_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 假设文件是每行一个序列的格式
                    sequences = [line.strip() for line in lines if line.strip()]
                    logger.info(f"加载了{len(sequences)}个文本序列")
                    
                    # 使用tokenizer处理序列
                    logger.info("使用tokenizer处理序列...")
                    input_ids = []
                    attention_mask = []
                    for seq in sequences:
                        encoded = tokenizer(seq, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                        input_ids.append(encoded["input_ids"][0])
                        attention_mask.append(encoded["attention_mask"][0])
                    
                    data["input_ids"] = torch.stack(input_ids)
                    data["attention_mask"] = torch.stack(attention_mask)
                    logger.info(f"处理完成，生成了形状为{data['input_ids'].shape}的张量")
            
            logger.info(f"数据加载成功，包含键: {list(data.keys())}")
            logger.info(f"输入ID形状: {data['input_ids'].shape}")
            
        except Exception as e:
            logger.error(f"所有加载方法都失败: {e}")
            logger.error(f"请检查数据文件格式: {data_path}")
            raise ValueError(f"无法加载数据文件: {data_path}")

        # 加载配置
        logger.info(f"从{data_config_path}加载配置...")
        with open(data_config_path, "r") as f:
            config = json.load(f)
        logger.info(f"配置信息: {config}")
        
        # 检查哈希值
        if config["mlm_data_symlink"] is not None and check_hash:
            assert config["mlm_data_hash_val"] is not None
            logger.info(f"检查数据文件{data_path}的MD5哈希值...")
            hash_identical = check_hash_of_file_md5(data_path, config["mlm_data_hash_val"])
            if not hash_identical:
                logger.error(f"哈希值不匹配！原始数据可能已被修改")
                raise ValueError(
                    f"尝试打开文件 {data_path}, ",
                    "但原始数据似乎已被修改。",
                )
            logger.info("哈希值匹配，数据完整性验证通过")
        elif check_hash:
            logger.warning("当不使用符号链接时，无法支持哈希检查")

        # 创建数据集实例
        logger.info("创建MLM数据集实例...")
        dataset = MlmDataset(
            tokens=data["input_ids"],
            attn_mask=data["attention_mask"],
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=set(),  # 暂时使用空集合
            whole_ngram_masking=config.get("whole_ngram_masking", False),
            mlm_prob=config["mlm_prob"],
            mlm_data_symlink=config["mlm_data_symlink"],
        )
        logger.info("MLM数据集加载完成!")

        return dataset
    

def main():
    """主函数：执行预训练流程"""
    start_time = time.time()
    
    # 步骤0: 初始化
    logger.info("="*80)
    logger.info("DNA序列预训练开始")
    logger.info("="*80)
    
    # 解析命令行参数
    logger.info("步骤1: 解析命令行参数...")
    args = parse_args()
    
    # 设置随机种子
    logger.info(f"设置随机种子: {args.seed}")
    set_random_seed(args.seed)
    
    # 提取参数并添加日志记录
    logger.info("提取命令行参数...")
    
    # 数据文件和目录参数
    train_data_file = args.train
    dev_data_file = args.dev
    train_dir = args.train_dir
    dev_dir = args.dev_dir
    output_dir = args.out
    cache_dir = args.cache_dir
    
    # 模型参数
    num_ngram_hidden_layer = args.num_ngram_hidden_layer
    learning_rate = args.lr
    
    # 检查必要参数
    logger.info("检查必要参数...")
    if not os.path.exists(train_data_file):
        logger.error(f"训练数据文件不存在: {train_data_file}")
        raise FileNotFoundError(f"训练数据文件不存在: {train_data_file}")
        
    if not os.path.exists(dev_data_file):
        logger.error(f"验证数据文件不存在: {dev_data_file}")
        raise FileNotFoundError(f"验证数据文件不存在: {dev_data_file}")
    
    # 创建输出目录
    logger.info(f"创建输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取并打印文件大小
    train_file_size = os.path.getsize(train_data_file) / (1024 * 1024)  # 转换为MB
    dev_file_size = os.path.getsize(dev_data_file) / (1024 * 1024)  # 转换为MB
    
    logger.info("-"*50)
    logger.info("数据文件信息:")
    logger.info(f"训练数据文件: {train_data_file} (大小: {train_file_size:.2f} MB)")
    logger.info(f"验证数据文件: {dev_data_file} (大小: {dev_file_size:.2f} MB)")
    logger.info(f"输出目录: {output_dir}")
    
    # 打印训练参数
    logger.info("-"*50)
    logger.info("训练参数:")
    logger.info(f"  每设备批量大小: {args.per_device_train_batch_size}")
    logger.info(f"  梯度累积步数: {args.grad_accumulation_steps}")
    logger.info(f"  实际批大小: {args.per_device_train_batch_size * args.grad_accumulation_steps}")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  训练轮数: {args.n_epoch}")
    logger.info(f"  随机种子: {args.seed}")
    logger.info(f"  N-gram隐藏层数: {num_ngram_hidden_layer}")
    logger.info(f"  数据加载器工作线程数: {args.num_workers}")
    logger.info(f"  数据加载模式: {('流式加载' if args.streaming else '标准加载')}")
    
    if args.streaming:
        logger.info(f"  流式加载缓冲区大小: {args.buffer_size}")
    logger.info("-"*50)
    
    # 1. 加载分词器
    logger.info("步骤2: 加载分词器...")
    tokenizer_start_time = time.time()
    try:
        # 尝试直接从Hugging Face加载预训练分词器
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        logger.info("成功从HuggingFace加载zhihan1996/DNABERT-2-117M分词器")
    except Exception as e:
        logger.warning(f"无法从HuggingFace加载分词器: {e}")
        # 如果无法从HuggingFace加载，尝试从本地路径加载
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bert_config_path = os.path.join(current_dir, "..", "resources", "DNABERT-2-117M")
        
        if os.path.exists(os.path.join(bert_config_path, "tokenizer.json")):
            logger.info(f"从本地路径加载分词器: {bert_config_path}")
            tokenizer = AutoTokenizer.from_pretrained(bert_config_path)
        else:
            # 如果本地也没有，使用基本的BERT分词器
            logger.info("使用基本的BERT分词器")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # 保存到本地路径以便后续使用
            os.makedirs(bert_config_path, exist_ok=True)
            tokenizer.save_pretrained(bert_config_path)
    
    tokenizer_time = time.time() - tokenizer_start_time
    logger.info(f"分词器加载完成，用时: {tokenizer_time:.2f}秒")
    logger.info(f"分词器词汇表大小: {len(tokenizer)}")
    
      
    # 加载或创建N-gram编码器
    logger.info("步骤2: 加载N-gram编码器...")
    ngram_start_time = time.time()
    if args.ngram_encoder_path and os.path.exists(args.ngram_encoder_path):
        logger.info(f"从文件加载N-gram编码器: {args.ngram_encoder_path}")
        ngram_encoder = NgramEncoder.from_file(args.ngram_encoder_path)
        logger.info(f"N-gram词汇表大小: {ngram_encoder.get_vocab_size()}")
    else:
        logger.error("N-gram编码器未指定且无法找到默认编码器，请使用--ngram-encoder-path指定路径或使用--train-ngram训练新的编码器")
        exit(1)
    
    # 获取N-gram词汇表大小
    ngram_vocab_size = ngram_encoder.get_vocab_size()
    logger.info(f"N-gram词汇表大小: {ngram_vocab_size}")
    ngram_time = time.time() - ngram_start_time
    logger.info(f"N-gram编码器加载完成，用时: {ngram_time:.2f}秒")
    
    
    # 3. 加载数据集
    logger.info("步骤3: 加载训练数据集...")
    dataset_start_time = time.time()
    # if args.streaming:
    #     logger.info("使用流式数据加载...")
    #     train_dataset = StreamingDNADataset(train_data_file, tokenizer, buffer_size=args.buffer_size)
    #     logger.info(f"训练数据集初始化完成 (流式)")
        
    #     val_dataset = StreamingDNADataset(dev_data_file, tokenizer, buffer_size=args.buffer_size)
    #     logger.info(f"验证数据集初始化完成 (流式)")
    # else:
    #     logger.info(f"从文件加载训练数据集: {train_data_file}")
    #     train_dataset = load_data_from_file(train_data_file, tokenizer, cache_dir=args.cache_dir)
    #     logger.info(f"训练数据集加载完成，大小: {len(train_dataset)} 条序列")
        
    #     logger.info(f"从文件加载验证数据集: {dev_data_file}")
    #     val_dataset = load_data_from_file(dev_data_file, tokenizer, cache_dir=args.cache_dir)
    #     logger.info(f"验证数据集加载完成，大小: {len(val_dataset)} 条序列")
    
    
    
    train_dataset = mlmdataset_builder(save_dir=train_dir, 
                                        tokenizer=tokenizer, 
                                        ngram_encoder=ngram_encoder, 
                                        data_path=train_data_file,
                                        data_config_path=train_dir+"/config.json",
                                        check_hash=False)
    val_dataset = mlmdataset_builder(save_dir=dev_dir, 
                                      tokenizer=tokenizer, 
                                      ngram_encoder=ngram_encoder, 
                                      data_path=dev_data_file,
                                      data_config_path=dev_dir+"/config.json",
                                      check_hash=False)

    
    dataset_time = time.time() - dataset_start_time
    logger.info(f"数据集加载完成，用时: {dataset_time:.2f}秒")
    
    # 3. 配置模型
    logger.info("步骤4: 配置模型...")
    model_config_start_time = time.time()
    try:
        # 尝试直接从Hugging Face加载预训练模型配置
        logger.info("尝试从HuggingFace加载DNABERT-2-117M模型配置...")
        bert_config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        logger.info("成功从HuggingFace加载zhihan1996/DNABERT-2-117M模型BertConfig配置")
    except Exception as e:
        logger.warning(f"无法从HuggingFace加载模型配置: {e}")
        # 如果无法从HuggingFace加载，创建一个基本的BERT配置
        logger.info("创建基本的BERT配置")
        bert_config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            model_type="bert",  # 确保设置model_type
        )
        # 保存到本地路径以便后续使用
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bert_config_path = os.path.join(current_dir, "..", "resources", "DNABERT-2-117M")
        os.makedirs(bert_config_path, exist_ok=True)
        bert_config.save_pretrained(bert_config_path)
  
    
    # 创建ZEN配置（扩展的BERT配置，包含Ngram信息）
    logger.info("步骤6: 创建ZEN配置...")
    zen_config = ZenConfig(
        num_word_hidden_layers=num_ngram_hidden_layer,
        ngram_vocab_size=ngram_vocab_size,
        **bert_config.to_dict(),
    )
    
    # 打印ZEN配置信息
    logger.info("ZEN配置详情:")
    logger.info(f"  词汇表大小: {zen_config.vocab_size}")
    logger.info(f"  隐藏层大小: {zen_config.hidden_size}")
    logger.info(f"  注意力头数: {zen_config.num_attention_heads}")
    logger.info(f"  隐藏层数量: {zen_config.num_hidden_layers}")
    logger.info(f"  N-gram词汇表大小: {zen_config.ngram_vocab_size}")
    logger.info(f"  N-gram隐藏层数量: {zen_config.num_word_hidden_layers}")
    
    # 加载预训练模型或从检查点恢复
    logger.info("步骤7: 加载预训练模型...")
    model_load_start_time = time.time()
    logger.info("从预训练模型初始化...")
    try:
        logger.info("尝试从HuggingFace下载预训练模型...")
        model = BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", config=zen_config)
        logger.info("成功从HuggingFace下载预训练模型")
    except Exception as e:
        logger.error(f"无法从HuggingFace下载模型: {e}")
        exit(1)

    
    model_load_time = time.time() - model_load_start_time
    logger.info(f"模型加载完成，用时: {model_load_time:.2f}秒")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    logger.info(f"参数比例: {trainable_params/total_params*100:.2f}%")

    # 3. 配置优化器
    logger.info("步骤8: 配置优化器...")
    # 排除LayerNorm参数，因为它们通常不需要权重衰减
    model_params = [param for name, param in model.named_parameters() if ("Norm" not in name)]
    optimizer = torch.optim.AdamW(model_params, learning_rate, weight_decay=0.01)
    logger.info(f"使用AdamW优化器，学习率: {learning_rate}, 权重衰减: 0.01")
    
    # 4. 配置训练参数
    logger.info("步骤9: 配置训练参数...")
    
    # 处理流式数据集的max_steps问题
    if args.streaming:
        # 流式数据集需要明确指定max_steps
        estimated_train_size = 1_000_000  # 估计的训练样本数
        max_steps = math.ceil(estimated_train_size / (args.per_device_train_batch_size * args.grad_accumulation_steps) * args.n_epoch)
        logger.info(f"流式数据集模式: 估计的训练样本数: {estimated_train_size}, 设置max_steps: {max_steps}")
        
        train_args = transformers.training_args.TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=1_000,
            save_steps=1_000,
            max_grad_norm=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            max_steps=max_steps,  # 使用计算的最大步数
            logging_steps=100,
            dataloader_num_workers=args.num_workers,
            dataloader_prefetch_factor=2,
            warmup_steps=100,
            save_safetensors=False,
            seed=args.seed,
            data_seed=args.seed,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_first_step=True,
        )
    else:
        # 标准数据集使用epochs
        train_args = transformers.training_args.TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=1_000,
            save_steps=1_000,
            max_grad_norm=1,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            num_train_epochs=args.n_epoch,
            logging_steps=100,
            dataloader_num_workers=args.num_workers,
            dataloader_prefetch_factor=2,
            warmup_steps=100,
            save_safetensors=False,
            seed=args.seed,
            data_seed=args.seed,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_first_step=True,
        )
    
    # 创建日志目录
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    logger.info(f"训练日志将保存到: {os.path.join(output_dir, 'logs')}")
    
    # 5. 创建Trainer实例
    logger.info("步骤10: 创建Trainer实例...")
    
    # 定义回调函数，用于记录训练进度
    class TrainingProgressCallback(transformers.TrainerCallback):
        def __init__(self, total_steps):
            self.total_steps = total_steps
            self.start_time = time.time()
            self.last_log_time = time.time()
            
        def on_step_end(self, args, state, control, **kwargs):
            current_time = time.time()
            # 每100步或者时间间隔超过60秒记录一次进度
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
    if args.streaming:
        total_steps = max_steps  # 使用之前计算的max_steps
    else:
        total_steps = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.grad_accumulation_steps) * args.n_epoch)
    
    logger.info(f"总训练步数: {total_steps}")
    progress_callback = TrainingProgressCallback(total_steps)
    
    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        optimizers=(optimizer, None),  # 提供优化器和调度器
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # 评估指标计算函数
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # logits预处理函数
        callbacks=[progress_callback],  # 添加进度回调
    )
    
    # 6. 开始训练
    logger.info("="*80)
    logger.info("步骤11: 开始训练...")
    logger.info("="*80)
    train_start_time = time.time()
    trainer.train()
    train_time = time.time() - train_start_time
    
    # 7. 训练完成，保存最终模型
    logger.info("="*80)
    logger.info(f"训练完成！总用时: {train_time/60:.2f}分钟")
    logger.info(f"最终模型已保存到 {output_dir}")
    
    # 计算总用时
    total_time = time.time() - start_time
    logger.info(f"总运行时间: {total_time/60:.2f}分钟")
    logger.info("="*80)


if __name__ == "__main__":
    main()

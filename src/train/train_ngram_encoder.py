#!/usr/bin/env python
"""
训练N-gram编码器并保存配置文件

此脚本从文本文件或GUE数据集加载DNA序列数据，训练N-gram编码器，
并将结果保存为正确格式的配置文件。
"""

import os
import argparse
import logging
from typing import List, Dict, Tuple, Set, Optional
import glob
import time
from datetime import timedelta
from tqdm import tqdm
import pandas as pd
import numpy as np
import traceback
import sys
import gc
import re

import torch
from transformers import AutoTokenizer
from tools.get_seq_form_dir import *

from dnazen.ngram import NgramEncoder
try:
    from scripts.ngram_encoder_analyze import analyze_ngrams_coverage
except ModuleNotFoundError:
    # 当直接运行脚本时，尝试相对导入
    from ngram_encoder_analyze import analyze_ngrams_coverage
import ipdb

# 配置日志
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# def find_csv_files(directory: str) -> List[str]:
#     """查找指定目录下的所有CSV文件"""
#     logger.info(f"正在查找 {directory} 目录下的所有CSV文件")
#     csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
#     logger.info(f"找到 {len(csv_files)} 个CSV文件")
#     return csv_files


# def find_gue_files(gue_dir: str) -> List[str]:
#     """查找GUE数据集中的序列文件"""
#     if not gue_dir or not os.path.exists(gue_dir):
#         return []
    
#     logger.info(f"查找GUE数据集文件: {gue_dir}")
#     sequence_files = []
    
#     # 先查找我们处理过的汇总文件
#     gue_seq_file = os.path.join(os.path.dirname(gue_dir), "pretrain", "gue_sequences.txt")
#     if os.path.exists(gue_seq_file):
#         logger.info(f"找到GUE汇总序列文件: {gue_seq_file}")
#         return [gue_seq_file]
    
#     # 如果没有汇总文件，则查找所有CSV文件
#     csv_files = find_csv_files(gue_dir)
#     return csv_files


# def find_mspecies_files(data_dir: str) -> List[str]:
#     """查找mspecies数据集文件"""
#     mspecies_file = os.path.join(data_dir, "pretrain", "dev", "dev.txt")
#     if os.path.exists(mspecies_file):
#         logger.info(f"找到mspecies数据集文件: {mspecies_file}")
#         return [mspecies_file]
#     return []


# def is_dna_sequence(text: str) -> bool:
#     """判断一个字符串是否为DNA序列"""
#     if not isinstance(text, str) or not text:
#         return False
    
#     # 计算ATGC比例
#     dna_chars = set("ATGCNatgcn")
#     dna_count = sum(c in dna_chars for c in text)
#     return dna_count / len(text) > 0.9


# def extract_dna_sequences_from_csv(csv_file: str, min_length: int = 20) -> List[str]:
#     """从CSV文件中提取DNA序列数据"""
#     sequences = []
#     try:
#         # 读取CSV文件
#         df = pd.read_csv(csv_file)
        
#         # 查找可能包含序列的列
#         for column in df.columns:
#             # 检查列名是否包含"seq"或"sequence"
#             if any(keyword in column.lower() for keyword in ["seq", "sequence", "dna"]):
#                 # 提取该列中的DNA序列
#                 for seq in df[column].dropna():
#                     if isinstance(seq, str) and is_dna_sequence(seq) and len(seq) >= min_length:
#                         sequences.append(seq.upper())  # 转换为大写
        
#         # 如果没有找到明显的序列列，检查每一列
#         if not sequences:
#             for column in df.columns:
#                 if df[column].dtype == object:  # 字符串类型的列
#                     # 检查前几行是否像DNA序列
#                     sample = df[column].dropna().head(5).astype(str)
#                     if any(is_dna_sequence(s) for s in sample):
#                         for seq in df[column].dropna():
#                             if isinstance(seq, str) and is_dna_sequence(seq) and len(seq) >= min_length:
#                                 sequences.append(seq.upper())
        
#         logger.info(f"文件 {csv_file} 提取的DNA序列长度统计: 序列数量:【{len(sequences)}】 平均长度:【{np.mean([len(s) for s in sequences]) if sequences else 0:.2f}】 最大长度:【{max([len(s) for s in sequences]) if sequences else 0}】 最小长度:【{min([len(s) for s in sequences]) if sequences else 0}】 95%分位数长度:【{np.percentile([len(s) for s in sequences], 95) if sequences else 0:.0f}】")
    
#     except Exception as e:
#         logger.error(f"处理文件 {csv_file} 时出错: {str(e)}")
    
#     return sequences


# def print_sequence_stats(sequences: List[str], source_name: str):
#     """计算并打印DNA序列的统计信息"""
#     if not sequences:
#         logger.warning(f"{source_name} 中没有找到有效的DNA序列")
#         return
    
#     seq_lengths = [len(seq) for seq in sequences]
#     avg_length = sum(seq_lengths) / len(seq_lengths)
#     max_length = max(seq_lengths)
#     min_length = min(seq_lengths)
    
#     # 计算95%分位数长度
#     seq_lengths.sort()
#     percentile_95_idx = int(len(seq_lengths) * 0.95)
#     percentile_95_length = seq_lengths[percentile_95_idx]
    
#     logger.info(f"{source_name} 提取的DNA序列长度统计: 序列数量:【{len(sequences)}】 平均长度:【{avg_length:.2f}】 最大长度:【{max_length}】 最小长度:【{min_length}】 95%分位数长度:【{percentile_95_length}】")




# def extract_dna_sequences(csv_file: str) -> List[str]:
#     """从CSV文件中提取DNA序列"""
#     try:
#         sequences = extract_dna_sequences_from_csv(csv_file)
#         return sequences
#     except Exception as e:
#         logger.error(f"从CSV文件提取DNA序列时出错: {str(e)}")
#         return []



# def batch_process_sequences(sequences, batch_size=10000):
#     """批量处理序列，以避免内存问题"""
#     total = len(sequences)
#     logger.info(f"开始批量处理 {total} 条序列，批次大小: {batch_size}")
    
#     for i in range(0, total, batch_size):
#         end = min(i + batch_size, total)
#         logger.info(f"处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}，序列 {i}-{end-1}")
#         yield sequences[i:end]


def tokenize_data(texts: List[str], tokenizer) -> List[List[int]]:
    """将文本数据转换为token ID"""
    logger.info("将文本转换为token ID")
    tokenized_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenized_data.append(tokens)
    logger.info(f"转换完成，共 {len(tokenized_data)} 个序列")
    return tokenized_data


# def get_gue_sequences(gue_dir: str,sequences):
#     logger.info(f"从GUE数据集加载数据: {args.gue_dir}")
#     csv_files = find_csv_files(args.gue_dir)
    
#     # 尝试查找已处理的GUE序列文件
#     gue_seq_file = os.path.join(os.path.dirname(args.gue_dir), "pretrain", "gue_sequences.txt")
#     if os.path.exists(gue_seq_file):
#         logger.info(f"找到已处理的GUE序列文件: {gue_seq_file}")
#         with open(gue_seq_file, 'r') as f:
#             gue_sequences = [line.strip() for line in f if line.strip()]
#         logger.info(f"从GUE汇总文件加载了 {len(gue_sequences)} 条序列")
#         sequences.extend(gue_sequences)
#     else:
#         # 处理所有CSV文件
#         total_extracted = 0
#         for csv_file in csv_files:
#             logger.info(f"处理CSV文件: {csv_file}")
#             file_sequences = extract_dna_sequences(csv_file)
#             sequences.extend(file_sequences)
#             total_extracted += len(file_sequences)
#             logger.info(f"从 {csv_file} 提取了 {len(file_sequences)} 条DNA序列，累计: {total_extracted}")
            


# def get_mspecies_sequences(data_dir: str,sequences):
#     logger.info(f"从输入文件加载数据: {args.input}")
#     start_time = time.time()
    
#     with open(args.input, 'r') as f:
#         file_sequences = [line.strip() for line in f if line.strip()]
#         sequences.extend(file_sequences)
        
#     elapsed = time.time() - start_time
#     logger.info(f"从 {args.input} 加载了 {len(file_sequences)} 条序列，耗时: {timedelta(seconds=elapsed)}")


def main():
    parser = argparse.ArgumentParser(description="训练N-gram编码器")
    
    # 输入数据源选项
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", type=str, help="输入文本文件路径")
    input_group.add_argument("--gue-dir", type=str, help="GUE数据集目录路径")
    
    # 输出和配置选项
    parser.add_argument("--output", type=str, required=True, help="输出N-gram编码器配置文件路径")
    parser.add_argument("--min-ngram-len", type=int, default=2, help="最小N-gram长度")
    parser.add_argument("--max-ngram-len", type=int, default=5, help="最大N-gram长度")
    parser.add_argument("--max-ngrams", type=int, default=30, help="最大N-gram数量")
    parser.add_argument("--min-pmi", type=float, default=0, help="最小PMI阈值")
    parser.add_argument("--min-token-count", type=int, default=2, help="最小token计数阈值")
    parser.add_argument("--min-ngram-freq", type=int, default=2, help="最小N-gram频率阈值")
    parser.add_argument("--method", type=str, choices=["freq", "pmi"], default="pmi", help="N-gram选择方法")
    parser.add_argument("--num-workers", type=int, default=1, help="并行处理的工作线程数")
    parser.add_argument("--tok", type=str, default="zhihan1996/DNABERT-2-117M", help="使用的tokenizer名称")
    parser.add_argument("--analyze-only", action="store_true", help="只分析现有的N-gram编码器，不进行训练")
    parser.add_argument("--plots-only", action="store_true", help="只重新生成N-gram分布图，不进行其他分析")
    parser.add_argument("--coverage-files", type=str, nargs='+', help="用于分析N-gram覆盖率的DNA序列数据文件")
    parser.add_argument("--auto-coverage", action="store_true", help="自动分析GUE和mspecies数据集的覆盖率")
    parser.add_argument("--data-dir", type=str, default="../data", help="数据根目录")
    parser.add_argument("--batch-size", type=int, default=50000, help="训练时的批次大小")
    
    args = parser.parse_args()
    # 加载数据
    sequences = []
    
    # 从GUE数据集加载数据
    if args.gue_dir:
        get_gue_sequences(args.gue_dir,sequences)
    
    # 从输入文件加载数据
    if args.input:
        get_mspecies_sequences(args.input,sequences)
    
    # 如果没有数据，报错并退出
    if not sequences:
        logger.error("没有找到任何DNA序列数据，请检查输入参数")
        return
    
    # 去重
    unique_sequences = list(set(sequences))
    logger.info(f"序列去重: 原始序列数 {len(sequences)}，去重后 {len(unique_sequences)}")
    sequences = unique_sequences
    
    # 打印序列统计信息
    print_sequence_stats(sequences, "所有数据源")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    
    # 加载tokenizer  将序列转换为token ID
    logger.info(f"加载tokenizer: {args.tok}")
    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    tokenized_data = tokenize_data(sequences, tokenizer)

    # 训练N-gram编码器
    logger.info("开始训练N-gram编码器...")
    start_time = time.time()
    
    # 初始化N-gram编码器
    logger.info("初始化N-gram编码器")
    encoder = NgramEncoder(
        vocab_dict={},  # 空词典，将通过训练填充
        min_ngram_len=args.min_ngram_len,
        max_ngram_len=args.max_ngram_len,
        max_ngrams=args.max_ngrams,
    )
    
   # 训练N-gram编码器
    logger.info(f"使用 {args.method} 方法训练N-gram编码器")
    logger.info(f"参数: min_pmi={args.min_pmi}, min_token_count={args.min_token_count}, min_ngram_freq={args.min_ngram_freq}")
    
    ngram_freqs = encoder.train(
        tokens=tokenized_data,
        min_pmi=args.min_pmi,
        min_token_count=args.min_token_count,
        min_ngram_freq=args.min_ngram_freq,
        num_workers=args.num_workers,
        returns_freq=True,
        method=args.method,
    )
    
    training_elapsed = time.time() - start_time  
    vocab_size = encoder.get_vocab_size()
    logger.info(f"训练完成，N-gram词汇表大小: {vocab_size},总耗时: {timedelta(seconds=training_elapsed)}")
    
    try:
        # 显示一些N-gram示例
        logger.info("前5个N-gram示例:")
        for i, (ngram, id_) in enumerate(list(encoder.get_vocab().items())[:5]):
            freq = ngram_freqs.get(ngram, "N/A")
            # 将数字token转换为碱基对序列
            ngram_text = tokenizer.decode(ngram)
            logger.info(f"  {i+1}. {ngram} (碱基: {ngram_text}) -> ID {id_}, 频率 {freq}")
            
        # 保存编码器
        encoder.save(args.output)
        logger.info(f"N-gram编码器已保存到: {args.output}")
        
        
        # 获取输出路径的目录部分
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        
        # 写入N-gram列表和频率信息
        ngram_file_path = os.path.join(output_dir, "ngram_list.txt")
        with open(ngram_file_path, "w") as f:
            f.write("序号\t频率\t字符长度\tBPE分词长度\tID\tN-gram\n")
            
            # 按频率排序（从高到低）
            sorted_ngrams = sorted(
                [(ngram, ngram_id) for ngram, ngram_id in encoder.get_vocab().items()],
                key=lambda x: ngram_freqs.get(x[0], 0) if ngram_freqs else 0,
                reverse=True
            )
            
            for idx, (ngram, ngram_id) in enumerate(sorted_ngrams, 1):
                freq = ngram_freqs.get(ngram, 0) if ngram_freqs else 0
                # 将数字token转换为碱基对序列
                ngram_text = tokenizer.decode(ngram)
                # 获取ngram的BPE分词表示
                bpe_tokens = tokenizer.tokenize(ngram_text)
                # 只保存BPE分词长度，不保存分词文本
                f.write(f"{idx}\t{freq}\t{len(ngram_text)}\t{len(bpe_tokens)}\t{ngram_id}\t{ngram_text}\n")
        
        logger.info(f"N-gram列表已保存到: {ngram_file_path}")
        

    except Exception as e:
        logger.error(f"训练N-gram编码器时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    logger.info("N-gram编码器训练与分析完成")


if __name__ == "__main__":
    main()
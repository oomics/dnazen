#!/usr/bin/env python
"""
训练N-gram编码器并保存配置文件

此脚本从文本文件或GUE数据集加载DNA序列数据，训练N-gram编码器，
并将结果保存为正确格式的配置文件。
"""

import os
import argparse
import logging
from typing import List
import time
from datetime import timedelta
import traceback

from transformers import AutoTokenizer
from tools.get_seq_form_dir import (
    print_sequence_stats,
    get_mspecies_sequences,
    get_gue_sequences,
    get_gue_sequences_type
)

from dnazen.ngram import NgramEncoder


# 配置日志
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def tokenize_data(texts: List[str], tokenizer) -> List[List[int]]:
    """将文本数据转换为token ID"""
    logger.info("将文本转换为token ID")
    tokenized_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenized_data.append(tokens)
    logger.info(f"转换完成，共 {len(tokenized_data)} 个序列")
    return tokenized_data


def main():
    parser = argparse.ArgumentParser(description="提取预训练数据集")

    # 输入数据源选项 - 移除互斥组，允许同时指定多个输入
    parser.add_argument("--gue-dir", type=str, help="GUE数据集目录路径")
    parser.add_argument("--mspecies-dir", type=str, help="mspecies数据集目录路径")

    # 至少需要一个输入源
    parser.add_argument("--output-train", type=str, required=True, help="输出train.txt文件路径")
    parser.add_argument("--output-dev", type=str, required=True, help="输出dev.txt文件路径")
    args = parser.parse_args()

    # 检查是否至少提供了一个输入源
    if not args.input and not args.gue_dir:
        logger.error("错误：必须提供至少一个输入源（--mspecies-dir 或 --gue-dir）")
        parser.print_help()
        return

    # 加载数据
    gue_sequences_train = []
    gue_sequences_dev = []
    gue_sequences_test = []
    mspecies_sequences_dev = []
    mspecies_sequences_train = []
    
    sequences_train = []
    sequences_dev = []
    

    

    # 从GUE数据集加载数据
    if args.gue_dir:
        get_gue_sequences_type(args.gue_dir, gue_sequences_train,gue_sequences_dev,gue_sequences_test)
        sequences_train.extend(gue_sequences_train)
        sequences_train.extend(gue_sequences_dev)
        sequences_train.extend(gue_sequences_test)
        
        sequences_dev.extend(gue_sequences_dev)
        sequences_dev.extend(gue_sequences_test)


    # 从输入文件加载数据
    if args.mspecies_dir:
        get_mspecies_sequences(args.input, mspecies_sequences_train)
        sequences_train.extend(mspecies_sequences_train)
        sequences_dev.extend(mspecies_sequences_train[int(len(mspecies_sequences_train) * 0.9):])


    # 去重
    unique_sequences_train = list(set(sequences_train))
    logger.info(f"序列去重: train原始序列数 {len(sequences_train)}，去重后 {len(unique_sequences_train)}")
    sequences_train = unique_sequences_train
    
    unique_sequences_dev = list(set(sequences_dev))
    logger.info(f"序列去重: dev原始序列数 {len(sequences_dev)}，去重后 {len(unique_sequences_dev)}")
    sequences_dev = unique_sequences_dev

    # 打印序列统计信息
    print_sequence_stats(unique_sequences_train, "所有train数据源")
    print_sequence_stats(unique_sequences_dev, "所有dev数据源")


    # 创建输出目录
    output_tarin_dir = os.path.dirname(args.output_train)
    if output_tarin_dir and not os.path.exists(output_tarin_dir):
        os.makedirs(output_tarin_dir, exist_ok=True)
    
    output_dev_dir = os.path.dirname(args.output_dev)
    if output_dev_dir and not os.path.exists(output_dev_dir):
        os.makedirs(output_dev_dir, exist_ok=True)

    


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
训练N-gram编码器并保存配置文件

此脚本从文本文件加载数据，训练N-gram编码器，
并将结果保存为正确格式的配置文件。
"""

import os
import argparse
import logging
from typing import List

import torch
from transformers import AutoTokenizer

from dnazen.ngram import NgramEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_text_data(file_path: str) -> List[str]:
    """加载文本数据"""
    logger.info(f"从 {file_path} 加载文本数据")
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    logger.info(f"加载了 {len(lines)} 行文本")
    return lines


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
    parser = argparse.ArgumentParser(description="训练N-gram编码器并保存配置文件")
    parser.add_argument("--input", type=str, required=True, help="输入文本文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出配置文件路径")
    parser.add_argument("--tokenizer", type=str, default="zhihan1996/DNABERT-2-117M", help="Tokenizer名称或路径")
    parser.add_argument("--min-ngram-len", type=int, default=2, help="最小N-gram长度")
    parser.add_argument("--max-ngram-len", type=int, default=5, help="最大N-gram长度")
    parser.add_argument("--max-ngrams", type=int, default=30, help="每个序列最多匹配的N-gram数量")
    parser.add_argument("--min-pmi", type=float, default=1.0, help="最小PMI阈值")
    parser.add_argument("--min-token-count", type=int, default=5, help="最小token计数阈值")
    parser.add_argument("--min-ngram-freq", type=int, default=5, help="最小N-gram频率阈值")
    parser.add_argument("--method", type=str, choices=["pmi", "freq"], default="pmi", help="训练方法")
    parser.add_argument("--num-workers", type=int, default=4, help="训练使用的工作线程数")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # 加载并处理数据
    texts = load_text_data(args.input)
    tokenized_data = tokenize_data(texts, tokenizer)
    
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
    
    vocab_size = encoder.get_vocab_size()
    logger.info(f"训练完成，N-gram词汇表大小: {vocab_size}")
    
    if vocab_size > 0:
        # 显示一些N-gram示例
        logger.info("前5个N-gram示例:")
        for i, (ngram, id_) in enumerate(list(encoder.get_vocab().items())[:5]):
            freq = ngram_freqs.get(ngram, "N/A")
            logger.info(f"  {i+1}. {ngram} -> ID {id_}, 频率 {freq}")
        
        # 保存N-gram编码器配置
        logger.info(f"保存N-gram编码器配置到: {args.output}")
        encoder.save(args.output, pretty=True)
        logger.info("保存完成")
    else:
        logger.warning("没有找到任何N-gram，请调整训练参数")


if __name__ == "__main__":
    main()
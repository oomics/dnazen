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
    parser = argparse.ArgumentParser(description="训练N-gram编码器")

    # 输入数据源选项 - 移除互斥组，允许同时指定多个输入
    parser.add_argument("--input", type=str, help="输入文本文件路径")
    parser.add_argument("--gue-dir", type=str, help="GUE数据集目录路径")

    # 至少需要一个输入源
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
    parser.add_argument("--data-dir", type=str, default="../data", help="数据根目录")
    parser.add_argument("--batch-size", type=int, default=50000, help="训练时的批次大小")

    args = parser.parse_args()

    # 检查是否至少提供了一个输入源
    if not args.input and not args.gue_dir:
        logger.error("错误：必须提供至少一个输入源（--input 或 --gue-dir）")
        parser.print_help()
        return

    # 加载数据
    sequences = []

    # 从GUE数据集加载数据
    if args.gue_dir:
        get_gue_sequences(args.gue_dir, sequences)

    # 从输入文件加载数据
    if args.input:
        get_mspecies_sequences(args.input, sequences)

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
    logger.info(
        f"参数: min_pmi={args.min_pmi}, min_token_count={args.min_token_count}, min_ngram_freq={args.min_ngram_freq}"
    )

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
            logger.info(f"  {i + 1}. {ngram} (碱基: {ngram_text}) -> ID {id_}, 频率 {freq}")

        # 保存编码器
        encoder.save(args.output)
        logger.info(f"N-gram编码器已保存到: {args.output}")

        # 获取输出路径的目录部分
        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."

        # 写入N-gram列表和频率信息
        ngram_file_path = os.path.join(output_dir, "ngram_list.txt")
        with open(ngram_file_path, "w") as f:
            f.write("序号\t频率\t字符长度\tBPE分词长度\tID\tN-gram\ttoken_ids\n")

            # 按频率排序（从高到低）
            sorted_ngrams = sorted(
                [(ngram, ngram_id) for ngram, ngram_id in encoder.get_vocab().items()],
                key=lambda x: ngram_freqs.get(x[0], 0) if ngram_freqs else 0,
                reverse=True,
            )

            for idx, (ngram, ngram_id) in enumerate(sorted_ngrams, 1):
                freq = ngram_freqs.get(ngram, 0) if ngram_freqs else 0
                # 将数字token转换为碱基对序列
                ngram_text = tokenizer.decode(ngram)
                # 获取ngram的BPE分词表示
                bpe_tokens = tokenizer.tokenize(ngram_text)
                # 只保存BPE分词长度，不保存分词文本
                f.write(
                    f"{idx}\t{freq}\t{len(ngram_text)}\t{len(bpe_tokens)}\t{ngram_id}\t{ngram_text}\t{ngram}\n"
                )

        logger.info(f"N-gram列表已保存到: {ngram_file_path}")

    except Exception as e:
        logger.error(f"训练N-gram编码器时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return

    logger.info("N-gram编码器训练与分析完成")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
训练N-gram编码器并保存配置文件

此脚本从文本文件或GUE数据集加载DNA序列数据，训练N-gram编码器，
并将结果保存为正确格式的配置文件。
"""

import os
import logging
from typing import List, Dict
import glob
import time
from datetime import timedelta
import pandas as pd


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_csv_files(directory: str) -> List[str]:
    """查找指定目录下的所有CSV文件"""
    logger.info(f"正在查找 {directory} 目录下的所有CSV文件")
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    return csv_files


def find_gue_files(gue_dir: str) -> List[str]:
    """查找GUE数据集中的序列文件"""
    if not gue_dir or not os.path.exists(gue_dir):
        return []

    logger.info(f"查找GUE数据集文件: {gue_dir}")
    # sequence_files = []

    # 先查找我们处理过的汇总文件
    gue_seq_file = os.path.join(os.path.dirname(gue_dir), "pretrain", "gue_sequences.txt")
    if os.path.exists(gue_seq_file):
        logger.info(f"找到GUE汇总序列文件: {gue_seq_file}")
        return [gue_seq_file]

    # 如果没有汇总文件，则查找所有CSV文件
    csv_files = find_csv_files(gue_dir)
    return csv_files


def find_mspecies_files(data_dir: str) -> List[str]:
    """查找mspecies数据集文件"""
    mspecies_file = os.path.join(data_dir, "pretrain", "dev", "dev.txt")
    if os.path.exists(mspecies_file):
        logger.info(f"找到mspecies数据集文件: {mspecies_file}")
        return [mspecies_file]
    return []


def is_dna_sequence(text):
    """
    检查文本是否为DNA序列（只包含A、T、G、C字符）

    Args:
        text: 要检查的文本

    Returns:
        bool: 如果是DNA序列则返回True，否则返回False
    """
    if not isinstance(text, str):
        return False

    # 防止处理None或空字符串
    if not text:
        return False

    dna_chars = set("ATGC")
    # 转换为大写并移除可能的空白字符
    text = text.upper().strip()

    # 计算DNA字符的数量
    dna_count = sum(c in dna_chars for c in text)

    # 如果至少95%的字符是DNA字符，则认为是DNA序列
    return len(text) > 0 and dna_count / len(text) >= 0.95


def extract_dna_sequences_from_csv(csv_file: str, min_length: int = 20) -> List[str]:
    """从CSV文件中提取DNA序列数据"""
    sequences = []
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 查找可能包含序列的列
        for column in df.columns:
            # 检查列名是否包含"seq"或"sequence"
            if any(keyword in column.lower() for keyword in ["seq", "sequence", "dna"]):
                # 提取该列中的DNA序列
                for seq in df[column].dropna():
                    if isinstance(seq, str) and is_dna_sequence(seq) and len(seq) >= min_length:
                        sequences.append(seq.upper())  # 转换为大写

        # 如果没有找到明显的序列列，检查每一列
        if not sequences:
            for column in df.columns:
                if df[column].dtype == object:  # 字符串类型的列
                    # 检查前几行是否像DNA序列
                    sample = df[column].dropna().head(5).astype(str)
                    if any(is_dna_sequence(s) for s in sample):
                        for seq in df[column].dropna():
                            if isinstance(seq, str) and is_dna_sequence(seq) and len(seq) >= min_length:
                                sequences.append(seq.upper())

        # logger.info(f"文件 {csv_file} 提取的DNA序列长度统计: 序列数量:【{len(sequences)}】 平均长度:【{np.mean([len(s) for s in sequences]) if sequences else 0:.2f}】 最大长度:【{max([len(s) for s in sequences]) if sequences else 0}】 最小长度:【{min([len(s) for s in sequences]) if sequences else 0}】 95%分位数长度:【{np.percentile([len(s) for s in sequences], 95) if sequences else 0:.0f}】")

    except Exception as e:
        logger.error(f"处理文件 {csv_file} 时出错: {str(e)}")

    return sequences


def print_sequence_stats(sequences: List[str], source_name: str):
    """计算并打印DNA序列的统计信息"""
    if not sequences:
        logger.warning(f"{source_name} 中没有找到有效的DNA序列")
        return

    seq_lengths = [len(seq) for seq in sequences]
    avg_length = sum(seq_lengths) / len(seq_lengths)
    max_length = max(seq_lengths)
    min_length = min(seq_lengths)

    # 计算95%分位数长度
    seq_lengths.sort()
    percentile_95_idx = int(len(seq_lengths) * 0.95)
    percentile_95_length = seq_lengths[percentile_95_idx]

    logger.info(
        f"{source_name} 提取的DNA序列长度统计: 序列数量:【{len(sequences)}】 平均长度:【{avg_length:.2f}】 最大长度:【{max_length}】 最小长度:【{min_length}】 95%分位数长度:【{percentile_95_length}】"
    )


def extract_dna_sequences(csv_file: str) -> List[str]:
    """从CSV文件中提取DNA序列"""
    try:
        sequences = extract_dna_sequences_from_csv(csv_file)
        return sequences
    except Exception as e:
        logger.error(f"从CSV文件提取DNA序列时出错: {str(e)}")
        return []


def get_gue_each_sequences(gue_dir: str) -> Dict[str, List[str]]:
    """从GUE数据集加载数据，返回每个种类的序列映射

    Args:
        gue_dir: GUE数据集目录

    Returns:
        Dict[str, List[str]]: 键为种类名称（文件夹名），值为对应的序列列表
    """
    logger.info(f"从GUE数据集加载数据: {gue_dir}")
    csv_files = find_csv_files(gue_dir)
    each_sequences = {}

    # 尝试查找已处理的GUE序列文件
    gue_seq_file = os.path.join(os.path.dirname(gue_dir), "pretrain", "gue_sequences.txt")
    if os.path.exists(gue_seq_file):
        logger.info(f"找到已处理的GUE序列文件: {gue_seq_file}")
        # 注意：汇总文件中可能没有种类信息，这里将所有序列归为一类
        with open(gue_seq_file, "r") as f:
            gue_sequences = [line.strip() for line in f if line.strip()]
        logger.info(f"从GUE汇总文件加载了 {len(gue_sequences)} 条序列")
        each_sequences["gue_combined"] = gue_sequences
    else:
        # 处理所有CSV文件，按文件夹分组
        for csv_file in csv_files:
            # 获取文件相对于gue_dir的路径，作为种类名称
            rel_path = os.path.relpath(os.path.dirname(csv_file), gue_dir)
            species_name = rel_path.replace(os.path.sep, "_")
            # logger.info(f"处理CSV文件: {csv_file}，种类: {species_name}")

            file_sequences = extract_dna_sequences(csv_file)

            # 将序列添加到对应种类的列表中
            if species_name not in each_sequences:
                each_sequences[species_name] = []
            each_sequences[species_name].extend(file_sequences)

            # logger.info(f"从 {csv_file} 提取了 {len(file_sequences)} 条DNA序列，种类 {species_name} 现有序列数: {len(each_sequences[species_name])}")

    # 打印每个种类的序列统计信息
    # for species, seqs in each_sequences.items():
    #     print_sequence_stats(seqs, f"种类 {species}")

    return each_sequences


def get_gue_sequences(gue_dir: str, sequences):
    logger.info(f"从GUE数据集加载数据: {gue_dir}")
    csv_files = find_csv_files(gue_dir)

    # 尝试查找已处理的GUE序列文件
    gue_seq_file = os.path.join(os.path.dirname(gue_dir), "pretrain", "gue_sequences.txt")
    if os.path.exists(gue_seq_file):
        logger.info(f"找到已处理的GUE序列文件: {gue_seq_file}")
        with open(gue_seq_file, "r") as f:
            gue_sequences = [line.strip() for line in f if line.strip()]
        logger.info(f"从GUE汇总文件加载了 {len(gue_sequences)} 条序列")
        sequences.extend(gue_sequences)
    else:
        # 处理所有CSV文件
        total_extracted = 0
        for csv_file in csv_files:
            logger.info(f"处理CSV文件: {csv_file}")
            file_sequences = extract_dna_sequences(csv_file)
            sequences.extend(file_sequences)
            total_extracted += len(file_sequences)
            logger.info(f"从 {csv_file} 提取了 {len(file_sequences)} 条DNA序列，累计: {total_extracted}")


def get_gue_sequences_type(gue_dir: str, sequences_dev,sequences_train,sequences_test):
    logger.info(f"从GUE数据集加载数据: {gue_dir}")
    csv_files = find_csv_files(gue_dir)
     # 处理所有CSV文件
    total_extracted = 0
    for csv_file in csv_files:
        logger.info(f"处理CSV文件: {csv_file}")
        file_sequences = extract_dna_sequences(csv_file)
        if csv_file.endswith("dev.csv"):
            sequences_dev.extend(file_sequences)
        elif csv_file.endswith("train.csv"):
            sequences_train.extend(file_sequences)
        elif csv_file.endswith("test.csv"):
            sequences_test.extend(file_sequences)
            
        total_extracted += len(file_sequences)
        logger.info(f"从 {csv_file} 提取了 {len(file_sequences)} 条DNA序列，累计: {total_extracted}")
    logger.info(f"共提取了 {len(sequences_train)} 条DNA序列用于训练，{len(sequences_dev)} 条DNA序列用于验证，{len(sequences_test)} 条DNA序列用于测试")

    
    

def get_mspecies_sequences(data_dir: str, sequences):
    logger.info(f"从输入文件加载mspecies数据: {data_dir}")
    start_time = time.time()

    with open(data_dir, "r") as f:
        file_sequences = [line.strip() for line in f if line.strip()]
        sequences.extend(file_sequences)

    elapsed = time.time() - start_time
    logger.info(f"从 {data_dir} 加载了 {len(file_sequences)} 条序列，耗时: {timedelta(seconds=elapsed)}")

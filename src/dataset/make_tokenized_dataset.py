"""
分词数据集生成工具

本脚本用于将原始文本数据使用特定分词器进行分词处理，并保存为PyTorch兼容的格式，
便于后续使用MaskedLM等模型进行训练。该脚本特别适用于DNA序列等生物数据的预处理。

使用方法:
    python make_tokenized_dataset.py --data [原始数据路径] --tok [分词器名称] -o [输出路径]

参数:
    --data: 原始文本数据文件路径，每行一个序列
    --tok: 使用的分词器名称或路径，默认为"zhihan1996/DNABERT-2-117M"
    --out/-o: 处理后数据的保存路径
    --batch-size: 每批处理的序列数，默认为10000
    --resume: 是否从上次中断处继续处理，默认为False
    --max-length: 手动指定最大序列长度，默认为None (自动计算)

输出:
    处理后的数据将保存为包含input_ids和attention_mask的PyTorch张量字典
"""

import click
import time
import logging
import os
import psutil
import json
from datetime import timedelta

from tqdm import tqdm
import torch
from transformers import AutoTokenizer


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """
    获取当前进程的内存使用情况。

    Returns:
        str: 当前内存使用量的字符串表示，以MB为单位
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.2f} MB"


def rm_empty_lines(data_list: list[str]) -> list[str]:
    """
    移除文本数据中的空行。

    步骤:
    1. 遍历所有文本行
    2. 保留非空行
    3. 返回清理后的数据列表

    Args:
        data_list (list[str]): 包含可能有空行的原始文本行列表

    Returns:
        list[str]: 移除空行后的文本行列表
    """
    start_time = time.time()
    logger.info(f"开始移除空行，原始行数: {len(data_list)}")

    data_text = []
    for d in tqdm(data_list, desc="移除空行进度"):
        if len(d) != 0:
            data_text.append(d)

    elapsed = time.time() - start_time
    logger.info(
        f"空行移除完成，耗时: {timedelta(seconds=elapsed)}，"
        f"保留行数: {len(data_text)}，"
        f"移除了 {len(data_list) - len(data_text)} 行空行"
    )
    logger.info(f"当前内存使用: {get_memory_usage()}")

    return data_text


def save_checkpoint(processed_count, total_count, out_path, max_seq_length=None):
    """
    保存处理进度检查点

    Args:
        processed_count (int): 已处理的序列数量
        total_count (int): 总序列数量
        out_path (str): 输出文件路径
        max_seq_length (int, optional): 最大序列长度
    """
    checkpoint_path = f"{out_path}.checkpoint"
    checkpoint_data = {
        "processed_count": processed_count,
        "total_count": total_count,
        "timestamp": time.time(),
        "max_seq_length": max_seq_length,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)
    progress_percentage = (processed_count / total_count) * 100 if total_count > 0 else 0
    logger.info(f"保存检查点: 已处理 {processed_count}/{total_count} 条序列 ({progress_percentage:.2f}%)")


def load_checkpoint(out_path):
    """
    加载处理进度检查点

    Args:
        out_path (str): 输出文件路径

    Returns:
        tuple: (已处理的序列数量, 最大序列长度)，如果没有检查点则返回(0, None)
    """
    checkpoint_path = f"{out_path}.checkpoint"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        processed_count = checkpoint_data["processed_count"]
        total_count = checkpoint_data["total_count"]
        timestamp = checkpoint_data.get("timestamp", 0)
        max_seq_length = checkpoint_data.get("max_seq_length", None)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        logger.info(f"找到检查点: 已处理 {processed_count}/{total_count} 条序列，保存时间: {time_str}")
        if max_seq_length:
            logger.info(f"从检查点恢复最大序列长度: {max_seq_length}")
        return processed_count, max_seq_length
    return 0, None


def calculate_max_length(data_text, tokenizer, sample_size=1000):
    """
    计算数据集中的最大序列长度

    为了提高效率，可以基于随机样本来估算最大长度，而不是扫描整个数据集

    Args:
        data_text (list): 文本数据列表
        tokenizer: 分词器
        sample_size (int): 用于估算的样本数量

    Returns:
        int: 估算的最大序列长度
    """
    logger.info(f"正在估算最大序列长度 (基于 {sample_size} 个样本)...")
    start_time = time.time()

    # 如果数据量小于样本数，则使用全部数据
    if len(data_text) <= sample_size:
        sample_texts = data_text
    else:
        # 从数据集中随机抽取样本
        import random

        sample_indices = random.sample(range(len(data_text)), sample_size)
        sample_texts = [data_text[i] for i in sample_indices]

    # 对样本进行分词，获取长度
    sample_lengths = []
    for text in tqdm(sample_texts, desc="计算样本长度"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        sample_lengths.append(len(tokens))

    # 计算最大长度
    max_length = max(sample_lengths)

    # 增加一些冗余，确保能容纳所有序列
    max_length = int(max_length * 1.1)  # 增加10%的冗余

    elapsed = time.time() - start_time
    logger.info(f"最大序列长度估算完成: {max_length}，耗时: {timedelta(seconds=elapsed)}")
    logger.info(
        f"样本序列长度统计: 最小={min(sample_lengths)}, 平均={sum(sample_lengths) / len(sample_lengths):.1f}, 最大={max(sample_lengths)}"
    )

    return max_length


def process_batch(batch_texts, tokenizer, temp_dir, batch_idx, max_length):
    """
    处理一批文本数据

    Args:
        batch_texts (list): 一批文本数据
        tokenizer: 分词器
        temp_dir (str): 临时文件目录
        batch_idx (int): 批次索引
        max_length (int): 最大序列长度，用于统一所有批次的张量形状

    Returns:
        str: 临时文件路径
    """
    logger.info(f"处理批次 {batch_idx}，包含 {len(batch_texts)} 条序列")

    # 对批次文本进行分词，使用统一的最大长度
    tokenize_start = time.time()
    data_tokenized = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding="max_length",  # 使用固定的最大长度填充
        max_length=max_length,  # 指定最大长度
        truncation=True,  # 如果序列超出最大长度则截断
        return_token_type_ids=False,
    )
    tokenize_elapsed = time.time() - tokenize_start

    # 准备输出数据
    input_ids = data_tokenized["input_ids"]
    attn_mask = data_tokenized["attention_mask"]

    batch_data = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
    }

    # 保存批次数据到临时文件
    temp_file = os.path.join(temp_dir, f"batch_{batch_idx}.pt")
    torch.save(batch_data, temp_file)

    logger.info(
        f"批次 {batch_idx} 处理完成，耗时: {timedelta(seconds=tokenize_elapsed)}，"
        f"形状: {input_ids.shape}，保存到: {temp_file}"
    )
    logger.info(f"当前内存使用: {get_memory_usage()}")

    # 释放内存
    del data_tokenized, input_ids, attn_mask, batch_data

    return temp_file


def merge_batch_files(temp_files, out_path):
    """
    合并所有批次文件为最终输出文件

    Args:
        temp_files (list): 临时文件路径列表
        out_path (str): 最终输出文件路径
    """
    logger.info(f"开始合并 {len(temp_files)} 个批次文件...")

    # 初始化合并后的数据结构
    all_input_ids = []
    all_attn_masks = []

    # 记录第一个批次的张量形状作为参考
    reference_shape = None
    skipped_batches = 0

    # 逐个加载批次文件并合并
    for temp_file in tqdm(temp_files, desc="合并批次文件"):
        try:
            batch_data = torch.load(temp_file)
            current_shape = batch_data["input_ids"].shape[1]  # 获取序列长度维度

            # 如果是第一个批次，记录形状作为参考
            if reference_shape is None:
                reference_shape = current_shape

            # 检查当前批次的形状是否与参考形状匹配
            if current_shape != reference_shape:
                logger.warning(
                    f"批次文件 {temp_file} 的形状不匹配: 预期 {reference_shape}，实际 {current_shape}，将跳过此批次"
                )
                skipped_batches += 1
                continue

            all_input_ids.append(batch_data["input_ids"])
            all_attn_masks.append(batch_data["attention_mask"])
        except Exception as e:
            logger.error(f"处理文件 {temp_file} 时出错: {str(e)}")
            skipped_batches += 1

    if skipped_batches > 0:
        logger.warning(f"合并过程中共跳过了 {skipped_batches} 个批次文件")

    # 合并所有张量
    logger.info(f"正在合并 {len(all_input_ids)} 个批次的张量...")
    merged_input_ids = torch.cat(all_input_ids, dim=0)
    merged_attn_masks = torch.cat(all_attn_masks, dim=0)

    # 准备最终输出数据
    final_data = {
        "input_ids": merged_input_ids,
        "attention_mask": merged_attn_masks,
    }

    # 保存最终数据
    logger.info(f"正在保存最终数据到 {out_path}...")
    torch.save(final_data, out_path)
    logger.info(f"合并完成，最终数据形状: {merged_input_ids.shape}")
    logger.info(f"最终文件大小: {os.path.getsize(out_path) / (1024 * 1024):.2f} MB")

    # 清理临时文件
    logger.info("正在清理临时文件...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"无法删除临时文件 {temp_file}: {str(e)}")
    logger.info("临时文件已清理")


@click.command()
@click.option("--data", type=str, help="原始数据文件路径，文本格式，每行一个序列")
@click.option("--tok", type=str, help="分词器名称或路径", default="zhihan1996/DNABERT-2-117M")
@click.option("-o", "--out", type=str, help="输出文件路径")
@click.option("--batch-size", type=int, default=10000, help="每批处理的序列数")
@click.option("--resume", is_flag=True, help="是否从上次中断处继续处理")
@click.option("--max-length", type=int, default=None, help="手动指定最大序列长度")

def main(data, tok, out, batch_size, resume, max_length):
    tokenize_batch_text(data, tok, out, batch_size, resume, max_length)


def tokenize_batch_text(data, tok, out, batch_size, resume, max_length):
    """
    将原始文本数据使用指定分词器进行分词处理，并保存为PyTorch格式。

    工作流程:
    1. 读取原始文本数据
    2. 清理数据，移除空行
    3. 加载指定的分词器
    4. 计算或使用指定的最大序列长度
    5. 分批对文本数据进行分词处理
    6. 合并所有批次结果并保存为PyTorch张量
    """
    total_start_time = time.time()

    # 创建输出目录和临时目录
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)

    temp_dir = os.path.join(out_dir, "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)

    # 检查是否需要恢复处理
    start_idx = 0
    checkpoint_max_length = None
    if resume:
        start_idx, checkpoint_max_length = load_checkpoint(out)

    # 步骤1: 读取原始数据
    logger.info(f"开始处理，从 {data} 读取原始数据")
    read_start = time.time()
    with open(data, "r") as f:
        data_text = f.read().split("\n")
    read_elapsed = time.time() - read_start
    logger.info(f"数据读取完成，耗时: {timedelta(seconds=read_elapsed)}，读取了 {len(data_text)} 行数据")
    logger.info(f"当前内存使用: {get_memory_usage()}")

    # 步骤2: 清理数据，移除空行
    data_text = rm_empty_lines(data_text)

    # 步骤3: 加载分词器
    logger.info(f"正在加载分词器: {tok}")
    tok_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok)
    tok_elapsed = time.time() - tok_start
    logger.info(f"分词器加载完成，耗时: {timedelta(seconds=tok_elapsed)}")
    logger.info(f"当前内存使用: {get_memory_usage()}")

    # 步骤4: 确定最大序列长度
    # 优先使用命令行参数，其次是检查点中的值，最后是计算得到的值
    if max_length is not None:
        logger.info(f"使用命令行指定的最大序列长度: {max_length}")
    elif checkpoint_max_length is not None:
        max_length = checkpoint_max_length
        logger.info(f"使用检查点中的最大序列长度: {max_length}")
    else:
        max_length = calculate_max_length(data_text, tokenizer)
        logger.info(f"计算得到的最大序列长度: {max_length}")

    # 如果有检查点，跳过已处理的部分
    if start_idx > 0:
        logger.info(f"从检查点恢复，跳过前 {start_idx} 条序列")
        data_text = data_text[start_idx:]

    # 步骤5: 分批处理文本数据
    logger.info(f"开始分批处理文本数据，批次大小: {batch_size}，最大序列长度: {max_length}")

    temp_files = []
    total_count = len(data_text) + start_idx

    # 获取已存在的临时文件
    existing_temp_files = []
    if resume:
        for i in range(start_idx // batch_size):
            temp_file = os.path.join(temp_dir, f"batch_{i}.pt")
            if os.path.exists(temp_file):
                existing_temp_files.append(temp_file)
                logger.info(f"找到已处理的批次文件: {temp_file}")

    # 分批处理
    for i in range(0, len(data_text), batch_size):
        batch_idx = i // batch_size + (start_idx // batch_size)
        batch_texts = data_text[i : i + batch_size]

        # 处理当前批次，使用统一的最大长度
        temp_file = process_batch(batch_texts, tokenizer, temp_dir, batch_idx, max_length)
        temp_files.append(temp_file)

        # 更新处理进度并保存检查点（包含最大长度信息）
        processed_count = start_idx + i + len(batch_texts)
        save_checkpoint(processed_count, total_count, out, max_length)

        # 释放内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 合并所有临时文件
    all_temp_files = existing_temp_files + temp_files
    merge_batch_files(all_temp_files, out)

    # 删除检查点文件
    checkpoint_path = f"{out}.checkpoint"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # 清理临时目录
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
        os.rmdir(temp_dir)

    # 总结处理过程
    total_elapsed = time.time() - total_start_time
    logger.info(f"全部处理完成，总耗时: {timedelta(seconds=total_elapsed)}")
    logger.info(f"分词后的数据已保存到: {out}")
    logger.info(f"最终内存使用: {get_memory_usage()}")


if __name__ == "__main__":
    main()

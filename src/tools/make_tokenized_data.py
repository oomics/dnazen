"""
DNA数据收集与分词工具

本脚本用于从不同数据源收集DNA序列数据，对其进行分词处理，
并将结果保存为可用于后续处理的文本格式。支持多种DNA数据源的组合使用。

使用方法:
    python make_tokenized_data.py -d [数据类型] -o [输出路径]

参数:
    -d/--data: 逗号分隔的数据类型列表。可用类型: all, gue_all, gue_test, hg38, hg38_all, mspecies
    -o/--out: 处理后数据的保存路径，默认为"/data3/peter/pretrain-tokenized.txt"

输出:
    处理后的数据将保存为文本文件，每行一个序列，序列中的token ID用冒号分隔
"""

import click
import time
import logging
import os
import psutil
from datetime import timedelta
from tqdm import tqdm
from transformers import AutoTokenizer


from utils.datas import (
    get_all_hg38_data,
    get_useful_gue_data,
    get_multi_species_data,
)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 数据目录配置
DATA_DIR = "/data1/peter"


def get_memory_usage():
    """
    获取当前进程的内存使用情况。

    Returns:
        str: 当前内存使用量的字符串表示，以MB为单位
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.2f} MB"


def get_rvs_complementary(text):
    """
    获取DNA序列的反向互补序列。

    在DNA中，A与T互补，C与G互补。
    反向互补序列是指原序列的反向序列，并将每个碱基替换为其互补碱基。

    Args:
        text (str): 原始DNA序列

    Returns:
        str: 反向互补序列
    """
    tmp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([tmp[c] for c in reversed(text)])


def get_gue_all():
    """
    获取所有GUE数据集的DNA序列（训练集、开发集和测试集）。

    Returns:
        list[str]: 包含所有GUE序列的列表
    """
    logger.info("正在获取所有GUE数据...")
    start_time = time.time()

    result = (
        get_useful_gue_data(f"{DATA_DIR}/GUE", type="train")
        + get_useful_gue_data(f"{DATA_DIR}/GUE", type="dev")
        + get_useful_gue_data(f"{DATA_DIR}/GUE", type="test")
    )

    elapsed = time.time() - start_time
    logger.info(f"GUE数据获取完成，共{len(result)}个序列，耗时: {timedelta(seconds=elapsed)}")

    return result


def get_gue_test():
    """
    获取GUE测试集数据。

    Returns:
        list[str]: 包含GUE测试集序列的列表
    """
    logger.info("正在获取GUE测试集数据...")
    start_time = time.time()

    result = get_useful_gue_data(f"{DATA_DIR}/GUE", type="test")

    elapsed = time.time() - start_time
    logger.info(f"GUE测试集数据获取完成，共{len(result)}个序列，耗时: {timedelta(seconds=elapsed)}")

    return result


def get_hg38():
    """
    获取人类基因组HG38数据。

    Returns:
        list[str]: 包含人类基因组序列的列表
    """
    logger.info("正在获取HG38基因组数据...")
    start_time = time.time()

    result = get_all_hg38_data(f"{DATA_DIR}/hg38.fa")

    elapsed = time.time() - start_time
    logger.info(f"HG38基因组数据获取完成，共{len(result)}个序列，耗时: {timedelta(seconds=elapsed)}")

    return result


def get_hg38_hg38rvs():
    """
    获取人类基因组HG38数据及其反向互补序列。

    包含原始HG38序列和每个序列的反向互补序列。

    Returns:
        list[str]: 包含HG38序列及其反向互补序列的列表
    """
    logger.info("正在获取HG38基因组数据及其反向互补序列...")
    start_time = time.time()

    data_hg38 = get_all_hg38_data(f"{DATA_DIR}/hg38.fa")
    logger.info(f"HG38原始数据获取完成，共{len(data_hg38)}个序列")

    # 计算反向互补序列
    logger.info("正在生成反向互补序列...")
    rvs_start_time = time.time()
    data_hg38_rvs = [get_rvs_complementary(d) for d in tqdm(data_hg38, desc="生成反向互补序列")]
    rvs_elapsed = time.time() - rvs_start_time
    logger.info(f"反向互补序列生成完成，耗时: {timedelta(seconds=rvs_elapsed)}")

    result = data_hg38 + data_hg38_rvs
    elapsed = time.time() - start_time
    logger.info(f"HG38+反向互补数据准备完成，共{len(result)}个序列，耗时: {timedelta(seconds=elapsed)}")

    return result


def get_multi_species_all():
    """
    获取多物种数据集（包括训练集和开发集）。

    Returns:
        list[str]: 包含多物种序列的列表
    """
    logger.info("正在获取多物种数据...")
    start_time = time.time()

    dev_data = get_multi_species_data(f"{DATA_DIR}/dev.txt")
    train_data = get_multi_species_data(f"{DATA_DIR}/train.txt")
    result = dev_data + train_data

    elapsed = time.time() - start_time
    logger.info(
        f"多物种数据获取完成，共{len(result)}个序列（开发集:{len(dev_data)}，训练集:{len(train_data)}），"
        f"耗时: {timedelta(seconds=elapsed)}"
    )

    return result


def get_all_data():
    """
    获取所有数据源的DNA序列组合。

    包括多物种数据、GUE数据和HG38数据（含反向互补）。

    Returns:
        list[str]: 包含所有数据源序列的列表
    """
    logger.info("正在获取所有数据源的组合...")
    start_time = time.time()

    mspecies_data = get_multi_species_all()
    gue_data = get_gue_all()
    hg38_data = get_hg38_hg38rvs()

    result = mspecies_data + gue_data + hg38_data

    elapsed = time.time() - start_time
    logger.info(f"所有数据获取完成，共{len(result)}个序列，耗时: {timedelta(seconds=elapsed)}")
    logger.info(
        f"数据组成: 多物种数据({len(mspecies_data)}), GUE数据({len(gue_data)}), HG38数据({len(hg38_data)})"
    )

    return result


# 数据类型到函数的映射
data_mapping = {
    "all": get_all_data,
    "gue_all": get_gue_all,
    "gue_test": get_gue_test,
    "hg38": get_hg38,
    "hg38_all": get_hg38_hg38rvs,
    "mspecies": get_multi_species_all,
}


def get_tokenized_data(
    data_types: list[str],
):
    """
    获取并分词指定类型的DNA数据。

    步骤:
    1. 根据指定的数据类型收集DNA序列
    2. 使用DNABERT-2分词器对序列进行分词
    3. 移除每个序列的特殊token（CLS和SEP）
    4. 返回分词结果

    Args:
        data_types (list[str]): 数据类型列表

    Returns:
        list[list[int]]: 分词后的token ID列表的列表
    """
    logger.info(f"开始获取数据，指定类型: {', '.join(data_types)}")
    start_time = time.time()

    # 处理data_types参数，如果包含'all'则忽略其他类型
    if "all" in data_types and len(data_types) > 1:
        data_types = ["all"]
        logger.warning("'all'包含所有数据类型。其他指定的数据类型将被忽略。")

    # 收集数据
    all_data = []
    for data_type in data_types:
        type_start_time = time.time()
        logger.info(f"正在收集{data_type}类型的数据...")

        if data_type not in data_mapping:
            logger.error(f"未知的数据类型: {data_type}")
            continue

        data_func = data_mapping[data_type]
        data = data_func()
        all_data.extend(data)

        type_elapsed = time.time() - type_start_time
        logger.info(
            f"{data_type}类型数据收集完成，获取了{len(data)}个序列，耗时: {timedelta(seconds=type_elapsed)}"
        )

    # 计算数据总长度
    total_len = 0
    for d in all_data:
        total_len += len(d)

    data_elapsed = time.time() - start_time
    logger.info(
        f"所有数据收集完成，共{len(all_data)}个序列，总长度={total_len}个字符，"
        f"耗时: {timedelta(seconds=data_elapsed)}"
    )
    logger.info(f"当前内存使用: {get_memory_usage()}")

    # 加载分词器
    logger.info("正在加载DNABERT-2分词器...")
    tok_start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True, use_fast=True
    )
    tok_elapsed = time.time() - tok_start_time
    logger.info(f"分词器加载完成，耗时: {timedelta(seconds=tok_elapsed)}")

    # 分词处理
    logger.info(f"开始对{len(all_data)}个序列进行分词...")
    tokenize_start_time = time.time()
    data_tokenized: list[list[int]] = tokenizer(
        all_data, return_attention_mask=False, return_token_type_ids=False
    )["input_ids"]  # type: ignore
    tokenize_elapsed = time.time() - tokenize_start_time
    logger.info(f"分词处理完成，耗时: {timedelta(seconds=tokenize_elapsed)}")

    # 移除特殊token（CLS和SEP）
    logger.info("正在移除特殊token（CLS和SEP）...")
    clean_start_time = time.time()
    for idx, data in enumerate(tqdm(data_tokenized, desc="移除特殊token")):
        data.pop(0)  # 移除CLS token
        data.pop(-1)  # 移除SEP token
    clean_elapsed = time.time() - clean_start_time
    logger.info(f"特殊token移除完成，耗时: {timedelta(seconds=clean_elapsed)}")

    # 释放不再需要的数据
    del all_data
    del tokenizer

    total_elapsed = time.time() - start_time
    logger.info(f"数据获取和分词完成，总耗时: {timedelta(seconds=total_elapsed)}")
    logger.info(f"当前内存使用: {get_memory_usage()}")

    # 显示各步骤耗时占比
    logger.info("各步骤耗时占比:")
    logger.info(
        f"- 数据收集: {data_elapsed / total_elapsed * 100:.1f}% ({timedelta(seconds=data_elapsed)})"
    )
    logger.info(
        f"- 分词器加载: {tok_elapsed / total_elapsed * 100:.1f}% ({timedelta(seconds=tok_elapsed)})"
    )
    logger.info(
        f"- 分词处理: {tokenize_elapsed / total_elapsed * 100:.1f}% ({timedelta(seconds=tokenize_elapsed)})"
    )
    logger.info(
        f"- 移除特殊token: {clean_elapsed / total_elapsed * 100:.1f}% ({timedelta(seconds=clean_elapsed)})"
    )

    return data_tokenized


@click.command()
@click.option(
    "-d",
    "--data",
    type=str,
    help="逗号分隔的数据类型列表。可用类型: all, gue_all, gue_test, hg38, hg38_all, mspecies",
)
@click.option(
    "-o",
    "--out",
    default="/data3/peter/pretrain-tokenized.txt",
    help="分词数据的保存路径",
    type=str,
)
def main(data: str, out: str):
    """
    主函数: 收集DNA数据，进行分词，并保存处理结果。

    工作流程:
    1. 根据指定类型收集DNA序列数据
    2. 对序列进行分词处理
    3. 将处理后的数据保存为文本文件
    """
    total_start_time = time.time()
    logger.info(f"开始处理，数据类型: {data}, 输出路径: {out}")

    # 验证输入参数
    if not data:
        logger.error("必须指定数据类型。使用 --data 选项。")
        return

    # 解析数据类型
    data_types = data.split(",")
    logger.info(f"指定的数据类型: {', '.join(data_types)}")

    # 获取并分词数据
    data_tokenized: list[list[int]] = get_tokenized_data(data_types)

    # 保存处理后的数据
    logger.info(f"正在将{len(data_tokenized)}个分词序列保存到: {out}")
    save_start_time = time.time()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    with open(out, "w") as f:
        for i, d in enumerate(tqdm(data_tokenized[:-1], desc="保存进度")):
            # 将token ID列表转换为冒号分隔的字符串
            token_str = ":".join([str(num) for num in d]) + "\n"
            f.write(token_str)

            # 每处理1000个序列显示一次进度
            if (i + 1) % 1000 == 0:
                logger.info(f"已保存 {i + 1}/{len(data_tokenized)} 个序列")

    save_elapsed = time.time() - save_start_time
    logger.info(f"数据保存完成，耗时: {timedelta(seconds=save_elapsed)}")

    # 输出文件信息
    file_size = os.path.getsize(out) / (1024 * 1024)  # 转换为MB
    logger.info(f"保存文件大小: {file_size:.2f} MB")

    # 统计处理过程
    total_elapsed = time.time() - total_start_time
    logger.info(f"全部处理完成，总耗时: {timedelta(seconds=total_elapsed)}")
    logger.info(f"最终内存使用: {get_memory_usage()}")


if __name__ == "__main__":
    main()

"""Make pretrained dataset from .txt/.fa files.

本脚本用于从原始文本文件或预处理后的token文件构建预训练数据集，
并将训练集和验证集保存到指定的输出目录中。"""

import random  # 用于生成随机数，确保实验可复现
from typing import Literal  # 用于类型注解，限制变量取值
import time  # 用于计算加载和下载时间

import logging  # 日志记录模块，用于输出调试信息

import click  # 用于简化命令行界面开发
from transformers import AutoTokenizer  # 加载预训练的Tokenizer

from dnazen.ngram import NgramEncoder  # 导入n-gram编码器模块，用于处理n-gram特征
from dnazen.data.mlm_dataset import (
    MlmDataset,
    _load_core_ngrams,
)  # 导入构建MLM数据集的工具和加载核心n-gram的函数

# 配置日志输出格式和级别，方便调试和记录运行信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--data-source",
    type=click.Choice(["raw", "tokenized"]),
    default="raw",
    help="数据类型：原始数据(raw)或已分词数据(tokenized)",
)
@click.option(
    "-d",
    "--data",
    "data_dir",
    type=str,
    default="/data2/peter/dnazen_pretrain_v3",
    help="数据文件的路径目录",
)
@click.option(
    "--tok-source",
    "tokenizer_source",
    type=click.Choice(["file", "huggingface"]),
    default="huggingface",
    help="Tokenizer的加载方式：文件或Huggingface模型",
)
@click.option(
    "--tok",
    "tokenizer_cfg",
    type=str,
    default="zhihan1996/DNABERT-2-117M",
    help="Tokenizer的配置，支持模型名称或路径",
)
@click.option("--ngram", "ngram_file", type=str, help="n-gram解码器的配置文件路径")
@click.option("--core-ngram", type=str, default=None, help="核心n-gram文件路径，可选")
@click.option("--max-ngrams", default=30, type=int, help="最大允许匹配的n-gram数量")
@click.option("--out", "output_dir", type=str, help="保存数据集输出目录的路径")
@click.option("--seed", type=int, default=42, help="随机种子")
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
):
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

    if core_ngram is not None:
        # 开始记录加载核心n-gram文件的时间
        start_time = time.time()
        core_ngrams = _load_core_ngrams(core_ngram)
        end_time = time.time()
        logger.info("核心n-gram加载耗时: %.2f秒", end_time - start_time)
    else:
        core_ngrams = set()

    # 根据数据类型进行不同的数据加载操作
    if data_source == "raw":
        # 当数据源为原始文本时，从原始文本创建验证数据集
        val_dataset = MlmDataset.from_raw_data(
            f"{data_dir}/dev/dev.txt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,  # 设置蒙面语言模型中mask的概率为20%
        )
        # 保存验证数据集到指定输出目录
        val_dataset.save(f"{output_dir}/dev")

        # 从原始文本创建训练数据集
        train_dataset = MlmDataset.from_raw_data(
            f"{data_dir}/train/train.txt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        # 保存训练数据集到输出目录
        train_dataset.save(f"{output_dir}/train")
    else:
        # 当数据源为预处理后的分词数据时，从tokenized数据创建验证数据集
        val_dataset = MlmDataset.from_tokenized_data(
            data_dir=f"{data_dir}/dev/dev.pt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        # 保存验证数据集
        val_dataset.save(f"{output_dir}/dev")

        # 从tokenized数据创建训练数据集
        train_dataset = MlmDataset.from_tokenized_data(
            data_dir=f"{data_dir}/train/train.pt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        # 保存训练数据集
        train_dataset.save(f"{output_dir}/train")


if __name__ == "__main__":
    # 当脚本作为主程序执行时，调用main函数
    main()

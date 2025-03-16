"""Make pretrained dataset from .txt/.fa files.

本脚本用于从原始文本文件或预处理后的token文件构建预训练数据集，
并将训练集和验证集保存到指定的输出目录中。"""

import random  # 用于生成随机数，确保实验可复现
from typing import Literal  # 用于类型注解，限制变量取值
import time  # 用于计算加载和下载时间
import os  # 操作系统相关功能，用于文件路径处理
import datetime  # 日期时间操作，用于生成时间戳
from pathlib import Path  # 文件路径处理，提供面向对象的文件系统路径处理

import logging  # 日志记录模块，用于输出调试信息
from tqdm import tqdm  # 进度条显示，提供训练过程可视化

import click  # 用于简化命令行界面开发
import torch  # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler  # 数据加载和处理工具

from transformers import (
    AutoTokenizer,  # 加载预训练的Tokenizer
    PreTrainedTokenizer,  # 预训练Tokenizer基类
    CONFIG_NAME,  # 模型配置文件名常量
    WEIGHTS_NAME,  # 模型权重文件名常量
)

# 导入ZEN模型相关组件
from ZEN.modeling import ZenForPreTraining, ZenConfig

# 导入自定义工具和模块
from dnazen.ngram import NgramEncoder  # n-gram编码器模块，用于处理n-gram特征
from dnazen.data.mlm_dataset import (
    MlmDataset,
    _load_core_ngrams,
)  # 用于构建掩码语言模型数据集的工具

# 导入优化器和学习率调度器
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

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
@click.option(
    "--ngram", 
    "ngram_file", 
    type=str, 
    required=True,
    help="n-gram解码器的配置文件路径"
)
@click.option(
    "--core-ngram", 
    type=str, 
    default=None, 
    help="核心n-gram文件路径，可选"
)
@click.option(
    "--max-ngrams", 
    default=30, 
    type=int, 
    help="最大允许匹配的n-gram数量"
)
@click.option(
    "--out", 
    "output_dir", 
    type=str, 
    required=True,
    help="保存数据集输出目录的路径"
)
@click.option(
    "--seed", 
    type=int, 
    default=42, 
    help="随机种子，用于确保实验可重复"
)
# 添加新的训练相关选项
@click.option(
    "--model",
    "bert_model",
    type=str,
    default="zhihan1996/DNABERT-2-117M",
    help="预训练模型路径或名称",
)
@click.option(
    "--lr",
    "learning_rate",
    type=float,
    default=2e-5,
    help="学习率，默认2e-5",
)
@click.option(
    "--loss-scale",
    type=float,
    default=0,
    help="FP16训练的损失缩放系数，0表示动态缩放",
)
@click.option(
    "--warmup",
    "warmup_proportion",
    type=float,
    default=0.1,
    help="学习率预热比例，默认0.1",
)
@click.option(
    "--data-epochs",
    "num_data_epochs",
    type=int,
    default=1,
    help="数据训练轮数",
)
@click.option(
    "--reduce-mem/--no-reduce-mem",
    "reduce_memory",
    default=True,
    help="是否启用内存优化",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="训练总轮数",
)
@click.option(
    "--batch-size",
    "train_batch_size",
    type=int,
    default=128,
    help="训练批次大小",
)
@click.option(
    "--grad-accum",
    "gradient_accumulation_steps",
    type=int,
    default=16,
    help="梯度累积步数",
)
@click.option(
    "--local-rank",
    type=int,
    default=-1,
    help="分布式训练的本地排名，-1表示单机训练",
)
@click.option(
    "--fp16/--no-fp16",
    default=False,
    help="是否使用半精度(FP16)训练",
)
@click.option(
    "--scratch/--no-scratch",
    default=False,
    help="是否从零开始训练，不使用预训练模型",
)
@click.option(
    "--save-prefix",
    "save_name",
    type=str,
    default="dnazen_",
    help="保存模型的名称前缀",
)
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
    
    示例：
        # 使用默认配置进行训练
        python src/train/run_pretrain_zen.py --ngram ../out/exp1_pmi5/ngram_encoder.json --out ./data/
        
        # 使用自定义配置进行训练
        python src/train/run_pretrain_zen.py  \\
            --data-source tokenized \\
            --data ../mspecies/train/train.txt \\
            --ngram ../out/exp1_pmi5/ngram_encoder.json \\
            --out ./data/ \\
            --lr 3e-5 \\
            --epochs 20 \\
            --batch-size 256 \\
            --fp16
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
    data_source: Literal["raw", "tokenized"],  # 数据源类型：原始数据或已分词数据
    data_dir: str,  # 数据文件的路径目录
    tokenizer_source: Literal["file", "huggingface"],  # Tokenizer来源：文件或Huggingface
    tokenizer_cfg: str,  # Tokenizer配置，可以是模型名称或路径
    ngram_file: str,  # n-gram解码器配置文件路径
    output_dir: str,  # 模型和数据保存的输出目录
    seed: int,  # 随机种子，用于确保实验可重复
    bert_model: str,  # 预训练BERT模型的路径或名称
    learning_rate: float,  # 学习率，影响模型收敛速度
    loss_scale: float,  # 损失缩放系数，用于FP16训练
    warmup_proportion: float,  # 预热比例，控制学习率预热阶段
    num_data_epochs: int,  # 数据训练的轮数
    reduce_memory: bool,  # 是否减少内存使用，适用于资源受限情况
    epochs: int,  # 训练总轮数
    train_batch_size: int,  # 训练批次大小
    gradient_accumulation_steps: int,  # 梯度累积步数，用于模拟更大批次
    local_rank: int,  # 分布式训练的本地排名
    scratch: bool,  # 是否从零开始训练
    fp16: bool,  # 是否使用半精度浮点训练
    save_name: str,  # 保存模型的名称前缀
    already_trained_epoch: int,  # 已经训练的轮数，用于继续训练
    pregenerated_data: str | None,  # 预生成数据的路径，如果为None则使用data_dir
):
    """准备预训练数据并训练模型
    
    本函数实现了DNA序列预训练的完整流程，包括：
    1. 加载数据和tokenizer
    2. 构建和初始化模型
    3. 设置优化器和学习率调度器
    4. 执行训练循环
    5. 保存训练好的模型

    Args:
        data_source: 数据类型，原始数据(raw)或已分词数据(tokenized)
        data_dir: 数据文件的路径目录
        tokenizer_source: Tokenizer的加载方式：文件或Huggingface模型
        tokenizer_cfg: Tokenizer的配置，支持模型名称或路径
        ngram_file: n-gram解码器的配置文件路径
        output_dir: 保存数据集输出目录的路径
        seed: 随机种子，确保结果可复现
        bert_model: 预训练模型路径或名称
        learning_rate: 学习率，影响模型收敛速度
        loss_scale: 损失缩放系数，用于FP16训练
        warmup_proportion: 预热比例，控制学习率预热阶段
        num_data_epochs: 数据训练的轮数
        reduce_memory: 是否减少内存使用，适用于资源受限情况
        epochs: 训练总轮数
        train_batch_size: 训练批次大小
        gradient_accumulation_steps: 梯度累积步数，用于模拟更大批次
        local_rank: 分布式训练的本地排名
        scratch: 是否从零开始训练
        fp16: 是否使用半精度浮点训练
        save_name: 保存模型的名称前缀
        already_trained_epoch: 已训练的轮数，用于继续训练
        pregenerated_data: 预生成数据的路径
    """

    
    # 设置随机种子，确保实验结果可复现
    random.seed(seed)
    
    # 如果未指定预生成数据路径，则使用输入数据目录
    if pregenerated_data is None:
        pregenerated_data = data_dir
    
    # 获取logger对象，用于记录训练过程信息
    logger = logging.getLogger(__name__)

    # 记录训练配置信息
    logger.info("="*50)
    logger.info("训练配置信息:")
    logger.info(f"数据源类型: {data_source}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"预训练模型: {bert_model}")
    logger.info(f"学习率: {learning_rate}")
    logger.info(f"批次大小: {train_batch_size}")
    logger.info(f"梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"训练轮数: {epochs}")
    logger.info(f"是否使用FP16: {fp16}")
    logger.info(f"是否从零开始训练: {scratch}")
    logger.info(f"已训练轮数: {already_trained_epoch}")
    logger.info("="*50)
    
    # 设置计算设备，优先使用GPU加速训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取可用的GPU数量，用于多GPU并行训练
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 记录硬件信息
    logger.info("硬件配置信息:")
    logger.info(f"使用设备: {device}")
    logger.info(f"GPU数量: {n_gpu}")
    if torch.cuda.is_available():
        for i in range(n_gpu):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("="*50)
    
    # 从Huggingface加载预训练的tokenizer
    logger.info(f"加载tokenizer: {tokenizer_cfg}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg)
    logger.info(f"Tokenizer加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 获取每个epoch的样本数，用于计算优化步数
    #samples_per_epoch = get_samples_per_epoch(pregenerated_data, num_data_epochs)
    
    # 检查是否成功获取样本数，如果没有则需要先生成预训练数据
    if not samples_per_epoch:
        logger.error(f"未在路径 {pregenerated_data} 找到预生成的训练数据")
        logger.error("请先使用 create_pretraining_data.py 生成预训练数据集")
        return

    # 确保samples_per_epoch非空
    if len(samples_per_epoch) == 0:
        logger.error("samples_per_epoch为空，请检查数据集是否正确生成")
        return

    logger.info(f"找到预训练数据: {pregenerated_data}")
    logger.info(f"每个epoch的样本数: {samples_per_epoch}")

    # 计算总训练样本数，考虑到可能循环使用有限的数据
    total_train_examples = 0
    num_data_epochs = len(samples_per_epoch)
    
    for i in range(epochs):
        epoch_index = i % num_data_epochs
        total_train_examples += samples_per_epoch[epoch_index]
        logger.info(f"Epoch {i+1}: 使用数据集 {epoch_index+1}，样本数: {samples_per_epoch[epoch_index]}")

    logger.info(f"总训练样本数: {total_train_examples}")

    # 计算总优化步数，用于设置学习率调度
    num_train_optimization_steps = int(
        total_train_examples / train_batch_size / gradient_accumulation_steps)
    
    # 分布式训练时，需要除以节点数量
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # 模型初始化：从零开始或加载预训练模型
    if scratch:
        # 从零开始训练，创建新的ZEN模型配置和模型实例
        config = ZenConfig(21128, 104089)  # 词汇表大小和n-gram大小
        model = ZenForPreTraining(config)
    else:
        # 加载预训练模型，复用已有参数
        model = ZenForPreTraining.from_pretrained(bert_model)

    # 启用半精度训练（如果指定）
    if fp16:
        model.half()  # 将模型参数转换为FP16格式
    
    # 将模型移动到指定设备(GPU/CPU)
    model.to(device)
    
    # 设置分布式训练或多GPU训练
    if local_rank != -1:
        # 分布式训练配置
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "请从https://www.github.com/nvidia/apex安装apex库以使用分布式和FP16训练。")
        model = DDP(model)
    elif n_gpu > 1:
        # 多GPU数据并行训练
        model = torch.nn.DataParallel(model)

    # 准备优化器参数组，针对不同参数设置不同的权重衰减
    param_optimizer = list(model.named_parameters())
    # LayerNorm和偏置项不应用权重衰减
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 将参数分为两组：应用权重衰减和不应用权重衰减的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},  # 应用0.01的权重衰减
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}  # 不应用权重衰减
    ]

    # 根据是否使用半精度训练选择不同的优化器
    if fp16:
        # 半精度训练需要特殊的优化器
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "请从https://www.github.com/nvidia/apex安装apex库以使用分布式和FP16训练。")

        # 使用FusedAdam优化器，专为半精度训练设计
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        
        # 配置FP16优化器包装器
        if loss_scale == 0:
            # 使用动态损失缩放
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            # 使用静态损失缩放
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        
        # 创建学习率预热调度器
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        # 全精度训练使用BertAdam优化器
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    # 初始化全局步数计数器
    global_step = 0
    
    # 打印训练信息
    logger.info("***** 开始训练 *****")
    logger.info("  样本总数 = %d", total_train_examples)
    logger.info("  批次大小 = %d", train_batch_size)
    logger.info("  优化步数 = %d", num_train_optimization_steps)
    
    # 将模型设置为训练模式
    model.train()
    
    # 导入预生成数据集类
    from dnazen.data.pregenerated_dataset import PregeneratedDataset
    
    # 开始训练循环，共训练epochs轮
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
        
        # 计算当前epoch使用的数据集索引
        epoch_dataset_index = epoch % num_data_epochs
        logger.info(f"使用数据集 {epoch_dataset_index + 1}/{num_data_epochs}")
        
        # 为当前epoch创建数据集
        logger.info("创建数据集...")
        dataset_start_time = time.time()
        epoch_dataset = PregeneratedDataset(
            epoch=epoch,  # 当前训练轮次
            training_path=pregenerated_data,  # 训练数据路径
            tokenizer=tokenizer,  # 使用的tokenizer
            num_data_epochs=num_data_epochs,  # 数据训练的轮数
            reduce_memory=reduce_memory,  # 是否减少内存使用
            fp16=fp16  # 是否使用半精度格式
        )
        logger.info(f"数据集创建完成，耗时: {time.time() - dataset_start_time:.2f}秒")
        logger.info(f"数据集大小: {len(epoch_dataset)}样本")
        
        # 根据训练模式选择不同的采样器
        if local_rank == -1:
            # 单机训练使用随机采样器
            train_sampler = RandomSampler(epoch_dataset)
        else:
            # 分布式训练使用分布式采样器
            train_sampler = DistributedSampler(epoch_dataset)
        
        # 创建数据加载器，用于批量加载训练数据
        train_dataloader = DataLoader(
            epoch_dataset, 
            sampler=train_sampler, 
            batch_size=train_batch_size
        )
        
        # 初始化当前epoch的训练损失和样本计数
        tr_loss = 0  # 累计训练损失
        nb_tr_examples, nb_tr_steps = 0, 0  # 训练样本数和步数
        
        # 使用tqdm创建进度条，用于可视化训练进度
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            # 遍历每个批次的数据
            for step, batch in enumerate(train_dataloader):
                batch_start_time = time.time()
                
                # 将批次数据移动到指定设备(GPU/CPU)
                batch = tuple(t.to(device) for t in batch)
                
                # 解包批次数据，获取输入特征和标签
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, ngram_ids, ngram_masks, ngram_positions, \
                ngram_starts, \
                ngram_lengths, ngram_segment_ids = batch

                # 前向传播，计算损失值
                forward_start_time = time.time()
                loss = model(
                    input_ids,  # 输入token IDs
                    ngram_ids,  # n-gram IDs
                    ngram_positions,  # n-gram在序列中的位置
                    segment_ids,  # 分段IDs，用于区分不同的序列
                    ngram_segment_ids,  # n-gram的分段IDs
                    input_mask,  # 注意力掩码，用于忽略填充token
                    ngram_masks,  # n-gram掩码
                    lm_label_ids,  # 语言模型标签IDs
                    is_next  # 下一句预测标签
                )
                forward_time = time.time() - forward_start_time

                # 多GPU训练时，需要对损失取平均
                if n_gpu > 1:
                    loss = loss.mean()  # 多GPU平均损失
                
                # 梯度累积：将损失除以累积步数
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                # 反向传播，计算梯度
                backward_start_time = time.time()
                if fp16:
                    # 半精度训练使用优化器的backward方法
                    optimizer.backward(loss)
                else:
                    # 全精度训练直接调用loss的backward方法
                    loss.backward()
                backward_time = time.time() - backward_start_time
                
                # 累计训练损失和样本数
                tr_loss += loss.item()  # 累加当前批次的损失
                nb_tr_examples += input_ids.size(0)  # 累加样本数
                nb_tr_steps += 1  # 累加步数
                
                # 更新进度条
                pbar.update(1)
                
                # 计算平均损失并显示在进度条上
                mean_loss = tr_loss * gradient_accumulation_steps / nb_tr_steps
                batch_time = time.time() - batch_start_time
                
                # 每100步记录一次详细信息
                if step % 100 == 0:
                    logger.info(f"Step {step}/{len(train_dataloader)}:")
                    logger.info(f"  损失值: {mean_loss:.5f}")
                    logger.info(f"  批次处理时间: {batch_time:.3f}秒")
                    logger.info(f"  前向传播时间: {forward_time:.3f}秒")
                    logger.info(f"  反向传播时间: {backward_time:.3f}秒")
                    if torch.cuda.is_available():
                        logger.info(f"  GPU内存使用: {torch.cuda.memory_allocated()/1024/1024:.1f}MB")
                
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f} | Batch: {batch_time:.3f}s")
                
                # 梯度累积达到指定步数后，更新模型参数
                if (step + 1) % gradient_accumulation_steps == 0:
                    # 半精度训练时，需要手动调整学习率
                    if fp16:
                        # 使用BERT专用的预热学习率调整方法
                        lr_this_step = learning_rate * warmup_linear.get_lr(global_step, warmup_proportion)
                        # 更新优化器中所有参数组的学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    
                    # 更新模型参数
                    optimizer.step()
                    # 清零梯度，准备下一次累积
                    optimizer.zero_grad()
                    # 增加全局步数
                    global_step += 1

        # 每个epoch结束后记录统计信息
        epoch_time = time.time() - epoch_start_time
        logger.info(f"\nEpoch {epoch+1} 统计信息:")
        logger.info(f"  总耗时: {epoch_time:.2f}秒")
        logger.info(f"  平均每步时间: {epoch_time/len(train_dataloader):.3f}秒")
        logger.info(f"  最终损失值: {mean_loss:.5f}")
        logger.info(f"  处理样本数: {nb_tr_examples}")
        
        # 保存模型相关日志
        logger.info("\n开始保存模型...")
        save_start_time = time.time()
        
        # 获取当前时间戳，用于生成唯一的模型保存路径
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')

        # 基础保存路径
        saving_path = output_dir
        
        # 创建模型保存目录，包含时间戳和轮次信息
        model_save_dir = Path(os.path.join(saving_path, save_name + st + "_epoch_" + str(epoch + already_trained_epoch)))

        # 检查保存目录是否已存在且非空
        if model_save_dir.is_dir() and list(model_save_dir.iterdir()):
            logger.warning(f"输出目录 ({ model_save_dir }) 已存在且不为空!")
        
        # 创建保存目录（如果不存在）
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型、配置和词汇表
        logger.info("** ** * 保存微调后的模型 ** ** * ")
        
        # 如果使用了DataParallel，需要获取内部模型
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # 设置模型权重和配置文件的保存路径
        output_model_file = os.path.join(model_save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(model_save_dir, CONFIG_NAME)

        # 保存模型状态字典
        torch.save(model_to_save.state_dict(), output_model_file)
        # 保存模型配置
        model_to_save.config.to_json_file(output_config_file)
        # 保存tokenizer词汇表
        tokenizer.save_vocabulary(model_save_dir)

        # 保存完成后记录信息
        save_time = time.time() - save_start_time
        logger.info(f"模型保存完成，耗时: {save_time:.2f}秒")
        logger.info(f"保存路径: {model_save_dir}")
        logger.info("="*50)

if __name__ == "__main__":
    # 当脚本作为主程序执行时，调用main函数
    main()

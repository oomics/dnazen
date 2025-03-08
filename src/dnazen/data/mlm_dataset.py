"""
MLM数据集模块

本模块实现了用于掩码语言模型(MLM)预训练的数据集类，支持N-gram特征提取和全N-gram掩码。
主要包含以下功能：
1. 数据加载和保存
2. 运行时token掩码
3. N-gram特征提取和编码
4. 核心N-gram保护（防止被掩码）
"""

from typing import TypedDict
import random
import os
import json
import logging

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from dnazen.ngram import NgramEncoder
from dnazen.misc import hash_file_md5, check_hash_of_file_md5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MlmData(TypedDict):
    """MLM数据的输出格式，包含所有模型输入所需的字段"""

    input_ids: torch.Tensor  # 输入token IDs
    ngram_input_ids: torch.Tensor  # N-gram IDs
    attention_mask: torch.Tensor  # 注意力掩码
    ngram_attention_mask: torch.Tensor  # N-gram注意力掩码
    ngram_position_matrix: torch.Tensor  # N-gram位置矩阵
    labels: torch.Tensor  # MLM任务的标签


class MlmDataSaved(TypedDict):
    """保存到磁盘的MLM数据格式"""

    input_ids: torch.Tensor  # 输入token IDs
    attention_mask: torch.Tensor  # 注意力掩码


class MlmDataConfig(TypedDict):
    """MLM数据集配置"""

    whole_ngram_masking: bool  # 是否使用全N-gram掩码
    mlm_prob: float  # MLM掩码概率
    mlm_data_symlink: str | None  # MLM数据符号链接
    mlm_data_hash_val: str | None  # MLM数据哈希值


# --- 工具函数 ---


def _save_core_ngrams(
    path: str,
    core_ngrams: set[tuple[int, ...]],
):
    """
    保存核心N-gram到文件

    参数:
        path: 保存路径
        core_ngrams: 核心N-gram集合，每个N-gram是token ID的元组
    """
    # 将元组转换为列表以便JSON序列化
    core_ngrams_ = [list(ngram) for ngram in list(core_ngrams)]
    
    logger.info(f"正在保存{len(core_ngrams)}个核心N-gram到{path}")
    with open(path, "w") as f:
        json.dump(core_ngrams_, f)
    logger.info(f"核心N-gram保存完成")


def _load_core_ngrams(
    path: str,
):
    """
    从文件加载核心N-gram

    参数:
        path: 加载路径

    返回:
        核心N-gram集合，每个N-gram是token ID的元组
    """
    logger.info(f"正在从{path}加载核心N-gram")
    with open(path, "r") as f:
        ngrams = json.load(f)

    # 将列表转换回元组
    result = set(tuple(ngram) for ngram in ngrams)
    logger.info(f"成功加载{len(result)}个核心N-gram")
    return result


class TokenMasker:
    """
    Token掩码器，用于MLM任务中的token掩码

    支持两种掩码策略：
    1. 随机token掩码：随机选择token进行掩码
    2. 全N-gram掩码：整个N-gram一起掩码

    同时支持核心N-gram保护，防止重要的N-gram被掩码
    """

    def __init__(
        self,
        core_ngrams: set[tuple[int, ...]],
        cls_token: int,
        sep_token: int,
        pad_token: int,
        mask_token: int,
        whole_ngram_masking: bool = True,
        verbose: bool = True,
    ):
        """
        初始化Token掩码器

        参数:
            core_ngrams: 核心N-gram集合，这些N-gram不会被掩码
            cls_token: CLS token的ID
            sep_token: SEP token的ID
            pad_token: PAD token的ID
            mask_token: MASK token的ID
            whole_ngram_masking: 是否使用全N-gram掩码
            verbose: 是否输出详细日志
        """
        self.core_ngrams = core_ngrams
        self.whole_ngram_masking = whole_ngram_masking
        self.CLS = cls_token
        self.SEP = sep_token
        self.PAD = pad_token
        self.MASK = mask_token

        # 计算核心N-gram的长度范围
        self.core_ngram_min_len = 128
        self.core_ngram_max_len = 0
        for ngram in core_ngrams:
            self.core_ngram_min_len = min(self.core_ngram_min_len, len(ngram))
            self.core_ngram_max_len = max(self.core_ngram_max_len, len(ngram))

        if verbose:
            logger.info(
                f"TokenMasker初始化完成。"
                f"核心N-gram最小长度={self.core_ngram_min_len}; "
                f"核心N-gram最大长度={self.core_ngram_max_len}"
            )

    def create_mlm_predictions(
        self,
        token_seq: torch.Tensor,
        mlm_prob: float,
        vocab_list: list[int],
        ngram_encoder: NgramEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        创建MLM预测任务的输入和标签

        参数:
            token_seq: 输入token序列
            mlm_prob: 掩码概率
            vocab_list: 词汇表列表
            ngram_encoder: N-gram编码器

        返回:
            masked_token_seq: 掩码后的token序列
            labels: MLM任务的标签，被掩码的位置为原始token ID，其他位置为-100
        """
        assert token_seq.dim() == 1, f"token_seq应为一维数组，但维度为{token_seq.dim()}"

        # 找出可以被掩码的候选位置（排除特殊token）
        candidate_idxes_list = (
            torch.nonzero(
                (token_seq != self.CLS) & (token_seq != self.SEP) & (token_seq != self.MASK),
                as_tuple=False,
            )
            .squeeze()
            .tolist()
        )

        # 找出核心N-gram的位置，这些位置不会被掩码
        token_seq_list = token_seq.tolist()
        non_candidiate_idxes = []
        if len(self.core_ngrams) > 0:
            for idx in candidate_idxes_list:
                for len_ in range(self.core_ngram_min_len, self.core_ngram_max_len + 1):
                    if idx + len_ > token_seq.shape[0]:
                        continue
                    # 检查是否匹配核心N-gram
                    if tuple(token_seq_list[idx : idx + len_]) in self.core_ngrams:
                        non_candidiate_idxes += range(idx, idx + len_)

        # 创建掩码序列和标签
        masked_token_seq = token_seq.clone()
        labels = torch.empty_like(token_seq).fill_(-100)  # 默认标签为-100（不计算损失）

        # 初始化ngram_mask，无论是否使用全N-gram掩码
        ngram_mask = torch.zeros_like(token_seq, dtype=torch.bool)
        num_ngram_mask = 0
        
        # 全N-gram掩码策略
        if self.whole_ngram_masking:
            # 创建N-gram掩码
            for ngram, idx in ngram_encoder.get_matched_ngrams(token_seq, pad_token_id=self.PAD):
                # 随机决定是否掩码这个N-gram
                num = random.random()
                if num < mlm_prob:
                    ngram_mask[idx : idx + len(ngram)] = 1

            # 统计N-gram掩码的数量
            num_ngram_mask = int(ngram_mask.sum().item())

            # 应用掩码策略：80%替换为[MASK]，10%保持不变，10%替换为随机token
            rand = torch.rand_like(token_seq, dtype=torch.float)
            ngram_mask_80 = ngram_mask & (rand < 0.8)  # 80%替换为[MASK]
            ngram_mask_10 = ngram_mask & (rand >= 0.8) & (rand < 0.9)  # 10%保持不变
            ngram_mask_10_rand = ngram_mask & (rand >= 0.9)  # 10%替换为随机token

            # 应用掩码
            masked_token_seq[ngram_mask_80] = self.MASK
            masked_token_seq[ngram_mask_10] = token_seq[ngram_mask_10]
            masked_token_seq[ngram_mask_10_rand] = torch.tensor(
                random.choices(vocab_list, k=int(ngram_mask_10_rand.sum().item())),
                dtype=torch.long,
            )

        # 计算序列长度和候选位置数量
        seq_len = len(candidate_idxes_list)
        candidate_idxes = torch.tensor(
            [idx for idx in candidate_idxes_list if idx not in non_candidiate_idxes],
            dtype=torch.int32,
        )

        # 调整掩码概率，考虑已经被N-gram掩码的token
        len_prop = (seq_len - num_ngram_mask) / len(candidate_idxes) if len(candidate_idxes) > 0 else 0
        mlm_prob *= len_prop  # 修改掩码概率

        # 为候选位置创建掩码
        candidate_mask = torch.zeros_like(token_seq, dtype=torch.bool)
        candidate_mask[candidate_idxes] = 1

        # 随机选择要掩码的token
        mask_prob = torch.full_like(token_seq, mlm_prob, dtype=torch.float)
        mask_prob[~candidate_mask] = 0
        mask_mask = torch.bernoulli(mask_prob).bool()  # 1=掩码；0=不掩码

        # 应用掩码
        labels[mask_mask] = masked_token_seq[mask_mask]  # 记录原始token作为标签
        rand = torch.rand_like(token_seq, dtype=torch.float)
        mask_mask_80 = mask_mask & (rand < 0.8)  # 80%替换为[MASK]
        mask_mask_10 = mask_mask & (rand >= 0.8) & (rand < 0.9)  # 10%保持不变
        mask_mask_10_rand = mask_mask & (rand >= 0.9)  # 10%替换为随机token

        # 应用掩码策略
        masked_token_seq[mask_mask_80] = self.MASK
        masked_token_seq[mask_mask_10] = token_seq[mask_mask_10]
        masked_token_seq[mask_mask_10_rand] = torch.tensor(
            random.choices(vocab_list, k=int(mask_mask_10_rand.sum().item())),
            dtype=torch.long,
        )

        # 统计掩码信息
        total_masked = int((mask_mask | ngram_mask).sum().item())
        total_tokens = int((token_seq != self.PAD).sum().item())
        mask_percentage = total_masked / total_tokens * 100 if total_tokens > 0 else 0

        return masked_token_seq, labels


class MlmDataset(Dataset):
    """
    MLM任务数据集

    特点：
    1. 支持运行时token掩码
    2. 支持N-gram特征提取
    3. 支持全N-gram掩码
    4. 支持核心N-gram保护
    5. 支持数据保存和加载
    """

    # 文件名常量
    NGRAM_ENCODER_FNAME = "ngram_encoder.json"
    TOKENIZER_DIR = "tokenizer"
    DATA_FNAME = "data.pt"
    CORE_NGRAMS_FNAME = "core_ngrams.txt"
    CONFIG_FNAME = "config.json"

    def __init__(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        mlm_prob: float = 0.15,
        whole_ngram_masking: bool = True,
        mlm_data_symlink: str | None = None,
        verbose: bool = True,
    ):
        """
        初始化MLM数据集

        参数:
            tokens: 分词器生成的token ID
            attn_mask: 分词器生成的注意力掩码
            tokenizer: transformers分词器
            ngram_encoder: N-gram编码器
            core_ngrams: 核心N-gram集合，这些N-gram不会被掩码
            mlm_prob: 掩码概率，默认为0.15
            whole_ngram_masking: 是否使用全N-gram掩码，默认为True
            mlm_data_symlink: 原始MLM数据在磁盘上的路径，用于创建符号链接节省空间
            verbose: 是否输出详细日志
        """
        super().__init__()
        self.ngram_encoder = ngram_encoder
        self.tokenizer = tokenizer

        # 分词器相关
        self.PAD: int = tokenizer.convert_tokens_to_ids("[PAD]")
        self.token_masker = TokenMasker(
            core_ngrams,
            cls_token=tokenizer.convert_tokens_to_ids("[CLS]"),
            sep_token=tokenizer.convert_tokens_to_ids("[SEP]"),
            pad_token=self.PAD,
            mask_token=tokenizer.convert_tokens_to_ids("[MASK]"),
            whole_ngram_masking=whole_ngram_masking,
            verbose=verbose,
        )

        self.mlm_prob = mlm_prob
        self.tokens = tokens
        self.core_ngrams = core_ngrams  # 修复token_masker引入的破坏性更改
        self.whole_ngram_masking = whole_ngram_masking
        self.attn_mask = attn_mask
        self.mlm_data_symlink = mlm_data_symlink

    @property
    def ngram_vocab_size(self):
        """获取N-gram词汇表大小"""
        return self.ngram_encoder.get_vocab_size()

    @classmethod
    def from_raw_data(
        cls,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        whole_ngram_masking: bool = False,
        mlm_prob: float = 0.15,
    ):
        """
        从原始文本数据创建MLM数据集

        参数:
            data_path: 原始数据文件路径
            tokenizer: transformers分词器
            ngram_encoder: N-gram编码器
            core_ngrams: 核心N-gram集合
            whole_ngram_masking: 是否使用全N-gram掩码
            mlm_prob: 掩码概率

        返回:
            MLM数据集实例
        """
        logger.info(f"正在从原始文本数据创建MLM数据集: {data_path}")
        with open(data_path, "r") as f:
            texts = f.readlines()

        logger.info(f"读取了{len(texts)}行文本，开始分词处理...")
        logger.info(f"分词器最大长度: {tokenizer.model_max_length}")
        
        outputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
        )
        tokens = outputs["input_ids"]
        attn_mask = outputs["attention_mask"]
        
        logger.info(f"分词完成，生成了{tokens.shape[0]}个序列，最大长度为{tokens.shape[1]}")
        
        return cls(
            tokens=tokens,
            attn_mask=attn_mask,
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            whole_ngram_masking=whole_ngram_masking,
            mlm_prob=mlm_prob,
        )

    @classmethod
    def from_tokenized_data(
        cls,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        whole_ngram_masking: bool = False,
        mlm_prob: float = 0.15,
    ):
        """
        从已分词数据创建MLM数据集

        参数:
            data_dir: 已分词数据目录
            tokenizer: transformers分词器
            ngram_encoder: N-gram编码器
            core_ngrams: 核心N-gram集合
            whole_ngram_masking: 是否使用全N-gram掩码
            mlm_prob: 掩码概率

        返回:
            MLM数据集实例
        """
        logger.info(f"正在从已分词数据创建MLM数据集: {data_dir}")
        data: MlmDataSaved = torch.load(data_dir)
        token_ids = data["input_ids"]
        attn_mask = data["attention_mask"]
        
        logger.info(f"加载了{token_ids.shape[0]}个序列，最大长度为{token_ids.shape[1]}")

        return cls(
            tokens=token_ids,
            attn_mask=attn_mask,
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            whole_ngram_masking=whole_ngram_masking,
            mlm_prob=mlm_prob,
            mlm_data_symlink=data_dir,
        )

    def save(self, save_dir: str):
        """
        保存数据集到磁盘

        参数:
            save_dir: 保存目录
        """
        logger.info(f"正在保存MLM数据集到{save_dir}...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.warning(f"目录{save_dir}不存在，已创建。")

        # 构建文件路径
        ngram_encoder_path = os.path.join(save_dir, self.NGRAM_ENCODER_FNAME)
        data_path = os.path.join(save_dir, self.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, self.CORE_NGRAMS_FNAME)
        tokenizer_path = os.path.join(save_dir, self.TOKENIZER_DIR)
        config_path = os.path.join(save_dir, self.CONFIG_FNAME)

        # 保存各组件
        logger.info(f"保存N-gram编码器到{ngram_encoder_path}...")
        self.ngram_encoder.save(ngram_encoder_path)
        
        logger.info(f"保存分词器到{tokenizer_path}...")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"保存{len(self.core_ngrams)}个核心N-gram到{core_ngram_path}...")
        _save_core_ngrams(core_ngram_path, core_ngrams=self.core_ngrams)

        # 保存数据或创建符号链接
        if self.mlm_data_symlink is None:
            logger.info(f"保存数据到{data_path}...")
            mlm_data_hash_val = None
            torch.save(
                {
                    "input_ids": self.tokens,
                    "attention_mask": self.attn_mask,
                },
                data_path,
            )
            logger.info(f"数据保存完成，共{self.tokens.shape[0]}个序列")
        else:
            # 使用符号链接节省空间
            logger.info(f"使用符号链接节省空间，正在计算{self.mlm_data_symlink}的MD5值...")
            mlm_data_hash_val = hash_file_md5(self.mlm_data_symlink)
            logger.info(f"MD5计算完成: {mlm_data_hash_val}")
            
            # 检查目标文件是否已存在，如果存在则先删除
            if os.path.exists(data_path) or os.path.islink(data_path):
                logger.warning(f"目标文件{data_path}已存在，正在删除...")
                os.remove(data_path)
            
            # 创建符号链接
            logger.info(f"创建符号链接: {self.mlm_data_symlink} -> {data_path}")
            os.symlink(self.mlm_data_symlink, dst=data_path)
            logger.info("符号链接创建完成")

        # 保存配置
        logger.info(f"保存配置到{config_path}...")
        data_cfg: MlmDataConfig = {
            "whole_ngram_masking": self.whole_ngram_masking,
            "mlm_prob": self.mlm_prob,
            "mlm_data_symlink": self.mlm_data_symlink,
            "mlm_data_hash_val": mlm_data_hash_val,
        }
        with open(config_path, "w") as f:
            json.dump(data_cfg, f, indent=2)
        
        logger.info(f"MLM数据集保存完成: {save_dir}")

    @classmethod
    def from_data_path(
        cls,
        data_path: str,
        ngram_encoder: NgramEncoder,
        data_config_path: str,
        tokenizer=None,
        max_ngrams=20,
        check_hash: bool = False,
    ):
        """从数据的保存目录加载MLM数据集。

        Args:
            data_path: 保存目录路径
            ngram_encoder: ngram编码器
            data_config_path: 数据集配置文件，是一个json文件，字典内容为MlmDataConfig
            tokenizer: 分词器，如果为None则从保存目录加载
            max_ngrams: 每个序列最多匹配的N-gram数量
            check_hash: 如果data_path是软连接，那么检查软链接的目标的哈希值，判断原文件是否被修改

        Returns:
            MLMDataset实例

        示例：
            >>> mlm_dataset = MlmDataset.from_data_path(
            >>>     data_path,
            >>>     ngram_encoder,
            >>>     data_config_path,
            >>>     tokenizer,
            >>>     max_ngrams,
            >>>     check_hash
            >>> )
            
            >>> mlm_dataset[1]
            
            >>> {
                "input_ids": xxx,
                "ngram_input_ids": xxx,
                "attention_mask": xxx,
                "ngram_attention_mask": xxx,
                "ngram_position_matrix": xxx,
                "labels": xxx
            }
            
        """
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

    @classmethod
    def from_dir(
        cls,
        save_dir,
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
        
        # 构建文件路径
        ngram_encoder_path = os.path.join(save_dir, cls.NGRAM_ENCODER_FNAME)
        tokenizer_path = os.path.join(save_dir, cls.TOKENIZER_DIR)
        data_path = os.path.join(save_dir, cls.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, cls.CORE_NGRAMS_FNAME)
        data_config_path = os.path.join(save_dir, cls.CONFIG_FNAME)

        # 加载N-gram编码器
        logger.info(f"加载N-gram编码器: {ngram_encoder_path}")
        ngram_encoder = NgramEncoder.from_file(ngram_encoder_path)
        logger.info(f"N-gram词汇表大小: {ngram_encoder.get_vocab_size()}")
        logger.info(f"N-gram长度范围: {ngram_encoder._min_ngram_len}-{ngram_encoder._max_ngram_len}")
        
        # 设置最大N-gram匹配数
        logger.info(f"设置最大N-gram匹配数: {max_ngrams}")
        ngram_encoder.set_max_ngram_match(max_ngrams)

        # 加载分词器
        if tokenizer is None:
            logger.info(f"从{tokenizer_path}加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"分词器词汇表大小: {len(tokenizer)}")
        else:
            logger.info("使用提供的分词器")

        # 加载数据
        logger.info(f"从{data_path}加载数据...")
        data = torch.load(data_path, weights_only=True)
        logger.info(f"加载了{len(data['input_ids'])}个序列，形状为{data['input_ids'].shape}")

        # 加载核心N-gram
        logger.info(f"从{core_ngram_path}加载核心N-gram...")
        core_ngrams = _load_core_ngrams(core_ngram_path)
        logger.info(f"加载了{len(core_ngrams)}个核心N-gram")

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
        dataset = cls(
            tokens=data["input_ids"],
            attn_mask=data["attention_mask"],
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            whole_ngram_masking=config.get("whole_ngram_masking", False),
            mlm_prob=config["mlm_prob"],
            mlm_data_symlink=config["mlm_data_symlink"],
        )
        logger.info("MLM数据集加载完成!")

        return dataset

    def __len__(self):
        """返回数据集大小"""
        return self.tokens.size(0)

    def __getitem__(self, index) -> MlmData:
        """
        获取数据集中的一个样本

        参数:
            index: 样本索引

        返回:
            包含所有模型输入所需字段的字典
        """
        # 运行时进行token掩码
        input_ids_, labels_ = self.token_masker.create_mlm_predictions(
            token_seq=self.tokens[index],
            vocab_list=list(self.tokenizer.get_vocab().values()),
            mlm_prob=self.mlm_prob,
            ngram_encoder=self.ngram_encoder,
        )

        # 提取N-gram特征
        ngram_encoder_outputs = self.ngram_encoder.encode(
            input_ids_,
            pad_token_id=self.PAD,
        )

        # 返回所有模型输入所需的字段
        return {
            "input_ids": input_ids_,
            "labels": labels_,
            "attention_mask": self.attn_mask[index],
            "ngram_attention_mask": ngram_encoder_outputs["ngram_attention_mask"],
            "ngram_input_ids": ngram_encoder_outputs["ngram_ids"],
            "ngram_position_matrix": ngram_encoder_outputs["ngram_position_matrix"],
        }


class MlmDatasetV2(Dataset):
    def __init__(
        self,
        sequences: list[str],
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        mlm_prob: float = 0.15,
        whole_ngram_masking: bool = True,
    ):
        super().__init__()

        if whole_ngram_masking:
            raise NotImplementedError("全N-gram掩码尚未实现")

        self.tokenizer = tokenizer
        self.ngram_encoder = ngram_encoder
        self.core_ngrams = core_ngrams
        self.mlm_prob = mlm_prob
        self.sequences = sequences
        self.token_masker = TokenMasker(
            core_ngrams,
            cls_token=tokenizer.cls_token_id,
            sep_token=tokenizer.sep_token_id,
            pad_token=tokenizer.pad_token_id,
            mask_token=tokenizer.mask_token_id,
            whole_ngram_masking=whole_ngram_masking,
        )

    @classmethod
    def from_raw_data_file(
        cls,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        whole_ngram_masking: bool = False,
        mlm_prob: float = 0.15,
    ):
        """从原始数据文件创建MLM数据集。

        Args:
            data_path (str): _description_
            tokenizer (PreTrainedTokenizer): _description_
            ngram_encoder (NgramEncoder): _description_
            core_ngrams (set[tuple[int, ...]]): _description_
            whole_ngram_masking (bool, optional): _description_. Defaults to False.
            mlm_prob (float, optional): _description_. Defaults to 0.15.

        Returns:
            _type_: _description_
        """
        with open(data_path, "r") as f:
            sequences = f.read().split("\n")
        # 移除空字符串
        sequences = [seq for seq in sequences if seq]

        return cls(
            sequences=sequences,
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            whole_ngram_masking=whole_ngram_masking,
            mlm_prob=mlm_prob,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        tokens = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True)
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()

        input_ids, labels = self.token_masker.create_mlm_predictions(
            token_seq=input_ids,
            mlm_prob=self.mlm_prob,
            vocab_list=list(self.tokenizer.get_vocab().values()),
            ngram_encoder=self.ngram_encoder,
        )

        ngram_encoder_outputs = self.ngram_encoder.encode(
            input_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "ngram_attention_mask": ngram_encoder_outputs["ngram_attention_mask"],
            "ngram_input_ids": ngram_encoder_outputs["ngram_ids"],
            "ngram_position_matrix": ngram_encoder_outputs["ngram_position_matrix"],
        }

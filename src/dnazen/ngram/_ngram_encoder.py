"""
N-gram编码与统计分析工具。

本模块提供了N-gram的编码、匹配和统计分析功能，支持基于PMI和频率的N-gram提取方法。
"""

from typing import TypedDict, Literal
import json
import warnings
from copy import deepcopy
import pathlib

import torch
from transformers import AutoTokenizer

from ._find_ngram import (
    find_ngrams_by_pmi,
    find_ngrams_by_freq,
    PmiNgramFinderConfig,
    FreqNgramFinderConfig,
)


class _NgramEncoderConfig(TypedDict):
    """N-gram编码器配置的类型定义"""
    vocab: dict[str, int]           # N-gram词汇表，键为N-gram字符串表示，值为索引ID
    min_ngram_len: int              # 最小N-gram长度
    max_ngram_len: int              # 最大N-gram长度
    max_ngrams: int                 # 每个序列最多匹配的N-gram数量


class EncodedNgram(TypedDict):
    """编码后的N-gram结果的类型定义"""
    ngram_ids: torch.Tensor         # N-gram的ID张量
    ngram_attention_mask: torch.Tensor  # 注意力掩码，表示有效的N-gram位置
    ngram_position_matrix: torch.Tensor  # 位置矩阵，映射N-gram到原始token位置


class NgramEncoder:
    """
    N-gram编码器，用于N-gram的提取、编码和统计分析。
    
    该类支持从预定义词典初始化，或从原始数据训练N-gram词典。
    提供了N-gram匹配、编码和统计功能。
    """
    
    def __init__(
        self,
        vocab_dict: dict[tuple[int, ...], int],
        min_ngram_len: int,
        max_ngram_len: int,
        max_ngrams: int,
    ):
        """
        初始化N-gram编码器。
        
        Args:
            vocab_dict: N-gram词典，键为token元组，值为对应的ID
            min_ngram_len: 最小N-gram长度
            max_ngram_len: 最大N-gram长度
            max_ngrams: 每个序列最多匹配的N-gram数量
        """
        # N-gram词典，键为token元组，值为ID
        self._vocab: dict[tuple[int, ...], int] = vocab_dict
        # ID到N-gram的反向映射
        self._id2ngrams = {}
        for k, v in vocab_dict.items():
            self._id2ngrams[v] = k

        self._min_ngram_len = min_ngram_len
        self._max_ngram_len = max_ngram_len
        self._max_ngrams = max_ngrams

    def _get_ngram_id(self, tokens: tuple[int, ...]) -> int | None:
        """
        获取给定token序列对应的N-gram ID。
        
        Args:
            tokens: token元组
            
        Returns:
            对应的N-gram ID，如果不在词典中则返回None
        """
        return self._vocab.get(tokens, None)

    def get_matched_ngrams(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """
        从token序列中获取匹配的N-gram及其位置。
        
        步骤:
        1. 验证输入格式并移除填充token
        2. 遍历不同长度的N-gram窗口
        3. 查找每个窗口是否有匹配的N-gram
        4. 生成匹配的N-gram及其起始位置
        
        Args:
            token_ids: 输入token序列
            pad_token_id: 填充token的ID，默认为3
            
        Yields:
            (ngram, position)元组，表示匹配的N-gram及其起始位置
        """
        assert token_ids.dim() == 1, f"token_ids应为一维张量，但得到的维度为{token_ids.dim()}"

        token_len = token_ids.shape[0]
        # 移除填充token
        token_ids = token_ids[token_ids != pad_token_id]
        token_ids_list = token_ids.tolist()

        # 遍历所有可能的N-gram长度
        for ngram_len in range(self._min_ngram_len, self._max_ngram_len + 1):
            # 对每个长度，遍历所有可能的起始位置
            for q in range(token_len - ngram_len + 1):
                # 提取当前窗口的token序列
                ngram: tuple[int, ...] = tuple(token_ids_list[q : q + ngram_len])
                # 查找该序列是否在N-gram词典中
                ngram_id = self._get_ngram_id(ngram)
                if ngram_id is None:
                    continue
                # 找到匹配项，生成(N-gram, 位置)元组
                yield ngram, q

    def get_num_matches(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """
        计算token序列中匹配的N-gram数量。
        
        Args:
            token_ids: 输入token序列
            pad_token_id: 填充token的ID，默认为3
            
        Returns:
            匹配的N-gram数量
        """
        return len([0 for _, _ in self.get_matched_ngrams(token_ids, pad_token_id)])

    def get_total_ngram_len(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """
        计算token序列中所有匹配N-gram的总长度。
        
        Args:
            token_ids: 输入token序列
            pad_token_id: 填充token的ID，默认为3
            
        Returns:
            所有匹配N-gram的总长度
        """
        total_ngram_len = 0
        for ngram, _ in self.get_matched_ngrams(token_ids, pad_token_id):
            total_ngram_len += len(ngram)
        return total_ngram_len

    def encode(self, token_ids: torch.Tensor, pad_token_id: int = 3) -> EncodedNgram:
        """
        将token序列编码为N-gram ID和位置矩阵。
        
        步骤:
        1. 验证输入格式并移除填充token
        2. 初始化返回结果的张量
        3. 遍历不同长度的N-gram窗口
        4. 对每个匹配的N-gram，记录其ID和位置信息
        5. 达到最大N-gram数量时提前返回
        
        Args:
            token_ids: 输入token序列
            pad_token_id: 填充token的ID，默认为3
            
        Returns:
            包含以下字段的字典:
            - ngram_ids: N-gram ID张量
            - ngram_attention_mask: 注意力掩码张量
            - ngram_position_matrix: 位置矩阵，表示每个N-gram覆盖的token位置
        """
        assert token_ids.dim() == 1

        token_len = token_ids.shape[0]
        # 移除填充token
        token_ids = token_ids[token_ids != pad_token_id]
        token_ids_list = token_ids.tolist()

        # 当前N-gram ID在结果中的索引
        _cur_ngram_id_idx = 0
        # 初始化返回结果
        ret_val: EncodedNgram = {
            "ngram_ids": torch.zeros(self._max_ngrams, dtype=torch.int),
            "ngram_attention_mask": torch.zeros(self._max_ngrams, dtype=torch.int),
            "ngram_position_matrix": torch.zeros(token_len, self._max_ngrams, dtype=torch.bool),
        }

        # 遍历所有可能的N-gram长度
        for ngram_len in range(self._min_ngram_len, self._max_ngram_len + 1):
            # 对每个长度，遍历所有可能的起始位置
            for q in range(token_len - ngram_len + 1):
                # 提取当前窗口的token序列
                ngram: tuple[int, ...] = tuple(token_ids_list[q : q + ngram_len])
                # 查找该序列是否在N-gram词典中
                ngram_id = self._get_ngram_id(ngram)
                if ngram_id is None:
                    continue
                
                # 找到匹配项，记录N-gram ID和位置信息
                ret_val["ngram_ids"][_cur_ngram_id_idx] = ngram_id
                ret_val["ngram_attention_mask"][_cur_ngram_id_idx] = 1
                # 标记该N-gram覆盖的所有token位置
                ret_val["ngram_position_matrix"][q : q + ngram_len, _cur_ngram_id_idx] = 1
                _cur_ngram_id_idx += 1
                
                # 达到最大N-gram数量时提前返回
                if _cur_ngram_id_idx >= self._max_ngrams:
                    return ret_val
        return ret_val

    def set_max_ngram_match(self, max_ngrams: int):
        """
        设置每个序列最多匹配的N-gram数量。
        
        Args:
            max_ngrams: 新的最大N-gram数量
        """
        self._max_ngrams = max_ngrams

    @classmethod
    def from_list(cls, ngram_list: list[list[int]], max_ngrams: int = 20):
        """
        从N-gram列表创建编码器实例。
        
        步骤:
        1. 遍历N-gram列表，构建词典
        2. 计算最小和最大N-gram长度
        3. 使用构建的词典和参数创建编码器实例
        
        Args:
            ngram_list: N-gram列表，每个N-gram是一个整数列表
            max_ngrams: 每个序列最多匹配的N-gram数量，默认为20
            
        Returns:
            新创建的N-gram编码器实例
        """
        vocab_dict = {}
        min_ngram_len = 100
        max_ngram_len = 0
        
        # 遍历N-gram列表，构建词典
        for idx, vocab in enumerate(ngram_list):
            k_ = tuple(vocab)
            min_ngram_len = min(len(k_), min_ngram_len)
            max_ngram_len = max(len(k_), max_ngram_len)
            vocab_dict[k_] = idx

        return cls(
            vocab_dict=vocab_dict,
            min_ngram_len=min_ngram_len,
            max_ngram_len=max_ngram_len,
            max_ngrams=max_ngrams,
        )

    @classmethod
    def from_file(cls, path):
        """
        从文件加载N-gram编码器。
        
        步骤:
        1. 从JSON文件加载配置
        2. 转换N-gram字符串表示为元组
        3. 使用加载的配置创建编码器实例
        
        Args:
            path: 配置文件路径
            
        Returns:
            加载的N-gram编码器实例
        """
        with open(path, "r") as f:
            config: _NgramEncoderConfig = json.load(f)

        # 将N-gram字符串表示转换为元组
        vocab_dict = {}
        for k, v in config["vocab"].items():
            k_ = tuple(int(s) for s in k.split(":"))
            vocab_dict[k_] = v

        return cls(
            vocab_dict=vocab_dict,
            min_ngram_len=config["min_ngram_len"],
            max_ngram_len=config["max_ngram_len"],
            max_ngrams=config["max_ngrams"],
        )

    def get_new_ngram_encoder_from_data(self, tokenized_data: list[list[int]], freq_threshold: int = 0):
        """
        基于给定数据集创建新的N-gram编码器，仅包含在数据集中能够匹配的N-gram。
        
        步骤:
        1. 在数据集中识别所有能够匹配的N-gram及其频率
        2. 根据频率阈值筛选N-gram
        3. 构建新的词典并计算N-gram长度范围
        4. 创建新的编码器实例
        
        Args:
            tokenized_data: 已分词的数据列表
            freq_threshold: 频率阈值，只保留频率大于等于该值的N-gram，默认为0
            
        Returns:
            新创建的N-gram编码器实例
        """
        # 统计数据集中匹配的N-gram及其频率
        matched_ngram_dict = {}
        for d in tokenized_data:
            for matched_ngram, _ in self.get_matched_ngrams(torch.tensor(d)):
                if matched_ngram_dict.get(matched_ngram) is None:
                    matched_ngram_dict[matched_ngram] = 1
                else:
                    matched_ngram_dict[matched_ngram] += 1

        # 根据频率排序并筛选N-gram
        sorted_ngram_dict = {
            k: v
            for k, v in sorted(matched_ngram_dict.items(), key=lambda item: item[1], reverse=True)
            if v >= freq_threshold
        }

        # 构建新的词典
        vocab = {}
        min_ngram_len = self._max_ngram_len + 1
        max_ngram_len = 0
        for idx, k in enumerate(sorted_ngram_dict.keys()):
            vocab[k] = idx
            min_ngram_len = min(min_ngram_len, len(k))
            max_ngram_len = max(max_ngram_len, len(k))

        # 创建新的编码器实例
        return NgramEncoder(
            vocab_dict=vocab,
            min_ngram_len=min_ngram_len,
            max_ngram_len=max_ngram_len,
            max_ngrams=self._max_ngrams,
        )

    def train(
        self,
        tokens: list[list[int]],
        min_pmi: float | None = None,
        min_token_count: int | None = None,
        secondary_filter: bool = False,
        min_ngram_freq: int = 5,
        num_workers: int = 64,
        returns_freq: bool = False,
        method: Literal["pmi", "freq"] = "pmi",
    ):
        """
        从原始token数据训练N-gram编码器。
        
        支持两种训练方法:
        1. PMI (点互信息): 基于词组成分之间的互信息提取N-gram
        2. 频率: 基于N-gram在语料中的出现频率提取
        
        步骤:
        1. 根据选择的方法和参数配置N-gram查找器
        2. 从token数据中提取N-gram
        3. 将提取的N-gram转换为编码器的词典格式
        4. 更新编码器的词典和ID到N-gram的映射
        
        Args:
            tokens: 分词后的token列表，每个元素是一个整数列表
            min_pmi: PMI方法的最小PMI阈值
            min_token_count: PMI方法的最小token计数阈值
            secondary_filter: 频率方法是否使用二次过滤
            min_ngram_freq: 最小N-gram频率阈值
            num_workers: 训练使用的工作线程数，默认为64
            returns_freq: 是否返回N-gram频率信息，默认为False
            method: 训练方法，可选 "pmi" 或 "freq"，默认为 "pmi"
            
        Returns:
            如果returns_freq为True，返回N-gram频率字典；否则返回None
        """
        if self._vocab != {}:
            warnings.warn("词典非空，训练后将被覆盖。")

        # 根据选择的方法配置N-gram查找器
        if method == "pmi":
            if min_pmi is None:
                raise ValueError("使用PMI方法时必须设置min_pmi")
            if min_token_count is None:
                raise ValueError("使用PMI方法时必须设置min_token_count")

            # 配置PMI N-gram查找器
            ngram_finder_config = PmiNgramFinderConfig()
            ngram_finder_config.min_pmi = min_pmi
            ngram_finder_config.max_ngram_len = self._max_ngram_len
            ngram_finder_config.min_ngram_len = self._min_ngram_len
            ngram_finder_config.min_ngram_freq = min_ngram_freq
            ngram_finder_config.min_token_count = min_token_count
            ngram_finder_config.num_workers = num_workers

            # 使用PMI方法提取N-gram
            self._vocab = find_ngrams_by_pmi(ngram_finder_config, tokens=tokens)
        elif method == "freq":
            if min_pmi is not None:
                print("[警告] 使用频率方法时，min_pmi参数不会被使用。")
            if min_token_count is not None:
                print("[警告] 使用频率方法时，min_token_count参数不会被使用。")

            # 配置频率N-gram查找器
            ngram_finder_config = FreqNgramFinderConfig()
            ngram_finder_config.min_freq = min_ngram_freq
            ngram_finder_config.max_ngram_len = self._max_ngram_len
            ngram_finder_config.min_ngram_len = self._min_ngram_len
            ngram_finder_config.secondary_filter = secondary_filter
            ngram_finder_config.num_workers = num_workers

            # 使用频率方法提取N-gram
            self._vocab = find_ngrams_by_freq(ngram_finder_config, tokens=tokens)
        else:
            raise NotImplementedError(f"不支持的方法 {method}。")

        # 复制词典用于返回频率信息(如果需要)
        if returns_freq:
            vocab_freq = deepcopy(self._vocab)
        else:
            vocab_freq = None

        # 将N-gram频率字典转换为ID映射
        for idx, (k, v) in enumerate(self._vocab.items()):
            self._vocab[k] = idx

        # 更新ID到N-gram的反向映射
        self._id2ngrams = {}
        for k, v in self._vocab.items():
            self._id2ngrams[v] = k

        return vocab_freq

    def train_from_file(
        self,
        fname: str,
        min_pmi: float,
        min_token_count: int,
        min_ngram_freq: int,
        num_workers: int = 64,
        returns_freq: bool = False,
    ):
        """
        从文件直接训练N-gram编码器（仅支持PMI方法）。
        
        此方法适用于处理大型文件，无需将所有数据加载到内存中。
        
        步骤:
        1. 配置PMI N-gram查找器
        2. 直接从文件读取数据并提取N-gram
        3. 将提取的N-gram转换为编码器的词典格式
        4. 更新编码器的词典和ID到N-gram的映射
        
        Args:
            fname: 输入文件路径
            min_pmi: 最小PMI阈值
            min_token_count: 最小token计数阈值
            min_ngram_freq: 最小N-gram频率阈值
            num_workers: 训练使用的工作线程数，默认为64
            returns_freq: 是否返回N-gram频率信息，默认为False
            
        Returns:
            如果returns_freq为True，返回N-gram频率字典；否则返回None
        """
        # 配置PMI N-gram查找器
        ngram_finder_config = PmiNgramFinderConfig()
        ngram_finder_config.min_pmi = min_pmi
        ngram_finder_config.max_ngram_len = self._max_ngram_len
        ngram_finder_config.min_ngram_len = self._min_ngram_len
        ngram_finder_config.min_ngram_freq = min_ngram_freq
        ngram_finder_config.min_token_count = min_token_count
        ngram_finder_config.num_workers = num_workers

        if self._vocab != {}:
            warnings.warn("词典非空，训练后将被覆盖。")

        import _ngram

        # 初始化查找器并从文件提取N-gram
        finder = _ngram.PmiNgramFinder(ngram_finder_config)
        finder.find_ngrams_from_file(fname)
        ngrams: list[list[int]] = finder.get_ngram_list([])

        # 转换结果为字典格式
        ngram_dict = {}
        for ngram in ngrams:
            freq = ngram.pop()
            ngram_dict[tuple(ngram)] = freq

        self._vocab = ngram_dict
        
        # 复制词典用于返回频率信息(如果需要)
        if returns_freq:
            vocab_freq = deepcopy(self._vocab)
        else:
            vocab_freq = None

        # 将N-gram频率字典转换为ID映射
        for idx, (k, v) in enumerate(self._vocab.items()):
            self._vocab[k] = idx

        # 更新ID到N-gram的反向映射
        self._id2ngrams = {}
        for k, v in self._vocab.items():
            self._id2ngrams[v] = k

        return vocab_freq

    def save(self, path, pretty=True):
        """
        保存N-gram编码器配置到文件。
        
        步骤:
        1. 将N-gram元组转换为字符串表示
        2. 构建配置字典
        3. 确保目录存在
        4. 将配置保存为JSON文件
        
        Args:
            path: 保存路径
            pretty: 是否美化JSON输出，默认为True
        """
        # 将N-gram元组转换为字符串表示
        vocab_dict = {}
        for k, v in self._vocab.items():
            k_ = ":".join(str(i) for i in k)
            vocab_dict[k_] = v

        # 构建配置字典
        config: _NgramEncoderConfig = {
            "vocab": vocab_dict,
            "max_ngram_len": self._max_ngram_len,
            "min_ngram_len": self._min_ngram_len,
            "max_ngrams": self._max_ngrams,
        }
        
        # 设置JSON缩进
        indent = 2 if pretty else None
        
        # 确保目录存在
        _dir = pathlib.Path(path).parent
        _dir.mkdir(parents=True, exist_ok=True)

        # 保存配置为JSON文件
        with open(path, "w") as f:
            json.dump(config, f, indent=indent)

    def get_vocab(self):
        """
        获取N-gram词典。
        
        Returns:
            N-gram词典，键为token元组，值为ID
        """
        return self._vocab

    def get_vocab_size(self):
        """
        获取N-gram词典大小。
        
        Returns:
            词典中N-gram的数量
        """
        return len(self._vocab)

    def get_id(self) -> int:
        """
        获取N-gram编码器的唯一标识。
        
        标识仅基于N-gram词典计算。
        
        Returns:
            编码器的哈希值作为唯一标识
        """
        return hash(frozenset(self._vocab.items()))

# 在文件末尾添加测试代码
if __name__ == "__main__":
    """
    N-gram编码器测试代码
    
    本测试展示了如何：
    1. 创建N-gram编码器
    2. 使用PMI和频率方法训练N-gram
    3. 进行N-gram匹配和编码
    4. 保存和加载N-gram编码器
    """
    import tempfile
    import os
    
    print("===== N-gram编码器测试 =====")
    
    # 1. 准备测试数据
    print("\n1. 准备测试数据")
    # 创建一些示例DNA序列
    test_data_raw = [
        "ATGACATGCATGCGCATGCATCGCATTGCATGCATGCACATGCATGCATG",
        "AATGCATGCATGCGCATGCATGCATGGTGCATGCATGCACATGCATGCATG",
        "ATTGCATGCATGCGCATGCATGCATTGCATGCATGCACATGCATGCATG",
        "AGGGTGCATGCATGCGCATGCATGCATTGCATGCACCTGCACATGCATGCATG"
    ]
    
    # 加载DNABERT-2 tokenizer
    print("加载DNABERT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    
    # 将DNA序列转换为token ID
    test_data = []
    for seq in test_data_raw:
        tokens = tokenizer.encode(seq, add_special_tokens=False)
        test_data.append(tokens)
    
    print(f"生成了{len(test_data)}个测试序列")
    print(f"第一个序列原始文本: {test_data_raw[0]}")
    print(f"第一个序列token IDs: {test_data[0]}")
    
    # 2. 初始化N-gram编码器
    print("\n2. 初始化N-gram编码器")
    encoder = NgramEncoder(
        vocab_dict={},  # 初始化为空词典，稍后训练
        min_ngram_len=2,  # 最小n-gram长度
        max_ngram_len=5,  # 最大n-gram长度
        max_ngrams=10,    # 每个序列最多匹配的n-gram数量
    )
    print(f"N-gram长度范围: {encoder._min_ngram_len}-{encoder._max_ngram_len}")
    
    # 3. 使用PMI方法训练N-gram
    print("\n3. 使用PMI方法训练N-gram")
    print("训练参数:")
    print("  - min_pmi = 1.0")
    print("  - min_token_count = 2")
    print("  - min_ngram_freq = 2")
    
    ngram_freqs = encoder.train(
        tokens=test_data,
        min_pmi=1.0,           # PMI阈值
        min_token_count=2,     # 最小token频率
        min_ngram_freq=2,      # 最小n-gram频率
        num_workers=1,         # 单线程处理
        returns_freq=True,     # 返回频率信息
        method="pmi",          # 使用PMI方法
    )
    
    print(f"PMI方法提取的N-gram数量: {encoder.get_vocab_size()}")
    if encoder.get_vocab_size() > 0:
        print("前5个N-gram及其ID:")
        for i, (ngram, id_) in enumerate(list(encoder.get_vocab().items())[:5]):
            print(f"  {i+1}. {ngram} -> ID {id_}, 频率 {ngram_freqs.get(ngram, 'N/A')}")
    
    # 4. 使用频率方法训练新的N-gram编码器
    print("\n4. 使用频率方法训练新的N-gram编码器")
    encoder_freq = NgramEncoder(
        vocab_dict={}, 
        min_ngram_len=2, 
        max_ngram_len=5,
        max_ngrams=10,
    )
    
    ngram_freqs = encoder_freq.train(
        tokens=test_data,
        min_ngram_freq=2,      # 最小n-gram频率
        num_workers=1,         
        returns_freq=True,
        method="freq",         # 使用频率方法
    )
    
    print(f"频率方法提取的N-gram数量: {encoder_freq.get_vocab_size()}")
    if encoder_freq.get_vocab_size() > 0:
        print("前5个N-gram及其ID:")
        for i, (ngram, id_) in enumerate(list(encoder_freq.get_vocab().items())[:5]):
            print(f"  {i+1}. {ngram} -> ID {id_}, 频率 {ngram_freqs.get(ngram, 'N/A')}")
    
    # 5. 测试N-gram匹配
    print("\n5. 测试N-gram匹配")
    # 使用训练好的编码器进行后续测试（选择元素较多的那个）
    if encoder.get_vocab_size() >= encoder_freq.get_vocab_size():
        test_encoder = encoder
        print("使用PMI训练的编码器进行测试")
    else:
        test_encoder = encoder_freq
        print("使用频率训练的编码器进行测试")
    
    test_sequence = torch.tensor(test_data[0])
    print(f"测试序列: {test_data_raw[0]}")
    print(f"测试序列token IDs: {test_sequence.tolist()}")
    
    matches = list(test_encoder.get_matched_ngrams(test_sequence))
    print(f"匹配到的N-gram数量: {len(matches)}")
    for i, (ngram, pos) in enumerate(matches):
        # 将ngram元组中的token ID转换回原始文本（如果可能）
        try:
            ngram_text = tokenizer.decode([t for t in ngram])
            print(f"  {i+1}. 位置 {pos}: {ngram} (文本: {ngram_text})")
        except:
            print(f"  {i+1}. 位置 {pos}: {ngram}")
    
    total_len = test_encoder.get_total_ngram_len(test_sequence)
    print(f"所有匹配N-gram的总长度: {total_len}")
    
    # 6. 测试N-gram编码
    print("\n6. 测试N-gram编码")
    encoded_ngram = test_encoder.encode(test_sequence)
    
    print("编码结果:")
    print(f"  N-gram IDs: {encoded_ngram['ngram_ids'][:len(matches)].tolist()}")
    print(f"  注意力掩码: {encoded_ngram['ngram_attention_mask'][:len(matches)].tolist()}")
    print("  位置矩阵 (形状): {shape}".format(shape=encoded_ngram['ngram_position_matrix'].shape))
    
    # 打印位置矩阵的可视化表示
    print("\n  位置矩阵可视化 (行=token位置, 列=N-gram索引):")
    for i in range(min(test_sequence.shape[0], 10)):  # 最多显示10行
        row = encoded_ngram['ngram_position_matrix'][i, :len(matches)].tolist()
        print(f"  位置 {i}: {''.join(['1' if x else '0' for x in row])}")
    
    # 7. 保存和加载模型
    print("\n7. 保存和加载模型")
    # 创建临时文件保存模型
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    print(f"保存模型到: {model_path}")
    test_encoder.save(model_path)
    
    print("加载保存的模型")
    loaded_encoder = NgramEncoder.from_file(model_path)
    print(f"加载的模型N-gram数量: {loaded_encoder.get_vocab_size()}")
    
    # 检查是否与原始模型相同
    print(f"加载的模型与原始模型相同: {loaded_encoder.get_id() == test_encoder.get_id()}")
    
    # 清理临时文件
    os.unlink(model_path)
    
    print("\n===== 测试完成 =====")

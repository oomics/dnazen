#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件名: n_gram.py
描述: DNA序列N-gram特征提取和编码模块
作者: DnaZen Team

该模块实现了N-gram提取和编码的完整流程：
1. 准备N-gram词典：从大型语料库中提取N-gram
2. 为每个训练实例提取N-gram：根据词典选择相关N-gram并创建匹配矩阵
3. 使用多层Transformer编码N-gram：对N-gram进行多层次表示
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional, Union

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class NGramExtractor:
    """
    N-gram提取器，用于从DNA序列中提取N-gram特征
    
    该类实现了两步N-gram提取方法：
    1. 准备N-gram词典：从大型语料库中提取N-gram
    2. 为每个训练实例提取N-gram：根据词典选择相关N-gram并创建匹配矩阵
    """
    
    def __init__(self, n_sizes: List[int] = [3, 4, 5], max_ngrams: int = 20000):
        """
        初始化N-gram提取器
        
        参数:
            n_sizes: 要提取的N-gram大小列表，例如[3,4,5]表示提取3-gram、4-gram和5-gram
            max_ngrams: 词典中保留的最大N-gram数量
        """
        self.n_sizes = n_sizes
        self.max_ngrams = max_ngrams
        self.ngram_dict = {}  # N-gram到ID的映射
        self.id_to_ngram = {}  # ID到N-gram的映射
        self.ngram_vocab_size = 0
        
        logger.info(f"初始化N-gram提取器: n_sizes={n_sizes}, max_ngrams={max_ngrams}")
    
    def build_ngram_dict(self, sequences: List[str], min_freq: int = 5) -> Dict[str, int]:
        """
        从序列集合中构建N-gram词典（步骤1）
        
        参数:
            sequences: DNA序列列表
            min_freq: 包含N-gram的最小频率阈值
            
        返回:
            构建的N-gram词典，将N-gram映射到唯一ID
        """
        start_time = time.time()
        logger.info(f"开始构建N-gram词典，序列数量: {len(sequences)}")
        
        # 计数器，用于统计所有N-gram的频率
        ngram_counter = Counter()
        
        # 从所有序列中提取N-gram并计数
        for i, seq in enumerate(sequences):
            if i % 10000 == 0 and i > 0:
                logger.info(f"已处理 {i} 条序列...")
            
            # 对每个N-gram大小进行处理
            for n in self.n_sizes:
                # 提取当前序列的所有N-gram
                for j in range(len(seq) - n + 1):
                    ngram = seq[j:j+n]
                    ngram_counter[ngram] += 1
        
        logger.info(f"共提取了 {len(ngram_counter)} 个不同的N-gram")
        
        # 根据频率筛选N-gram
        filtered_ngrams = [ngram for ngram, count in ngram_counter.items() 
                          if count >= min_freq]
        logger.info(f"频率≥{min_freq}的N-gram数量: {len(filtered_ngrams)}")
        
        # 按频率排序并限制数量
        sorted_ngrams = sorted(filtered_ngrams, 
                              key=lambda x: ngram_counter[x], 
                              reverse=True)[:self.max_ngrams]
        
        # 构建N-gram词典
        self.ngram_dict = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}
        self.id_to_ngram = {idx: ngram for ngram, idx in self.ngram_dict.items()}
        self.ngram_vocab_size = len(self.ngram_dict)
        
        logger.info(f"N-gram词典构建完成，词典大小: {self.ngram_vocab_size}")
        logger.info(f"词典构建用时: {time.time() - start_time:.2f}秒")
        
        return self.ngram_dict
    
    def save_ngram_dict(self, output_path: str):
        """
        保存N-gram词典到文件
        
        参数:
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for ngram, idx in self.ngram_dict.items():
                f.write(f"{ngram}\t{idx}\n")
        logger.info(f"N-gram词典已保存到: {output_path}")
    
    def load_ngram_dict(self, dict_path: str):
        """
        从文件加载N-gram词典
        
        参数:
            dict_path: 词典文件路径
            
        返回:
            加载的N-gram词典
        """
        self.ngram_dict = {}
        self.id_to_ngram = {}
        
        with open(dict_path, 'r') as f:
            for line in f:
                ngram, idx = line.strip().split('\t')
                idx = int(idx)
                self.ngram_dict[ngram] = idx
                self.id_to_ngram[idx] = ngram
        
        self.ngram_vocab_size = len(self.ngram_dict)
        logger.info(f"已加载N-gram词典，词典大小: {self.ngram_vocab_size}")
        return self.ngram_dict
    
    def extract_ngrams_for_sequence(self, sequence: str) -> Tuple[List[str], List[int]]:
        """
        为单个序列提取N-gram（步骤2的一部分）
        
        参数:
            sequence: 输入DNA序列
            
        返回:
            提取的N-gram列表和对应的ID列表
        """
        extracted_ngrams = []
        ngram_ids = []
        
        # 对每个N-gram大小进行处理
        for n in self.n_sizes:
            # 提取当前序列的所有N-gram
            for j in range(len(sequence) - n + 1):
                ngram = sequence[j:j+n]
                # 只保留词典中存在的N-gram
                if ngram in self.ngram_dict:
                    extracted_ngrams.append(ngram)
                    ngram_ids.append(self.ngram_dict[ngram])
        
        return extracted_ngrams, ngram_ids
    
    def create_matching_matrix(self, sequence: str, ngrams: List[str]) -> np.ndarray:
        """
        创建N-gram匹配矩阵M（步骤2）
        
        矩阵M的大小为 len(sequence) × len(ngrams)
        M[i,j] = 1 表示字符sequence[i]属于ngrams[j]
        M[i,j] = 0 表示字符sequence[i]不属于ngrams[j]
        
        参数:
            sequence: 输入DNA序列
            ngrams: 从序列中提取的N-gram列表
            
        返回:
            N-gram匹配矩阵
        """
        seq_len = len(sequence)
        ngram_len = len(ngrams)
        
        # 初始化匹配矩阵
        matching_matrix = np.zeros((seq_len, ngram_len), dtype=np.int8)
        
        # 填充匹配矩阵
        for j, ngram in enumerate(ngrams):
            n = len(ngram)
            # 找到ngram在序列中的所有位置
            for i in range(seq_len - n + 1):
                if sequence[i:i+n] == ngram:
                    # 将ngram覆盖的所有位置标记为1
                    for k in range(n):
                        matching_matrix[i+k, j] = 1
        
        return matching_matrix
    
    def process_sequence(self, sequence: str) -> Dict:
        """
        处理单个序列，提取N-gram并创建匹配矩阵
        
        参数:
            sequence: 输入DNA序列
            
        返回:
            包含提取的N-gram和匹配矩阵的字典
        """
        # 提取N-gram
        ngrams, ngram_ids = self.extract_ngrams_for_sequence(sequence)
        
        # 创建匹配矩阵
        matching_matrix = self.create_matching_matrix(sequence, ngrams)
        
        return {
            'ngrams': ngrams,
            'ngram_ids': ngram_ids,
            'matching_matrix': matching_matrix
        }


class NGramEncoder(nn.Module):
    """
    N-gram编码器，使用多层Transformer对N-gram进行编码
    
    该编码器将N-gram表示为多层次的向量，以便与BERT的不同层次匹配。
    它使用Transformer架构，通过多头自注意力机制对N-gram之间的交互进行建模。
    """
    
    def __init__(self, 
                 ngram_vocab_size: int, 
                 hidden_size: int = 768, 
                 num_hidden_layers: int = 6,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1):
        """
        初始化N-gram编码器
        
        参数:
            ngram_vocab_size: N-gram词汇表大小
            hidden_size: 隐藏层维度
            num_hidden_layers: Transformer层数
            num_attention_heads: 注意力头数量
            intermediate_size: 前馈网络中间层维度
            hidden_dropout_prob: 隐藏层dropout概率
            attention_probs_dropout_prob: 注意力概率dropout概率
        """
        super().__init__()
        
        # N-gram嵌入层
        self.ngram_embeddings = nn.Embedding(ngram_vocab_size, hidden_size)
        
        # 层归一化
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        
        # 初始化参数
        self.apply(self._init_weights)
        
        logger.info(f"初始化N-gram编码器: vocab_size={ngram_vocab_size}, "
                   f"hidden_size={hidden_size}, layers={num_hidden_layers}")
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, 
                ngram_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                output_all_encoded_layers: bool = True) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        前向传播，编码N-gram
        
        参数:
            ngram_ids: N-gram ID张量 [batch_size, num_ngrams]
            attention_mask: 注意力掩码 [batch_size, num_ngrams]
            output_all_encoded_layers: 是否输出所有编码层的结果
            
        返回:
            如果output_all_encoded_layers为True，返回所有层的输出列表
            否则，返回最后一层的输出
        """
        # 获取N-gram嵌入
        embedding_output = self.ngram_embeddings(ngram_ids)
        embedding_output = self.LayerNorm(embedding_output)
        embedding_output = self.dropout(embedding_output)
        
        # 创建注意力掩码（如果未提供）
        if attention_mask is None:
            attention_mask = torch.ones_like(ngram_ids)
        
        # 转换为PyTorch Transformer所需的掩码格式
        # 在PyTorch Transformer中，1表示需要关注的位置，0表示需要掩码的位置
        extended_attention_mask = (attention_mask == 0)
        
        # 编码N-gram
        encoded_layers = []
        hidden_states = embedding_output
        
        for i, layer_module in enumerate(self.encoder.layers):
            hidden_states = layer_module(hidden_states, src_key_padding_mask=extended_attention_mask)
            
            if output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            encoded_layers = hidden_states
        
        return encoded_layers


class DNADatasetWithNGrams(torch.utils.data.Dataset):
    """带有N-gram特征的DNA序列数据集"""
    
    def __init__(self, sequences, tokenizer, ngram_extractor, max_length=512):
        """
        初始化数据集
        
        参数:
            sequences: DNA序列列表
            tokenizer: 用于序列标记化的分词器
            ngram_extractor: N-gram提取器
            max_length: 最大序列长度
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.ngram_extractor = ngram_extractor
        self.max_length = max_length
        
    def __getitem__(self, idx):
        """获取单个数据样本"""
        seq = self.sequences[idx]
        
        # 使用tokenizer处理序列
        inputs = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 提取N-gram特征
        ngram_features = self.ngram_extractor.process_sequence(seq)
        
        # 准备MLM任务的输入和标签
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        
        # 创建MLM标签（复制输入ID）
        labels = input_ids.clone()
        
        # 将一部分token替换为[MASK]
        mask_token_id = self.tokenizer.mask_token_id
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        
        # 不掩码特殊token
        probability_matrix = torch.full(input_ids.shape, 0.15)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # 随机选择要掩码的token
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码token的损失
        
        # 80%的情况下用[MASK]替换
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id
        
        # 10%的情况下用随机token替换
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 处理匹配矩阵，确保与输入序列长度匹配
        matching_matrix = ngram_features['matching_matrix']
        if matching_matrix.shape[0] > self.max_length:
            matching_matrix = matching_matrix[:self.max_length, :]
        else:
            # 填充匹配矩阵
            padded_matrix = np.zeros((self.max_length, matching_matrix.shape[1]), dtype=np.int8)
            padded_matrix[:matching_matrix.shape[0], :] = matching_matrix
            matching_matrix = padded_matrix
        
        # 转换为张量
        matching_matrix = torch.tensor(matching_matrix, dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'ngram_ids': torch.tensor(ngram_features['ngram_ids'], dtype=torch.long),
            'matching_matrix': matching_matrix
        }
            
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)


def load_data_with_ngrams(file_path, tokenizer, ngram_extractor, max_length=512, cache_dir=None):
    """
    加载数据并提取N-gram特征
    
    参数:
        file_path: 数据文件路径
        tokenizer: 分词器
        ngram_extractor: N-gram提取器
        max_length: 最大序列长度
        cache_dir: 缓存目录
        
    返回:
        带有N-gram特征的数据集
    """
    # 生成缓存文件路径
    cache_file = None
    if cache_dir:
        import hashlib
        os.makedirs(cache_dir, exist_ok=True)
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"dna_ngram_dataset_{file_hash}.pt")
        
        # 如果缓存文件存在，直接加载
        if os.path.exists(cache_file):
            logger.info(f"从缓存加载数据: {cache_file}")
            return torch.load(cache_file)
    
    logger.info(f"从文件加载数据: {file_path}")
    
    # 读取文件内容
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):  # 跳过FASTA格式的标题行
                sequences.append(line)
    
    logger.info(f"加载了 {len(sequences)} 条DNA序列")
    
    # 如果N-gram词典为空，则构建词典
    if ngram_extractor.ngram_vocab_size == 0:
        logger.info("构建N-gram词典...")
        ngram_extractor.build_ngram_dict(sequences)
        
        # 保存N-gram词典
        if cache_dir:
            dict_path = os.path.join(cache_dir, "ngram_dict.txt")
            ngram_extractor.save_ngram_dict(dict_path)
    
    # 创建数据集
    logger.info("创建带有N-gram特征的数据集...")
    dataset = DNADatasetWithNGrams(sequences, tokenizer, ngram_extractor, max_length)
    
    # 保存到缓存
    if cache_file:
        logger.info(f"保存数据到缓存: {cache_file}")
        torch.save(dataset, cache_file)
    
    return dataset


# 如果直接运行此脚本，执行示例代码
if __name__ == "__main__":
    # 示例用法
    print("N-gram提取器和编码器示例")
    
    # 创建一些示例DNA序列
    example_sequences = [
        "ATGCATGCATGC",
        "GCATGCATGCAT",
        "TGCATGCATGCA",
        "CATGCATGCATG"
    ]
    
    # 初始化N-gram提取器
    extractor = NGramExtractor(n_sizes=[3, 4], max_ngrams=100)
    
    # 构建N-gram词典
    extractor.build_ngram_dict(example_sequences, min_freq=2)
    
    # 处理单个序列
    seq = "ATGCATGCATGC"
    result = extractor.process_sequence(seq)
    
    print(f"序列: {seq}")
    print(f"提取的N-gram: {result['ngrams']}")
    print(f"N-gram IDs: {result['ngram_ids']}")
    print(f"匹配矩阵形状: {result['matching_matrix'].shape}")
    print("匹配矩阵:")
    print(result['matching_matrix'])
    
    # 测试N-gram编码器
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用GPU进行N-gram编码")
    else:
        device = torch.device("cpu")
        print("使用CPU进行N-gram编码")
    
    # 创建N-gram编码器
    encoder = NGramEncoder(
        ngram_vocab_size=extractor.ngram_vocab_size,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12
    ).to(device)
    
    # 准备输入
    batch_size = 2
    ngram_ids = torch.tensor([result['ngram_ids'], result['ngram_ids']], dtype=torch.long).to(device)
    
    # 前向传播
    with torch.no_grad():
        encoded_layers = encoder(ngram_ids)
    
    # 打印结果
    print(f"N-gram编码器输出层数: {len(encoded_layers)}")
    for i, layer in enumerate(encoded_layers):
        print(f"第{i+1}层输出形状: {layer.shape}") 
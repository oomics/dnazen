# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""创建预训练数据集。

本脚本用于从原始文本创建用于预训练的数据集，主要功能包括：
1. 文本分词和处理
2. 生成掩码语言模型(MLM)训练样本
3. 生成下一句预测(NSP)训练样本
4. 处理n-gram特征
5. 保存为JSON格式的训练数据
"""
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
import logging
import time
import os
from random import random, randrange, randint, shuffle, choice
from ZEN import BertTokenizer, ZenNgramDict
import numpy as np
import json
import collections
from transformers import (
    AutoTokenizer,
)
from dnazen.ngram import NgramEncoder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dataset.make_tokenized_dataset import tokenize_batch_text
import torch
# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



NGRAM_DICT_NAME="ngram_list.txt"
NGRAM_ENCODER_NAME="ngram_encoder.json"

class ZenDNANgramDict(object):
    """
    Dict class to store the ngram
    """
    def __init__(self, ngram_freq_path, tokenizer, max_ngram_in_seq=128):
        """Constructs ZenNgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        
        if os.path.isdir(ngram_freq_path):
            ngram_encoder_dir = os.path.dirname(ngram_freq_path) if os.path.dirname(ngram_freq_path) else "."
            ngram_list_path = os.path.join(ngram_encoder_dir, NGRAM_DICT_NAME)
            ngram_encoder_path = os.path.join(ngram_encoder_dir, NGRAM_ENCODER_NAME)
                        
        ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)

        self.ngram_freq_path = ngram_encoder_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = ["[pad]"]
        self.ngram_to_id_dict = {"[pad]": 0}
        self.ngram_to_freq_dict = {}
        

        # 加载N-gram编码器
        try:
            encoder = NgramEncoder.from_file(ngram_encoder_path)
            logger.info(f"从文件加载N-gram编码器: {ngram_encoder_path}")
            logger.info(f"编码器包含 {len(encoder.get_vocab())} 个N-gram")
        except Exception as e:
            logger.error(f"加载N-gram编码器失败: {str(e)}")
        
        ngram_df = pd.read_csv(ngram_list_path, sep="\t")
        logger.info("loading ngram frequency file {}".format(ngram_freq_path))

        if ngram_df is not None:
        # logger.info("创建N-gram频率字典...")
            i = 0
            for _, row in ngram_df.iterrows():
                if "N-gram" in row and "频率" in row:
                    ngram = row["N-gram"]  # noqa: F841
                    freq = row["频率"]
                    #tokens = tuple(tokenizer.tokenize(ngram))
                    tokens = row["token_ids"]
                    self.ngram_to_freq_dict[ngram] = freq
                    self.id_to_ngram_list.append(tokens)
                    self.ngram_to_id_dict[tokens] = i + 1

            logger.info(
                f"创建了 {len(self.id_to_ngram_list)} 个token ID映射（总N-gram数: {len(self.ngram_to_freq_dict)}）"
            )
        else:
            logger.warning("没有N-gram列表文件，将不进行频率过滤")
            

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram,freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))
                

class DocumentDatabase:
    """文档数据库类，用于管理和采样训练文档。
    
    特点：
    1. 支持内存优化模式，可以将文档存储在磁盘上
    2. 支持按句子长度加权的文档采样
    3. 提供文档的随机访问功能
    """
    
    def __init__(self, reduce_memory=False):
        """初始化文档数据库。
        
        Args:
            reduce_memory: 是否启用内存优化模式
        """
        if reduce_memory:
            logger.info("启用内存优化模式，文档将存储在临时文件中")
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            logger.info("使用内存模式存储文档")
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        """添加文档到数据库。
        
        Args:
            document: 要添加的文档
        """
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))
        
        # 每添加100个文档记录一次日志
        if len(self.doc_lengths) % 100 == 0:
            logger.info(f"已添加 {len(self.doc_lengths)} 个文档")

    def _precalculate_doc_weights(self):
        """预计算文档权重，用于加权采样。"""
        logger.info("开始预计算文档权重...")
        start_time = time.time()
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]
        logger.info(f"文档权重计算完成，耗时: {time.time() - start_time:.2f}秒")

    def sample_doc(self, current_idx, sentence_weighted=True):
        """采样一个文档。
        
        Args:
            current_idx: 当前文档索引
            sentence_weighted: 是否按句子长度加权采样
            
        Returns:
            采样的文档
        """
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, ngram_dict):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)


                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                for p in range(2, 8):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q+p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in ngram_dict.ngram_to_id_dict:
                            ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment])

                shuffle(ngram_matches)
                if len(ngram_matches) > ngram_dict.max_ngram_in_seq:
                    ngram_matches = ngram_matches[:ngram_dict.max_ngram_in_seq]
                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                    "ngram_ids": ngram_ids,
                    "ngram_positions": ngram_positions,
                    "ngram_lengths": ngram_lengths,
                    "ngram_tuples": ngram_tuples,
                    "ngram_segment_ids": ngram_seg_ids
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def tokenize_text(train_corpus,bert_model,output_dir, tokenizer,docs):

    # 创建临时目录用于存储处理后的数据
    temp_dir = output_dir / "temp_tokenized"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "tokenized_data.pt"
    
    # 使用tokenize_batch_text处理文本
    tokenize_batch_text(
        data=train_corpus,
        tok=bert_model,
        out=str(temp_file),
        batch_size=1000000,  # 可以根据需要调整批处理大小
        resume=True,
        max_length=256
    )
    
    # 加载处理后的数据
    logger.info("加载处理后的数据...")
    tokenized_data = torch.load(temp_file)
    
    # 将处理后的数据添加到DocumentDatabase
    current_doc = []
    for i in tqdm(range(len(tokenized_data["input_ids"])), desc="构建文档数据库"):
        # 获取非padding的token
        mask = tokenized_data["attention_mask"][i].bool()
        tokens = tokenized_data["input_ids"][i][mask].tolist()
        
        # 转换为token字符串
        tokens = tokenizer.convert_ids_to_tokens(tokens)
        
        if len(tokens) > 0:
            current_doc.append(tokens)
            
        # 每1000个序列作为一个文档
        if len(current_doc) >= 1000:
            docs.add_document(current_doc)
            current_doc = []
    
    # 添加最后一个文档
    if current_doc:
        docs.add_document(current_doc)
        
    # 清理临时文件
    os.remove(temp_file)
    os.rmdir(temp_dir)
    
    logger.info(f"文档处理完成，共有 {len(docs)} 个文档")
    
def main():
    """主函数，处理命令行参数并执行预训练数据生成流程。"""
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True,
                       help='训练语料库文件路径')
    parser.add_argument("--output_dir", type=Path, required=True,
                       help='输出目录路径')
    parser.add_argument("--bert_model", type=str, required=True,
                       help="预训练BERT模型名称或路径")
    parser.add_argument("--do_lower_case", action="store_true",
                       help="是否将文本转换为小写")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                       help="是否使用全词掩码")
    parser.add_argument("--reduce_memory", action="store_true",
                       help="是否启用内存优化模式")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                       help="要生成的数据轮数")
    parser.add_argument("--max_seq_len", type=int, default=128,
                       help="最大序列长度")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                       help="生成短序列的概率")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                       help="掩码标记的概率")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                       help="每个序列最大掩码标记数")
    parser.add_argument("--ngram_list_dir", type=str, default="./",
                       help="n-gram列表文件路径")
    parser.add_argument("--max_ngram_in_sequence", type=int, default=20,
                       help="每个序列最大n-gram数量")

    args = parser.parse_args()

    # 记录配置信息
    logger.info("="*50)
    logger.info("预训练数据生成配置:")
    logger.info(f"训练语料库: {args.train_corpus}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"BERT模型: {args.bert_model}")
    logger.info(f"最大序列长度: {args.max_seq_len}")
    logger.info(f"生成轮数: {args.epochs_to_generate}")
    logger.info(f"掩码概率: {args.masked_lm_prob}")
    logger.info(f"每序列最大预测数: {args.max_predictions_per_seq}")
    logger.info(f"是否全词掩码: {args.do_whole_word_mask}")
    logger.info(f"是否内存优化: {args.reduce_memory}")
    logger.info("="*50)

    # 加载tokenizer
    logger.info("开始加载tokenizer...")
    start_time = time.time()
    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    logger.info(f"Tokenizer加载完成，词表大小: {len(vocab_list)}，耗时: {time.time() - start_time:.2f}秒")

    # 加载n-gram字典
    logger.info("开始加载n-gram字典...")
    start_time = time.time()
    #ngram_dict = ZenNgramDict(args.bert_model, tokenizer=tokenizer)
    ngram_dict = ZenDNANgramDict(args.ngram_list_dir, tokenizer=tokenizer)
    logger.info(f"N-gram字典加载完成，耗时: {time.time() - start_time:.2f}秒")

    # 处理训练语料库
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        logger.info("开始加载训练语料库...")
        corpus_start_time = time.time()
        
        # 使用tokenize_text批量进行tokenize处理文本
        tokenize_text(args.train_corpus,args.bert_model, args.output_dir, tokenizer,docs)
        # with args.train_corpus.open() as f:
        #     doc = []
        #     for line in tqdm(f, desc="加载数据集", unit=" lines"):
        #         line = line.strip()
        #         if line == "":
        #             docs.add_document(doc)
        #             doc = []
        #         else:
        #             tokens = tokenizer.tokenize(line)
        #             doc.append(tokens)
        #     if doc:
        #         docs.add_document(doc)
        
        corpus_load_time = time.time() - corpus_start_time
        logger.info(f"语料库加载完成，文档数: {len(docs)}，耗时: {corpus_load_time:.2f}秒")

        if len(docs) <= 1:
            logger.error("错误：输入文件中未找到文档分隔符！")
            logger.error("请在文档之间添加空行作为分隔符。")
            exit(1)

        # 创建输出目录
        args.output_dir.mkdir(exist_ok=True)
        
        # 生成每个epoch的数据
        for epoch in trange(args.epochs_to_generate, desc="生成数据"):
            epoch_start_time = time.time()
            logger.info(f"\n{'='*20} 开始生成第 {epoch+1} 轮数据 {'='*20}")
            
            epoch_filename = args.output_dir / f"epoch_{epoch}.json"
            num_instances = 0
            
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc="处理文档"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len,
                        short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob,
                        max_predictions_per_seq=args.max_predictions_per_seq,
                        whole_word_mask=args.do_whole_word_mask,
                        vocab_list=vocab_list,
                        ngram_dict=ngram_dict
                    )
                    
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
                    
                    # 每处理1000个文档记录一次进度
                    if doc_idx % 1000 == 0:
                        logger.info(f"已处理 {doc_idx}/{len(docs)} 个文档，生成 {num_instances} 个训练实例")
            
            # 保存本轮数据的统计信息
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len,
                    "max_ngram_in_sequence": args.max_ngram_in_sequence
                }
                metrics_file.write(json.dumps(metrics))
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\n第 {epoch+1} 轮数据生成完成:")
            logger.info(f"  生成训练实例数: {num_instances}")
            logger.info(f"  数据文件: {epoch_filename}")
            logger.info(f"  统计文件: {metrics_file}")
            logger.info(f"  耗时: {epoch_time:.2f}秒")
            logger.info(f"  平均速度: {num_instances/epoch_time:.2f}实例/秒")
            logger.info("="*50)

if __name__ == '__main__':
    main()

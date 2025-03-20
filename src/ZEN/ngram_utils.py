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
"""utils for ngram for ZEN model."""
import pandas as pd
import os
import logging
from dnazen.ngram import NgramEncoder
# NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)

# class ZenNgramDict(object):
#     """
#     Dict class to store the ngram
#     """
#     def __init__(self, ngram_freq_path, tokenizer, max_ngram_in_seq=128):
#         """Constructs ZenNgramDict

#         :param ngram_freq_path: ngrams with frequency
#         """
#         if os.path.isdir(ngram_freq_path):
#             ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
#         self.ngram_freq_path = ngram_freq_path
#         self.max_ngram_in_seq = max_ngram_in_seq
#         self.id_to_ngram_list = ["[pad]"]
#         self.ngram_to_id_dict = {"[pad]": 0}
#         self.ngram_to_freq_dict = {}

#         logger.info("loading ngram frequency file {}".format(ngram_freq_path))
#         with open(ngram_freq_path, "r", encoding="utf-8") as fin:
#             for i, line in enumerate(fin):
#                 import ipdb; ipdb.set_trace()
#                 ngram,freq = line.split(",")
#                 tokens = tuple(tokenizer.tokenize(ngram))
#                 self.ngram_to_freq_dict[ngram] = freq
#                 self.id_to_ngram_list.append(tokens)
#                 self.ngram_to_id_dict[tokens] = i + 1

#     def save(self, ngram_freq_path):
#         with open(ngram_freq_path, "w", encoding="utf-8") as fout:
#             for ngram,freq in self.ngram_to_freq_dict.items():
#                 fout.write("{},{}\n".format(ngram, freq))



NGRAM_DICT_NAME="ngram_list.txt"
NGRAM_ENCODER_NAME="ngram_encoder.json"

# class ZenNgramDict(object):
class ZenNgramDict(object):
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
                    freq = row["频率"]
                    #tokens = tuple(tokenizer.tokenize(ngram))
                    ngram = row["token_ids"]
                    tokens = row["N-gram"]
                    tok = tuple(tokens.split(" ")) # 将ngram的'GA CTCATT'转换为('GA', 'CTCATT')
                    #ipdb.set_trace()
                    self.ngram_to_freq_dict[ngram] = freq
                    self.id_to_ngram_list.append(tok)
                    self.ngram_to_id_dict[tok] = i + 1

            logger.info(
                f"创建了 {len(self.id_to_ngram_list)} 个token ID映射（总N-gram数: {len(self.ngram_to_freq_dict)}）"
            )
        else:
            logger.warning("没有N-gram列表文件，将不进行频率过滤")
            

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram,freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))
  
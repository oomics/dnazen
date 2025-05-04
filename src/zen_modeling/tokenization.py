import os
import json
import numpy as np
import random
from transformers import AutoTokenizer, PreTrainedTokenizer


class NgramTokenizer(PreTrainedTokenizer):
    """
    基于 HuggingFace 的 PreTrainedTokenizer 封装了一个包含 n-gram 匹配功能的 tokenizer，
    内部使用标准的 tokenizer，同时加载 ngram 配置文件（固定文件名为 ngram.json）。
    """

    def __init__(self, pretrained_tokenizer, ngram_json_file, **kwargs):
        # 初始化内部的 tokenizer（例如 BertTokenizer）
        self.bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, **kwargs)
        # 加载 ngram 的配置信息
        with open(ngram_json_file, "r", encoding="utf-8") as f:
            self.ngram_data = json.load(f)
        self.min_ngram_len = self.ngram_data.get("min_ngram_len", 2)
        self.max_ngram_len = self.ngram_data.get("max_ngram_len", 5)
        self.max_ngram_count = self.ngram_data.get("max_ngram_count", 30)
        self.pad_ngram_id = self.ngram_data.get("pad_ngram_id", 0)
        self.pad_ngram = self.ngram_data.get("pad_ngram", "<PAD>")
        self.ngram_size = self.ngram_data.get("ngram_size", None)
        self.ngram_vocab = self.ngram_data.get("vocab", {})

        # 设置 special tokens（一般为 [CLS]、[SEP]、[PAD]）
        self.cls_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.pad_token_id = self.bert_tokenizer.pad_token_id

    def save_pretrained(self, save_directory, **kwargs):
        """
        保存内部 tokenizer 及 ngram 配置文件（固定文件名为 ngram.json）。
        """
        self.bert_tokenizer.save_pretrained(save_directory, **kwargs)
        ngram_path = os.path.join(save_directory, "ngram.json")
        with open(ngram_path, "w", encoding="utf-8") as f:
            json.dump(self.ngram_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        从预训练模型目录加载 tokenizer，同时加载目录下的 ngram.json 文件。
        """
        ngram_file = os.path.join(pretrained_model_name_or_path, "ngram.json")
        if not os.path.exists(ngram_file):
            raise ValueError("ngram.json 文件不存在，请检查预训练目录。")
        instance = cls(pretrained_model_name_or_path, ngram_json_file=ngram_file, **kwargs)
        return instance

    def find_ngrams_in_string(self, ngram_dict, text, min_num, max_num):
        """
        在 text（tokens 用空格隔开）中查找所有在 ngram_dict 中出现的 n-gram。

        返回：
          found_ngrams: 每个元素为 (n-gram, start_index, end_index)
          matrix: numpy 数组，形状为 (len(found_ngrams), len(tokens))，匹配位置为 1
        """
        tokens = text.split()
        found_ngrams = []
        for start in range(len(tokens)):
            for length in range(min_num, max_num + 1):
                end = start + length
                if end > len(tokens):
                    break
                candidate = " ".join(tokens[start:end])
                if candidate in ngram_dict:
                    found_ngrams.append((candidate, start, end))
        matrix = np.zeros((len(found_ngrams), len(tokens)), dtype=int)
        for i, (_, start, end) in enumerate(found_ngrams):
            matrix[i, start:end] = 1
        return found_ngrams, matrix

    def _tokenize_and_match_ngrams(self, text, is_split_into_words=False, max_seq_length=None,
                                   pad_to_max_length=True):
        """
        核心处理函数：
          1. 判断输入是原始字符串还是已分词列表（is_split_into_words=True 时要求输入为 list）。
          2. 利用内部 tokenizer 得到 tokens，并添加 [CLS] 和 [SEP] 特殊 token。
          3. 若未指定 max_seq_length，则默认使用内部 tokenizer 的 model_max_length，
             对 tokens（不含特殊 token）先截断，再对整体序列（含特殊 token）进行 pad 或截断，
             确保最终长度等于 max_seq_length。
          4. 在截断后的 tokens（不含特殊 token）上匹配 n-gram，
             并对匹配位置做特殊 token 的偏移调整，过滤掉超出最终序列长度的匹配。
          5. 返回包含 input_ids、attention_mask、ngram_input_ids（去重后的 ngram id 列表）、
             ngram_attention_match（匹配矩阵）和 ngram_attention_mask（对应于 ngram_input_ids 的 mask）。
        """
        if max_seq_length is None:
            max_seq_length = self.bert_tokenizer.model_max_length

        if not is_split_into_words:
            tokens = self.bert_tokenizer.tokenize(text)
        else:
            if not isinstance(text, list):
                raise ValueError("当 is_split_into_words=True 时，输入必须为 list 类型。")
            tokens = text

        max_tokens = max_seq_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        tokens_with_special = [self.cls_token] + tokens + [self.sep_token]
        cur_seq_len = len(tokens_with_special)
        if pad_to_max_length and cur_seq_len < max_seq_length:
            pad_length = max_seq_length - cur_seq_len
            tokens_with_special = tokens_with_special + [self.pad_token] * pad_length
        if cur_seq_len > max_seq_length:
            tokens_with_special = tokens_with_special[:max_seq_length]
        new_seq_len = len(tokens_with_special)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_with_special)
        attention_mask = [1 if token != self.bert_tokenizer.pad_token else 0 for token in tokens_with_special]

        text_for_ngram = " ".join(tokens)
        found_ngrams, _ = self.find_ngrams_in_string(
            self.ngram_vocab, text_for_ngram, self.min_ngram_len, self.max_ngram_len
        )

        ngram_to_positions = {}
        for ng, start, end in found_ngrams:
            ngram_id = self.ngram_vocab[ng]
            # 整体右移 1（因为 [CLS] 在最前面）
            positions = set(range(start + 1, end + 1))
            if ngram_id in ngram_to_positions:
                ngram_to_positions[ngram_id].update(positions)
            else:
                ngram_to_positions[ngram_id] = positions

        for ngram_id in ngram_to_positions:
            ngram_to_positions[ngram_id] = {pos for pos in ngram_to_positions[ngram_id] if pos < new_seq_len}

        sorted_ngram_ids = sorted(
            ngram_to_positions.keys(),
            key=lambda nid: min(ngram_to_positions[nid]) if ngram_to_positions[nid] else new_seq_len
        )
        if len(sorted_ngram_ids) > self.max_ngram_count:
            sorted_ngram_ids = sorted_ngram_ids[:self.max_ngram_count]
        ngram_input_ids = sorted_ngram_ids

        ngram_attention_match = np.zeros((len(ngram_input_ids), new_seq_len), dtype=int)
        for idx, ngram_id in enumerate(ngram_input_ids):
            for pos in sorted(ngram_to_positions[ngram_id]):
                if pos < new_seq_len:
                    ngram_attention_match[idx, pos] = 1

        if pad_to_max_length:
            if len(ngram_input_ids) < self.max_ngram_count:
                pad_rows = self.max_ngram_count - len(ngram_input_ids)
                pad_matrix = np.zeros((pad_rows, new_seq_len), dtype=int)
                ngram_attention_match = np.concatenate([ngram_attention_match, pad_matrix], axis=0)
                ngram_input_ids += [self.pad_ngram_id] * pad_rows
        else:
            if len(ngram_input_ids) < 1:
                pad_rows = 1
                pad_matrix = np.zeros((pad_rows, new_seq_len), dtype=int)
                ngram_attention_match = np.concatenate([ngram_attention_match, pad_matrix], axis=0)
                ngram_input_ids += [self.pad_ngram_id] * pad_rows

        ngram_attention_mask = [1 if ngram_id != self.pad_ngram_id else 0 for ngram_id in ngram_input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ngram_input_ids": ngram_input_ids,
            "ngram_attention_match": ngram_attention_match.tolist(),
            "ngram_attention_mask": ngram_attention_mask,
        }

    def masking(self, outputs, mask_prob=0.15, whole_ngram_masking=True):
        """
        对 _tokenize_and_match_ngrams 的输出进行随机 masking（采用 BERT 的 80/10/10 策略），
        并生成用于计算 MLM loss 的 labels。要求是：
          - 如果被 mask 的 token 落在某个 ngram 内，则将该 ngram 所涉及的所有 token 都 mask 掉，
            同时从 ngram 输出中“去掉”该 ngram（对应 ngram_input_ids 置为 pad_ngram_id，
            ngram_attention_mask 置 0，ngram_attention_match 置零）。
          - 最后将 ngram 部分重新排序，将非 pad 的部分放前面，pad 的部分放后面。

        返回更新后的字典，包含 keys：
          input_ids, attention_mask, labels, ngram_input_ids, ngram_attention_match, ngram_attention_mask
        """
        original_input_ids = outputs["input_ids"]
        new_input_ids = original_input_ids.copy()
        seq_length = len(new_input_ids)
        # 初始化 labels，未 mask 的位置设为 -100
        labels = [-100] * seq_length

        cls_id = self.bert_tokenizer.cls_token_id
        sep_id = self.bert_tokenizer.sep_token_id
        pad_id = self.bert_tokenizer.pad_token_id
        mask_id = self.bert_tokenizer.mask_token_id
        vocab_size = self.bert_tokenizer.vocab_size

        candidate_masked_positions = set()
        # 首先，对每个非特殊 token按照 mask_prob 进行候选 mask（80/10/10策略）
        for pos in range(seq_length):
            if new_input_ids[pos] in [cls_id, sep_id, pad_id]:
                continue
            if random.random() < mask_prob:
                candidate_masked_positions.add(pos)
                labels[pos] = new_input_ids[pos]
                p = random.random()
                if p < 0.8:
                    new_input_ids[pos] = mask_id
                elif p < 0.9:
                    new_input_ids[pos] = random.randint(0, vocab_size - 1)
                else:
                    # 保持原样，但 label 已记录原始 token
                    new_input_ids[pos] = new_input_ids[pos]

        # 然后，对于每个 ngram，若其涉及的任一 token被 mask，则对该 ngram所有 token执行同样的 mask 策略
        ngram_input_ids = outputs["ngram_input_ids"][:]  # 复制列表
        ngram_attention_match = np.array(outputs["ngram_attention_match"])
        ngram_attention_mask = outputs["ngram_attention_mask"][:]
        max_ngram_count = len(ngram_input_ids)

        if whole_ngram_masking:
            for i in range(max_ngram_count):
                if ngram_input_ids[i] == self.pad_ngram_id:
                    continue
                # 获取该 ngram 涉及的所有 token 位置
                positions = [j for j, val in enumerate(ngram_attention_match[i]) if val == 1]
                if any(pos in candidate_masked_positions for pos in positions):
                    for pos in positions:
                        if pos not in candidate_masked_positions:
                            labels[pos] = original_input_ids[pos]
                            p = random.random()
                            if p < 0.8:
                                new_input_ids[pos] = mask_id
                            elif p < 0.9:
                                new_input_ids[pos] = random.randint(0, vocab_size - 1)
                            else:
                                new_input_ids[pos] = new_input_ids[pos]
                            candidate_masked_positions.add(pos)
                    # 将该 ngram “去掉”
                    ngram_input_ids[i] = self.pad_ngram_id
                    ngram_attention_mask[i] = 0
                    ngram_attention_match[i, :] = 0

            # 重新排序 ngram 部分，将非 pad 的放前面，pad 的放后面
            non_pad_indices = [i for i, ng in enumerate(ngram_input_ids) if ng != self.pad_ngram_id]
            pad_indices = [i for i, ng in enumerate(ngram_input_ids) if ng == self.pad_ngram_id]
            ngram_input_ids = [ngram_input_ids[i] for i in non_pad_indices] + [ngram_input_ids[i] for i in pad_indices]
            ngram_attention_match = np.concatenate([ngram_attention_match[non_pad_indices],
                                                        ngram_attention_match[pad_indices]], axis=0)
            ngram_attention_mask = [ngram_attention_mask[i] for i in non_pad_indices] + [ngram_attention_mask[i] for i
                                                                                             in pad_indices]

        new_outputs = {
            "input_ids": new_input_ids,
            "attention_mask": outputs["attention_mask"],
            "labels": labels,
            "ngram_input_ids": ngram_input_ids,
            "ngram_attention_match": ngram_attention_match.tolist(),
            "ngram_attention_mask": ngram_attention_mask,
        }
        return new_outputs

    def __call__(self, text, is_split_into_words=False, max_seq_length=None, pad_to_max_length=True, **kwargs):
        """
        调用时直接返回 _tokenize_and_match_ngrams 的结果。
        如果未传入 max_seq_length，则默认使用内部 tokenizer 的 model_max_length。
        """
        return self._tokenize_and_match_ngrams(text, is_split_into_words, max_seq_length=max_seq_length,
                                               pad_to_max_length=pad_to_max_length)


# 示例用法：
if __name__ == "__main__":
    pretrained_tokenizer = "/data1/user1/llm/DNABERT-2-117M"
    ngram_json_file = "/data1/user1/project/zen_train/data/pretrain/human_ms_gue/ngram_encoders/pmi_1_all_union_ngram_encoder.json"

    tokenizer = NgramTokenizer(pretrained_tokenizer, ngram_json_file)

    tokens = "A GGTT AAATA GGGGTT GAGA TA TGATG CTCA GGAGAA GCGCTT TCTT TCGC GAGCA CCCTGAA CCA GACC".split()
    outputs = tokenizer(tokens, is_split_into_words=True, max_seq_length=20)
    print("原始输出:")
    print("input_ids:", outputs["input_ids"])
    print("attention_mask:", outputs["attention_mask"])
    print("ngram_input_ids:", outputs["ngram_input_ids"])
    print("ngram_attention_match:", outputs["ngram_attention_match"])
    print("ngram_attention_mask:", outputs["ngram_attention_mask"])

    masked_outputs = tokenizer.masking(outputs, mask_prob=0.15)
    print("\n经过 whole_ngram_masking 后:")
    print("input_ids:", masked_outputs["input_ids"])
    print("attention_mask:", masked_outputs["attention_mask"])
    print("labels:", masked_outputs["labels"])
    print("ngram_input_ids:", masked_outputs["ngram_input_ids"])
    print("ngram_attention_match:", masked_outputs["ngram_attention_match"])
    print("ngram_attention_mask:", masked_outputs["ngram_attention_mask"])

import os
import json
import numpy as np
import random
from transformers import AutoTokenizer, PreTrainedTokenizer


class NgramTokenizer(PreTrainedTokenizer):
    """
    A wrapper around HuggingFace's PreTrainedTokenizer that adds n-gram matching support.
    It uses a standard tokenizer internally and loads n-gram configuration from a JSON file (named ngram.json).
    """

    def __init__(self, pretrained_tokenizer, ngram_json_file, **kwargs):
        # Initialize the internal tokenizer (e.g., BertTokenizer)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, **kwargs)

        # Load n-gram configuration
        with open(ngram_json_file, "r", encoding="utf-8") as f:
            self.ngram_data = json.load(f)

        # Read n-gram parameters
        self.min_ngram_len = self.ngram_data.get("min_ngram_len", 2)
        self.max_ngram_len = self.ngram_data.get("max_ngram_len", 5)
        self.max_ngram_count = self.ngram_data.get("max_ngram_count", 30)
        self.pad_ngram_id = self.ngram_data.get("pad_ngram_id", 0)
        self.pad_ngram = self.ngram_data.get("pad_ngram", "<PAD>")
        self.ngram_size = self.ngram_data.get("ngram_size", None)
        self.ngram_vocab = self.ngram_data.get("vocab", {})

        # Set special tokens ([CLS], [SEP], [PAD])
        self.cls_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.pad_token_id = self.bert_tokenizer.pad_token_id

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the internal tokenizer and the n-gram config file (ngram.json) to the specified directory.
        """
        self.bert_tokenizer.save_pretrained(save_directory, **kwargs)
        ngram_path = os.path.join(save_directory, "ngram.json")
        with open(ngram_path, "w", encoding="utf-8") as f:
            json.dump(self.ngram_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load the tokenizer from a pretrained model directory and read ngram.json from the same.
        """
        ngram_file = os.path.join(pretrained_model_name_or_path, "ngram.json")
        if not os.path.exists(ngram_file):
            raise ValueError("ngram.json file not found. Please check the pretrained directory.")
        instance = cls(pretrained_model_name_or_path, ngram_json_file=ngram_file, **kwargs)
        return instance

    def find_ngrams_in_string(self, ngram_dict, text, min_num, max_num):
        """
        Find all n-grams in `text` (where tokens are space-separated) that appear in `ngram_dict`.

        Returns:
          found_ngrams: List of tuples (ngram, start_index, end_index)
          matrix: numpy array of shape (len(found_ngrams), len(tokens)), with 1s indicating matching positions
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

    def _tokenize_and_match_ngrams(
        self,
        text,
        is_split_into_words=False,
        max_seq_length=None,
        pad_to_max_length=True
    ):
        """
        Core processing routine:
          1. Determine if `text` is a raw string or a list of tokens (if is_split_into_words=True, text must be a list).
          2. Tokenize using internal tokenizer and add [CLS] and [SEP].
          3. If max_seq_length is not specified, use the internal tokenizer's model_max_length.
             Truncate tokens (excluding special tokens) to fit, then pad or truncate the full sequence to max_seq_length.
          4. Match n-grams on the truncated tokens (excluding special tokens), adjust for special token offsets,
             and filter out matches that exceed the final sequence length.
          5. Return a dictionary containing input_ids, attention_mask,
             ngram_input_ids (unique n-gram IDs), ngram_attention_match (matching matrix),
             and ngram_attention_mask (mask for ngram_input_ids).
        """
        if max_seq_length is None:
            max_seq_length = self.bert_tokenizer.model_max_length

        # Tokenization
        if not is_split_into_words:
            tokens = self.bert_tokenizer.tokenize(text)
        else:
            if not isinstance(text, list):
                raise ValueError("When is_split_into_words=True, text must be a list.")
            tokens = text

        # Reserve space for [CLS] and [SEP]
        max_tokens = max_seq_length - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Add special tokens
        tokens_with_special = [self.cls_token] + tokens + [self.sep_token]
        cur_seq_len = len(tokens_with_special)

        # Padding or truncation
        if pad_to_max_length and cur_seq_len < max_seq_length:
            pad_length = max_seq_length - cur_seq_len
            tokens_with_special += [self.pad_token] * pad_length
        if cur_seq_len > max_seq_length:
            tokens_with_special = tokens_with_special[:max_seq_length]
        new_seq_len = len(tokens_with_special)

        # Convert tokens to IDs
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens_with_special)
        attention_mask = [1 if token != self.pad_token else 0 for token in tokens_with_special]

        # Find n-grams in the token sequence (excluding special tokens)
        text_for_ngram = " ".join(tokens)
        found_ngrams, _ = self.find_ngrams_in_string(
            self.ngram_vocab,
            text_for_ngram,
            self.min_ngram_len,
            self.max_ngram_len
        )

        # Map n-grams to their positions with offset for [CLS]
        ngram_to_positions = {}
        for ng, start, end in found_ngrams:
            ngram_id = self.ngram_vocab[ng]
            positions = set(range(start + 1, end + 1))  # shift by 1 for [CLS]
            if ngram_id in ngram_to_positions:
                ngram_to_positions[ngram_id].update(positions)
            else:
                ngram_to_positions[ngram_id] = positions

        # Filter out positions beyond sequence length
        for ngram_id in list(ngram_to_positions):
            ngram_to_positions[ngram_id] = {pos for pos in ngram_to_positions[ngram_id] if pos < new_seq_len}

        # Sort n-gram IDs by their earliest position
        sorted_ngram_ids = sorted(
            ngram_to_positions.keys(),
            key=lambda nid: min(ngram_to_positions[nid]) if ngram_to_positions[nid] else new_seq_len
        )

        # Limit to max_ngram_count
        if len(sorted_ngram_ids) > self.max_ngram_count:
            sorted_ngram_ids = sorted_ngram_ids[:self.max_ngram_count]
        ngram_input_ids = sorted_ngram_ids

        # Build matching matrix
        ngram_attention_match = np.zeros((len(ngram_input_ids), new_seq_len), dtype=int)
        for idx, ngram_id in enumerate(ngram_input_ids):
            for pos in sorted(ngram_to_positions[ngram_id]):
                if pos < new_seq_len:
                    ngram_attention_match[idx, pos] = 1

        # Padding for n-gram matrix and IDs
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

        ngram_attention_mask = [1 if ng != self.pad_ngram_id else 0 for ng in ngram_input_ids]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ngram_input_ids": ngram_input_ids,
            "ngram_attention_match": ngram_attention_match.tolist(),
            "ngram_attention_mask": ngram_attention_mask,
        }

    def masking(self, outputs, mask_prob=0.15, whole_ngram_masking=True):
        """
        Apply random masking following BERT's 80/10/10 strategy to the tokenized output.
        If `whole_ngram_masking` is True and any token in an n-gram is chosen for masking,
        mask all tokens in that n-gram and remove that n-gram from the n-gram output (set to pad).

        Returns an updated dict with keys:
          input_ids, attention_mask, labels,
          ngram_input_ids, ngram_attention_match, ngram_attention_mask
        """
        original_input_ids = outputs["input_ids"]
        new_input_ids = original_input_ids.copy()
        seq_length = len(new_input_ids)
        labels = [-100] * seq_length  # default no-loss positions

        cls_id = self.bert_tokenizer.cls_token_id
        sep_id = self.bert_tokenizer.sep_token_id
        pad_id = self.bert_tokenizer.pad_token_id
        mask_id = self.bert_tokenizer.mask_token_id
        vocab_size = self.bert_tokenizer.vocab_size

        candidate_positions = set()

        # Select candidate positions for token masking
        for pos in range(seq_length):
            token_id = new_input_ids[pos]
            if token_id in [cls_id, sep_id, pad_id]:
                continue
            if random.random() < mask_prob:
                candidate_positions.add(pos)
                labels[pos] = new_input_ids[pos]
                p = random.random()
                if p < 0.8:
                    new_input_ids[pos] = mask_id
                elif p < 0.9:
                    new_input_ids[pos] = random.randint(0, vocab_size - 1)
                # else keep original

        # Whole n-gram masking
        ngram_input_ids = outputs["ngram_input_ids"][:]
        ngram_attention_match = np.array(outputs["ngram_attention_match"])
        ngram_attention_mask = outputs["ngram_attention_mask"][...]
        max_ngrams = len(ngram_input_ids)

        if whole_ngram_masking:
            for i in range(max_ngrams):
                ng_id = ngram_input_ids[i]
                if ng_id == self.pad_ngram_id:
                    continue
                # positions of this n-gram
                positions = [idx for idx, val in enumerate(ngram_attention_match[i]) if val]
                if any(pos in candidate_positions for pos in positions):
                    for pos in positions:
                        if pos not in candidate_positions:
                            labels[pos] = original_input_ids[pos]
                            p = random.random()
                            if p < 0.8:
                                new_input_ids[pos] = mask_id
                            elif p < 0.9:
                                new_input_ids[pos] = random.randint(0, vocab_size - 1)
                            candidate_positions.add(pos)
                    # Remove this n-gram
                    ngram_input_ids[i] = self.pad_ngram_id
                    ngram_attention_mask[i] = 0
                    ngram_attention_match[i, :] = 0

            # Reorder n-grams: keep non-pad first
            non_pad_idxs = [i for i, ng in enumerate(ngram_input_ids) if ng != self.pad_ngram_id]
            pad_idxs = [i for i, ng in enumerate(ngram_input_ids) if ng == self.pad_ngram_id]
            ngram_input_ids = [ngram_input_ids[i] for i in non_pad_idxs + pad_idxs]
            ngram_attention_match = np.concatenate([ngram_attention_match[non_pad_idxs],
                                                    ngram_attention_match[pad_idxs]], axis=0)
            ngram_attention_mask = [ngram_attention_mask[i] for i in non_pad_idxs + pad_idxs]

        return {
            "input_ids": new_input_ids,
            "attention_mask": outputs["attention_mask"],
            "labels": labels,
            "ngram_input_ids": ngram_input_ids,
            "ngram_attention_match": ngram_attention_match.tolist(),
            "ngram_attention_mask": ngram_attention_mask,
        }

    def __call__(
        self,
        text,
        is_split_into_words=False,
        max_seq_length=None,
        pad_to_max_length=True,
        **kwargs
    ):
        """
        Invoke tokenization and n-gram matching. Returns the result of `_tokenize_and_match_ngrams`.
        """
        return self._tokenize_and_match_ngrams(
            text,
            is_split_into_words,
            max_seq_length=max_seq_length,
            pad_to_max_length=pad_to_max_length
        )


# Example usage
def main():
    pretrained = "/data1/user1/llm/DNABERT-2-117M"
    ngram_file = "/data1/user1/project/zen_train/data/pretrain/human_ms_gue/ngram_encoders/pmi_1_all_union_ngram_encoder.json"

    tokenizer = NgramTokenizer(pretrained, ngram_file)

    tokens = "A GGTT AAATA GGGGTT GAGA TA TGATG CTCA GGAGAA GCGCTT TCTT TCGC GAGCA CCCTGAA CCA GACC".split()
    outputs = tokenizer(tokens, is_split_into_words=True, max_seq_length=20)
    print("Original outputs:")
    print("input_ids:", outputs["input_ids"])
    print("attention_mask:", outputs["attention_mask"])
    print("ngram_input_ids:", outputs["ngram_input_ids"])
    print("ngram_attention_match:", outputs["ngram_attention_match"])
    print("ngram_attention_mask:", outputs["ngram_attention_mask"])

    masked = tokenizer.masking(outputs, mask_prob=0.15)
    print("\nAfter whole n-gram masking:")
    print("input_ids:", masked["input_ids"])
    print("attention_mask:", masked["attention_mask"])
    print("labels:", masked["labels"])
    print("ngram_input_ids:", masked["ngram_input_ids"])
    print("ngram_attention_match:", masked["ngram_attention_match"])
    print("ngram_attention_mask:", masked["ngram_attention_mask"])

if __name__ == "__main__":
    main()

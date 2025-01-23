from typing import TypedDict
import json
import warnings

import torch

from ._find_ngram import find_ngrams, NgramFinderConfig


class _NgramEncoderConfig(TypedDict):
    vocab: dict[str, int]
    min_ngram_len: int
    max_ngram_len: int
    max_ngrams: int


class EncodedNgram(TypedDict):
    ngram_ids: torch.Tensor
    ngram_position_matrix: torch.Tensor


class NgramEncoder:
    def __init__(
        self,
        vocab_dict: dict[tuple[int, ...], int],
        min_ngram_len: int,
        max_ngram_len: int,
        max_ngrams: int,
    ):
        self._vocab: dict[tuple[int, ...], int] = vocab_dict
        self._id2ngrams = {}
        for k, v in vocab_dict.items():
            self._id2ngrams[v] = k

        self._min_ngram_len = min_ngram_len
        self._max_ngram_len = max_ngram_len
        self._max_ngrams = max_ngrams

    def _get_ngram_id(self, tokens: tuple[int, ...]) -> int | None:
        return self._vocab.get(tokens, None)

    def encode(self, token_ids: torch.Tensor, pad_token_id: int = 3) -> EncodedNgram:
        """
        Encode token ids into ngram ids and it's position matrix.

        Returns:
            dict["ngram_ids"]: the id of ngrams
            dict["ngram_position_matrix"]: the position matrix ( max_ngrams*len(token_ids) ) mapping ngram to the original token position.
        """
        assert token_ids.dim() == 1

        token_ids_list = token_ids.tolist()
        token_len = len(token_ids_list)
        # Trim trailing pad tokens
        while token_ids_list and token_ids_list[-1] == pad_token_id:
            token_ids_list.pop()

        _cur_ngram_id_idx = 0  # index to ngram_ids
        ret_val: EncodedNgram = {
            "ngram_ids": torch.zeros(self._max_ngrams, dtype=torch.int),
            "ngram_position_matrix": torch.zeros(
                token_len, self._max_ngrams, dtype=torch.bool
            ),
        }

        for ngram_len in range(self._min_ngram_len, self._max_ngram_len):
            for q in range(token_len - ngram_len + 1):
                ngram: tuple[int, ...] = tuple(token_ids_list[q : q + ngram_len])
                ngram_id = self._get_ngram_id(ngram)
                if ngram_id is None:
                    continue
                # found the match
                ret_val["ngram_ids"][_cur_ngram_id_idx] = ngram_id
                ret_val["ngram_position_matrix"][
                    q : q + ngram_len, _cur_ngram_id_idx
                ] = 1
                _cur_ngram_id_idx += 1
                if _cur_ngram_id_idx >= self._max_ngrams:
                    return ret_val
        return ret_val

    @classmethod
    def from_file(cls, path):
        """Instantiate a new :class:`~dnazen.ngram.NgramTokenizer` from the file at the given path."""
        with open(path, "r") as f:
            config: _NgramEncoderConfig = json.load(f)

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

    def train(
        self,
        tokens: list[list[int]],
        min_pmi: float,
        min_token_count: int,
        min_ngram_freq: int,
        num_workers: int = 64,
    ):
        """Train the ngram encoder using the given token and config.

        Args:
            tokens (list[list[int]]): list of pre-tokenized tokens.
            min_pmi (float): the pmi threshold of ngram
            min_token_count (int): the minimum token frequency for filtering ngram
            min_ngram_freq (int): the minimum ngram frequency for filtering ngram
            num_workers (int, optional): number of workers for training. Defaults to 64.
        """
        ngram_finder_config = NgramFinderConfig()
        ngram_finder_config.min_pmi = min_pmi
        ngram_finder_config.max_ngram_len = self._max_ngram_len
        ngram_finder_config.min_ngram_len = self._min_ngram_len
        ngram_finder_config.min_ngram_freq = min_ngram_freq
        ngram_finder_config.min_token_count = min_token_count
        ngram_finder_config.num_workers = num_workers

        if self._vocab != {}:
            warnings.warn("the vocab is non-empty. Would be overloaded after training.")

        self._vocab = find_ngrams(ngram_finder_config, tokens=tokens)
        for idx, (k, v) in enumerate(self._vocab.items()):
            self._vocab[k] = idx

        self._id2ngrams = {}
        for k, v in self._vocab.items():
            self._id2ngrams[v] = k

    def save(self, path, pretty=True):
        vocab_dict = {}
        for k, v in self._vocab.items():
            k_ = ":".join(str(i) for i in k)
            vocab_dict[k_] = v

        config: _NgramEncoderConfig = {
            "vocab": vocab_dict,
            "max_ngram_len": self._max_ngram_len,
            "min_ngram_len": self._min_ngram_len,
            "max_ngrams": self._max_ngrams,
        }
        indent = 2 if pretty else None
        with open(path, "w") as f:
            json.dump(config, f, indent=indent)

    def get_vocab(self):
        return self._vocab

    def get_vocab_size(self):
        return len(self._vocab)

    def get_id(self) -> int:
        """
        Get the unique identifier of ngram encoder.

        The identifier is soloy based on the ngram vocabulary.
        """
        return hash(frozenset(self._vocab.items()))

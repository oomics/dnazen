"""
Utility for encoding, and do statical analysis of ngrams.
"""

from typing import TypedDict, Literal
import json
import warnings
from copy import deepcopy

import torch

from ._find_ngram import (
    find_ngrams_by_pmi,
    find_ngrams_by_freq,
    PmiNgramFinderConfig,
    FreqNgramFinderConfig,
)


class _NgramEncoderConfig(TypedDict):
    vocab: dict[str, int]
    min_ngram_len: int
    max_ngram_len: int
    max_ngrams: int


class EncodedNgram(TypedDict):
    ngram_ids: torch.Tensor
    ngram_attention_mask: torch.Tensor
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

    def get_matched_ngrams(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """Get matched ngrams and it's index from token_ids."""
        assert token_ids.dim() == 1, (
            f"token_ids should have dim=1, but got {token_ids.dim()}"
        )

        token_len = token_ids.shape[0]
        token_ids = token_ids[token_ids != pad_token_id]
        token_ids_list = token_ids.tolist()

        for ngram_len in range(self._min_ngram_len, self._max_ngram_len):
            for q in range(token_len - ngram_len + 1):
                ngram: tuple[int, ...] = tuple(token_ids_list[q : q + ngram_len])
                ngram_id = self._get_ngram_id(ngram)
                if ngram_id is None:
                    continue
                # found the match
                yield ngram, q

    def get_num_matches(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """Calculate the number of ngram matches from token_ids."""
        return len([0 for _, _ in self.get_matched_ngrams(token_ids, pad_token_id)])

    def get_total_ngram_len(self, token_ids: torch.Tensor, pad_token_id: int = 3):
        """Calculate average ngram length."""
        total_ngram_len = 0
        for ngram, _ in self.get_matched_ngrams(token_ids, pad_token_id):
            total_ngram_len += len(ngram)
        return total_ngram_len

    def encode(self, token_ids: torch.Tensor, pad_token_id: int = 3) -> EncodedNgram:
        """
        Encode token ids into ngram ids and it's position matrix.

        Returns:
            dict["ngram_ids"]: the id of ngrams
            dict["ngram_position_matrix"]: the position matrix ( max_ngrams*len(token_ids) ) mapping ngram to the original token position.
        """
        assert token_ids.dim() == 1

        token_len = token_ids.shape[0]
        token_ids = token_ids[token_ids != pad_token_id]
        token_ids_list = token_ids.tolist()
        # token_len = len(token_ids_list)

        _cur_ngram_id_idx = 0  # index to ngram_ids
        ret_val: EncodedNgram = {
            "ngram_ids": torch.zeros(self._max_ngrams, dtype=torch.int),
            "ngram_attention_mask": torch.zeros(self._max_ngrams, dtype=torch.int),
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
                ret_val["ngram_attention_mask"][_cur_ngram_id_idx] = 1
                ret_val["ngram_position_matrix"][
                    q : q + ngram_len, _cur_ngram_id_idx
                ] = 1
                _cur_ngram_id_idx += 1
                if _cur_ngram_id_idx >= self._max_ngrams:
                    return ret_val
        return ret_val

    def set_max_ngram_match(self, max_ngrams: int):
        self._max_ngrams = max_ngrams

    @classmethod
    def from_list(cls, ngram_list: list[list[int]], max_ngrams: int = 20):
        """Instantiate a new :class:`~dnazen.ngram.NgramTokenizer` from list of ngram list.

        Do not question why this method is even here. It is 20:00 and the day after tmw is Spring Festival.
        It's the only and least ugly way to finish my one-time-only-but-very-urgent job.
        """
        vocab_dict = {}
        min_ngram_len = 100
        max_ngram_len = 0
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
        min_pmi: float | None = None,
        min_token_count: int | None = None,
        min_ngram_freq: int = 5,
        num_workers: int = 64,
        returns_freq: bool = False,
        method: Literal["pmi", "freq"] = "pmi",
    ):
        """Train the ngram encoder using the given token, config and method.

        Args:
            tokens (list[list[int]]): list of pre-tokenized tokens.
            min_pmi (float): the pmi threshold of ngram (used only using pmi method)
            min_token_count (int): the minimum token frequency for filtering ngram (used only using pmi method)
            min_ngram_freq (int): the minimum ngram frequency for filtering ngram
            num_workers (int, optional): number of workers for training. Defaults to 64.
            returns_freq (bool, optional): whether return the frequency info. Defaults to false.
            method (Literal["pmi", "freq"], optional): the method to train. Defaults to pmi.
        """
        if self._vocab != {}:
            warnings.warn("the vocab is non-empty. Would be overloaded after training.")

        if method == "pmi":
            if min_pmi is None:
                raise ValueError("min_pmi not set")
            if min_token_count is None:
                raise ValueError("min_token_count not set")

            ngram_finder_config = PmiNgramFinderConfig()
            ngram_finder_config.min_pmi = min_pmi
            ngram_finder_config.max_ngram_len = self._max_ngram_len
            ngram_finder_config.min_ngram_len = self._min_ngram_len
            ngram_finder_config.min_ngram_freq = min_ngram_freq
            ngram_finder_config.min_token_count = min_token_count
            ngram_finder_config.num_workers = num_workers

            self._vocab = find_ngrams_by_pmi(ngram_finder_config, tokens=tokens)
        elif method == "freq":
            if min_pmi is not None:
                print("[Warning] min_pmi not used when using freq method to train.")
            if min_token_count is not None:
                print(
                    "[Warning] min_token_count not used when using freq method to train."
                )

            ngram_finder_config = FreqNgramFinderConfig()
            ngram_finder_config.min_freq = min_ngram_freq
            ngram_finder_config.max_ngram_len = self._max_ngram_len
            ngram_finder_config.min_ngram_len = self._min_ngram_len
            ngram_finder_config.num_workers = num_workers

            self._vocab = find_ngrams_by_freq(ngram_finder_config, tokens=tokens)
        else:
            raise NotImplementedError(f"Method {method} not supported.")

        if returns_freq:
            vocab_freq = deepcopy(self._vocab)
        else:
            vocab_freq = None

        for idx, (k, v) in enumerate(self._vocab.items()):
            self._vocab[k] = idx

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
        ngram_finder_config = PmiNgramFinderConfig()
        ngram_finder_config.min_pmi = min_pmi
        ngram_finder_config.max_ngram_len = self._max_ngram_len
        ngram_finder_config.min_ngram_len = self._min_ngram_len
        ngram_finder_config.min_ngram_freq = min_ngram_freq
        ngram_finder_config.min_token_count = min_token_count
        ngram_finder_config.num_workers = num_workers

        if self._vocab != {}:
            warnings.warn("the vocab is non-empty. Would be overloaded after training.")

        import _ngram

        finder = _ngram.PmiNgramFinder(ngram_finder_config)
        finder.find_ngrams_from_file(fname)
        ngrams: list[list[int]] = finder.get_ngram_list([])
        # del finder  # save memory

        ngram_dict = {}
        for ngram in ngrams:
            freq = ngram.pop()
            ngram_dict[tuple(ngram)] = freq

        self._vocab = ngram_dict
        if returns_freq:
            vocab_freq = deepcopy(self._vocab)
        else:
            vocab_freq = None

        for idx, (k, v) in enumerate(self._vocab.items()):
            self._vocab[k] = idx

        self._id2ngrams = {}
        for k, v in self._vocab.items():
            self._id2ngrams[v] = k

        return vocab_freq

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

        The identifier is solely based on the ngram vocabulary.
        """
        return hash(frozenset(self._vocab.items()))

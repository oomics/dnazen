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

logger = logging.getLogger(__name__)


class MlmData(TypedDict):
    input_ids: torch.Tensor
    ngram_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ngram_attention_mask: torch.Tensor
    ngram_position_matrix: torch.Tensor
    labels: torch.Tensor


class MlmDataSaved(TypedDict):
    """The Mlm data that is saved on disk."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class MlmDataConfig(TypedDict):
    whole_ngram_masking: bool
    mlm_prob: float
    mlm_data_symlink: str | None
    mlm_data_hash_val: str | None


# --- utils ---


def _save_core_ngrams(
    path: str,
    core_ngrams: set[tuple[int, ...]],
):
    core_ngrams_ = [list(ngram) for ngram in list(core_ngrams)]

    with open(path, "w") as f:
        json.dump(core_ngrams_, f)


def _load_core_ngrams(
    path: str,
):
    with open(path, "r") as f:
        ngrams = json.load(f)

    return set(tuple(ngram) for ngram in ngrams)


class TokenMasker:
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
        self.core_ngrams = core_ngrams
        self.whole_ngram_masking = whole_ngram_masking
        self.CLS = cls_token
        self.SEP = sep_token
        self.PAD = pad_token
        self.MASK = mask_token

        self.core_ngram_min_len = 128
        self.core_ngram_max_len = 0
        for ngram in core_ngrams:
            self.core_ngram_min_len = min(self.core_ngram_min_len, len(ngram))
            self.core_ngram_max_len = max(self.core_ngram_max_len, len(ngram))

        if verbose:
            logger.info(
                f"MlmDataset initialized. "
                f"minimum core ngram len={self.core_ngram_min_len}; "
                f"maximum core ngram len={self.core_ngram_max_len}"
            )

    def create_mlm_predictions(
        self,
        token_seq: torch.Tensor,
        mlm_prob: float,
        vocab_list: list[int],
        ngram_encoder: NgramEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert token_seq.dim() == 1, f"token_seq should be a 1d array, but got dimension {token_seq.dim()}"

        candidate_idxes_list = (
            torch.nonzero(
                (token_seq != self.CLS) & (token_seq != self.SEP) & (token_seq != self.MASK),
                as_tuple=False,
            )
            .squeeze()
            .tolist()
        )

        # get non-candidate indexes
        token_seq_list = token_seq.tolist()
        non_candidiate_idxes = []
        if len(self.core_ngrams) > 0:
            for idx in candidate_idxes_list:
                for len_ in range(self.core_ngram_min_len, self.core_ngram_max_len):
                    if idx + len_ > token_seq.shape[0]:
                        continue
                    # find core ngram matches
                    if tuple(token_seq_list[idx : idx + len_]) in self.core_ngrams:
                        non_candidiate_idxes += range(idx, idx + len_)

        masked_token_seq = token_seq.clone()
        labels = torch.empty_like(token_seq).fill_(-100)

        # tries to mask whole ngram
        if self.whole_ngram_masking:
            ngram_mask = torch.zeros_like(token_seq, dtype=torch.bool)
            for ngram, idx in ngram_encoder.get_matched_ngrams(token_seq, pad_token_id=self.PAD):
                num = random.random()
                if num < mlm_prob:
                    ngram_mask[idx : idx + len(ngram)] = 1
            num_ngram_mask = int(ngram_mask.sum().item())
            rand = torch.rand_like(token_seq, dtype=torch.float)
            ngram_mask_80 = ngram_mask & (rand < 0.8)
            ngram_mask_10 = ngram_mask & (rand >= 0.8) & (rand < 0.9)
            ngram_mask_10_rand = ngram_mask & (rand >= 0.9)
            masked_token_seq[ngram_mask_80] = self.MASK
            masked_token_seq[ngram_mask_10] = token_seq[ngram_mask_10]
            masked_token_seq[ngram_mask_10_rand] = torch.tensor(
                random.choices(vocab_list, k=int(ngram_mask_10_rand.sum().item())),
                dtype=torch.long,
            )
        else:
            num_ngram_mask = 0

        seq_len = len(candidate_idxes_list)
        candidate_idxes = torch.tensor(
            [idx for idx in candidate_idxes_list if idx not in non_candidiate_idxes],
            dtype=torch.int32,
        )
        len_prop = (seq_len - num_ngram_mask) / len(candidate_idxes)
        mlm_prob *= len_prop  # modify the mlm prob

        # Create a mask for candidate indices
        candidate_mask = torch.zeros_like(token_seq, dtype=torch.bool)
        candidate_mask[candidate_idxes] = 1

        # Sample which tokens to mask
        mask_prob = torch.full_like(token_seq, mlm_prob, dtype=torch.float)
        mask_prob[~candidate_mask] = 0
        mask_mask = torch.bernoulli(mask_prob).bool()  # 1 = sample; 0 = not sample

        # Apply masking
        labels[mask_mask] = masked_token_seq[mask_mask]
        rand = torch.rand_like(token_seq, dtype=torch.float)
        mask_mask_80 = mask_mask & (rand < 0.8)
        mask_mask_10 = mask_mask & (rand >= 0.8) & (rand < 0.9)
        mask_mask_10_rand = mask_mask & (rand >= 0.9)

        masked_token_seq[mask_mask_80] = self.MASK
        masked_token_seq[mask_mask_10] = token_seq[mask_mask_10]
        masked_token_seq[mask_mask_10_rand] = torch.tensor(
            random.choices(vocab_list, k=int(mask_mask_10_rand.sum().item())),
            dtype=torch.long,
        )

        return masked_token_seq, labels


class MlmDataset(Dataset):
    """
    Dataset for mlm task.

    Do masking when sampling.
    """

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
        """Initialize a Mlm dataset.

        Args:
            tokens (torch.Tensor): tokens from a tokenizer
            attn_mask (torch.Tensor): attention mask from a tokenizer.
            tokenizer (PreTrainedTokenizer): tokenizer from `transformers`.
            ngram_encoder (NgramEncoder): a ngram encoder for encoding.
            mlm_prob (float, optional): The proportion of masked tokens. Defaults to 0.15.
            whole_ngram_masking (bool, optional): whether to perform whole ngram masking from ZEN2. Defaults to False.
            mlm_data_symlink (str, optional): the path to the original mlm data on disk.
                If provided a path, we would try to create a symlink to the original data when saving to save memory.
                Defaults to None.
            verbose (bool, optional): Whether initialize with loggings. Defaults to True.
        """
        super().__init__()
        self.ngram_encoder = ngram_encoder
        self.tokenizer = tokenizer

        # tokenizer things
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
        self.core_ngrams = core_ngrams  # fix breaking change introduced by token_masker
        self.whole_ngram_masking = whole_ngram_masking
        self.attn_mask = attn_mask
        self.mlm_data_symlink = mlm_data_symlink

    @property
    def ngram_vocab_size(self):
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
        with open(data_path, "r") as f:
            texts = f.readlines()

        print(f"tokenizing {len(texts)} lines. model_max_length={tokenizer.model_max_length}")
        outputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
        )
        tokens = outputs["input_ids"]
        attn_mask = outputs["attention_mask"]
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
        """Build dataset from already tokenized data.

        We assume the token_ids are already padded.
        """

        data: MlmDataSaved = torch.load(data_dir)
        token_ids = data["input_ids"]
        attn_mask = data["attention_mask"]

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
        logger.info(f"Saving dataset to {save_dir}...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.warning(f"Directory {save_dir} did not exist and was created.")
        ngram_encoder_path = os.path.join(save_dir, self.NGRAM_ENCODER_FNAME)
        data_path = os.path.join(save_dir, self.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, self.CORE_NGRAMS_FNAME)
        tokenizer_path = os.path.join(save_dir, self.TOKENIZER_DIR)
        config_path = os.path.join(save_dir, self.CONFIG_FNAME)

        self.ngram_encoder.save(ngram_encoder_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        _save_core_ngrams(core_ngram_path, core_ngrams=self.core_ngrams)
        if self.mlm_data_symlink is None:
            mlm_data_hash_val = None
            torch.save(
                {
                    "input_ids": self.tokens,
                    "attention_mask": self.attn_mask,
                },
                data_path,
            )
        else:
            logger.info("Using symlink, calculating md5 value...")
            mlm_data_hash_val = hash_file_md5(self.mlm_data_symlink)
            logger.info(f"Calculation done. value={mlm_data_hash_val}")
            os.symlink(self.mlm_data_symlink, dst=data_path)

        data_cfg: MlmDataConfig = {
            "whole_ngram_masking": self.whole_ngram_masking,
            "mlm_prob": self.mlm_prob,
            "mlm_data_symlink": self.mlm_data_symlink,
            "mlm_data_hash_val": mlm_data_hash_val,
        }
        with open(config_path, "w") as f:
            json.dump(data_cfg, f, indent=2)

    @classmethod
    def from_dir(
        cls,
        save_dir: str,  # path to the save directory
        check_hash: bool = True,
    ):
        ngram_encoder_path = os.path.join(save_dir, cls.NGRAM_ENCODER_FNAME)
        tokenizer_path = os.path.join(save_dir, cls.TOKENIZER_DIR)
        data_path = os.path.join(save_dir, cls.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, cls.CORE_NGRAMS_FNAME)
        data_config_path = os.path.join(save_dir, cls.CONFIG_FNAME)

        with open(data_config_path, "r") as f:
            data_cfg: MlmDataConfig = json.load(f)
        if data_cfg["mlm_data_symlink"] is not None and check_hash:
            assert data_cfg["mlm_data_hash_val"] is not None
            # check hashval
            logger.info("Using the symlink when loading data. Checking the md5 value.")
            hash_identical = check_hash_of_file_md5(data_path, data_cfg["mlm_data_hash_val"])
            if not hash_identical:
                raise ValueError(
                    f"Trying to open file {data_path}, ",
                    "but the original data seems to be modified.",
                )
        elif check_hash:
            logger.warning("Checking hash is not supported when we are not using symlink.")

        data = torch.load(data_path, weights_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return cls(
            tokens=data["input_ids"],
            attn_mask=data["attention_mask"],
            tokenizer=tokenizer,
            ngram_encoder=NgramEncoder.from_file(ngram_encoder_path),
            core_ngrams=_load_core_ngrams(core_ngram_path),
            whole_ngram_masking=data_cfg.get("whole_ngram_masking", False),
            mlm_prob=data_cfg["mlm_prob"],
            mlm_data_symlink=data_cfg["mlm_data_symlink"],
        )

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, index) -> MlmData:
        # do masking during run-time
        input_ids_, labels_ = self.token_masker.create_mlm_predictions(
            token_seq=self.tokens[index],
            vocab_list=list(self.tokenizer.get_vocab().values()),
            mlm_prob=self.mlm_prob,
            ngram_encoder=self.ngram_encoder,
        )

        ngram_encoder_outputs = self.ngram_encoder.encode(
            input_ids_,
            pad_token_id=self.PAD,
        )

        return {
            "input_ids": input_ids_,
            "labels": labels_,
            "attention_mask": self.attn_mask[index],
            "ngram_attention_mask": ngram_encoder_outputs["ngram_attention_mask"],
            "ngram_input_ids": ngram_encoder_outputs["ngram_ids"],
            "ngram_position_matrix": ngram_encoder_outputs["ngram_position_matrix"],
        }

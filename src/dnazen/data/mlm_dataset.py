from typing import TypedDict
import random
import os
import json
import logging

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from dnazen.ngram import NgramEncoder

logger = logging.getLogger(__name__)


class MlmData(TypedDict):
    input_ids: torch.Tensor
    ngram_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ngram_attention_mask: torch.Tensor
    ngram_position_matrix: torch.Tensor
    labels: torch.Tensor


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


class MlmDataset(Dataset):
    """
    Dataset for mlm task.

    Do masking when sampling.
    """

    NGRAM_ENCODER_FNAME = "ngram_encoder.json"
    TOKENIZER_DIR = "tokenizer"
    DATA_FNAME = "data.pt"
    CORE_NGRAMS_FNAME = "core_ngrams.txt"

    def __init__(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder,
        core_ngrams: set[tuple[int, ...]],
        mlm_prob: float = 0.15,
        verbose: bool = True,
    ):
        """Initialize a Mlm dataset.

        Args:
            tokens (torch.Tensor): tokens from a tokenizer
            attn_mask (torch.Tensor): attention mask from a tokenizer.
            tokenizer (PreTrainedTokenizer): tokenizer from `transformers`.
            ngram_encoder (NgramEncoder): a ngram encoder for encoding.
            mlm_prob (float, optional): The proportion of masked tokens. Defaults to 0.15.
            verbose (bool, optional): Whether initialize with loggings. Defaults to True.
        """
        super().__init__()
        self.ngram_encoder = ngram_encoder
        self.tokenizer = tokenizer
        self.core_ngrams = core_ngrams

        # tokenizer things
        self.CLS: int = tokenizer.convert_tokens_to_ids("[CLS]")
        self.SEP: int = tokenizer.convert_tokens_to_ids("[SEP]")
        self.PAD: int = tokenizer.convert_tokens_to_ids("[PAD]")
        self.MASK: int = tokenizer.convert_tokens_to_ids("[MASK]")

        self.core_ngram_min_len = 128
        self.core_ngram_max_len = 0
        for ngram in core_ngrams:
            self.core_ngram_min_len = min(self.core_ngram_min_len, len(ngram))
            self.core_ngram_max_len = max(self.core_ngram_max_len, len(ngram))

        self.mlm_prob = mlm_prob
        self.tokens = tokens
        self.attn_mask = attn_mask

        if verbose:
            logger.info(
                f"MlmDataset initialized. "
                f"minimum core ngram len={self.core_ngram_min_len}; "
                f"maximum core ngram len={self.core_ngram_max_len}"
            )

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
        mlm_prob: float = 0.15,
    ):
        with open(data_path, "r") as f:
            texts = f.readlines()

        print(
            f"tokenizing {len(texts)} tokens. model_max_length={tokenizer.model_max_length}"
        )
        outputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
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
            mlm_prob=mlm_prob,
        )

    def save(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Warning: Directory {save_dir} did not exist and was created.")
        ngram_encoder_path = os.path.join(save_dir, self.NGRAM_ENCODER_FNAME)
        data_path = os.path.join(save_dir, self.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, self.CORE_NGRAMS_FNAME)
        tokenizer_path = os.path.join(save_dir, self.TOKENIZER_DIR)

        self.ngram_encoder.save(ngram_encoder_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        _save_core_ngrams(core_ngram_path, core_ngrams=self.core_ngrams)
        torch.save(
            {
                "mlm_prob": self.mlm_prob,
                "tokens": self.tokens,
                "attention_mask": self.attn_mask,
            },
            data_path,
        )

    @classmethod
    def from_dir(
        cls,
        save_dir: str,  # path to the save directory
    ):
        # data = torch.load(path)
        ngram_encoder_path = os.path.join(save_dir, cls.NGRAM_ENCODER_FNAME)
        tokenizer_path = os.path.join(save_dir, cls.TOKENIZER_DIR)
        data_path = os.path.join(save_dir, cls.DATA_FNAME)
        core_ngram_path = os.path.join(save_dir, cls.CORE_NGRAMS_FNAME)
        data = torch.load(data_path, weights_only=True)

        # tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return cls(
            tokens=data["tokens"],
            attn_mask=data["attention_mask"],
            tokenizer=tokenizer,
            ngram_encoder=NgramEncoder.from_file(ngram_encoder_path),
            core_ngrams=_load_core_ngrams(core_ngram_path),
            mlm_prob=data["mlm_prob"],
        )

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, index) -> MlmData:
        # do masking during run-time
        input_ids_, labels_ = MlmDataset._create_mlm_predictions(
            token_seq=self.tokens[index],
            sep_token=self.SEP,
            cls_token=self.CLS,
            mask_token=self.MASK,
            vocab_list=list(self.tokenizer.get_vocab().values()),
            mlm_prob=self.mlm_prob,
            core_ngrams=self.core_ngrams,
            max_core_ngram_len=self.core_ngram_max_len,
            min_core_ngram_len=self.core_ngram_min_len,
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

    @staticmethod
    def _create_mlm_predictions(
        token_seq: torch.Tensor,
        mlm_prob: float,
        cls_token: int,
        sep_token: int,
        mask_token: int,
        vocab_list: list[int],
        core_ngrams: set[tuple[int, ...]],
        min_core_ngram_len: int,
        max_core_ngram_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert token_seq.dim() == 1, (
            f"token_seq should be a 1d array, but got dimension {token_seq.dim()}"
        )
        CLS = cls_token
        SEP = sep_token
        MASK = mask_token

        candidate_idxes_list = (
            torch.nonzero(
                (token_seq != CLS) & (token_seq != SEP) & (token_seq != MASK),
                as_tuple=False,
            )
            .squeeze()
            .tolist()
        )

        # get non-candidate indexes
        token_seq_list = token_seq.tolist()
        non_candidiate_idxes = []
        for idx in candidate_idxes_list:
            for len_ in range(min_core_ngram_len, max_core_ngram_len):
                if idx + len_ > token_seq.shape[0]:
                    continue
                # find core ngram matches
                if tuple(token_seq_list[idx : idx + len_]) in core_ngrams:
                    non_candidiate_idxes += range(idx, idx + len_)

        seq_len = len(candidate_idxes_list)
        candidate_idxes = torch.tensor(
            [idx for idx in candidate_idxes_list if idx not in non_candidiate_idxes],
            dtype=torch.int32,
        )
        len_prop = seq_len / len(candidate_idxes)
        mlm_prob *= len_prop  # modify the mlm prob

        masked_token_seq = token_seq.clone()
        # labels = torch.zeros_like(token_seq)
        labels = torch.empty_like(token_seq).fill_(-100)

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

        masked_token_seq[mask_mask_80] = MASK
        masked_token_seq[mask_mask_10] = token_seq[mask_mask_10]
        masked_token_seq[mask_mask_10_rand] = torch.tensor(
            random.choices(vocab_list, k=int(mask_mask_10_rand.sum().item())),
            dtype=torch.long,
        )

        return masked_token_seq, labels

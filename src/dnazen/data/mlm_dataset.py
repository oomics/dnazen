from typing import TypedDict
# from random import choice
import random

from tqdm import tqdm
import csv
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from dnazen.ngram import NgramEncoder

class MlmData(TypedDict):
    input_ids: torch.Tensor
    ngram_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    ngram_attention_mask: torch.Tensor
    ngram_position_matrix: torch.Tensor
    labels: torch.Tensor
    

class MlmDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder | None,
        mlm_prob: float = 0.15,
    ):
        super().__init__()
        PAD: int = tokenizer.convert_tokens_to_ids("[PAD]")

        with open(data_path, "r") as f:
            texts = f.readlines()

        print(f"tokenizing {len(texts)} tokens. model_max_length={tokenizer.model_max_length}")
        outputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.num_classes = len(tokenizer.get_vocab().values())
        self.input_ids = torch.empty_like(outputs["input_ids"])
        self.labels    = torch.empty_like(outputs["input_ids"])
        
        CLS = tokenizer.convert_tokens_to_ids("[CLS]")
        SEP = tokenizer.convert_tokens_to_ids("[SEP]")
        MASK= tokenizer.convert_tokens_to_ids("[MASK]")
        vocab_list=list(
            tokenizer
            .get_vocab()
            .values()
        )
        for idx in tqdm(range(self.input_ids.size(0)), desc="masking"):
            input_ids_, labels_ = MlmDataset._create_mlm_predictions(
                outputs["input_ids"][idx], 
                mlm_prob=mlm_prob,
                # max_pred_per_seq=int(512 * mlm_prob),
                # tokenizer=tokenizer,
                cls_token=CLS,
                sep_token=SEP,
                mask_token=MASK,
                vocab_list=vocab_list
            )
            self.input_ids[idx] = input_ids_
            self.labels[idx] = labels_

        self.attention_mask: torch.Tensor = outputs["attention_mask"]  # type: ignore

        if ngram_encoder is None:
            self.ngram_id_list = None
            self.ngram_position_matrix_list = None
            return
        else:
            self.ngram_id_list: list[torch.Tensor] = []
            self.ngram_position_matrix_list: list[torch.Tensor] = []

        for i in tqdm(range(self.input_ids.size(0)), desc="ngram matching"):
            ngram_encoder_outputs = ngram_encoder.encode(self.input_ids[i], pad_token_id=PAD)
            self.ngram_id_list.append(ngram_encoder_outputs["ngram_ids"])
            self.ngram_position_matrix_list.append(
                ngram_encoder_outputs["ngram_position_matrix"]
            )

    def save(
        self,
        path: str
    ):
        torch.save({
            'num_classes': self.num_classes,
            'input_ids': self.input_ids,
            'labels': self.labels,
            'attention_mask': self.attention_mask,
            'ngram_id_list': self.ngram_id_list,
            'ngram_position_matrix_list': self.ngram_position_matrix_list
        }, path)
        
    def __len__(self):
        return self.input_ids.size(0)
    
    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            # "labels": self.labels[index],
            "labels": F.one_hot(self.labels[index], num_classes=self.num_classes).to(dtype=torch.float),
            "attention_mask": self.attention_mask[index],
            "ngram_input_ids": self.ngram_id_list[index],
            "ngram_position_matrix": self.ngram_position_matrix_list[index],
        }
    
    @classmethod
    def from_file(
        cls,
        path: str,
    ):
        data = torch.load(path)
        dataset = cls.__new__(cls)
        dataset.num_classes = data['num_classes']
        dataset.input_ids = data['input_ids']
        dataset.labels = data['labels']
        dataset.attention_mask = data['attention_mask']
        dataset.ngram_id_list = data['ngram_id_list']
        dataset.ngram_position_matrix_list = data['ngram_position_matrix_list']
        return dataset

    @staticmethod
    def _create_mlm_predictions(
        token_seq: torch.Tensor,
        mlm_prob: float,
        # max_pred_per_seq: int,
        # tokenizer: PreTrainedTokenizer,
        cls_token: int,
        sep_token: int,
        mask_token: int,
        vocab_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert token_seq.dim() == 1, f"token_seq should be a 1d array, but got dimension {token_seq.dim()}"
        CLS = cls_token
        SEP = sep_token
        MASK= mask_token

        candidate_idxes = torch.nonzero(
            (token_seq != CLS) & (token_seq != SEP) & (token_seq != MASK), 
            as_tuple=False
        ).squeeze()

        masked_token_seq = token_seq.clone()
        # labels = torch.zeros_like(token_seq)
        labels = torch.empty_like(token_seq).fill_(-100)

        # Create a mask for candidate indices
        candidate_mask = torch.zeros_like(token_seq, dtype=torch.bool)
        candidate_mask[candidate_idxes] = 1

        # Sample which tokens to mask
        mask_prob = torch.full_like(token_seq, mlm_prob, dtype=torch.float)
        mask_prob[~candidate_mask] = 0
        mask_mask = torch.bernoulli(mask_prob).bool()

        # Apply masking
        labels[mask_mask] = masked_token_seq[mask_mask]
        rand = torch.rand_like(token_seq, dtype=torch.float)
        mask_mask_80 = mask_mask & (rand < 0.8)
        mask_mask_10 = mask_mask & (rand >= 0.8) & (rand < 0.9)
        mask_mask_10_rand = mask_mask & (rand >= 0.9)

        masked_token_seq[mask_mask_80] = MASK
        masked_token_seq[mask_mask_10] = token_seq[mask_mask_10]
        masked_token_seq[mask_mask_10_rand] = torch.tensor(random.choices(vocab_list, k=mask_mask_10_rand.sum().item()), dtype=torch.long)

        return masked_token_seq, labels

        # masked_token_seq = torch.empty_like(token_seq)
        # labels = torch.zeros_like(token_seq)
        # masked_token_seq.copy_(token_seq)
        # for idx in random.choices(candidate_idxes, k=num_masked_tokens):
        #     rand_num = random.random()
        #     if rand_num < 0.8: # 80% of the time, replace with [MASK]
        #         masked_token = MASK
        #     elif rand_num < 0.9:
        #         masked_token = token_seq[idx]
        #     else:
        #         masked_token = random.choice(vocab_list)
        #     labels[idx] = masked_token_seq[idx]
        #     masked_token_seq[idx] = masked_token

        # return labels, masked_token_seq
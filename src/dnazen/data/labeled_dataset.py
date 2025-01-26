from typing import TypedDict, no_type_check
import csv
import logging

import torch
from torch.utils.data import Dataset
# from tokenizers import Tokenizer, Encoding

from transformers import PreTrainedTokenizer

from dnazen.ngram import NgramEncoder


class LabeledData(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ZenLabeledData(LabeledData):
    # ngram specific
    ngram_input_ids: torch.Tensor | None
    ngram_attention_mask: torch.Tensor | None
    ngram_position_matrix: torch.Tensor | None


class LabeledDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        ngram_encoder: NgramEncoder | None,
    ):
        super().__init__()
        PAD: int = tokenizer.convert_tokens_to_ids("[PAD]")

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.info("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            raise NotImplementedError("not supported yet.")
            # data is in the format of [text1, text2, label]
            # logging.info("Perform sequence-pair classification...")
            # texts = [[d[0], d[1]] for d in data]
            # labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        outputs = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        self.labels = labels
        self.input_ids: torch.Tensor = outputs["input_ids"]  # type: ignore
        self.attention_mask: torch.Tensor = outputs["attention_mask"]  # type: ignore

        if ngram_encoder is None:
            self.ngram_id_list = None
            self.ngram_attention_mask = None
            self.ngram_position_matrix_list = None
            return

        self.ngram_id_list: list[torch.Tensor] = []
        self.ngram_position_matrix_list: list[torch.Tensor] = []
        self.ngram_attention_mask: list[torch.Tensor] = []

        for i in range(self.input_ids.size(0)):
            ngram_encoder_outputs = ngram_encoder.encode(
                self.input_ids[i], pad_token_id=PAD
            )
            self.ngram_id_list.append(ngram_encoder_outputs["ngram_ids"])
            self.ngram_attention_mask.append(ngram_encoder_outputs["ngram_attention_mask"])
            self.ngram_position_matrix_list.append(
                ngram_encoder_outputs["ngram_position_matrix"]
            )

    def __len__(self):
        return self.input_ids.shape[0]

    # @no_type_check
    def __getitem__(self, i) -> ZenLabeledData | LabeledData:
        if self.ngram_id_list is not None:
            return {
                "input_ids": self.input_ids[i],
                "labels": self.labels[i],
                "attention_mask": self.attention_mask[i],
                "ngram_input_ids": self.ngram_id_list[i],
                "ngram_attention_mask": self.ngram_attention_mask[i],
                "ngram_position_matrix": self.ngram_position_matrix_list[i],
            }
        else:
            return {
                "input_ids": self.input_ids[i],
                "labels": self.labels[i],
                "attention_mask": self.attention_mask[i],
            }

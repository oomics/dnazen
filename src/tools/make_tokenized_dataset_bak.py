"""Tokenize the raw text data using an tokenizer."""

import click

from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from dnazen.data.mlm_dataset import MlmDataSaved


def rm_empty_lines(data_list: list[str]):
    data_text = []
    for d in tqdm(data_list, "removing empty lines"):
        if len(d) != 0:
            data_text.append(d)
    return data_text


@click.command()
@click.option("--data", type=str, help="Path to the raw data in text format.")
@click.option("--tok", type=str, help="Name of tokenizer.", default="zhihan1996/DNABERT-2-117M")
@click.option("-o", "--out", type=str, help="out directory")
def main(data, tok, out):
    with open(data, "r") as f:
        data_text = f.read().split("\n")
    # check if there are empty lines in text
    data_text = rm_empty_lines(data_text)

    tokenizer = AutoTokenizer.from_pretrained(tok)
    print("tokenizing...")
    data_tokenized = tokenizer(
        data_text, return_tensors="pt", padding="longest", return_token_type_ids=False
    )
    del tokenizer  # save mem
    print("saving...")

    input_ids: torch.Tensor = data_tokenized["input_ids"]  # type: ignore
    attn_mask: torch.Tensor = data_tokenized["attention_mask"]  # type: ignore
    print(f"{type(input_ids)=}")
    out_data: MlmDataSaved = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
    }

    torch.save(out_data, out)


if __name__ == "__main__":
    main()

"""Make pretrained dataset from .txt/.fa files."""

import random
from typing import Literal
from argparse import ArgumentParser
import logging

import torch
import click
from transformers import AutoTokenizer, PreTrainedTokenizer

from dnazen.ngram import NgramEncoder
from dnazen.data.mlm_dataset import MlmDataset, _load_core_ngrams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option("--data-source", type=click.Choice(["raw", "tokenized"]), default="raw", help="type of data")
@click.option(
    "-d",
    "--data",
    "data_dir",
    type=str,
    default="/data2/peter/dnazen_pretrain_v3",
    help="path to data file",
)
@click.option(
    "--tok-source", "tokenizer_source", type=click.Choice(["file", "huggingface"]), default="huggingface"
)
@click.option("--tok", "tokenizer_cfg", type=str, default="zhihan1996/DNABERT-2-117M")
@click.option("--ngram", "ngram_file", type=str, help="path to ngram decoder config file.")
@click.option("--core-ngram", type=str, default=None)
@click.option("--max-ngrams", default=30, type=int, help="Maximum number of ngram matches.")
@click.option("--out", "output_dir", type=str, help="Path to save dir.")
@click.option("--seed", type=int, default=42)
def main(
    data_source: Literal["raw", "tokenized"],
    data_dir: str,
    tokenizer_source: Literal["file", "huggingface"],
    tokenizer_cfg: str,
    ngram_file: str,
    core_ngram: str | None,
    max_ngrams: int,
    output_dir: str,
    seed: int,
):
    random.seed(seed)
    assert output_dir is not None

    if tokenizer_source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg)
    else:
        tokenizer = AutoTokenizer.from_file(tokenizer_cfg)
    tokenizer.model_max_length = 256

    # ngram
    ngram_encoder = NgramEncoder.from_file(ngram_file)
    ngram_encoder.set_max_ngram_match(max_ngrams=max_ngrams)

    # core ngram
    if core_ngram is not None:
        core_ngrams = _load_core_ngrams(core_ngram)
    else:
        core_ngrams = set()

    if data_source == "raw":
        val_dataset = MlmDataset.from_raw_data(
            f"{data_dir}/dev.txt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        val_dataset.save(f"{output_dir}/dev")

        train_dataset = MlmDataset.from_raw_data(
            f"{data_dir}/train.txt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        train_dataset.save(f"{output_dir}/train")
    else:
        val_dataset = MlmDataset.from_tokenized_data(
            data_dir=f"{data_dir}/dev.pt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        val_dataset.save(f"{output_dir}/dev")

        train_dataset = MlmDataset.from_tokenized_data(
            data_dir=f"{data_dir}/train.pt",
            tokenizer=tokenizer,
            ngram_encoder=ngram_encoder,
            core_ngrams=core_ngrams,
            mlm_prob=0.20,
        )
        train_dataset.save(f"{output_dir}/train")


if __name__ == "__main__":
    main()

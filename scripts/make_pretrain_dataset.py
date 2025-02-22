"""Make pretrained dataset from .txt/.fa files."""

import random
from typing import Literal
from argparse import ArgumentParser

import torch
import click
from transformers import AutoTokenizer, PreTrainedTokenizer

from dnazen.ngram import NgramEncoder
from dnazen.data.mlm_dataset import MlmDataset, _load_core_ngrams

# TODO: use click instead
parser = ArgumentParser()

parser.add_argument("--data_source",choices=["raw", "tokenized"] type=str, default="raw", help="the type of data.")
parser.add_argument("-d", "--data", type=str, default="/data2/peter/dnazen_pretrain_v3", help="path to data file")
parser.add_argument("--tok_source", choices=["file", "huggingface"], default="huggingface")
parser.add_argument("--tok", type=str, default="zhihan1996/DNABERT-2-117M")
parser.add_argument("--ngram", type=str, help="path to ngram decoder config file.", default="/home/peter/llm_projects/ZENforDNA/resources/ngram-encoder-hg38-gue-v0.json")
parser.add_argument("--core-ngram", type=str, default=None)
parser.add_argument("--max-ngrams", default=30, type=int, help="Maximum number of ngram matches")
parser.add_argument("--out", type=str, help="path to save dir")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)

@click.command()
@click.option("--data-source", type=click.Choice(["raw", "tokenized"]),
              default="raw", help="type of data")
@click.option("-d", "--data", "data_dir", type=str, 
              default="/data2/peter/dnazen_pretrain_v3", help="path to data file")
@click.option("--tok-source", type=click.Choice(["file", "huggingface"]), default="huggingface")
def main(data_source: Literal["raw", "tokenized"],
         data_dir: str,
         tok_source: Literal["file", "huggingface"],
         tok: str,
         ngram: str,
         core_ngram: str,
         max_ngrams: int,
         out: str,
         seed: int
         ):
    pass

tokenizer_source = args.tok_source
tokenizer_cfg = args.tok
data_dir = args.data
ngram_file = args.ngram
output_dir = args.out

assert output_dir is not None

if tokenizer_source == "huggingface":
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg)
else:
    tokenizer = AutoTokenizer.from_file(tokenizer_cfg)
tokenizer.model_max_length = 256

# ngram
ngram_encoder = NgramEncoder.from_file(ngram_file)
ngram_encoder.set_max_ngram_match(max_ngrams=args.max_ngrams)

# core ngram
if args.core_ngram is not None:
    core_ngrams = _load_core_ngrams(args.core_ngram)
else:
    core_ngrams = set()

if args.data_source == "raw":
    val_dataset = MlmDataset.from_raw_data(
        f"{data_dir}/dev.txt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=core_ngrams,
        mlm_prob=0.20
    )
    val_dataset.save(f"{output_dir}/dev")

    train_dataset = MlmDataset.from_raw_data(
        f"{data_dir}/train.txt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=core_ngrams,
        mlm_prob=0.20
    )
    train_dataset.save(f"{output_dir}/train")
else:
    val_dataset = MlmDataset.from_tokenized_data(
        data_dir=f"{data_dir}/dev.pt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=core_ngrams,
        mlm_prob=0.20
    )
    val_dataset.save(f"{output_dir}/dev")

    data_tok = torch.load(f"{data_dir}/train.pt")
    train_dataset = MlmDataset.from_tokenized_data(
        data_dir=f"{data_dir}/dev.pt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=core_ngrams,
        mlm_prob=0.20
    )
    train_dataset.save(f"{output_dir}/train")
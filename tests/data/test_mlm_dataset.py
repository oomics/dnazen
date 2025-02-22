import shutil

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from dnazen.data.mlm_dataset import MlmDataset
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
test_csv_file = current_dir + "/resources/test.csv"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        model_max_length=128,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )


@pytest.fixture
def ngram_encoder():
    from dnazen.ngram import NgramEncoder

    return NgramEncoder.from_file(current_dir + "/resources/ngram-encoder.json")


@pytest.mark.usefixtures("tokenizer", "ngram_encoder")
@pytest.mark.parametrize(("whole_ngram_masking"), (True, False))
def test_mlm_dataset(
    tokenizer: PreTrainedTokenizer,
    ngram_encoder,
    whole_ngram_masking: bool,
):
    dataset = MlmDataset.from_raw_data(
        current_dir + "/resources/test_mlm.txt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=set([(1, 10), (23, 24, 25)]),
        whole_ngram_masking=whole_ngram_masking,
    )
    for d in dataset:
        assert isinstance(d, dict)

    # test save
    dataset.save(current_dir + "/.cache")

    dataset = MlmDataset.from_dir(current_dir + "/.cache")
    for d in dataset:
        pass
    shutil.rmtree(current_dir + "/.cache")


@pytest.mark.usefixtures("tokenizer", "ngram_encoder")
@pytest.mark.parametrize(("whole_ngram_masking"), (True, False))
def test_mlm_dataset_from_tokenized(
    tokenizer: PreTrainedTokenizer,
    ngram_encoder,
    whole_ngram_masking: bool,
):
    dataset = MlmDataset.from_tokenized_data(
        current_dir + "/resources/test_mlm.pt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=set([(1, 10), (23, 24, 25)]),
        whole_ngram_masking=whole_ngram_masking,
    )
    for d in dataset:
        assert isinstance(d, dict)

    # test save
    dataset.save(current_dir + "/.cache")

    dataset = MlmDataset.from_dir(current_dir + "/.cache")
    for d in dataset:
        pass
    shutil.rmtree(current_dir + "/.cache")

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
def test_labeled_dataset(
    tokenizer: PreTrainedTokenizer,
    ngram_encoder,
):
    dataset = MlmDataset.from_raw_data(
        current_dir + "/resources/test_mlm.txt",
        tokenizer=tokenizer,
        ngram_encoder=ngram_encoder,
        core_ngrams=set([(1, 10), (23, 24, 25)]),
    )
    for d in dataset:
        assert isinstance(d, dict)

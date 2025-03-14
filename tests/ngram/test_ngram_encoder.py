import os
import random

import pytest

from dnazen.ngram import NgramEncoder

random.seed(42)
current_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def tokens() -> list[list[int]]:
    NUM_TOKEN_SEQ = 10_000

    MIN_SEQ_LEN = 512
    MAX_SEQ_LEN = 1_000
    token_seq_list = []
    for _ in range(NUM_TOKEN_SEQ):
        seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
        token_seq = [1, 2, 3] + [random.randint(0, 128) for _ in range(seq_len)]
        token_seq_list.append(token_seq)
    return token_seq_list


@pytest.fixture(autouse=True)
def tokens2() -> list[list[int]]:
    NUM_TOKEN_SEQ = 10_000

    token_seq_list = []
    for _ in range(NUM_TOKEN_SEQ):
        token_seq = [1, 2, 1, 2, 1, 2, 1, 2]
        token_seq_list.append(token_seq)
    token_seq_list.append([4, 5, 6])
    return token_seq_list


@pytest.mark.parametrize("method", ["pmi", "freq", "pmi-ref"])
def test_train(tokens, method):
    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    if method in ["pmi", "pmi-ref"]:
        ngram_encoder.train(tokens, min_pmi=1, min_token_count=3, min_ngram_freq=3, method=method)
    else:
        ngram_encoder.train(tokens, min_ngram_freq=5, method=method)

    assert ngram_encoder.get_vocab().get((1, 2, 3)) is not None, (
        "Should have (1,2,3) in vocab",
        f"The vocab={ngram_encoder.get_vocab()}",
    )


@pytest.mark.parametrize("num_workers", [1, 16])
def test_freq_count(tokens2, num_workers):
    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_freq = ngram_encoder.train(
        tokens2,
        min_ngram_freq=5,
        method="freq",
        num_workers=num_workers,
        returns_freq=True,
    )
    assert ngram_freq is not None, "Should return frequency"
    for k, v in ngram_freq.items():
        assert len(k) <= 6, f"key {k}'s length should exceed 6"
        assert v >= 5, f"key {k} has frequency {v}, but should be >= 5."

    assert ngram_freq.get((1, 2, 1, 2, 1, 2)) == 10_000, (
        f"(1,2,1,2,1,2) should have freq 10_000, but got {ngram_freq.get((1, 2, 1, 2))}. freq={ngram_freq}"
    )
    assert ngram_freq.get((4, 5)) is None


@pytest.mark.parametrize(
    "fname",
    [
        current_dir + "/resources/tokenized_data.txt",
        current_dir + "/resources/tokenized_data2.txt",
        current_dir + "/resources/tokenized_data3.txt",
    ],
)
def test_train_from_file(fname: str):
    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_encoder.train_from_file(
        fname,
        min_pmi=1,
        min_token_count=2,
        min_ngram_freq=2,
    )


def test_train_from_file_should_fail():
    fname = current_dir + "/resources/tokenized_data_should_fail.txt"
    with pytest.raises(ValueError):
        ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
        ngram_encoder.train_from_file(
            fname,
            min_pmi=1,
            min_token_count=2,
            min_ngram_freq=2,
        )


# @pytest.mark.skip(reason="two algos are now different!")
@pytest.mark.parametrize(
    "fname",
    [
        current_dir + "/resources/tokenized_data.txt",
        current_dir + "/resources/tokenized_data2.txt",
        current_dir + "/resources/tokenized_data3.txt",
    ],
)
def test_train_differential_test(fname):
    # test if two method of training is different
    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_encoder.train_from_file(
        fname,
        min_pmi=1,
        min_token_count=2,
        min_ngram_freq=2,
    )

    tokens = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                tokens.append([])
            else:
                tokens.append([int(s) for s in line.split(":")])
    ngram_encoder2 = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_encoder2.train(
        tokens,
        min_pmi=1,
        min_token_count=2,
        min_ngram_freq=2,
    )

    assert ngram_encoder.get_id() == ngram_encoder2.get_id()


@pytest.mark.parametrize(
    "fname",
    [
        current_dir + "/resources/tokenized_data.txt",
        # TODO: remove the below data
        # "/data1/peter/pretrain-tokenized-hg38-all-gue-all.txt"
    ],
)
def test_train_match_ref_impl(fname):
    from ngram_encoder_ref import FindNgrams
    # test if two method of training is different

    tokens = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                pass
            else:
                tokens.append([int(s) for s in line.split(":")])

    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_freq = ngram_encoder.train(
        tokens,
        min_pmi=1,
        min_token_count=4,
        min_ngram_freq=4,
        num_workers=100,
        returns_freq=True,
    )
    assert ngram_freq is not None
    print("finding ngram with pure python...")
    ngram_finder = FindNgrams(min_count=3, min_pmi=1)
    ngram_finder.find_ngrams_pmi(tokens, n=5, freq_threshold=3)

    target_ngrams = ngram_finder.ngrams
    k2del = []
    for k in target_ngrams.keys():
        if len(k) == 1:
            k2del.append(k)
    for k in k2del:
        del target_ngrams[k]
    target_ngrams = dict(sorted(target_ngrams.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))

    # save_ngram_dict_to_dir(ngram_freq, dir=current_dir + "/resources/my-ngram-freq.json")
    # save_ngram_dict_to_dir(target_ngrams, dir=current_dir + "/resources/target-ngram-freq.json")

    assert ngram_freq == target_ngrams, f"ngram_freq={ngram_freq}; actual should be {target_ngrams}"

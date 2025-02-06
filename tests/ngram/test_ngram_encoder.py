import pytest
import random

from dnazen.ngram import NgramEncoder

random.seed(42)


@pytest.fixture(autouse=True)
def tokens() -> list[list[int]]:
    NUM_TOKEN_SEQ = 10_000

    MIN_SEQ_LEN = 512
    MAX_SEQ_LEN = 1_000
    token_seq_list = []
    for i in range(NUM_TOKEN_SEQ):
        seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
        token_seq = [1, 2, 3] + [random.randint(0, 128) for _ in range(seq_len)]
        token_seq_list.append(token_seq)
    return token_seq_list


def test_train(tokens):
    ngram_encoder = NgramEncoder({}, min_ngram_len=2, max_ngram_len=6, max_ngrams=20)
    ngram_encoder.train(
        tokens,
        min_pmi=1,
        min_token_count=3,
        min_ngram_freq=3,
    )

    assert ngram_encoder.get_vocab().get((1, 2, 3)) is not None, (
        "Should have (1,2,3) in vocab"
    )

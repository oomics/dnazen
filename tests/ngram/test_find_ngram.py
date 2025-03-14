import os

import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.parametrize(
    "fname",
    [
        current_dir + "/resources/tokenized_data.txt",
    ],
)
def test_find_ngrams_by_pmi_ref_impl(fname):
    from dnazen.ngram._find_ngram import find_ngrams_by_pmi
    from _ngram import PmiNgramFinderConfig

    config = PmiNgramFinderConfig()
    config.max_ngram_len = 6
    config.min_ngram_freq = 2
    config.min_ngram_len = 2
    config.min_pmi = 1
    config.min_token_count = 2
    config.num_workers = 1
    
    tokens = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                pass
            else:
                tokens.append([int(s) for s in line.split(":")])

    ngram_dict = find_ngrams_by_pmi(config, tokens, use_ref_impl=True)
    print(ngram_dict)
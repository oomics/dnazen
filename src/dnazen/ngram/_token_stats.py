import os
import json

from transformers import PreTrainedTokenizer

from _ngram import NgramFinderConfig, DnaNgramFinder


def analyze_token_stats(
    tokens: list[list[int]],
    tokenizer: PreTrainedTokenizer,
    min_ngram_len: int,
    max_ngram_len: int,
    min_pmi: float,
    min_token_count: int,
    min_ngram_freq: int,
    num_workers: int = 64,
    save_dir: str = ".cache",
):
    ngram_finder_config = NgramFinderConfig()
    ngram_finder_config.min_pmi = min_pmi
    ngram_finder_config.max_ngram_len = max_ngram_len
    ngram_finder_config.min_ngram_len = min_ngram_len
    ngram_finder_config.min_ngram_freq = min_ngram_freq
    ngram_finder_config.min_token_count = min_token_count
    ngram_finder_config.num_workers = num_workers

    finder = DnaNgramFinder(ngram_finder_config)
    finder.find_ngrams_batched(tokens)

    ngrams: list[list[int]] = finder.get_ngram_list([])

    ngram_dict: dict[tuple[int, ...], int] = {}
    for ngram in ngrams:
        freq = ngram.pop()
        ngram_dict[tuple(ngram)] = freq

    ngram_decoded = {}
    for k, v in ngram_dict.items():
        k_ = tokenizer.decode(list(k))
        ngram_decoded[k_] = v

    token_decoded = {}
    for k, v in finder.token_freq.items():
        k_ = tokenizer.decode(k)
        token_decoded[k_] = v

    sorted_ngram_decoded = dict(
        sorted(ngram_decoded.items(), key=lambda item: item[1], reverse=True)
    )
    sorted_token_decoded = dict(
        sorted(token_decoded.items(), key=lambda item: item[1], reverse=True)
    )

    # save to dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/token-freq.txt", "w") as f:
        json.dump(sorted_token_decoded, f)
    with open(save_dir + "/ngram-freq.txt", "w") as f:
        json.dump(sorted_ngram_decoded, f)

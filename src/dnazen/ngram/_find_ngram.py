import _ngram
from _ngram import PmiNgramFinderConfig, FreqNgramFinderConfig


def find_ngrams_by_pmi(
    ngram_finder_config: PmiNgramFinderConfig,
    tokens: list[list[int]],
) -> dict[tuple[int, ...], int]:
    """
    Finds n-grams in a list of tokens based on the provided configuration.

    Args:
        ngram_finder_config (PmiNgramFinderConfig): Configuration for the n-gram finder.
        tokens (list[int]): List of integer tokens to find n-grams in.
    Returns:
        dict[tuple, int]: A dictionary where keys are n-grams (as tuples of integers) and values are their frequencies.
    """
    finder = _ngram.PmiNgramFinder(ngram_finder_config)
    finder.find_ngrams_batched(tokens)
    ngrams: list[list[int]] = finder.get_ngram_list([])

    ngram_dict = {}
    for ngram in ngrams:
        freq = ngram.pop()
        ngram_dict[tuple(ngram)] = freq

    return ngram_dict


def find_ngrams_by_freq(
    ngram_finder_config: FreqNgramFinderConfig, tokens: list[list[int]]
) -> dict[tuple[int, ...], int]:
    finder = _ngram.FreqNgramFinder(ngram_finder_config)
    finder.find_ngrams_batched(tokens)
    ngrams: list[list[int]] = finder.get_ngram_list([])

    ngram_dict = {}
    for ngram in ngrams:
        freq = ngram.pop()
        ngram_dict[tuple(ngram)] = freq

    return ngram_dict

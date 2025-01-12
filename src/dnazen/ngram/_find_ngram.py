import _ngram
from _ngram import NgramFinderConfig

# class NgramFinderConfig(_ngram.NgramFinderConfig):
#     @classmethod
#     def from_dict(cls, d: dict):
#         instance = cls()
#         for key, value in d.items():
#             if not getattr(instance, key):
#                 raise ValueError(f"attribute {key} not found in `NgramFinderConfig`")

#             setattr(instance, key, value)
#         return instance


def find_ngrams(
    ngram_finder_config: NgramFinderConfig,
    tokens: list[list[int]],
) -> dict[tuple[int, ...], int]:
    """
    Finds n-grams in a list of tokens based on the provided configuration.

    Args:
        ngram_finder_config (NgramFinderConfig): Configuration for the n-gram finder.
        tokens (list[int]): List of integer tokens to find n-grams in.
    Returns:
        dict[tuple, int]: A dictionary where keys are n-grams (as tuples of integers) and values are their frequencies.
    """
    finder = _ngram.DnaNgramFinder(ngram_finder_config)
    finder.find_ngrams_batched(tokens)
    ngrams: list[list[int]] = finder.get_ngram_list([])

    ngram_dict = {}
    for ngram in ngrams:
        freq = ngram.pop()
        ngram_dict[tuple(ngram)] = freq

    return ngram_dict

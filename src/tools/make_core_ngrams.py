import click

from dnazen.data.mlm_dataset import _save_core_ngrams
from dnazen.ngram import NgramEncoder


@click.command()
@click.option(
    "--ngram_file1", help="Path to the first ngram encoder file.", required=True,
)
@click.option(
    "--ngram_file2", help="Path to the second ngram encoder file.", default=None,
)
@click.option(
    "--out", help="Output directory for the core ngrams.",
)
def main(ngram_file1, ngram_file2, out):
    if ngram_file2 is not None:
        ngram_encoder1 = NgramEncoder.from_file(ngram_file1)
        ngram_encoder2 = NgramEncoder.from_file(ngram_file2)

        core_ngrams = set(ngram_encoder1.get_vocab().keys()) & set(ngram_encoder2.get_vocab().keys())
    else:
        ngram_encoder1 = NgramEncoder.from_file(ngram_file1)

        core_ngrams = set(ngram_encoder1.get_vocab().keys())

    print("number of core ngrams=", len(core_ngrams))

    _save_core_ngrams(out, core_ngrams=core_ngrams)

if __name__ == "__main__":
    main()

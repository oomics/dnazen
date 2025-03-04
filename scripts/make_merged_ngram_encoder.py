import click

from dnazen.ngram import NgramEncoder

def _parse_comma_separated(values):
    values = values.split(",")
    return [
        val.strip(" ") for val in values
    ]

@click.command()
@click.option(
    "--ngram-files", 
    type=str,
    multiple=True,
    help="Path to the ngram encoder file.",
)
@click.option(
    "-o", "--out",
    type=str,
    help="Directory to ngram encoder configuration. Should be ended with `.json`."
)
def main(ngram_files, out):
    ngram_encoders: list[NgramEncoder] = [
        NgramEncoder.from_file(ngram_file) for ngram_file in ngram_files
    ]
    merged_keys = set().union(*[set(ngram_encoder.get_vocab().keys()) for ngram_encoder in ngram_encoders])

    print("number of merged keys=", len(merged_keys))
    _vocab = {}
    for idx, k in enumerate(merged_keys):
        _vocab[k] = idx

    min_ngram_len = min(*[ngram_encoder._min_ngram_len for ngram_encoder in ngram_encoders])
    max_ngram_len = max(*[ngram_encoder._max_ngram_len for ngram_encoder in ngram_encoders])
    new_ngram_encoder = NgramEncoder(
        _vocab, min_ngram_len, max_ngram_len, max_ngrams=ngram_encoders[0]._max_ngrams
    )
    new_ngram_encoder.save(out)


if __name__ == "__main__":
    main()
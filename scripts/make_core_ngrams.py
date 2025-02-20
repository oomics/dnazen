import click

from dnazen.data.mlm_dataset import _load_core_ngrams, _save_core_ngrams
from dnazen.ngram import NgramEncoder



@click.command()
@click.option('--ngram_file1', default="/home/peter/llm_projects/ZENforDNA/resources/ngram-encoder-hg38-gue-v0.json", help='Path to the first ngram encoder file.')
@click.option('--ngram_file2', default="/home/peter/llm_projects/ZENforDNA/resources/ngram-encoder-gue-all-v0.json", help='Path to the second ngram encoder file.')
@click.option('--out', default="/home/peter/llm_projects/ZENforDNA/resources/core_ngrams2.txt", help='Output directory for the core ngrams.')
def main(ngram_file1, ngram_file2, out):
    ngram_encoder1 = NgramEncoder.from_file(ngram_file1)
    ngram_encoder2 = NgramEncoder.from_file(ngram_file2)

    overlap_keys = set(ngram_encoder1.get_vocab().keys()) & set(ngram_encoder2.get_vocab().keys())
    print("number of overlap keys=", len(overlap_keys))

    _save_core_ngrams(out, core_ngrams=overlap_keys)

if __name__ == '__main__':
    main()
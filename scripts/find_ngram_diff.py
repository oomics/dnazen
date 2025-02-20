import click
import json

from transformers import AutoTokenizer, PreTrainedTokenizer

from dnazen.ngram import NgramEncoder

def token_id_to_text(
    ngrams: set[str],
    tokenizer: PreTrainedTokenizer
)->list[str]:
    ret = []
    
    for ngram in ngrams:
        ngram_ = [int(s) for s in ngram.split(":")]
        ret.append(
            tokenizer.decode(ngram_)
        )
    return ret

@click.command()
@click.option("--n-gram1")
@click.option("--n-gram2")
@click.option("--tok", help="name of tokenizer")
@click.option("--out")
def main(
    n_gram1, n_gram2, tok, out
):
    tokenizer = AutoTokenizer.from_pretrained(
        tok
    )
    
    with open(n_gram1, "r") as f:
        ngram1 = json.load(f)["vocab"]

    with open(n_gram2, "r") as f:
        ngram2 = json.load(f)["vocab"]

    overlap_keys = set(ngram1.keys()) & set(ngram2.keys())

    ngram1_only = set(ngram1.keys()) - overlap_keys
    ngram2_only = set(ngram2.keys()) - overlap_keys

    ngram1_text = token_id_to_text(
        ngram1_only, tokenizer 
    )
    ngram2_text = token_id_to_text(
        ngram2_only, tokenizer
    )
    print("dumping...")
    print("len of first ngrams=", len(ngram1_text))
    print("len of second ngrams=", len(ngram2_text))
    with open(out + "/ngram-1-only.txt", "w") as f:
        json.dump(ngram1_text, f)
    with open(out + "/ngram-2-only.txt", "w") as f:
        json.dump(ngram2_text, f)
    

main()
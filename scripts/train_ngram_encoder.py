from argparse import ArgumentParser

import click
from transformers import AutoTokenizer, AutoModel

from dnazen.ngram import NgramEncoder

from utils.datas import get_all_gue_data, get_all_hg38_data, get_useful_gue_data, get_multi_species_data

DATA_DIR = "/data1/peter"

def get_rvs_complementary(text):
    tmp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([tmp[c] for c in reversed(text)])

def get_gue_all():
    return get_useful_gue_data(
        f"{DATA_DIR}/GUE",
        type="train"
    ) + get_useful_gue_data(
        f"{DATA_DIR}/GUE",
        type="dev"
    ) + get_useful_gue_data(
        f"{DATA_DIR}/GUE",
        type="test"
    )

def get_gue_test():
    return get_useful_gue_data(
        f"{DATA_DIR}/GUE",
        type="test"
    )

def get_hg38():
    return get_all_hg38_data(
        f"{DATA_DIR}/hg38.fa"
    )

def get_hg38_hg38rvs():
    data_hg38 = get_all_hg38_data(
        f"{DATA_DIR}/hg38.fa"
    )
    data_hg38_rvs = [get_rvs_complementary(d) for d in data_hg38]
    return data_hg38 + data_hg38_rvs

def get_multi_species_all():
    return get_multi_species_data(
        f"{DATA_DIR}/dev.txt"
    ) + get_multi_species_data(
        f"{DATA_DIR}/train.txt"
    )

def get_all_data():
    return get_multi_species_all() + get_gue_all() + get_hg38_hg38rvs()

data_mapping = {
    "all": get_all_data,
    "gue_all": get_gue_all,
    "gue_test": get_gue_test,
    "hg38":get_hg38,
    "hg38_all":get_hg38_hg38rvs,
    "mspecies":get_multi_species_all,
}

def get_tokenized_data(data_types: list[str], tok_name: str = "zhihan1996/DNABERT-2-117M"):
    print("Getting data...")
    if 'all' in data_types and len(data_types) > 1:
        data_types = ['all']
        print("[Warning] 'all' includes all data types. Other specified data types will be ignored.")
    all_data = []
    for data_type in data_types:
        data_func = data_mapping[data_type]
        all_data.extend(data_func())

    # calculate the total length of data
    total_len = 0
    for d in all_data:
        total_len += len(d)

    print(f"Tokenizing (length of all data={total_len})...")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True, use_fast=True)
    data_tokenized: list[list[int]] = tokenizer(all_data)["input_ids"]  # type: ignore
    for idx, data in enumerate(data_tokenized):
        data.pop(0)
        data.pop(-1)
    return data_tokenized

@click.command()
@click.option("-s", "--source", type=click.Choice(["raw", "tokenized"]), help="Data source: raw or tokenized.", default="raw")
@click.option("-d", "--data", type=str, 
              help="""(when raw) Comma-separated list of data types to use. Valid data types: all, gue_all, gue_test, hg38, hg38_all, mspecies.
(When tokenized) directory of alread-tokenized data.
""")
@click.option("-o", '--out', help='Save dir for ngram encoder', type=str)
@click.option("--tok-name", type=str, help="name of tokenizer", default="zhihan1996/DNABERT-2-117M")
@click.option("--min-ngram", help="Minimum ngram frequency", type=click.IntRange(1, 100), default=5)
@click.option("--min-ngram-len", help="Minimum ngram length", type=click.IntRange(1, 100), default=2)
@click.option("--max-ngram-len", help="Minimum ngram length", type=click.IntRange(1, 100), default=5)
@click.option("--min-tok", help="Minimum token count", type=click.IntRange(1, 100), default=5)
@click.option("--min-pmi", help="Minimum pmi", type=float, default=5)
@click.option("--mem-efficient", help="Memory efficient", type=bool, default=False)
@click.option("--train-method", help="Training method. freq or pmi", type=click.Choice(["pmi", "freq"]), default="pmi")
@click.option("--out-freq", help="The out directory for frequency. Would not output freq info if not specified.", type=str, default=None)
def main(source:str, data: str, out: str, tok_name: str,
    min_ngram: int, min_ngram_len: int, max_ngram_len: int, min_tok: int, min_pmi: float, mem_efficient: bool, 
    train_method: str,
    out_freq: str | None
    ):
    print("min_token_count=", min_tok)
    returns_freq = (out_freq != None)
    
    if not mem_efficient:
        if source == "raw":
            data_types = data.split(',')
            data_tokenized: list[list[int]] = get_tokenized_data(data_types, tok_name=tok_name)
        else:
            import tqdm
            data_tokenized: list[list[int]] = []
            with open(data, "r") as f:
                lines = f.readlines()
                for line in tqdm.tqdm(lines, "converting data to tokens"):
                    line_ = line.replace("\n", "")
                    if len(line_) > 0:
                        data_tokenized.append(
                            [int(n) for n in line_.split(":")]
                        )
            del lines # save mem

        ngram_encoder = NgramEncoder(vocab_dict={}, min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len, max_ngrams=20)
        ngram_freq = ngram_encoder.train(
            data_tokenized, min_pmi=min_pmi, min_token_count=min_tok,
            secondary_filter=False, # set to false for now
            min_ngram_freq=min_ngram, num_workers=100, returns_freq=returns_freq,
            method=train_method
            )
        ngram_encoder.save(out)
    else:
        if source == "raw":
            print("[Warning] cannot use raw data in mem-efficient mode.")
            exit(1)
        if train_method != "pmi":
            print("[Warning] Only pmi is support for mem-efficient version.")
        
        ngram_encoder = NgramEncoder(vocab_dict={}, min_ngram_len=min_ngram_len, max_ngram_len=max_ngram_len, max_ngrams=20)
        ngram_freq = ngram_encoder.train_from_file(data, min_pmi=min_pmi, min_token_count=min_tok, min_ngram_freq=min_ngram, num_workers=100, returns_freq=returns_freq)
        ngram_encoder.save(out)

    if out_freq is None:
        return

    import json
    with open(out_freq, "w") as f:
        output = {}
        for k, v in ngram_freq.items():
            k_ = ":".join([str(num) for num in list(k)])
            output[k_] = v
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    main()
import click
from transformers import AutoTokenizer, AutoModel

from dnazen.ngram import NgramEncoder

from utils.datas import (
    get_all_gue_data,
    get_all_hg38_data,
    get_useful_gue_data,
    get_multi_species_data,
)

DATA_DIR = "/data1/peter"


def get_rvs_complementary(text):
    tmp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([tmp[c] for c in reversed(text)])


def get_gue_all():
    return (
        get_useful_gue_data(f"{DATA_DIR}/GUE", type="train")
        + get_useful_gue_data(f"{DATA_DIR}/GUE", type="dev")
        + get_useful_gue_data(f"{DATA_DIR}/GUE", type="test")
    )


def get_gue_test():
    return get_useful_gue_data(f"{DATA_DIR}/GUE", type="test")


def get_hg38():
    return get_all_hg38_data(f"{DATA_DIR}/hg38.fa")


def get_hg38_hg38rvs():
    data_hg38 = get_all_hg38_data(f"{DATA_DIR}/hg38.fa")
    data_hg38_rvs = [get_rvs_complementary(d) for d in data_hg38]
    return data_hg38 + data_hg38_rvs


def get_multi_species_all():
    return get_multi_species_data(f"{DATA_DIR}/dev.txt") + get_multi_species_data(f"{DATA_DIR}/train.txt")


def get_all_data():
    return get_multi_species_all() + get_gue_all() + get_hg38_hg38rvs()


data_mapping = {
    "all": get_all_data,
    "gue_all": get_gue_all,
    "gue_test": get_gue_test,
    "hg38": get_hg38,
    "hg38_all": get_hg38_hg38rvs,
    "mspecies": get_multi_species_all,
}


def get_tokenized_data(
    data_types: list[str],
):
    print("Getting data...")
    if "all" in data_types and len(data_types) > 1:
        data_types = ["all"]
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
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True, use_fast=True
    )
    data_tokenized: list[list[int]] = tokenizer(
        all_data, return_attention_mask=False, return_token_type_ids=False
    )["input_ids"]  # type: ignore
    for idx, data in enumerate(data_tokenized):
        data.pop(0)
        data.pop(-1)
    return data_tokenized


@click.command()
@click.option(
    "-d",
    "--data",
    type=str,
    help="Comma-separated list of data types to use. Valid data types: all, gue_all, gue_test, hg38, hg38_all, mspecies.",
)
@click.option(
    "-o",
    "--out",
    default="/data3/peter/pretrain-tokenized.txt",
    help="Save dir for ngram encoder",
    type=str,
)
def main(data: str, out: str):
    data_types = data.split(",")
    data_tokenized: list[list[int]] = get_tokenized_data(data_types)
    with open(out, "w") as f:
        for d in data_tokenized[:-1]:
            f.write((":".join([str(num) for num in d]) + "\n"))


if __name__ == "__main__":
    main()

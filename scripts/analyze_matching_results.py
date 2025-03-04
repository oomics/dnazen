"""Take matching results and analyze the TATA-box found in ngram and the original sequence."""

import re

import click
import pandas as pd

def has_tata(text: str)-> list[int]:
    pattern = r".*TATA[AT][AT]"
    return re.match(pattern, text) is not None


@click.command()
@click.option("-f", "match_res_fname", help="The excel file having matching results.")
@click.option("-o", "out_dir", help="Output directory.")
def main(match_res_fname: str, out_dir: str):
    df_all = pd.read_excel(match_res_fname, sheet_name=None)
    for df_name, df in df_all.items():
        df["has_tata_in_text"] = 0
        df["has_tata_in_ngram"] = 0

        for idx, text in enumerate(df["text"]):
            text = text.replace(" ", "")

            flag_has_tata_in_seq = has_tata(text)
            df["has_tata_in_text"][idx] = flag_has_tata_in_seq
        
        for idx, matched_ngrams in enumerate(df["matched_ngrams"]):
            matched_ngrams = matched_ngrams.replace(" ", "")

            flag_has_tata_in_ngram = has_tata(matched_ngrams)
            df["has_tata_in_ngram"][idx] = flag_has_tata_in_ngram

    with pd.ExcelWriter(out_dir, mode="w") as writer:
        for df_name, df in df_all.items():
            df.to_excel(writer, sheet_name=df_name)

if __name__ == "__main__":
    main()
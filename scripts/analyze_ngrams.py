import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path

import click
from glob import glob
from transformers import AutoTokenizer, PreTrainedTokenizer

from dnazen.ngram import NgramEncoder
from dnazen.ngram.utils import ngram_str_to_tuple, ngram_tuple_to_str


def load_ngram_freq(fname):
    with open(fname, "r") as f:
        d = json.load(f)

    output = {}
    for k, v in d.items():
        output[ngram_str_to_tuple(k)] = v
    return output


def make_meta_data(df: pd.DataFrame):
    return {
        "total_num_matches": int(df["num_matches"].sum()),
        "total_num_data": len(df),
        "num_data_no_match": len(df[df["num_matches"] == 0]),
        "num_data_has_match": len(df[df["num_matches"] > 0]),
    }


def load_gue_results(
    fname,
    ngram_encoder: NgramEncoder,
    ngram_freq_dict: dict | None,
    tokenizer: PreTrainedTokenizer,
    min_freq: int = 5,
    includes_no_match: bool = False,
):
    """_summary_

    Args:
        fname (_type_): the results .csv file generated during finetuning
        ngram_encoder (NgramEncoder): match a consecutive token ids to ngram id
        ngram_freq_dict (dict | None): the corresponding frequency of ngram. Defaults to None.
        tokenizer (PreTrainedTokenizer): tokenizer
        min_freq (int, optional): The frequency threshold for `ngram_freq_dict`. 
            Effective only when `ngram_freq_dict` not None. Defaults to 5.
        includes_no_match (bool, optional): Whether include data with no ngram matches. Defaults to False.
    """
    results = pd.read_csv(fname)

    results["matched_ngrams"] = ""
    results["num_matches"] = 0

    meta_datas = {
        "total_num_matches": 0,
        "total_num_data": 0,
        "num_data_no_match": 0,
        "num_data_has_match": 0,
    }
    for idx, text in enumerate(results["text"]):
        token_ids = tokenizer(text, return_tensors="pt", return_attention_mask=False)["input_ids"].squeeze(
            0
        )
        if ngram_freq_dict is not None:
            matched_ngrams = [
                ngrams
                for ngrams, _ in ngram_encoder.get_matched_ngrams(token_ids)
                if ngram_freq_dict[ngrams] >= min_freq
            ]
        else:
            matched_ngrams = [ngrams for ngrams, _ in ngram_encoder.get_matched_ngrams(token_ids)]
        results["num_matches"][idx] = len(matched_ngrams)
        matched_ngrams_text = [
            tokenizer.decode(list(ngram)).replace("[CLS] ", "").replace(" [SEP]", "")
            for ngram in matched_ngrams
        ]
        results["matched_ngrams"][idx] = ":".join(matched_ngrams_text)

        meta_datas["total_num_matches"] += len(matched_ngrams)
        if len(matched_ngrams) == 0:
            meta_datas["num_data_no_match"] += 1
        else:
            meta_datas["num_data_has_match"] += 1
        meta_datas["total_num_data"] += 1

    results_correct = results[results["actual_label"] == results["prediction_label"]]
    results_wrong = results[results["actual_label"] != results["prediction_label"]]
    del results_correct["Unnamed: 0"]
    del results_wrong["Unnamed: 0"]
    if not includes_no_match:
        results_correct_ = results_correct[results_correct["num_matches"] > 0]
        results_wrong_ = results_wrong[results_wrong["num_matches"] > 0]
    else:
        results_correct_ = results_correct
        results_wrong_ = results_wrong

    return {
        "true": results_correct_, # results correctly classified by llm
        "false": results_wrong_, # results wrongly classified by llm
        "meta_data": meta_datas, # the general matching results of the dataset
        "meta_data_correct": make_meta_data(results_correct), # the general matching results of the dataset (where all data are correctly classified)
        "meta_data_wrong": make_meta_data(results_wrong), # the general matching results of the dataset (where all data are wrongly classified)
    }


def get_data_name(fpath: str):
    return fpath.split("/")[-2]


def put_all_results_into_excel(
    result_files: list[str],
    ngram_freq_dict: dict | None,
    out_dir: str,
    ngram_encoder: NgramEncoder,
    tokenizer: PreTrainedTokenizer,
    freq=5,
    includes_no_match=False,
):
    if ngram_freq_dict is not None:
        META_DATA_OUT_DIR = f"{out_dir}/meta-datas-freq-{freq}.xlsx"
        MATCHING_RESULTS_OUT_DIR = f"{out_dir}/matching-results-freq-{freq}.xlsx"
    else:
        META_DATA_OUT_DIR = f"{out_dir}/meta-datas.xlsx"
        MATCHING_RESULTS_OUT_DIR = f"{out_dir}/matching-results.xlsx"

    all_results = {}
    meta_data_res = {}
    meta_data_res_correct = {}
    meta_data_res_wrong = {}
    for finetune_result_fname in tqdm(result_files):
        name = get_data_name(finetune_result_fname)
        results = load_gue_results(
            finetune_result_fname,
            ngram_encoder=ngram_encoder,
            ngram_freq_dict=ngram_freq_dict,
            tokenizer=tokenizer,
            min_freq=freq,
            includes_no_match=includes_no_match,
        )
        all_results[name] = results
        meta_data_res[name] = results["meta_data"]
        meta_data_res_correct[name] = results["meta_data_correct"]
        meta_data_res_wrong[name] = results["meta_data_wrong"]

    with pd.ExcelWriter(META_DATA_OUT_DIR, mode="w") as writer:
        df = pd.DataFrame.from_dict(meta_data_res, orient="index")
        df.to_excel(writer, sheet_name=f"all")

        df = pd.DataFrame.from_dict(meta_data_res_correct, orient="index")
        df.to_excel(writer, sheet_name=f"all-true")

        df = pd.DataFrame.from_dict(meta_data_res_wrong, orient="index")
        df.to_excel(writer, sheet_name=f"all-false")

    with pd.ExcelWriter(MATCHING_RESULTS_OUT_DIR, mode="w") as writer:
        for name, res in all_results.items():
            res["true"].to_excel(writer, sheet_name=f"{name}-true")
            res["false"].to_excel(writer, sheet_name=f"{name}-false")

@click.command("Analyze ngram matching results for gue data. Requires finetune results.")
@click.option("--out")
@click.option("--tok-name", type=str, help="name of tokenizer", default="zhihan1996/DNABERT-2-117M")
@click.option("--results", "gue_results_dir", 
              help="the finetune results directory. Should have `pred_results.csv` generated by `run_finetune.py`")
@click.option("--ngram", "ngram_encoder_file")
@click.option("--ngram-freq", "ngram_freq_file", type=str, default=None)
def main(out, tok_name, gue_results_dir, ngram_encoder_file, ngram_freq_file: str|None):
    Path(out).mkdir(parents=True, exist_ok=True) # create output dir if not exist
    
    ngram_encoder = NgramEncoder.from_file(ngram_encoder_file)
    print("num ngrams=", len(ngram_encoder.get_vocab().keys()))
    if ngram_freq_file is not None:
        ngram_freq_dict = load_ngram_freq(ngram_freq_file)
        for k, v in ngram_freq_dict.items():
            print("k=", k, "; v=", v)
            break
        assert len(ngram_freq_dict.keys()) == len(ngram_encoder.get_vocab().keys())
    else:
        ngram_freq_dict = None

    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    result_files = glob(gue_results_dir + "/**/pred_results.csv")
    # currently freq is not used since we are no longer using 
    # frequency-based method. This param was used to analyze the change of ngram matching statistics
    # when we are slowly increasing the frequency threshold of ngram.
    for fq in [5]:
        put_all_results_into_excel(
            result_files, 
            ngram_freq_dict=ngram_freq_dict, out_dir=out, 
            ngram_encoder=ngram_encoder, tokenizer=tokenizer, freq=fq
        )



if __name__ == "__main__":
    main()
    
    # --------------
    # the comments are intentionally left 
    # to help you understand what these parameters means.
    # --------------
    
    # OUTPUT_DIR = "../results/2025-2-23/mspecies"

    # GUE_RESULTS_DIR = "/home/peter/llm_projects/ZENforDNA/resources/finetune-output-v5.0"
    # GUE_RESULTS_DIR = "/data2/peter/finetune-output-v9.0"
    # # GUE_RESULTS_DIR = "/data2/peter/finetune-output-2-phase-v9.1"

    # # NGRAM_ENCODER_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams-detail/ngram-encoder-freq-hg38-gue-all-v7.4.json"
    # # NGRAM_FREQ_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams-detail/ngram-encoder-freq-hg38-gue-all-min-ngram-5-v7.5-frequencies-2025-2-17.json"

    # # NGRAM_ENCODER_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams/ngram-encoder-freq-hg38-gue-all-v8.0.json"
    # # NGRAM_FREQ_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams-detail/ngram-encoder-freq-hg38-gue-all-min-ngram-5-v7.4-frequencies-2025-2-16.json"

    # # NGRAM_ENCODER_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams/ngram-encoder-freq-hg38-gue-all-v8.0.json"
    # # NGRAM_FREQ_FILE = None

    # NGRAM_ENCODER_FILE = (
    #     "/home/peter/llm_projects/ZENforDNA/resources/ngrams/ngram-encoder-pmi-mspecies-gue-all-v9.0.json"
    # )
    # NGRAM_FREQ_FILE = None

    # # NGRAM_ENCODER_FILE = "/home/peter/llm_projects/ZENforDNA/resources/ngrams/ngram-encoder-pmi-mspecies-gue-all-v9.0.json"
    # # NGRAM_FREQ_FILE = None



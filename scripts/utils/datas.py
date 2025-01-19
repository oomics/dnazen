"""Utilities for processing datas in dataset."""

from typing import Literal
from glob import glob
import os

from .util import find_file

# --- GUE ---
_gue_data_type = Literal["train", "test", "dev"]


def get_all_gue_data(dir_path: str, type: _gue_data_type = "train", with_label=-1) -> list:
    """Get all GUE data without label."""
    ret = []

    for data_file in find_file(dir_path, target_filename=f"{type}.csv"):
        with open(data_file, "r") as f:
            lines = f.readlines()

        for line in lines[1:]:
            data = line.split(",")[0]
            label = line.split(",")[1].strip("\n")
            if with_label == -1 or int(label) == with_label:
                ret.append(data)

    return ret

def get_useful_gue_data(
    dir_path: str, type: _gue_data_type = "train"
) -> list:
    assert "mouse" not in dir_path, "word mouse should not be in data path. Detail please see the code."
    ret = []

    for data_file in find_file(dir_path, target_filename=f"{type}.csv"):
        with open(data_file, "r") as f:
            lines = f.readlines()

        is_mouse = "mouse" in data_file
        if is_mouse:
            print(data_file)
        
        for line in lines[1:]:
            data = line.split(",")[0]
            label = line.split(",")[1].strip("\n")
            if is_mouse and int(label) == 1:
                ret.append(data)
            else:
                ret.append(data)

    return ret


# --- hg38 ---
def get_all_hg38_data(data_path: str) -> list[str]:
    """Get all hg38 data without comments."""
    from Bio import SeqIO

    if not data_path.endswith(".fa"):
        raise ValueError("data_path must be a .fa file!")

    ret: list = []
    for seq_record in SeqIO.parse(data_path, "fasta"):
        dna_seq = str(seq_record.seq).upper().replace("N", "")
        ret.append(dna_seq)

    return ret

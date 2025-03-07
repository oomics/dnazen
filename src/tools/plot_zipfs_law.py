import json
import numpy as np
import click

import matplotlib.pyplot as plt


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def check_zipfs_law(word_freq, out_dir: str):
    name = out_dir.split("/")[-1].split(".")[0]

    # Sort frequencies in descending order
    sorted_freq = sorted(word_freq.values(), reverse=True)

    # Rank of the words
    ranks = np.arange(1, len(sorted_freq) + 1)

    # Zipf's Law: frequency * rank = constant
    zipf_values = sorted_freq[0] / ranks

    # Plotting the actual frequencies vs Zipf's Law
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_freq, label="Actual Frequencies")
    plt.loglog(ranks, zipf_values, label="Zipf's Law", linestyle="--")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title(f"({name}) Word Frequencies vs Zipf's Law")
    plt.legend()
    plt.show()
    plt.savefig(out_dir)


@click.command()
@click.option("-f", "--file", type=str)
@click.option("-o", "--out", type=str)
def main(file, out):
    word_freq = load_json(file)
    check_zipfs_law(word_freq, out)


if __name__ == "__main__":
    main()

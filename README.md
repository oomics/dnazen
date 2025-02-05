# ZENforDNA

ZENforDNA is a project aimed at providing tools and resources for DNA sequence analysis and LLM pretraining, finetuning

## Installation

This project managed in the form of a library. To install this project:

```bash
pip install -e .
```

Or install with setup.py:
```
python setup.py install
```

## Directory Layout

`src`: the source code of `dnazen` library

`tests`: the tests of our library

`scripts`: the scripts you need to do pretraining, finetuning, making datasets, etc.

`resources`: directory for keeping results.


## Usage

We keep all scripts you might find useful in `sciripts` directory. The usage of script is self-explanatory. Here are some scripts you might find most helpful:

- `run_pretrain.py`: Run the pretraining process.

    - You need to prepare for the dataset first.

- `run_finetune.py` Run the finetuning process.

    - You need to have a pretrained model first.




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


## Getting Started (for Development)

- __1. install `pixi` and `makefile` if you haven't ready (optional)__
    > We use `pixi` to manage the project. The reason we don't use `conda` is it has some issue managing `gcc`. If you insist using `conda` for development, make sure don't editable install this package.

    > We are also using `makefile` in favor of the `task` feature from `pixi`. This is because some of our tasks is too complex to use `pixi-task` and we would like to keep an hierarcial structure of tasks.

- __2. install the package__
    > Do `pixi install` if you want to use `pixi`.

    > Otherwise, do `pip install -e .`

- __3. install pre-commit hook__
    > Do `pre-commit install`.
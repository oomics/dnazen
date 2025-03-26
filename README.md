# DNAZEN

DNAZEN is a project aimed at providing tools and resources for DNA sequence analysis with pretraining and finetuning

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

`resources`: directory for keeping various resources.

`results`: results from experiments


## Usage



We keep all scripts you might find useful in `scripts` directory. The usage of script is self-explanatory. Here are some scripts you might find most helpful:

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

    > Specifically do `pip install .` if you are using `conda` (see `1` for the reason)

- __3. install pre-commit hook__
    > Do `pre-commit install`.

## Getting Started (for train)
- `cd scripts && source env.sh && make -f makefile pretrain `


## 实验配置说明

本项目支持三种不同的实验配置，用于训练和评估N-gram编码器。每个实验使用不同的数据集组合，结果保存在独立的目录中，便于比较和分析。

### 实验配置

1. **实验1 (exp1_gue_mspecies)**: 使用GUE数据集 + mspecies/dev数据集
   - 结合了基因组和多物种数据
   - 输出目录: `../data/pretrain/exp1_gue_mspecies/`

2. **实验2 (exp2_mspecies)**: 仅使用mspecies/dev数据集
   - 只使用多物种数据
   - 输出目录: `../data/pretrain/exp2_mspecies/`

3. **实验3 (exp3_gue)**: 仅使用GUE数据集
   - 只使用基因组数据
   - 输出目录: `../data/pretrain/exp3_gue/`

### 使用方法

使用`prepare_data_and_config.sh`脚本可以执行不同的实验配置。通过`--experiment`参数指定要运行的实验ID（1-3）。

#### 基本用法

```bash
./scripts/prepare_data_and_config.sh [选项] --experiment <id>
```

#### 选项说明

- `--all`: 执行所有步骤（训练N-gram编码器、tokenize训练和验证数据、准备预训练数据集）
- `--train-ngram`: 训练N-gram编码器
- `--tokenize-train`: 为训练数据生成tokenized数据
- `--tokenize-dev`: 为验证数据生成tokenized数据
- `--prepare-dataset`: 准备预训练数据集
- `--plots-only`: 只重新生成N-gram分布图
- `--experiment <id>`: 指定实验ID (1-3)
- `-h, --help`: 显示帮助信息

#### 示例

1. 使用GUE数据集训练N-gram编码器（实验3）:
   ```bash
   ./scripts/prepare_data_and_config.sh --train-ngram --experiment 3
   ```

2. 使用GUE+mspecies/dev执行所有步骤（实验1）:
   ```bash
   ./scripts/prepare_data_and_config.sh --all --experiment 1
   ```

3. 只重新生成实验2的N-gram分布图:
   ```bash
   ./scripts/prepare_data_and_config.sh --plots-only --experiment 2
   ```

### 可视化结果

每个实验都会生成以下可视化结果，保存在对应实验目录的`plots`文件夹中：

1. **N-gram分布图**: 展示不同长度N-gram的频率分布
   - 文件: `plots/ngram_distribution.png`

2. **交互式N-gram点频图**: 展示N-gram长度与频率的关系
   - 交互式HTML: `plots/ngram_interactive_scatter.html`
   - 静态图像: `plots/ngram_scatter.png`

3. **Zipf's Law图**: 展示N-gram频率与排名的关系
   - 交互式HTML: `plots/ngram_zipf_law.html`
   - 静态图像: `plots/ngram_zipf_law.png`

交互式图表支持鼠标滚轮缩放、悬停查看详情等功能，便于深入分析N-gram分布特性。

### 比较不同实验结果

通过运行不同的实验配置，可以比较不同数据集组合对N-gram编码器性能的影响。建议关注以下几个方面：

1. N-gram词汇表大小和组成
2. 频率分布的差异
3. Zipf定律符合程度
4. 预训练模型在下游任务上的表现

通过这些比较，可以确定最适合特定应用场景的N-gram编码方案。






安装正确的 NVIDIA apex 库
卸载当前安装的 apex，并手动安装 NVIDIA 官方版本：
bash
pip uninstall apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./



## Data 目录结构：
./data/downloads : 原始下载的DNABert2数据，dnabert_2_pretrain.zip为ms数据集，GUE.zip 为GUE数据集合
./data/pretrain_model: 预训练模型保存目录
./data/output/finetune: 为微调数据和模型输出目录
./data/epoch_x_metrics.json 和./data/epoch_x.json: 为zen模型中用到的经过ngram和dnabert2模型tokenized的数据
./data/exp1_pmi5/ngram_encoder.json 和./data/exp1_pmi5/ngram_list.txt 为使用PMI=5，对GUE数据集+ms的dev数据抽取ngram的到的ngram code




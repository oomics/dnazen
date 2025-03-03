# How to run the scripts with makefile


## Prerequisite

1. 仿照 `env.sh.sample`写一个脚本用于注册环境变量

2. `source env.sh` 注意是source而不是sh，否则环境变量不会发生改变

## ngrams

`make -f makefile_ngram_merged.mak all -j3`

## pretrain

`make pretrain -j3`

## finetune

`make -f makefile_finetune.mak all -j3`

> 如果gpu资源吃紧，请不要并行进行finetune
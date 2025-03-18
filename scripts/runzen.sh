#!/bin/bash
###################################################################################
# 脚本名称: run.sh
###################################################################################
#EXPERIMENT_ID="exp1_pmi2"
EXPERIMENT_ID="exp1_pmi5"
#EXPERIMENT_ID="exp1_pmi10"

# echo "===============================run.sh step0: 开始tokenize分词==============================================="
# #bash  ./prepare_pretrain_data.sh 
# echo "===============================run.sh step0: tokenize分词完成==============================================="


# echo "===============================run.sh step1: ngram抽取==============================================="
# bash ./prepare_ngram.sh -e $EXPERIMENT_ID
# echo "===============================run.sh step1: ngram抽取完成==============================================="

# echo "===============================run.sh step2: 准备数据集dataset==============================================="
# bash  ./prepare_pretrain_data.sh -e $EXPERIMENT_ID
# echo "===============================run.sh step2: 准备数据集dataset完成==============================================="

# echo "===============================run.sh step3: 开始预训练==============================================="
# bash ./pretrain.sh -e $EXPERIMENT_ID
# echo "===============================run.sh step3: 预训练完成==============================================="

# echo "===============================run.sh step4: 开始微调==============================================="
# bash finetune2.sh -e $EXPERIMENT_ID
# echo "===============================run.sh step4: 微调完成==============================================="
   
# 创建预训练数据
# python ../src/train/create_pre_train_data_zen.py     \
#     --train_corpus ../../mspecies/train/train.txt     \
#     --output_dir ../data/     \
#     --bert_model zhihan1996/DNABERT-2-117M     \
#     --ngram_list_dir  ../../out/exp1_pmi5/  

# python ../src/train/create_pre_train_data_zen.py     \
#     --train_corpus ../../mspecies/dev/dev.txt     \
#     --output_dir ../data/     \
#     --bert_model zhihan1996/DNABERT-2-117M     \
#     --ngram_list_dir  ../../out/exp1_pmi5/  

# 运行预训练
python ../src/train/run_pretrain_zen.py \
    --data-source tokenized \
    --data ../data/ \
    --ngram ../../out/exp1_pmi5/ngram_encoder.json \
    --out ../data/ \
    --model ~/DNABERT-2-117M \
    --lr 5e-5 \
    --epochs 5 \
    --batch-size 1024 \
    --grad-accum 2 \
    --warmup 0.1 \
    --num-workers 8 \
    --pin-memory True \
    --prefetch-factor 2 
    
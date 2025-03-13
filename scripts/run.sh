#!/bin/bash
###################################################################################
# 脚本名称: run.sh
###################################################################################

echo "===============================run.sh ngram抽取==============================================="
 #./prepare_data_and_config.sh --train-ngram    --experiment 1
 #./prepare_data_and_config.sh --coverage-analysis   --experiment 1
echo "===============================run.sh ngram抽取完成==============================================="


echo "===============================run.sh 开始tokenize分词==============================================="
#./prepare_data_and_config.sh --tokenize-train   
#./prepare_data_and_config.sh --tokenize-dev  
echo "===============================run.sh tokenize分词完成==============================================="


echo "===============================run.sh 开始准备数据集dataset==============================================="
#./prepare_data_and_config.sh --prepare-dataset   --experiment 1
echo "===============================run.sh 准备数据集dataset完成==============================================="


echo "===============================run.sh 开始预训练==============================================="
#bash ./pretrain.sh --experiment 1
echo "===============================run.sh 预训练完成==============================================="

echo "===============================run.sh 开始微调==============================================="
#bash ./finetune.sh --experiment 1 --parallel --resume

FINETUNE_DATA_DIR=/data1/peter FINETUNE_OUT_DIR=/data3/peter/finetune-2025-3-13 MAIN_NGRAM_ENCODER_DIR=/data2/peter/dnazen-pretrain-v9.4/train/ngram_encoder.json DNAZEN_PRETRAIN_DATA_DIR=/data2/peter/dnazen-pretrain-v9.4 FINETUNE_CHECKPOINT_STEP=10000 make -f makefile_finetune.mak all -j3

echo "===============================run.sh 微调完成==============================================="

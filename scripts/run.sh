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
bash ./pretrain.sh
echo "===============================run.sh 预训练完成==============================================="



echo "===============================run.sh 开始微调==============================================="
#bash ./finetune.sh --experiment 1 --parallel --resume
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINETUNE_OUT_DIR="../data/output/finetune/output/parallel_${TIMESTAMP}" 
# 创建输出目录
mkdir -p "$FINETUNE_OUT_DIR"

FINETUNE_DATA_DIR="../data/pretrain/train/finetune" \
FINETUNE_OUT_DIR="$FINETUNE_OUT_DIR" \
MAIN_NGRAM_ENCODER_DIR="../data/pretrain/exp1_gue_mspecies/ngram_encoder.json" \
DNAZEN_PRETRAIN_DATA_DIR="../data/pretrain/exp1_gue_mspecies/output/" \
FINETUNE_CHECKPOINT_STEP=41 \
make -f makefile_finetune.mak all -j8

echo "===============================run.sh 微调完成==============================================="

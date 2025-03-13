#!/bin/bash
###################################################################################
# 脚本名称: run.sh
###################################################################################


 #./prepare_data_and_config.sh --train-ngram    --experiment 1
 #./prepare_data_and_config.sh --coverage-analysis   --experiment 1



#./prepare_data_and_config.sh --train-ngram    --experiment 3
#./prepare_data_and_config.sh --coverage-analysis   --experiment 3



#./prepare_data_and_config.sh --coverage-analysis   --experiment 4
#./prepare_data_and_config.sh --coverage-analysis   --experiment 5
#./prepare_data_and_config.sh --coverage-analysis   --experiment 6


echo "===============================开始分词==============================================="
./prepare_data_and_config.sh --tokenize-train   
./prepare_data_and_config.sh --tokenize-dev   

echo "===============================开始准备数据集==============================================="
./prepare_data_and_config.sh --prepare-dataset   --experiment 1
echo "===============================准备数据集完成==============================================="


echo "===============================开始预训练==============================================="
./prepare_data_and_config.sh --pretrain  --experiment 1
echo "===============================预训练完成==============================================="

echo "===============================开始微调==============================================="
#bash ./finetune.sh --experiment 1 --parallel --resume
echo "===============================微调完成==============================================="

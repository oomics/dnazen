#!/bin/bash
###################################################################################
# 脚本名称: run.sh
###################################################################################


./prepare_data_and_config.sh --train-ngram    --experiment 1
./prepare_data_and_config.sh --coverage-analysis   --experiment 1



./prepare_data_and_config.sh --train-ngram    --experiment 3
./prepare_data_and_config.sh --coverage-analysis   --experiment 3


#./prepare_data_and_config.sh --tokenize-train   
#./prepare_data_and_config.sh --tokenize-dev   

# ./prepare_data_and_config.sh --coverage-analysis   --experiment 1
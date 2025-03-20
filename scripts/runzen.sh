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

# # 运行预训练
# python ../src/train/run_pretrain_zen.py \
#     --data-source tokenized \
#     --data ../data/ \
#     --ngram ../../out/exp1_pmi5/ngram_encoder.json \
#     --out ../data/ \
#     --model ~/DNABERT-2-117M \
#     --lr 5e-5 \
#     --epochs 5 \
#     --batch-size 1024 \
#     --grad-accum 2 \
#     --warmup 0.1 \
#     --num-workers 8 \
#     --pin-memory True \
#     --prefetch-factor 2 


#MODEL_PATH=../../zen_train/data/dnazen_0319194420_epoch_0/
MODEL_PATH=~/zen-model/
#MODEL_PATH=~/DNABERT-2-117M
NGRAM_ENCODER_PATH=~/zen-model/ngram_encoder.json
NGRAM_ENCODER_LIST_PATH=../../out/exp1_pmi5/

# TF任务
TF_SUBTASKS=("0" "1" "2" "3" "4")
# Mouse任务
MOUSE_SUBTASKS=("0" "1" "2" "3" "4")
# PD任务
PD_SUBTASKS=("prom_300_all" "prom_core_all")
# EMP任务
EMP_SUBTASKS=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K9ac" "H4" "H4ac")



GUE_DIR="../../GUE/"                        # 数据根目录
FINETUNE_OUT_DIR="../output/finetune"        # 微调输出目录



###################################################################################
# 任务处理函数
###################################################################################
process_task() {
     local task_type=$1
     local sub_task=$2
     local num_epochs=$3

     # 检查是否需要跳过此任务（断点继续功能）
     if [ "$RESUME" = true ] && [ -f "$RESUME_FILE" ]; then
          if grep -q "${task_type}/${sub_task}" "$RESUME_FILE"; then
               echo "跳过已完成的任务: ${task_type}/${sub_task}"
               return 0
          fi
     fi


     echo "================================================"
     echo "任务: $task_type/$sub_task"
     echo "================================================"

     local data_path="${GUE_DIR}/${TASK_PATHS[$task_type]}/${sub_task}"
     local task_output_path="${FINETUNE_OUT_DIR}/${task_type}/${sub_task}"   

     TASK_NAME=$task_type+"_"+$sub_task

     echo "数据路径 DATA_PATH: $data_path"
     echo "训练任务输出路径 TASK_OUTPUT_PATH: $task_output_path"
     echo "训练轮数 NUM_TRAIN_EPOCHS: $num_epochs"

     # 运行预训练
     python ../src/train/run_sequence_level_classification.py \
          --data_dir $data_path \
          --bert_model $MODEL_PATH \
          --task_name $TASK_NAME \
          --do_train \
          --do_eval \
          --max_seq_length 128 \
          --train_batch_size 1024  \
          --num_train_epochs $num_epochs \
          --gradient_accumulation_steps 2 \
          --learning_rate 5e-5 \
          --output_dir $task_output_path \
          --save_steps 1000 \
          --seed 42 \
          --warmup_proportion 0.1 
}


###################################################################################
# 任务执行
###################################################################################

task1() {
    echo "Task 1 started"
    # 处理TF任务
    for subtask in "${TF_SUBTASKS[@]}"; do
      process_task "tf" "$subtask" "${TASK_EPOCHS[tf]}"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "任务失败，退出执行"
        exit 1
      fi
    done
    sleep 3
    echo "Task 1 completed"
}
# 运行预训练
#TASK_NAME=prom_300_all
# python ../src/train/run_sequence_level_classification.py \
#      --data_dir ../../GUE/prom/$TASK_NAME \
#      --bert_model $MODEL_PATH \
#      --task_name $TASK_NAME \
#      --do_train \
#      --do_eval \
#      --max_seq_length 128 \
#      --train_batch_size 1024  \
#      --num_train_epochs 1 \
#      --gradient_accumulation_steps 2 \
#      --learning_rate 5e-5 \
#      --output_dir ../output/finetune/pd/$TASK_NAME \
#      --save_steps 1000 \
#      --seed 42 \
#      --warmup_proportion 0.1 \
#      --ngram_list_dir $NGRAM_ENCODER_LIST_PATH



#   MODEL_PATH=~/zen-model/
#   #MODEL_PATH=~/DNABERT-2-117M
#   NGRAM_ENCODER_PATH=~/zen-model/ngram_encoder.json




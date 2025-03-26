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
   
# # 创建预训练数据
# CMD="python ../src/train/create_pre_train_data_zen.py \
#     --train_corpus ../../mspecies/train/train.txt \
#     --output_dir ../data/ \
#     --bert_model zhihan1996/DNABERT-2-117M \
#     --ngram_list_dir  ../../out/exp1_pmi5/"  

# echo $CMD
# eval $CMD

# CMD="python ../src/train/create_pre_train_data_zen.py \
#     --train_corpus ../../mspecies/dev/dev.txt \
#     --output_dir ../data/ \
#     --bert_model zhihan1996/DNABERT-2-117M \
#     --ngram_list_dir  ../../out/exp1_pmi5/"  

# echo $CMD
# eval $CMD

# # 运行预训练
# echo "===============================$0 step5: 开始预训练==============================================="
# CMD="python ../src/train/run_pretrain_zen.py \
#     --data-source tokenized \
#     --data ../data/ \
#     --ngram ../../out/exp1_pmi5/ngram_encoder.json \
#     --out ../data/pretrain_model/ \
#     --model ~/DNABERT-2-117M \
#     --lr 5e-5 \
#     --epochs 2 \
#     --batch-size 512 \
#     --grad-accum 2 \
#     --warmup 0.1 \
#     --num-workers 8 \
#     --reduce-mem  "

CMD="python ../src/train/run_pretrain_zen.py \
    --data-source tokenized \
    --data ../data/ \
    --ngram ../../out/exp1_pmi5/ngram_encoder.json \
    --out ../data/pretrain_model/ \
    --model ~/DNABERT-2-117M \
    --lr 5e-5 \
    --epochs 2 \
    --batch-size 512 \
    --grad-accum 2 \
    --warmup 0.1 \
    --num-workers 8 \
    --reduce-mem \
    --scratch "

echo $CMD
eval $CMD

echo "===============================$0 step5: 预训练完成==============================================="
#MODEL_PATH=../../zen_train/data/dnazen_0319194420_epoch_0/
#PRETRAINED_MODEL_PATH=~/zen-model/

#PRETRAINED_MODEL_PATH=../../zen_train/data/dnazen_0319194420_epoch_0/
PRETRAINED_MODEL_NAME=$(ls -t ../data/pretrain_model/ | head -n 1)
PRETRAINED_MODEL_PATH=../data/pretrain_model/$PRETRAINED_MODEL_NAME
echo "PRETRAINED_MODEL_PATH: "$PRETRAINED_MODEL_PATH

#MODEL_PATH=~/DNABERT-2-117M
NGRAM_ENCODER_PATH=~/zen-model/ngram_encoder.json
NGRAM_ENCODER_LIST_PATH=../../out/exp1_pmi5/

# 运行预训练
TASK_NAME=prom_300_all
echo "===============================run.sh step6: 开始微调==============================================="
CMD="python ../src/train/run_sequence_level_classification.py \
     --data_dir ../../GUE/prom/$TASK_NAME \
     --bert_model $PRETRAINED_MODEL_PATH \
     --task_name $TASK_NAME \
     --do_train \
     --do_eval \
     --max_seq_length 128 \
     --train_batch_size 1024  \
     --num_train_epochs 2 \
     --gradient_accumulation_steps 2 \
     --learning_rate 5e-5 \
     --output_dir ../output/finetune/pd/$TASK_NAME \
     --save_steps 1000 \
     --seed 42 \
     --warmup_proportion 0.1 \
     --ngram_list_dir $NGRAM_ENCODER_LIST_PATH  "

echo $CMD
#eval $CMD


# 运行全部微调任务
CMD="bash finetune_zen.sh --experiment 1 --parallel --max-workers 4 --pretrained-model $PRETRAINED_MODEL_PATH"
echo $CMD
eval $CMD
echo "===============================run.sh step6: 微调完成==============================================="





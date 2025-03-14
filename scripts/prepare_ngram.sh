#!/bin/bash
USE_GUE="../../GUE/"   
USE_MSPECIES="../../GUE/"          
USE_MSPECIES="../../mspecies/dev/dev.txt"

NGRAM_ENCODER_PATH="../../out/exp1_pmi2/ngram_encoder.json"
EXPERIMENT_DIR="../data/pretrain/exp1_pmi2"
COVERAGE_DIR="${EXPERIMENT_DIR}/coverage"

RUN_NGRAM_ENCODER=true
RUN_COVERAGE_ANALYSIS=true

###################################################################################
# 4. 数据预处理
###################################################################################
# Step1:提取N-gram编码
if [[ "$RUN_NGRAM_ENCODER" == "true" ]]; then
  echo "===== Step1 开始提取N-gram编码器  ====="  
  # 检查并删除已存在的ngram_encoder.json文件
  if [[ -f "${EXPERIMENT_DIR}/ngram_encoder.json" ]]; then
    echo "发现已存在的ngram_encoder.json文件，正在删除..."
    rm -f "${EXPERIMENT_DIR}/ngram_encoder.json"
  fi
  
  CMD="python ../src/train/train_ngram_encoder.py \
    --gue-dir ${USE_GUE} \
    --input ${USE_MSPECIES} \
    --output ${EXPERIMENT_DIR}/ngram_encoder.json \
    --tok zhihan1996/DNABERT-2-117M \
    --min-ngram-len 2 \
    --max-ngram-len 5 \
    --max-ngrams 30 \
    --min-pmi 2 \
    --min-token-count 5 \
    --min-ngram-freq 5 \
    --method pmi \
    --num-workers 4"
  
  echo "执行命令: $CMD"
  eval $CMD
  
  if [[ $? -ne 0 ]]; then
    echo "N-gram编码器训练失败"
    exit 1
  fi
  echo "===== N-gram编码器训练完成 ====="
fi

# Step1.1: N-gram编码在训练数据集上的覆盖率验证
if [[ "$RUN_COVERAGE_ANALYSIS" == "true" ]]; then
  echo "===== Step1.1 开始验证N-gram编码在训练数据集上的覆盖率 ====="

  if [[ -d "$COVERAGE_DIR" ]]; then
    echo "发现已存在的coverage目录，正在删除..."
    rm -rf "$COVERAGE_DIR"
  fi

  # 构建命令
  CMD="python ../src/dataset/ngram_encoder_analyze.py \
    --encoder ${EXPERIMENT_DIR}/ngram_encoder.json \
    --output-dir ${COVERAGE_DIR} \
    --tok zhihan1996/DNABERT-2-117M \
    --gue-dir ../data/GUE \
    --mspecies-dir ../data/pretrain/dev/dev.txt \
    --ngram-list ${EXPERIMENT_DIR}/ngram_list.txt "

  
  echo "执行命令: $CMD"
  eval $CMD
  
  
  if [[ $? -ne 0 ]]; then
    echo "N-gram编码覆盖率分析失败"
    exit 1
  fi
  
  echo "===== N-gram编码覆盖率分析完成 ====="
fi

echo "所有选定的任务已完成"

#!/bin/bash

# 设置默认值
RUN_NGRAM_ENCODER=false
RUN_TOKENIZE_TRAIN=false
RUN_TOKENIZE_DEV=false
RUN_PREPARE_DATASET=false

# 显示帮助信息
function show_help {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  --all                  执行所有步骤"
  echo "  --train-ngram          训练N-gram编码器"
  echo "  --tokenize-train       为训练数据生成tokenized数据"
  echo "  --tokenize-dev         为验证数据生成tokenized数据"
  echo "  --prepare-dataset      准备预训练数据集"
  echo "  -h, --help             显示此帮助信息"
  echo ""
  echo "示例:"
  echo "  $0 --train-ngram --tokenize-train  # 只执行训练N-gram编码器和tokenize训练数据"
  echo "  $0 --all                           # 执行所有步骤"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_NGRAM_ENCODER=true
      RUN_TOKENIZE_TRAIN=true
      RUN_TOKENIZE_DEV=true
      RUN_PREPARE_DATASET=true
      shift
      ;;
    --train-ngram)
      RUN_NGRAM_ENCODER=true
      shift
      ;;
    --tokenize-train)
      RUN_TOKENIZE_TRAIN=true
      shift
      ;;
    --tokenize-dev)
      RUN_TOKENIZE_DEV=true
      shift
      ;;
    --prepare-dataset)
      RUN_PREPARE_DATASET=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      show_help
      exit 1
      ;;
  esac
done

# 如果没有指定任何参数，显示帮助信息
if [[ "$RUN_NGRAM_ENCODER" == "false" && "$RUN_TOKENIZE_TRAIN" == "false" && "$RUN_TOKENIZE_DEV" == "false" && "$RUN_PREPARE_DATASET" == "false" ]]; then
  show_help
  exit 0
fi

# 创建必要的目录
# mkdir -p ../data/pretrain/tokenized
# mkdir -p ../data/pretrain/dev
# mkdir -p ../data/pretrain/train


###################################################################################
# 1. 环境准备
###################################################################################


###################################################################################
# 2. 数据下载
###################################################################################



###################################################################################
# 3. 数据准备
###################################################################################
# Step1:训练N-gram编码器
if [[ "$RUN_NGRAM_ENCODER" == "true" ]]; then
  echo "=====Step1 开始训练N-gram编码器 ====="
  python scripts/train_ngram_encoder.py \
    --input ../data/pretrain/raw/dev.txt \
    --output ../data/pretrain/ngram_encoder.json \
    --min-ngram-len 2 \
    --max-ngram-len 5 \
    --max-ngrams 30 \
    --min-pmi 0.5 \
    --min-token-count 2 \
    --min-ngram-freq 2 \
    --method pmi \
    --num-workers 4
  
  if [[ $? -ne 0 ]]; then
    echo "训练N-gram编码器失败"
    exit 1
  fi
  echo "===== N-gram编码器训练完成 ====="
fi

# Step2: 为训练数据生成tokenized数据
if [[ "$RUN_TOKENIZE_TRAIN" == "true" ]]; then
  echo "=====Step2 开始为训练数据生成tokenized数据 ====="
  python make_tokenized_dataset.py \
    --data ../data/pretrain/train/train.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ../data/pretrain/train/train.pt \
    --batch-size 500000 \
    --max-length 256 \
    --resume

  
  if [[ $? -ne 0 ]]; then
    echo "为训练数据生成tokenized数据失败"
    exit 1
  fi
  echo "===== 训练数据tokenized完成 ====="
fi

# Step3:  为验证数据生成tokenized数据
if [[ "$RUN_TOKENIZE_DEV" == "true" ]]; then
  echo "===== Step3 开始为验证数据生成tokenized数据 ====="
  python make_tokenized_dataset.py \
    --data ../data/pretrain/dev/dev.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ../data/pretrain/dev/dev.pt \
    --batch-size 500000 \
    --max-length 256
  
  if [[ $? -ne 0 ]]; then
    echo "为验证数据生成tokenized数据失败"
    exit 1
  fi
  echo "===== 验证数据tokenized完成 ====="
fi

# 准备预训练数据集，使用已经tokenized的数据
if [[ "$RUN_PREPARE_DATASET" == "true" ]]; then
  echo "===== 开始准备预训练数据集 ====="
  python scripts/make_pretrain_dataset.py \
    --data-source tokenized \
    #--data ../data/pretrain/tokenized \
    --data ../data/pretrain/train/train.pt \
    --tok-source huggingface \
    --tok zhihan1996/DNABERT-2-117M \
    --ngram ../data/pretrain/ngram_encoder.json \
    --max-ngrams 30 \
    --out ../data/pretrain \
    --seed 42
  
  if [[ $? -ne 0 ]]; then
    echo "准备预训练数据集失败"
    exit 1
  fi
  echo "===== 预训练数据集准备完成 ====="
fi


###################################################################################
# 4. 启动第一阶段训练进程
###################################################################################





###################################################################################
# 5. 启动第二阶段训练进程
###################################################################################





###################################################################################
# 6. 启动微调训练进程
###################################################################################


###################################################################################
# 7. 评估测试
###################################################################################


echo "所有选定的任务已完成"

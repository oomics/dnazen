#!/bin/bash

###################################################################################
# 全局参数配置
###################################################################################

# 执行控制参数
#RUN_TOKENIZE_TRAIN=true
#RUN_TOKENIZE_DEV=true
RUN_PREPARE_DATASET=true

# 数据路径参数
GUE_DIR="../../GUE/"                        # 数据根目录
PRETRAIN_CHECKPOINT="../../out/exp1_pmi2/checkpoint-198" # 预训练模型检查点
#NGRAM_ENCODER_PATH="../../out/exp1_pmi2/" # NGram编码器路径
NGRAM_ENCODER_PATH="../../out/exp1_pmi2/ngram_encoder.json" # NGram编码器路径
FINETUNE_OUT_DIR="../output/finetune"        # 微调输出目录
REPORT_OUT_DIR="../output/report"            # 报告输出目录
USE_MSPECIES="../../mspecies/dev/dev.txt"

# 数据路径参数
TRAIN_DATA="../../mspecies/train/train.txt"
DEV_DATA="../../mspecies/dev/dev.txt"

# 输出路径参数
EXPERIMENT_ID="exp1_pmi2"
EXPERIMENT_DIR="../../out/${EXPERIMENT_ID}"
PRETRAIN_OUTPUT_DIR="${EXPERIMENT_DIR}/pretrain"

TRAIN_OUTPUT="../../mspecies/train/train.pt"
DEV_OUTPUT="../../mspecies/dev/dev.pt"
PRETRAIN_TOKENIZED_DATA_DIR="../../mspecies/"


# 模型参数
TOKENIZER="zhihan1996/DNABERT-2-117M"
TOK_SOURCE="huggingface"

# 处理参数
BATCH_SIZE=1000000
MAX_LENGTH=256
MAX_NGRAMS=30
SEED=42
DATA_SOURCE="tokenized"

# 打印参数函数
print_parameters() {
  echo "========== 参数列表 =========="
  echo "# 执行控制参数"
  echo "执行训练数据tokenize: ${RUN_TOKENIZE_TRAIN}"
  echo "执行验证数据tokenize: ${RUN_TOKENIZE_DEV}"
  echo "执行预训练数据集准备: ${RUN_PREPARE_DATASET}"
  echo ""
  echo "# 数据路径参数"
  echo "训练数据: ${TRAIN_DATA}"
  echo "训练数据输出: ${TRAIN_OUTPUT}"
  echo "验证数据: ${DEV_DATA}"
  echo "验证数据输出: ${DEV_OUTPUT}"
  echo "预训练用的tokenized数据目录: ${PRETRAIN_TOKENIZED_DATA_DIR}"
  echo ""
  echo "# 输出路径参数"
  echo "实验ID: ${EXPERIMENT_ID}"
  echo "实验目录: ${EXPERIMENT_DIR}"
  echo "预训练输出目录: ${PRETRAIN_OUTPUT_DIR}"
  echo ""
  echo "# 模型参数"
  echo "分词器: ${TOKENIZER}"
  echo "分词器来源: ${TOK_SOURCE}"
  echo ""
  echo "# 处理参数"
  echo "批量大小: ${BATCH_SIZE}"
  echo "最大长度: ${MAX_LENGTH}"
  echo "最大N-grams: ${MAX_NGRAMS}"
  echo "随机种子: ${SEED}"
  echo "数据源类型: ${DATA_SOURCE}"
  echo "============================="
}

# 打印所有参数
print_parameters

# Step2: 为训练数据生成tokenized数据
if [[ "$RUN_TOKENIZE_TRAIN" == "true" ]]; then
  echo "===== Step2 开始为训练数据生成tokenized数据 ====="
  
  # 打印tokenize训练数据参数
  echo "Tokenize训练数据参数:"
  echo "  --data: ${TRAIN_DATA}"
  echo "  --tok: ${TOKENIZER}"
  echo "  --out: ${TRAIN_OUTPUT}"
  echo "  --batch-size: ${BATCH_SIZE}"
  echo "  --max-length: ${MAX_LENGTH}"
  echo "  --resume"
  
  python ../src/dataset/make_tokenized_dataset.py \
    --data ${TRAIN_DATA} \
    --tok ${TOKENIZER} \
    --out ${TRAIN_OUTPUT} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
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
  
  # 打印tokenize验证数据参数
  echo "Tokenize验证数据参数:"
  echo "  --data: ${DEV_DATA}"
  echo "  --tok: ${TOKENIZER}"
  echo "  --out: ${DEV_OUTPUT}"
  echo "  --max-length: ${MAX_LENGTH}"
  
  python ../src/dataset/make_tokenized_dataset.py \
    --data ${DEV_DATA} \
    --tok ${TOKENIZER} \
    --out ${DEV_OUTPUT} \
    --max-length ${MAX_LENGTH}
  
  if [[ $? -ne 0 ]]; then
    echo "为验证数据生成tokenized数据失败"
    exit 1
  fi
  echo "===== 验证数据tokenized完成 ====="
fi

# Step4:  准备预训练数据集，使用已经tokenized的数据
if [[ "$RUN_PREPARE_DATASET" == "true" ]]; then
  echo "===== Step4 开始准备预训练数据集 使用实验${EXPERIMENT_ID}的N-gram编码器 ====="
  
  # 打印预训练数据集准备参数
  echo "预训练数据集准备参数:"
  echo "  --data-source: ${DATA_SOURCE}"
  echo "  --data: ${PRETRAIN_TOKENIZED_DATA_DIR}"
  echo "  --tok-source: ${TOK_SOURCE}"
  echo "  --tok: ${TOKENIZER}"
  echo "  --ngram: ${EXPERIMENT_DIR}/ngram_encoder.json"
  echo "  --max-ngrams: ${MAX_NGRAMS}"
  echo "  --out: ${PRETRAIN_OUTPUT_DIR}"
  echo "  --seed: ${SEED}"
  
  python ../src/dataset/make_pretrain_dataset.py \
    --data-source ${DATA_SOURCE} \
    --data ${PRETRAIN_TOKENIZED_DATA_DIR} \
    --tok-source ${TOK_SOURCE} \
    --tok ${TOKENIZER} \
    --ngram ${EXPERIMENT_DIR}/ngram_encoder.json \
    --max-ngrams ${MAX_NGRAMS} \
    --out ${PRETRAIN_OUTPUT_DIR} \
    --seed ${SEED}
  
  if [[ $? -ne 0 ]]; then
    echo "准备预训练数据集失败"
    exit 1
  fi
  echo "===== 预训练数据集准备完成 ====="
fi

echo "预训练数据准备完成，输出目录: ${PRETRAIN_OUTPUT_DIR}"

#!/bin/bash

###################################################################################
# 全局参数配置
###################################################################################


# 实验名称参数
EXPERIMENT_ID="exp1_pmi2"


# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help                显示帮助信息"
    echo "  -e, --experiment-id ID    指定实验ID (默认: $EXPERIMENT_ID)"
    echo
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--experiment-id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "实验ID: ${EXPERIMENT_ID}"
EXPERIMENT_DIR="../../out/${EXPERIMENT_ID}"

NGRAM_ENCODER_PATH="${EXPERIMENT_DIR}/ngram_encoder.json" # NGram编码器路径
echo "NGram编码器路径: ${NGRAM_ENCODER_PATH}"
TRAIN_DATA="../../mspecies/train/train.txt"
DEV_DATA="../../mspecies/dev/dev.txt"


# 执行控制参数
RUN_TOKENIZE_TRAIN=true
RUN_TOKENIZE_DEV=true
#RUN_PREPARE_DATASET=true


# 输出路径参数
PRETRAIN_OUTPUT_DIR="${EXPERIMENT_DIR}/pretrain"

PRETRAIN_TOKENIZED_DATA_DIR="${EXPERIMENT_DIR}/pretrain"
TRAIN_OUTPUT="${PRETRAIN_TOKENIZED_DATA_DIR}/train/train.pt"
DEV_OUTPUT="${PRETRAIN_TOKENIZED_DATA_DIR}/dev/dev.pt"


# 模型参数
TOKENIZER="zhihan1996/DNABERT-2-117M"
TOK_SOURCE="huggingface"

# 处理参数
BATCH_SIZE=1000000
MAX_LENGTH=256
MAX_NGRAMS=30
SEED=42
DATA_SOURCE="tokenized"



echo "使用实验: $EXPERIMENT_NAME (ID: $EXPERIMENT_ID)"
echo "实验目录: $EXPERIMENT_DIR"


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
  
  # 检查并删除已存在的train.pt文件
  if [[ -f "${TRAIN_OUTPUT}" ]]; then
    echo "发现已存在的${TRAIN_OUTPUT}文件，正在删除..."
    rm -f "${TRAIN_OUTPUT}"
  fi
  
  # 确保输出目录存在
  mkdir -p "$(dirname "${TRAIN_OUTPUT}")"
  
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
  
    # 检查并删除已存在的train.pt文件
  if [[ -f "${DEV_OUTPUT}" ]]; then
    echo "发现已存在的${DEV_OUTPUT}文件，正在删除..."
    rm -f "${DEV_OUTPUT}"
  fi
  
  # 确保输出目录存在
  mkdir -p "$(dirname "${TRAIN_OUTPUT}")"

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
  echo "  --ngram: ${NGRAM_ENCODER_PATH}"
  echo "  --max-ngrams: ${MAX_NGRAMS}"
  echo "  --out: ${PRETRAIN_OUTPUT_DIR}"
  echo "  --seed: ${SEED}"
  
  python ../src/dataset/make_pretrain_dataset.py \
    --data-source ${DATA_SOURCE} \
    --data ${PRETRAIN_TOKENIZED_DATA_DIR} \
    --tok-source ${TOK_SOURCE} \
    --tok ${TOKENIZER} \
    --ngram ${NGRAM_ENCODER_PATH}  \
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

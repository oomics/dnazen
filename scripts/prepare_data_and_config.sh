#!/bin/bash

# 设置默认值
RUN_NGRAM_ENCODER=false
RUN_TOKENIZE_TRAIN=false
RUN_TOKENIZE_DEV=false
RUN_PREPARE_DATASET=false
RUN_PLOTS_ONLY=false
RUN_COVERAGE_ANALYSIS=false
EXPERIMENT_ID=1  # 默认实验ID

# 显示帮助信息
function show_help {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  --all                  执行所有步骤"
  echo "  --train-ngram          训练N-gram编码器"
  echo "  --tokenize-train       为训练数据生成tokenized数据"
  echo "  --tokenize-dev         为验证数据生成tokenized数据"
  echo "  --prepare-dataset      准备预训练数据集"
  echo "  --coverage-analysis    分析N-gram在GUE和mspecies数据集上的覆盖率"
  echo "  --experiment <id>      指定实验ID (1-10)"
  echo "                         1: GUE + mspecies/dev"
  echo "                         2: 仅 mspecies/dev"
  echo "                         3: 仅 GUE"
  echo "  -h, --help             显示此帮助信息"
  echo ""
  echo "示例:"
  echo "  $0 --train-ngram --experiment 3  # 使用GUE数据集训练N-gram编码器"
  echo "  $0 --all --experiment 1          # 使用GUE+mspecies/dev执行所有步骤"
  echo "  $0 --plots-only --experiment 2   # 只重新生成实验2的N-gram分布图"
  echo "  $0 --coverage-analysis --experiment 3  # 分析实验3的N-gram覆盖率"
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
    --coverage-analysis)
      RUN_COVERAGE_ANALYSIS=true
      shift
      ;;
    --experiment)
      EXPERIMENT_ID="$2"
      shift 2
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
if [[ "$RUN_NGRAM_ENCODER" == "false" && "$RUN_TOKENIZE_TRAIN" == "false" && "$RUN_TOKENIZE_DEV" == "false" && "$RUN_PREPARE_DATASET" == "false" && "$RUN_PLOTS_ONLY" == "false" && "$RUN_COVERAGE_ANALYSIS" == "false" ]]; then
  show_help
  exit 0
fi

# 检查实验ID是否有效
if [[ -n "$EXPERIMENT_ID" ]]; then
  if [[ "$EXPERIMENT_ID" -lt 1 || "$EXPERIMENT_ID" -gt 6 ]]; then
    echo "错误: 实验ID必须是1-6之间的数字"
    echo "用法: ./prepare_data_and_config.sh [选项]"
    echo "选项:"
    echo "  --experiment N    选择实验配置 (1-6)"
    show_help
    exit 1
  fi
else
  echo "错误: 需要指定实验ID"
  show_help
  exit 1
fi

GUE_DATA_DIR=${GUE_DATA_DIR:-"../data/GUE"}
PRETRAIN_DATA_DIR=${PRETRAIN_DATA_DIR:-"../data/pretrain"}

# 根据实验ID设置输出目录和参数
case "$EXPERIMENT_ID" in
  1)
    EXPERIMENT_NAME="exp1_gue_mspecies"
    EXPERIMENT_DESC="GUE数据集+mspecies/dev数据集抽取ngram"
    USE_GUE="--gue-dir ${GUE_DATA_DIR}"
    USE_INPUT="--input ${PRETRAIN_DATA_DIR}/dev/dev.txt"
    MIN_FREQ_FILTER=5    
    ;;
  2)
    EXPERIMENT_NAME="exp2_mspecies"
    EXPERIMENT_DESC="仅mspecies/dev数据集抽取ngram"
    USE_GUE=""
    USE_INPUT="--input ${PRETRAIN_DATA_DIR}/dev/dev.txt"
    MIN_FREQ_FILTER=5
    ;;
  3)
    EXPERIMENT_NAME="exp3_gue"
    EXPERIMENT_DESC="仅GUE数据集抽取ngram"
    USE_GUE="--gue-dir ${GUE_DATA_DIR}"
    USE_INPUT=""
    MIN_FREQ_FILTER=5
    ;;
  4)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_5"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为5"
    USE_GUE="--gue-dir ${GUE_DATA_DIR}"
    USE_INPUT=""
    MIN_FREQ_FILTER=5
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  5)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_1"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为1"
    USE_GUE="--gue-dir ${GUE_DATA_DIR}"
    USE_INPUT=""
    MIN_FREQ_FILTER=1
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  6)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_100"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为100"
    USE_GUE="--gue-dir ${GUE_DATA_DIR}"
    USE_INPUT=""
    MIN_FREQ_FILTER=100
    PARENT_EXPERIMENT="exp3_gue"
    ;;
esac

# 创建实验目录
if [[ -n "$PARENT_EXPERIMENT" ]]; then
  # 如果是子实验，使用父实验的目录
  EXPERIMENT_DIR="../data/pretrain/${PARENT_EXPERIMENT}"
  COVERAGE_DIR="${EXPERIMENT_DIR}/coverage_analysis_${EXPERIMENT_NAME}"
else
  # 否则创建新的实验目录
  EXPERIMENT_DIR="../data/pretrain/${EXPERIMENT_NAME}"
  COVERAGE_DIR="${EXPERIMENT_DIR}/coverage_analysis"
fi
mkdir -p ${EXPERIMENT_DIR}
mkdir -p ${COVERAGE_DIR}

# 创建必要的目录
mkdir -p ${PRETRAIN_DATA_DIR}/tokenized
mkdir -p ${PRETRAIN_DATA_DIR}/dev
mkdir -p ${PRETRAIN_DATA_DIR}/train


###################################################################################
# 1. 环境准备
###################################################################################


###################################################################################
# 2. 数据下载
###################################################################################


###################################################################################
# 3. 数据预处理
###################################################################################
# Step1:提取N-gram编码
if [[ "$RUN_NGRAM_ENCODER" == "true" ]]; then
  echo "===== Step1 开始提取N-gram编码器 (实验${EXPERIMENT_ID}: ${EXPERIMENT_DESC}) ====="  
  CMD="python ../src/train/train_ngram_encoder.py \
    ${USE_GUE} \
    ${USE_INPUT} \
    --output ${EXPERIMENT_DIR}/ngram_encoder.json \
    --tok zhihan1996/DNABERT-2-117M \
    --min-ngram-len 2 \
    --max-ngram-len 5 \
    --max-ngrams 30 \
    --min-pmi 0.5 \
    --min-token-count 2 \
    --min-ngram-freq 2 \
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
  
  # 构建命令
  CMD="python ../src/dataset/ngram_encoder_analyze.py \
    --encoder ${EXPERIMENT_DIR}/ngram_encoder.json \
    --output-dir ${COVERAGE_DIR} \
    --tok zhihan1996/DNABERT-2-117M \
    --gue-dir ${GUE_DATA_DIR} \
    --mspecies-dir ${PRETRAIN_DATA_DIR}/dev/dev.txt \
    --ngram-list ${EXPERIMENT_DIR}/ngram_list.txt \
    --min-freq-filter ${MIN_FREQ_FILTER}"
  
  echo "执行命令: $CMD"
  eval $CMD
  
  if [[ $? -ne 0 ]]; then
    echo "N-gram编码覆盖率分析失败"
    exit 1
  fi
  
  echo "===== N-gram编码覆盖率分析完成 ====="
fi





# Step2: 为训练数据生成tokenized数据
if [[ "$RUN_TOKENIZE_TRAIN" == "true" ]]; then
  echo "===== Step2 开始为训练数据生成tokenized数据 ====="
  python ../src/dataset/make_tokenized_dataset.py \
    --data ${PRETRAIN_DATA_DIR}/train/train.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ${PRETRAIN_DATA_DIR}/train/train.pt \
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
  python ../src/dataset/make_tokenized_dataset.py \
    --data ${PRETRAIN_DATA_DIR}/dev/dev.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ${PRETRAIN_DATA_DIR}/dev/dev.pt \
    --batch-size 500000 \
    --max-length 256
  
  if [[ $? -ne 0 ]]; then
    echo "为验证数据生成tokenized数据失败"
    exit 1
  fi
  echo "===== 验证数据tokenized完成 ====="
fi

# Step4:  准备预训练数据集，使用已经tokenized的数据
if [[ "$RUN_PREPARE_DATASET" == "true" ]]; then
  echo "===== Step4 开始准备预训练数据集 使用实验${EXPERIMENT_ID}的N-gram编码器 ====="
  python ../src/dataset/make_pretrain_dataset.py \
    --data-source tokenized \
    --data ${PRETRAIN_DATA_DIR}/train/train.pt \
    --tok-source huggingface \
    --tok zhihan1996/DNABERT-2-117M \
    --ngram ${EXPERIMENT_DIR}/ngram_encoder.json \
    --max-ngrams 30 \
    --out ${EXPERIMENT_DIR}/pretrain_data \
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

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

# 输出路径参数
EXPERIMENT_DIR="../../out/${EXPERIMENT_ID}"

echo "使用实验: $EXPERIMENT_NAME (ID: $EXPERIMENT_ID)"
echo "实验目录: $EXPERIMENT_DIR"


# 数据路径参数
USE_GUE="../../GUE/"        
USE_MSPECIES="../../mspecies/dev/dev.txt"


COVERAGE_DIR="${EXPERIMENT_DIR}/coverage"
NGRAM_ENCODER_PATH="${EXPERIMENT_DIR}/ngram_encoder.json"

# 模型参数
TOKENIZER="zhihan1996/DNABERT-2-117M"

RUN_NGRAM_ENCODER=true
RUN_COVERAGE_ANALYSIS=true

# 打印参数函数
print_parameters() {
  echo "========== 参数列表 =========="
  echo "GUE目录: ${USE_GUE}"
  echo "多物种数据: ${USE_MSPECIES}"
  echo "实验目录: ${EXPERIMENT_DIR}"
  echo "覆盖率目录: ${COVERAGE_DIR}"
  echo "N-gram编码器路径: ${NGRAM_ENCODER_PATH}"
  echo "分词器: ${TOKENIZER}"
  echo "执行N-gram编码器训练: ${RUN_NGRAM_ENCODER}"
  echo "执行覆盖率分析: ${RUN_COVERAGE_ANALYSIS}"
  echo "============================="
}

###################################################################################
# 4. 数据预处理
###################################################################################
# 打印所有参数
print_parameters

# Step1:提取N-gram编码
if [[ "$RUN_NGRAM_ENCODER" == "true" ]]; then
  echo "===== Step1 开始提取N-gram编码器  ====="  
  # 检查并删除已存在的ngram_encoder.json文件
  if [[ -f "${EXPERIMENT_DIR}/ngram_encoder.json" ]]; then
    echo "发现已存在的ngram_encoder.json文件，正在删除..."
    rm -f "${EXPERIMENT_DIR}/ngram_encoder.json"
  fi
  
  # 打印N-gram编码器训练参数
  echo "N-gram编码器训练参数:"
  echo "  --gue-dir: ${USE_GUE}"
  echo "  --input: ${USE_MSPECIES}"
  echo "  --output: ${EXPERIMENT_DIR}/ngram_encoder.json"
  echo "  --tok: ${TOKENIZER}"
  echo "  --min-ngram-len: 2"
  echo "  --max-ngram-len: 5"
  echo "  --max-ngrams: 30"
  echo "  --min-pmi: 2"
  echo "  --min-token-count: 5"
  echo "  --min-ngram-freq: 5"
  echo "  --method: pmi"
  echo "  --num-workers: 4"
  
  CMD="python ../src/train/train_ngram_encoder.py \
    --gue-dir ${USE_GUE} \
    --input ${USE_MSPECIES} \
    --output ${EXPERIMENT_DIR}/ngram_encoder.json \
    --tok ${TOKENIZER} \
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

  # 打印覆盖率分析参数
  echo "覆盖率分析参数:"
  echo "  --encoder: ${EXPERIMENT_DIR}/ngram_encoder.json"
  echo "  --output-dir: ${COVERAGE_DIR}"
  echo "  --tok: ${TOKENIZER}"
  echo "  --gue-dir: ${USE_GUE}"
  echo "  --mspecies-dir: ${USE_MSPECIES}"
  echo "  --ngram-list: ${EXPERIMENT_DIR}/ngram_list.txt"

  # 构建命令
  CMD="python ../src/dataset/ngram_encoder_analyze.py \
    --encoder ${EXPERIMENT_DIR}/ngram_encoder.json \
    --output-dir ${COVERAGE_DIR} \
    --tok ${TOKENIZER} \
    --gue-dir ${USE_GUE} \
    --mspecies-dir ${USE_MSPECIES} \
    --ngram-list ${EXPERIMENT_DIR}/ngram_list.txt "

  
  echo "执行命令: $CMD"
  eval $CMD
  
  
  if [[ $? -ne 0 ]]; then
    echo "N-gram编码覆盖率分析失败"
    exit 1
  fi
  
  echo "===== N-gram编码覆盖率分析完成 ====="
fi

echo "N-gram编码器路径: ${NGRAM_ENCODER_PATH}"
echo "N-gram编码覆盖率分析路径: ${COVERAGE_DIR}"
echo "所有选定的任务已完成"

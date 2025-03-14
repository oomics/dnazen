#!/bin/bash
#
# 微调脚本 - 支持多任务并行处理
#

###################################################################################
# 全局配置变量
###################################################################################

# 数据路径参数
EXPERIMENT_ID="exp1_pmi2"

# 输出路径参数
EXPERIMENT_DIR="../../out/${EXPERIMENT_ID}"
PRETRAIN_OUTPUT_DIR="${EXPERIMENT_DIR}/pretrain"

# 路径配置
GUE_DIR="../../GUE/"                        # 数据根目录
PRETRAIN_CHECKPOINT="${EXPERIMENT_DIR}/output/checkpoint-198" # 预训练模型检查点
#NGRAM_ENCODER_PATH="../../out/exp1_pmi2/" # NGram编码器路径
NGRAM_ENCODER_PATH="${EXPERIMENT_DIR}/ngram_encoder.json" # NGram编码器路径
#FINETUNE_OUT_DIR="../output/finetune"        # 微调输出目录
REPORT_OUT_DIR="../output/report"            # 报告输出目录


# 处理所有任务
echo "开始处理所有任务..."


echo "===============================run.sh 开始微调==============================================="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINETUNE_OUT_DIR="${EXPERIMENT_DIR}/parallel_${TIMESTAMP}" 
# 创建输出目录
mkdir -p "$FINETUNE_OUT_DIR"

FINETUNE_DATA_DIR="../data/pretrain/train/finetune" \
FINETUNE_OUT_DIR="$FINETUNE_OUT_DIR" \
MAIN_NGRAM_ENCODER_DIR="$NGRAM_ENCODER_PATH" \
DNAZEN_PRETRAIN_DATA_DIR="$PRETRAIN_OUTPUT_DIR" \
FINETUNE_CHECKPOINT_STEP=41 \
make -f makefile_finetune.mak all -j8

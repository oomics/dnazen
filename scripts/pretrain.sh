#!/bin/bash
###################################################################################
# 脚本名称: pretrain.sh
# 描述: DNA序列预训练模型训练启动脚本
# 用法: bash pretrain.sh [--experiment <id>] [--parent-experiment <name>]
# 作者: DnaZen Team
###################################################################################

# 解析命令行参数
EXPERIMENT_ID=""
PARENT_EXPERIMENT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    --parent-experiment)
      PARENT_EXPERIMENT="$2"
      shift 2
      ;;
    *)
      echo "未知选项: $1"
      echo "用法: bash pretrain.sh [--experiment <id>] [--parent-experiment <name>]"
      exit 1
      ;;
  esac
done

# 如果没有指定实验ID，使用默认值
if [[ -z "$EXPERIMENT_ID" ]]; then
  echo "警告: 未指定实验ID，使用默认值1"
  EXPERIMENT_ID="1"
fi

# 根据实验ID设置实验名称
case "$EXPERIMENT_ID" in
  1)
    EXPERIMENT_NAME="exp1_gue_mspecies"
    ;;
  2)
    EXPERIMENT_NAME="exp2_mspecies"
    ;;
  3)
    EXPERIMENT_NAME="exp3_gue"
    ;;
  4)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_5"
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  5)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_1"
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  6)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_100"
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  *)
    echo "错误: 无效的实验ID: $EXPERIMENT_ID"
    echo "有效的实验ID: 1-6"
    exit 1
    ;;
esac

# 如果指定了父实验，使用父实验目录
if [[ -n "$PARENT_EXPERIMENT" ]]; then
  EXPERIMENT_DIR="../data/pretrain/${PARENT_EXPERIMENT}"
else
  EXPERIMENT_DIR="../data/pretrain/${EXPERIMENT_NAME}"
fi

echo "使用实验: $EXPERIMENT_NAME (ID: $EXPERIMENT_ID)"
echo "实验目录: $EXPERIMENT_DIR"

###################################################################################
# 1. 数据和输出路径配置
###################################################################################

# 基础数据目录
DATA_DIR="../data/pretrain"
# 训练数据目录文件
TRAIN_DIR_FILE="$DATA_DIR/train/train.pt" 
# 验证数据目录文件
DEV_DIR_FILE="$DATA_DIR/dev/dev.pt"
# 模型保存输出目录
OUTPUT_DIR="$EXPERIMENT_DIR/output"
# 数据缓存目录
CACHE_DIR="$EXPERIMENT_DIR/cache"
# N-gram编码器路径
NGRAM_ENCODER_PATH="$EXPERIMENT_DIR/ngram_encoder.json"

# 获取训练和验证数据的目录路径
TRAIN_DIR=$(dirname "$TRAIN_DIR_FILE")
DEV_DIR=$(dirname "$DEV_DIR_FILE")

###################################################################################
# 2. 训练参数配置
###################################################################################
# GPU每批次样本数，H800可以使用128，A100/4090可以使用32
PER_DEVICE_BATCH_SIZE=16
# 梯度累积步数 (实际批大小 = PER_DEVICE_BATCH_SIZE * GRAD_ACCU_STEPS)，
# 4096=32*8*16，每过4096个sample（data）才做一次梯度下降batch
GRAD_ACCU_STEPS=16
# Ngram注意力层数
NUM_NGRAM_HIDDEN_LAYER=6
# 训练轮数
N_EPOCH=0.1
# 随机种子，确保实验可重复性
SEED=42
# 学习率
LEARNING_RATE=8e-5
# 数据加载器工作线程数
NUM_WORKERS=16
# 是否使用流式数据加载 (对于30G大文件推荐使用)
USE_STREAMING=true
# 流式数据加载缓冲区大小
BUFFER_SIZE=50000

###################################################################################
# 3. 目录准备
###################################################################################
# 创建模型输出目录(如果不存在)
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "创建输出目录: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

# 创建缓存目录(如果不存在且不使用流式加载)
if [ "$USE_STREAMING" = false ] && [ ! -d "$CACHE_DIR" ]; then
  echo "创建缓存目录: $CACHE_DIR"
  mkdir -p "$CACHE_DIR"
fi

###################################################################################
# 4. 输出训练参数信息
###################################################################################
echo "========================= DNA序列预训练开始 ========================="
echo "实验ID: $EXPERIMENT_ID"
echo "实验名称: $EXPERIMENT_NAME"
if [[ -n "$PARENT_EXPERIMENT" ]]; then
  echo "父实验: $PARENT_EXPERIMENT"
fi
echo "训练参数:"
echo "----------------------------------------"
echo "训练数据文件: $TRAIN_DIR_FILE"
echo "验证数据文件: $DEV_DIR_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "N-gram编码器路径: $NGRAM_ENCODER_PATH"
if [ "$USE_STREAMING" = false ]; then
  echo "缓存目录: $CACHE_DIR"
fi
echo "----------------------------------------"
echo "每设备批量大小: $PER_DEVICE_BATCH_SIZE"
echo "梯度累积步数: $GRAD_ACCU_STEPS"
echo "实际批大小: $((PER_DEVICE_BATCH_SIZE * GRAD_ACCU_STEPS))"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $N_EPOCH"
echo "随机种子: $SEED"
echo "Ngram隐藏层数: $NUM_NGRAM_HIDDEN_LAYER"
echo "数据加载器工作线程数: $NUM_WORKERS"
echo "数据加载模式: $([ "$USE_STREAMING" = true ] && echo "流式加载" || echo "标准加载")"
if [ "$USE_STREAMING" = true ]; then
  echo "流式加载缓冲区大小: $BUFFER_SIZE"
fi
echo "=================================================================="

###################################################################################
# 5. 启动训练进程
###################################################################################
# 获取脚本所在目录，确保可以正确找到训练脚本
SCRIPT_DIR=$(dirname "$0")
echo "正在启动训练..."

# 构建命令行参数
CMD="python ../src/train/run_pretrain.py \
  --train \"$TRAIN_DIR_FILE\" \
  --dev \"$DEV_DIR_FILE\" \
  --out \"$OUTPUT_DIR\" \
  --ngram-encoder-path \"$NGRAM_ENCODER_PATH\" \
  --num_ngram_hidden_layer $NUM_NGRAM_HIDDEN_LAYER \
  --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
  --grad-accumulation-steps $GRAD_ACCU_STEPS \
  --lr $LEARNING_RATE \
  --n-epoch $N_EPOCH \
  --seed $SEED \
  --num-workers $NUM_WORKERS \
  --train_dir \"$EXPERIMENT_DIR/pretrain_data/train\" \
  --dev_dir \"$EXPERIMENT_DIR/pretrain_data/dev\" "

  # --train_dir \"$TRAIN_DIR\" \
  # --dev_dir \"$DEV_DIR\"  "

# 根据数据加载模式添加相应参数
if [ "$USE_STREAMING" = true ]; then
  CMD="$CMD --streaming --buffer-size $BUFFER_SIZE"
else
  CMD="$CMD --cache-dir \"$CACHE_DIR\""
fi

# 输出完整命令
echo "执行命令: $CMD"

# 运行训练脚本
eval $CMD

# 检查训练是否成功完成
if [ $? -eq 0 ]; then
  echo "=================================================================="
  echo "训练成功完成！"
  echo "模型输出目录: $OUTPUT_DIR"
  echo "=================================================================="
else
  echo "=================================================================="
  echo "训练过程中出现错误，请检查日志。"
  echo "=================================================================="
  exit 1
fi 

#!/bin/bash
###################################################################################
# 脚本名称: pretrain.sh
# 描述: DNA序列预训练模型训练启动脚本
# 用法: bash pretrain.sh
# 作者: DnaZen Team
###################################################################################

###################################################################################
# 1. 数据和输出路径配置
###################################################################################
# 基础数据目录
DATA_DIR="../data/pretrain"
# 训练数据目录文件
TRAIN_DIR_FILE="$DATA_DIR/train/train.txt" 
# 验证数据目录文件
DEV_DIR_FILE="$DATA_DIR/dev/dev.txt"
# 模型保存输出目录
OUTPUT_DIR="$DATA_DIR/output"
# 数据缓存目录
CACHE_DIR="$DATA_DIR/cache"

###################################################################################
# 2. 训练参数配置
###################################################################################
# GPU每批次样本数
PER_DEVICE_BATCH_SIZE=128
# 梯度累积步数 (实际批大小 = PER_DEVICE_BATCH_SIZE * GRAD_ACCU_STEPS)
GRAD_ACCU_STEPS=4
# Ngram注意力层数
NUM_NGRAM_HIDDEN_LAYER=6
# 训练轮数
N_EPOCH=2
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
# 获取训练和验证数据的目录路径
TRAIN_DIR=$(dirname "$TRAIN_DIR_FILE")
DEV_DIR=$(dirname "$DEV_DIR_FILE")

# 创建训练数据目录(如果不存在)
if [ ! -d "$TRAIN_DIR" ]; then
  echo "训练数据目录不存在: $TRAIN_DIR"
  echo "正在创建目录..."
  mkdir -p "$TRAIN_DIR"
fi

# 创建验证数据目录(如果不存在)
if [ ! -d "$DEV_DIR" ]; then
  echo "验证数据目录不存在: $DEV_DIR"
  echo "正在创建目录..."
  mkdir -p "$DEV_DIR"
fi

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
echo "训练参数:"
echo "----------------------------------------"
echo "训练数据文件: $TRAIN_DIR_FILE"
echo "验证数据文件: $DEV_DIR_FILE"
echo "输出目录: $OUTPUT_DIR"
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
CMD="python \"$SCRIPT_DIR/run_pretrain.py\" \
  --train \"$TRAIN_DIR_FILE\" \
  --dev \"$DEV_DIR_FILE\" \
  --out \"$OUTPUT_DIR\" \
  --num_ngram_hidden_layer $NUM_NGRAM_HIDDEN_LAYER \
  --per-device-train-batch-size $PER_DEVICE_BATCH_SIZE \
  --grad-accumulation-steps $GRAD_ACCU_STEPS \
  --lr $LEARNING_RATE \
  --n-epoch $N_EPOCH \
  --seed $SEED \
  --num-workers $NUM_WORKERS \
  --resume ../resources/DNABERT-2-117M \
  --train_dir \"$TRAIN_DIR\" \
  --dev_dir \"$DEV_DIR\"  "

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

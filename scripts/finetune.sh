#!/bin/bash
###################################################################################
# 脚本名称: finetune.sh
# 描述: DNA序列微调训练模型训练启动脚本
# 备注: bash finetune.sh --experiment 1 emp H3K4me3
# 用法: bash finetune.sh [--experiment <id>] [--parent-experiment <name>] <任务类型> <子任务>
#       bash finetune.sh --experiment 1 --parallel [--max-workers <num>] [--gpu-ids <ids>]
# 作者: DnaZen Team
###################################################################################

# 解析命令行参数
EXPERIMENT_ID=""
PARENT_EXPERIMENT=""
TASK_TYPE=""
SUB_TASK=""
GUE_DIR="../data/GUE"
PARALLEL=false
MAX_WORKERS=4
GPU_IDS=""
RETRY_COUNT=1

# 首先处理选项参数
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
    --parallel)
      PARALLEL=true
      shift
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --retry-count)
      RETRY_COUNT="$2"
      shift 2
      ;;
    -*)
      echo "未知选项: $1"
      echo "用法: bash finetune.sh [--experiment <id>] [--parent-experiment <name>] <任务类型> <子任务>"
      echo "      bash finetune.sh --experiment <id> --parallel [--max-workers <num>] [--gpu-ids <ids>]"
      echo "任务类型: emp, pd, tf, mouse"
      echo "例如: bash finetune.sh --experiment 1 emp H3K4me3"
      echo "      bash finetune.sh --experiment 1 --parallel --max-workers 4 --gpu-ids 0,1,2,3"
      exit 1
      ;;
    *)
      # 如果不是选项参数，则认为是位置参数
      if [ -z "$TASK_TYPE" ]; then
        TASK_TYPE="$1"
      elif [ -z "$SUB_TASK" ]; then
        SUB_TASK="$1"
      else
        echo "错误: 提供了过多的参数"
        echo "用法: bash finetune.sh [--experiment <id>] [--parent-experiment <name>] <任务类型> <子任务>"
        exit 1
      fi
      shift
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
OUTPUT_DIR="../data/output"

# 微调评估结果保存目录
REPORT_OUT_DIR="~/jenkins/workspace/output/"

# 数据缓存目录
CACHE_DIR="$EXPERIMENT_DIR/cache"
# N-gram编码器路径
NGRAM_ENCODER_PATH="$EXPERIMENT_DIR/ngram_encoder.json"

# 获取训练和验证数据的目录路径
TRAIN_DIR=$(dirname "$TRAIN_DIR_FILE")
DEV_DIR=$(dirname "$DEV_DIR_FILE")

FINETUNE_DATA_DIR="$TRAIN_DIR/finetune"
FINETUNE_OUT_DIR="../data/output/finetune/output"
FINETUNE_CHECKPOINT_STEP=3000

###################################################################################
# 2. 训练参数配置
###################################################################################

PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=32
# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS=1
# 学习率
LEARNING_RATE=3e-5
# 默认训练轮数（根据任务类型可能会有不同设置）
NUM_TRAIN_EPOCHS=5
# 是否使用混合精度训练
USE_FP16=true

# 任务类型及其对应的训练轮数
declare -A TASK_EPOCHS
TASK_EPOCHS["emp"]=5
TASK_EPOCHS["pd"]=10
TASK_EPOCHS["tf"]=6
TASK_EPOCHS["mouse"]=6

# 任务类型及其对应的数据路径
declare -A TASK_PATHS
TASK_PATHS["emp"]="EMP"
TASK_PATHS["pd"]="prom"
TASK_PATHS["tf"]="tf"
TASK_PATHS["mouse"]="mouse"

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
echo "========================= DNA序列微调训练开始 ========================="
echo "实验ID EXPERIMENT_ID: $EXPERIMENT_ID"
echo "实验名称 EXPERIMENT_NAME: $EXPERIMENT_NAME"

if [ "$PARALLEL" = true ]; then
  echo "运行模式: 并行训练"
  echo "最大并行任务数: $MAX_WORKERS"
  if [ -n "$GPU_IDS" ]; then
    echo "指定GPU IDs: $GPU_IDS"
  fi
else
  echo "运行模式: 单任务训练"
  echo "任务类型 TASK_TYPE: $TASK_TYPE"
  echo "子任务 SUB_TASK: $SUB_TASK"
fi

echo "训练参数:"
echo "----------------------------------------"
echo "训练数据文件 TRAIN_DIR_FILE: $TRAIN_DIR_FILE"
echo "验证数据文件 DEV_DIR_FILE: $DEV_DIR_FILE"
echo "输出目录 OUTPUT_DIR: $OUTPUT_DIR"
echo "N-gram编码器路径 NGRAM_ENCODER_PATH: $NGRAM_ENCODER_PATH"
if [ "$USE_STREAMING" = false ]; then
  echo "缓存目录 CACHE_DIR: $CACHE_DIR"
fi

echo "=================================================================="

###################################################################################
# 5. 启动训练进程
###################################################################################
# 获取脚本所在目录，确保可以正确找到训练脚本
SCRIPT_DIR=$(dirname "$0")
echo "正在启动训练..."

# 检查必要的环境变量
if [ -z "$FINETUNE_DATA_DIR" ]; then
  echo "错误: 环境变量FINETUNE_DATA_DIR未设置"
  exit 1
fi

if [ -z "$FINETUNE_OUT_DIR" ]; then
  echo "错误: 环境变量FINETUNE_OUT_DIR未设置"
  exit 1
fi

if [ -z "$FINETUNE_CHECKPOINT_STEP" ]; then
  echo "错误: 环境变量FINETUNE_CHECKPOINT_STEP未设置"
  exit 1
fi

if [ -z "$EXPERIMENT_DIR" ]; then
  echo "错误: 环境变量EXPERIMENT_DIR未设置"
  exit 1
fi

if [ -z "$NGRAM_ENCODER_PATH" ]; then
  echo "错误: 环境变量NGRAM_ENCODER_PATH未设置"
  exit 1
fi

# 预训练检查点路径
#PRETRAIN_CHECKPOINT="${EXPERIMENT_DIR}/output/checkpoint-${FINETUNE_CHECKPOINT_STEP}"
PRETRAIN_CHECKPOINT="${OUTPUT_DIR}/checkpoint-${FINETUNE_CHECKPOINT_STEP}"

if [ "$PARALLEL" = true ]; then
  # 并行模式：创建任务配置文件
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  TASKS_CONFIG_PATH="${FINETUNE_OUT_DIR}/tasks_config_${TIMESTAMP}.json"
  PARALLEL_OUTPUT_DIR="${FINETUNE_OUT_DIR}/parallel_${TIMESTAMP}"
  
  echo "使用并行模式运行所有任务"
  echo "任务配置文件: $TASKS_CONFIG_PATH"
  echo "并行输出目录: $PARALLEL_OUTPUT_DIR"
  
  # 创建输出目录
  mkdir -p "$PARALLEL_OUTPUT_DIR"
  
  # 创建任务配置文件
#  cat > "$TASKS_CONFIG_PATH" << EOF
# {
#   "data_base_dir": "../data",
#   "tasks": [
#     {
#       "task_type": "tf",
#       "data_dir": "GUE/tf",
#       "sub_tasks": [ "3"],
#       "num_train_epochs": ${TASK_EPOCHS["tf"]},
#       "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
#       "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
#       "learning_rate": $LEARNING_RATE,
#       "fp16": $USE_FP16
#     },
#     {
#       "task_type": "pd",
#       "data_dir": "GUE/prom",
#       "sub_tasks": ["prom_300_all", "prom_core_all"],
#       "num_train_epochs": ${TASK_EPOCHS["pd"]},
#       "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
#       "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
#       "learning_rate": $LEARNING_RATE,
#       "fp16": $USE_FP16
#     },
#     {
#       "task_type": "emp",
#       "data_dir": "GUE/EMP",
#       "sub_tasks": ["H3"],
#       "num_train_epochs": ${TASK_EPOCHS["emp"]},
#       "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
#       "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
#       "learning_rate": $LEARNING_RATE,
#       "fp16": $USE_FP16
#     }
#   ]
# } 
# EOF

#  cat > "$TASKS_CONFIG_PATH" << EOF
# {
#   "data_base_dir": "../data",
#   "tasks": [
#     {
#       "task_type": "emp",
#       "data_dir": "GUE/EMP",
#       "sub_tasks": ["H3K14ac","H3K36me3","H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H4", "H4ac"],
#       "num_train_epochs": ${TASK_EPOCHS["emp"]},
#       "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
#       "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
#       "learning_rate": $LEARNING_RATE,
#       "fp16": $USE_FP16
#     }
#   ]
# } 
# EOF
  cat > "$TASKS_CONFIG_PATH" << EOF
{
  "data_base_dir": "../data",
  "tasks": [
    {
      "task_type": "tf",
      "data_dir": "GUE/tf",
      "sub_tasks": ["0", "1", "2", "3", "4"],
      "num_train_epochs": ${TASK_EPOCHS["tf"]},
      "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
      "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
      "learning_rate": $LEARNING_RATE,
      "fp16": $USE_FP16
    },
    {
      "task_type": "mouse",
      "data_dir": "GUE/mouse",
      "sub_tasks": ["0", "1", "2", "3", "4"],
      "num_train_epochs": ${TASK_EPOCHS["mouse"]},
      "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
      "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
      "learning_rate": $LEARNING_RATE,
      "fp16": $USE_FP16
    },
    {
      "task_type": "pd",
      "data_dir": "GUE/prom",
      "sub_tasks": ["prom_300_all", "prom_core_all"],
      "num_train_epochs": ${TASK_EPOCHS["pd"]},
      "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
      "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
      "learning_rate": $LEARNING_RATE,
      "fp16": $USE_FP16
    },
    {
      "task_type": "emp",
      "data_dir": "GUE/EMP",
      "sub_tasks": ["H3","H3K14ac","H3K36me3","H3K14me1", "H3K14me2", "H3K14me3", "H3K9ac", "H4", "H4ac"],
      "num_train_epochs": ${TASK_EPOCHS["emp"]},
      "per_device_train_batch_size": $PER_DEVICE_TRAIN_BATCH_SIZE,
      "per_device_eval_batch_size": $PER_DEVICE_EVAL_BATCH_SIZE,
      "learning_rate": $LEARNING_RATE,
      "fp16": $USE_FP16
    }
  ]
} 
EOF
echo "TASKS_CONFIG_PATH: $TASKS_CONFIG_PATH"
cat "$TASKS_CONFIG_PATH"

for task in $(jq -r '.tasks[] | .task_type + "/" + .sub_tasks[]' "$TASKS_CONFIG_PATH"); do
  echo "================================================"
  echo "任务: $task"
  echo "================================================"

  DATA_PATH="${GUE_DIR}/${TASK_PATHS[$TASK_TYPE]}/${SUB_TASK}"
  TASK_OUTPUT_PATH="${FINETUNE_OUT_DIR}/${TASK_TYPE}/${SUB_TASK}"   
  echo "数据路径 DATA_PATH: $DATA_PATH"
  echo "训练任务输出路径 TASK_OUTPUT_PATH: $TASK_OUTPUT_PATH"
  echo "训练轮数 NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"
  echo "微调评估结果保存目录 REPORT_OUT_DIR: $REPORT_OUT_DIR"
  
  # 创建输出目录
  mkdir -p "$TASK_OUTPUT_PATH"

  # 构建训练命令
  CMD="python ../src/train/run_finetune.py \
    --data_path $DATA_PATH \
    --checkpoint $PRETRAIN_CHECKPOINT \
    --ngram_encoder_dir $NGRAM_ENCODER_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS
    --out $TASK_OUTPUT_PATH"
  
  # 输出完整命令
  echo "执行命令: $CMD"
  
  # 运行训练脚本
  eval $CMD

  # 创建目标目录（如果不存在）
  mkdir -p "$OUTPUT_DIR/finetune/output"
  
  # 复制输出文件到报告保存文件夹，但排除checkpoint-*目录
  echo "复制输出文件到 $REPORT_OUT_DIR，排除checkpoint目录..."
  find "$TASK_OUTPUT_PATH" -type f -not -path "*/checkpoint-*/*" -exec cp --parents {} "$REPORT_OUT_DIR" \;
done
  
else
  # 单任务模式：运行单个任务
  # 检查任务类型是否有效
  if [[ ! "${!TASK_PATHS[@]}" =~ "$TASK_TYPE" ]]; then
    echo "错误: 无效的任务类型: $TASK_TYPE"
    echo "有效的任务类型: ${!TASK_PATHS[@]}"
    exit 1
  fi
  
  # 设置数据路径和输出路径
  DATA_PATH="${GUE_DIR}/${TASK_PATHS[$TASK_TYPE]}/${SUB_TASK}"
  TASK_OUTPUT_PATH="${FINETUNE_OUT_DIR}/${TASK_TYPE}/${SUB_TASK}"
  
  # 设置训练轮数
  NUM_TRAIN_EPOCHS=${TASK_EPOCHS[$TASK_TYPE]}
  
  echo "任务类型 TASK_TYPE: $TASK_TYPE"
  echo "任务类型 TASK_PATHS: ${TASK_PATHS[$TASK_TYPE]}"
  echo "子任务 SUB_TASK: $SUB_TASK"
  echo "数据路径 DATA_PATH: $DATA_PATH"
  echo "训练任务输出路径 TASK_OUTPUT_PATH: $TASK_OUTPUT_PATH"
  echo "训练轮数 NUM_TRAIN_EPOCHS: $NUM_TRAIN_EPOCHS"
  
  # 创建输出目录
  mkdir -p "$TASK_OUTPUT_PATH"
  
  # 构建训练命令
  CMD="python ../src/train/run_finetune.py \
    --data_path $DATA_PATH \
    --checkpoint $PRETRAIN_CHECKPOINT \
    --ngram_encoder_dir $NGRAM_ENCODER_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS"
  
  # 添加可选参数
  if [ "$USE_FP16" = true ]; then
    CMD="$CMD --fp16"
  fi
  
  # 添加输出目录
  CMD="$CMD --out $TASK_OUTPUT_PATH"
  
  # 输出完整命令
  echo "执行命令: $CMD"
  
  # 运行训练脚本
  eval $CMD
  
  
  # 检查训练是否成功完成
  if [ $? -eq 0 ]; then

    # 复制输出文件到报告保存文件夹，但排除checkpoint-*目录
    echo "复制输出文件到 $REPORT_OUT_DIR，排除checkpoint目录..."
    find "$TASK_OUTPUT_PATH" -type f -not -path "*/checkpoint-*/*" -exec cp --parents {} "$REPORT_OUT_DIR" \;

    echo "=================================================================="
    echo "微调训练成功完成！"
    echo "模型输出目录: $REPORT_OUT_DIR"
    echo "=================================================================="
  else
    echo "=================================================================="
    echo "训练过程中出现错误，请检查日志。"
    echo "=================================================================="
    exit 1
  fi
fi 

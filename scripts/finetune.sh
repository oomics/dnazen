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
EXPERIMENT_DIR="../data/pretrain/exp1_gue_mspecies"
RESUME=false  # 添加断点继续训练的标志
RESUME_FILE=""  # 添加断点记录文件路径
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
    --resume)
      RESUME=true
      shift
      ;;
    --resume-file)
      RESUME_FILE="$2"
      shift 2
      ;;
    -*)
      echo "未知选项: $1"
      echo "用法: bash finetune.sh [--experiment <id>] [--parent-experiment <name>] <任务类型> <子任务>"
      echo "      bash finetune.sh --experiment <id> --parallel [--max-workers <num>] [--gpu-ids <ids>]"
      echo "      bash finetune.sh --experiment <id> --parallel --resume [--resume-file <file>]"
      echo "任务类型: emp, pd, tf, mouse"
      echo "例如: bash finetune.sh --experiment 1 emp H3K4me3"
      echo "      bash finetune.sh --experiment 1 --parallel --max-workers 4 --gpu-ids 0,1,2,3"
      echo "      bash finetune.sh --experiment 1 --parallel --resume"
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
FINETUNE_CHECKPOINT_STEP=10000

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
# 任务处理函数
###################################################################################
process_task() {
  local task_type=$1
  local sub_task=$2
  local num_epochs=$3
  
  # 检查是否需要跳过此任务（断点继续功能）
  if [ "$RESUME" = true ] && [ -f "$RESUME_FILE" ]; then
    if grep -q "${task_type}/${sub_task}" "$RESUME_FILE"; then
      echo "跳过已完成的任务: ${task_type}/${sub_task}"
      return 0
    fi
  fi
  
  echo "================================================"
  echo "任务: $task_type/$sub_task"
  echo "================================================"

  local data_path="${GUE_DIR}/${TASK_PATHS[$task_type]}/${sub_task}"
  local task_output_path="${FINETUNE_OUT_DIR}/${task_type}/${sub_task}"   
  
  echo "数据路径 DATA_PATH: $data_path"
  echo "训练任务输出路径 TASK_OUTPUT_PATH: $task_output_path"
  echo "训练轮数 NUM_TRAIN_EPOCHS: $num_epochs"
  
  # 创建输出目录
  mkdir -p "$task_output_path"

  # 构建训练命令
  local cmd="python ../src/train/run_finetune.py \
    --data_path $data_path \
    --checkpoint $PRETRAIN_CHECKPOINT \
    --ngram_encoder_dir $NGRAM_ENCODER_PATH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr $LEARNING_RATE \
    --num_train_epochs $num_epochs \
    --out $task_output_path"
  
  # 添加可选参数
  if [ "$USE_FP16" = true ]; then
    cmd="$cmd --fp16"
  fi
  
  # 输出完整命令
  echo "执行命令: $cmd"
  
  # 运行训练脚本
  eval $cmd
  local result=$?
  
  if [ $result -eq 0 ]; then
    # 任务成功，记录到断点文件
    echo "${task_type}/${sub_task}" >> "${RESUME_FILE:-${FINETUNE_OUT_DIR}/completed_tasks.txt}"
    
    # 复制输出文件到报告保存文件夹，但排除checkpoint-*目录
    echo "复制输出文件到 $REPORT_OUT_DIR，排除checkpoint目录..."
    find "$task_output_path" -type f -not -path "*/checkpoint-*/*" -exec cp --parents {} "$REPORT_OUT_DIR" \;
    
    # 删除checkpoint目录以节省空间
    echo "删除checkpoint目录以节省空间..."
    find "$task_output_path" -type d -name "checkpoint-*" -exec rm -rf {} \; 2>/dev/null || true
    
    return 0
  else
    echo "任务 ${task_type}/${sub_task} 执行失败，退出代码: $result"
    return 1
  fi
}

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

# 设置断点记录文件
if [ "$RESUME" = true ] && [ -z "$RESUME_FILE" ]; then
  RESUME_FILE="${FINETUNE_OUT_DIR}/completed_tasks.txt"
  echo "使用默认断点记录文件: $RESUME_FILE"
fi

# 确保断点记录文件存在
if [ "$RESUME" = true ]; then
  if [ ! -f "$RESUME_FILE" ]; then
    echo "断点记录文件不存在，将创建新文件: $RESUME_FILE"
    mkdir -p "$(dirname "$RESUME_FILE")"
    touch "$RESUME_FILE"
  else
    echo "从断点记录文件继续: $RESUME_FILE"
    echo "已完成的任务:"
    cat "$RESUME_FILE"
  fi
  
  # 扫描任务目录，检查是否有已完成但未记录的任务
  echo "扫描任务目录，检查已完成但未记录的任务..."
  
  # 遍历任务类型目录
  for TASK_TYPE_DIR in "$FINETUNE_OUT_DIR"/*; do
    if [ -d "$TASK_TYPE_DIR" ]; then
      TASK_TYPE=$(basename "$TASK_TYPE_DIR")
      
      # 遍历子任务目录
      for SUB_TASK_DIR in "$TASK_TYPE_DIR"/*; do
        if [ -d "$SUB_TASK_DIR" ]; then
          SUB_TASK=$(basename "$SUB_TASK_DIR")
          TASK_ID="${TASK_TYPE}/${SUB_TASK}"
          
          # 检查是否存在eval_results.json文件
          if [ -f "$SUB_TASK_DIR/eval_results.json" ]; then
            # 检查任务是否已在断点记录文件中
            if ! grep -q "^$TASK_ID$" "$RESUME_FILE"; then
              echo "发现已完成但未记录的任务: $TASK_ID"
              echo "$TASK_ID" >> "$RESUME_FILE"
              echo "已添加到断点记录文件"
              
              # 打印评估结果内容
              echo "任务 $TASK_ID 的评估结果:"
              cat "$SUB_TASK_DIR/eval_results.json" | jq -C . || cat "$SUB_TASK_DIR/eval_results.json"
              echo "----------------------------------------"
            fi
          fi
        fi
      done
    fi
  done
  
  echo "断点记录文件更新完成，共记录 $(wc -l < "$RESUME_FILE") 个已完成任务"
fi

if [ "$PARALLEL" = true ]; then
  # 并行模式：使用数组管理任务
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  PARALLEL_OUTPUT_DIR="${FINETUNE_OUT_DIR}/parallel_${TIMESTAMP}"
  
  echo "使用并行模式运行所有任务"
  echo "并行输出目录: $PARALLEL_OUTPUT_DIR"
  echo "最大并行任务数: $MAX_WORKERS"
  
  # 创建输出目录
  mkdir -p "$PARALLEL_OUTPUT_DIR"
  
  # 定义任务数组
  declare -A TASK_CONFIGS
  
  # TF任务
  TF_SUBTASKS=("0" "1" "2" "3" "4")
  #TF_SUBTASKS=("1" "2" "3" "4")
  # Mouse任务
  MOUSE_SUBTASKS=("0" "1" "2" "3" "4")
  # PD任务
  PD_SUBTASKS=("prom_300_all" "prom_core_all")
  # EMP任务
  EMP_SUBTASKS=("H3" "H3K14ac" "H3K36me3" "H3K14me1" "H3K14me2" "H3K14me3" "H3K9ac" "H4" "H4ac")
  
  # 处理所有任务
  echo "开始处理所有任务..."
  
  # 处理TF任务
  for subtask in "${TF_SUBTASKS[@]}"; do
    process_task "tf" "$subtask" "${TASK_EPOCHS[tf]}"
    if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
      echo "任务失败，退出执行"
      exit 1
    fi
  done
  
  # 处理Mouse任务
  for subtask in "${MOUSE_SUBTASKS[@]}"; do
    process_task "mouse" "$subtask" "${TASK_EPOCHS[mouse]}"
    if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
      echo "任务失败，退出执行"
      exit 1
    fi
  done
  
  # 处理PD任务
  for subtask in "${PD_SUBTASKS[@]}"; do
    process_task "pd" "$subtask" "${TASK_EPOCHS[pd]}"
    if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
      echo "任务失败，退出执行"
      exit 1
    fi
  done
  
  # 处理EMP任务
  for subtask in "${EMP_SUBTASKS[@]}"; do
    process_task "emp" "$subtask" "${TASK_EPOCHS[emp]}"
    if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
      echo "任务失败，退出执行"
      exit 1
    fi
  done

  echo "所有任务处理完成！"

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
    # 记录完成的任务
    echo "${TASK_TYPE}/${SUB_TASK}" >> "${RESUME_FILE:-${FINETUNE_OUT_DIR}/completed_tasks.txt}"

    # 复制输出文件到报告保存文件夹，但排除checkpoint-*目录
    echo "复制输出文件到 $OUTPUT_DIR/finetune/output，排除checkpoint目录..."
    #find "$TASK_OUTPUT_PATH" -type f -not -path "*/checkpoint-*/*" -exec cp --parents {} "$REPORT_OUT_DIR" \;
    
    echo "=================================================================="
    echo "删除checkpoint目录以节省空间"
    echo "find ../data/output/finetune/output/ -type d -name "checkpoint-*" -exec du -sh {} \;"
    echo "find ../data/output/finetune/output/ -type d -name "checkpoint-*" -exec rm -rf {} \; 2>/dev/null || true;"
    echo "=================================================================="

    find "$TASK_OUTPUT_PATH" -type d -name "checkpoint-*" -exec du -sh {} \;
    find "$TASK_OUTPUT_PATH" -type d -name "checkpoint-*" -exec rm -rf {} \; 2>/dev/null || true

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

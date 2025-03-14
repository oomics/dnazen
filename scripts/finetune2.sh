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
FINETUNE_OUT_DIR="../output/finetune"        # 微调输出目录
REPORT_OUT_DIR="../output/report"            # 报告输出目录


# 运行模式配置
RESUME=true                                 # 是否从断点继续执行
RESUME_FILE="./finetune_progress.txt"        # 存储已完成任务的文件


# 微调参数配置
PER_DEVICE_TRAIN_BATCH_SIZE=32               # 训练批次大小
PER_DEVICE_EVAL_BATCH_SIZE=32                # 评估批次大小
GRADIENT_ACCUMULATION_STEPS=1                # 梯度累积步数
LEARNING_RATE=5e-5                           # 学习率
USE_FP16=true                                # 是否使用FP16精度训练

# 各任务训练轮数
declare -A TASK_EPOCHS
TASK_EPOCHS[tf]=5
TASK_EPOCHS[mouse]=5
TASK_EPOCHS[pd]=10
TASK_EPOCHS[emp]=5
TASK_EPOCHS[virus]=5
TASK_EPOCHS[splice]=5

# 各任务数据路径
declare -A TASK_PATHS
TASK_PATHS[tf]="tf"
TASK_PATHS[mouse]="mouse"
TASK_PATHS[pd]="pd"
TASK_PATHS[emp]="emp"
TASK_PATHS[virus]="virus"
TASK_PATHS[splice]="splice"

# 创建输出目录
mkdir -p "$FINETUNE_OUT_DIR"
mkdir -p "$REPORT_OUT_DIR"

# TF任务
TF_SUBTASKS=("0" "1" "2" "3" "4")
# Mouse任务
MOUSE_SUBTASKS=("0" "1" "2" "3" "4")
# PD任务
PD_SUBTASKS=("prom_300_all" "prom_core_all")
# EMP任务
EMP_SUBTASKS=("H3" "H3K14ac" "H3K36me3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K79me3" "H3K9ac" "H4" "H4ac")
# Virus任务
VIRUS_SUBTASKS=("covid")
# Splice任务
SPLICE_SUBTASKS=("reconstructed")

###################################################################################
# 打印配置信息函数
###################################################################################
print_config() {
    echo "============================================================"
    echo "                   微调脚本配置信息                         "
    echo "============================================================"
    
    echo -e "\n【运行模式配置】"
    echo "  断点继续执行: $RESUME"
    echo "  断点记录文件: $RESUME_FILE"
    
    echo -e "\n【路径配置】"
    echo "  数据根目录:                $GUE_DIR"
    echo "  预训练模型检查点:          $PRETRAIN_CHECKPOINT"
    echo "  NGram编码器路径:           $NGRAM_ENCODER_PATH"
    echo "  微调输出目录:              $FINETUNE_OUT_DIR"
    echo "  报告输出目录:              $REPORT_OUT_DIR"
    
    echo -e "\n【微调参数配置】"
    echo "  训练批次大小:              $PER_DEVICE_TRAIN_BATCH_SIZE"
    echo "  评估批次大小:              $PER_DEVICE_EVAL_BATCH_SIZE"
    echo "  梯度累积步数:              $GRADIENT_ACCUMULATION_STEPS"
    echo "  学习率:                    $LEARNING_RATE"
    echo "  FP16精度训练:              $USE_FP16"
    
    echo -e "\n【任务训练轮数】"
    for task in "${!TASK_EPOCHS[@]}"; do
        echo "  $task: ${TASK_EPOCHS[$task]}"
    done
    
    echo -e "\n【任务数据路径】"
    for task in "${!TASK_PATHS[@]}"; do
        echo "  $task: ${TASK_PATHS[$task]}"
    done
    
    echo -e "\n【任务子类型】"
    echo "  TF任务:     ${TF_SUBTASKS[*]}"
    echo "  Mouse任务:  ${MOUSE_SUBTASKS[*]}"
    echo "  PD任务:     ${PD_SUBTASKS[*]}"
    echo "  EMP任务:    ${EMP_SUBTASKS[*]}"
    
    echo -e "\n============================================================\n"
}

# 打印配置信息
print_config
  
# 处理所有任务
echo "开始处理所有任务..."


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
# 任务执行
###################################################################################

task1() {
    echo "Task 1 started"
    # 处理TF任务
    for subtask in "${TF_SUBTASKS[@]}"; do
      process_task "tf" "$subtask" "${TASK_EPOCHS[tf]}"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "任务失败，退出执行"
        exit 1
      fi
    done
    sleep 3
    echo "Task 1 completed"
}

task2() {
    echo "Task 2 started"
    # 处理Mouse任务
    for subtask in "${MOUSE_SUBTASKS[@]}"; do
      process_task "mouse" "$subtask" "${TASK_EPOCHS[mouse]}"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "任务失败，退出执行"
        exit 1
      fi
    done
    sleep 3
    echo "Task 2 completed"
}

task3() {
    echo "Task 3 started"
    # 处理PD任务
    for subtask in "${PD_SUBTASKS[@]}"; do
      process_task "pd" "$subtask" "${TASK_EPOCHS[pd]}"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "任务失败，退出执行"
        exit 1
      fi
    done
    sleep 3
    echo "Task 3 completed"
}

task4() {
    echo "Task 4 started"
    # 处理EMP任务
    for subtask in "${EMP_SUBTASKS[@]}"; do
      process_task "emp" "$subtask" "${TASK_EPOCHS[emp]}"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "任务失败，退出执行"
        exit 1
      fi
    done
    sleep 3
    echo "Task 4 completed"
}

task5() {
    echo "Task 5 started"
    # 处理Virus任务
    for subtask in "${VIRUS_SUBTASKS[@]}"; do
      process_task "virus" "$subtask" "${TASK_EPOCHS[virus]}"
    done
    sleep 3
    echo "Task 5 completed"
} 

task6() {
    echo "Task 6 started"
    # 处理Splice任务
    for subtask in "${SPLICE_SUBTASKS[@]}"; do
      process_task "splice" "$subtask" "${TASK_EPOCHS[splice]}"
    done
    sleep 3
    echo "Task 6 completed"
}

task1 
task2 
task3 
task4 
task5 
task6 

wait

echo "所有任务处理完成！"

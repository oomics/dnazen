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
TASK_PATHS[pd]="prom"
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

# 创建日志目录
LOG_DIR="${FINETUNE_OUT_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "日志将保存在目录: $LOG_DIR"

task1() {
    local LOG_FILE="${LOG_DIR}/task1_tf.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务1-TF] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 1 started" | tee -a "$LOG_FILE"
    # 处理TF任务
    for subtask in "${TF_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务1-TF] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "tf" "$subtask" "${TASK_EPOCHS[tf]}" 2>&1 | tee -a "$LOG_FILE"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务1-TF] 子任务失败，退出执行" | tee -a "$LOG_FILE"
        exit 1
      fi
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务1-TF] 完成" | tee -a "$LOG_FILE"
}

task2() {
    local LOG_FILE="${LOG_DIR}/task2_mouse.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务2-Mouse] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 2 started" | tee -a "$LOG_FILE"
    # 处理Mouse任务
    for subtask in "${MOUSE_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务2-Mouse] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "mouse" "$subtask" "${TASK_EPOCHS[mouse]}" 2>&1 | tee -a "$LOG_FILE"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务2-Mouse] 子任务失败，退出执行" | tee -a "$LOG_FILE"
        exit 1
      fi
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务2-Mouse] 完成" | tee -a "$LOG_FILE"
}

task3() {
    local LOG_FILE="${LOG_DIR}/task3_pd.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务3-PD] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 3 started" | tee -a "$LOG_FILE"
    # 处理PD任务
    for subtask in "${PD_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务3-PD] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "pd" "$subtask" "${TASK_EPOCHS[pd]}" 2>&1 | tee -a "$LOG_FILE"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务3-PD] 子任务失败，退出执行" | tee -a "$LOG_FILE"
        exit 1
      fi
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务3-PD] 完成" | tee -a "$LOG_FILE"
}

task4() {
    local LOG_FILE="${LOG_DIR}/task4_emp.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务4-EMP] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 4 started" | tee -a "$LOG_FILE"
    # 处理EMP任务
    for subtask in "${EMP_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务4-EMP] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "emp" "$subtask" "${TASK_EPOCHS[emp]}" 2>&1 | tee -a "$LOG_FILE"
      if [ $? -ne 0 ] && [ "$RESUME" = false ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务4-EMP] 子任务失败，退出执行" | tee -a "$LOG_FILE"
        exit 1
      fi
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务4-EMP] 完成" | tee -a "$LOG_FILE"
}

task5() {
    local LOG_FILE="${LOG_DIR}/task5_virus.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务5-Virus] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 5 started" | tee -a "$LOG_FILE"
    # 处理Virus任务
    for subtask in "${VIRUS_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务5-Virus] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "virus" "$subtask" "${TASK_EPOCHS[virus]}" 2>&1 | tee -a "$LOG_FILE"
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务5-Virus] 完成" | tee -a "$LOG_FILE"
} 

task6() {
    local LOG_FILE="${LOG_DIR}/task6_splice.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务6-Splice] 开始执行" | tee -a "$LOG_FILE"
    echo "Task 6 started" | tee -a "$LOG_FILE"
    # 处理Splice任务
    for subtask in "${SPLICE_SUBTASKS[@]}"; do
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务6-Splice] 处理子任务: $subtask" | tee -a "$LOG_FILE"
      process_task "splice" "$subtask" "${TASK_EPOCHS[splice]}" 2>&1 | tee -a "$LOG_FILE"
    done
    sleep 3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [任务6-Splice] 完成" | tee -a "$LOG_FILE"
}

# 并行执行所有任务
echo "开始并行执行所有任务..."
echo "可以通过查看 $LOG_DIR 目录下的日志文件跟踪各任务进度"

task1 > >(tee -a "${LOG_DIR}/stdout_task1.log") 2> >(tee -a "${LOG_DIR}/stderr_task1.log" >&2) &
task2 > >(tee -a "${LOG_DIR}/stdout_task2.log") 2> >(tee -a "${LOG_DIR}/stderr_task2.log" >&2) &
task3 > >(tee -a "${LOG_DIR}/stdout_task3.log") 2> >(tee -a "${LOG_DIR}/stderr_task3.log" >&2) &
task4 > >(tee -a "${LOG_DIR}/stdout_task4.log") 2> >(tee -a "${LOG_DIR}/stderr_task4.log" >&2) &
task5 > >(tee -a "${LOG_DIR}/stdout_task5.log") 2> >(tee -a "${LOG_DIR}/stderr_task5.log" >&2) &
task6 > >(tee -a "${LOG_DIR}/stdout_task6.log") 2> >(tee -a "${LOG_DIR}/stderr_task6.log" >&2) &

wait

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有任务处理完成！"
echo "详细日志可在 $LOG_DIR 目录下查看"

# 生成汇总报告
SUMMARY_LOG="${LOG_DIR}/summary.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 任务执行汇总" > "$SUMMARY_LOG"
echo "================================================" >> "$SUMMARY_LOG"
echo "任务完成统计:" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task1_tf.log" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task2_mouse.log" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task3_pd.log" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task4_emp.log" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task5_virus.log" >> "$SUMMARY_LOG"
grep -c "完成" "${LOG_DIR}/task6_splice.log" >> "$SUMMARY_LOG"
echo "================================================" >> "$SUMMARY_LOG"

echo "已生成任务汇总报告: $SUMMARY_LOG"

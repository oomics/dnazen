#!/bin/bash

# 设置默认值
RUN_NGRAM_ENCODER=false
RUN_TOKENIZE_TRAIN=false
RUN_TOKENIZE_DEV=false
RUN_PREPARE_DATASET=false
RUN_PLOTS_ONLY=false
RUN_COVERAGE_ANALYSIS=false
RUN_PRETRAIN=false  # 是否运行预训练
RUN_FINE_TUNE=false  # 新增：是否运行微调
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
  echo "  --pretrain             执行预训练"
  echo "  --fine-tune            执行微调训练"
  echo "  --experiment <id>      指定实验ID (1-6)"
  echo "                         1: GUE + mspecies/dev"
  echo "                         2: 仅 mspecies/dev"
  echo "                         3: 仅 GUE"
  echo "                         4: 基于实验3，ngram最小频率为5"
  echo "                         5: 基于实验3，ngram最小频率为1"
  echo "                         6: 基于实验3，ngram最小频率为100"
  echo "  -h, --help             显示此帮助信息"
  echo ""
  echo "示例:"
  echo "  $0 --train-ngram --experiment 3  # 使用GUE数据集训练N-gram编码器"
  echo "  $0 --all --experiment 1          # 使用GUE+mspecies/dev执行所有步骤"
  echo "  $0 --pretrain --experiment 3     # 使用实验3的配置执行预训练"
  echo "  $0 --fine-tune --experiment 3    # 使用实验3的配置执行微调训练"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_NGRAM_ENCODER=true
      RUN_TOKENIZE_TRAIN=true
      RUN_TOKENIZE_DEV=true
      RUN_PREPARE_DATASET=true
      RUN_PRETRAIN=true
      RUN_FINE_TUNE=true
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
    --pretrain)
      RUN_PRETRAIN=true
      shift
      ;;
    --fine-tune)
      RUN_FINE_TUNE=true
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
if [[ "$RUN_NGRAM_ENCODER" == "false" && "$RUN_TOKENIZE_TRAIN" == "false" && "$RUN_TOKENIZE_DEV" == "false" && "$RUN_PREPARE_DATASET" == "false" && "$RUN_PLOTS_ONLY" == "false" && "$RUN_COVERAGE_ANALYSIS" == "false" && "$RUN_PRETRAIN" == "false" && "$RUN_FINE_TUNE" == "false" ]]; then
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

# 根据实验ID设置输出目录和参数
case "$EXPERIMENT_ID" in
  1)
    EXPERIMENT_NAME="exp1_gue_mspecies"
    EXPERIMENT_DESC="GUE数据集+mspecies/dev数据集抽取ngram，使用mspecies/train数据进行pretrain"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT="--input ../data/pretrain/dev/dev.txt"
    MIN_FREQ_FILTER=5    
    ;;
  2)
    EXPERIMENT_NAME="exp2_mspecies"
    EXPERIMENT_DESC="仅mspecies/dev数据集抽取ngram"
    USE_GUE=""
    USE_INPUT="--input ../data/pretrain/dev/dev.txt"
    MIN_FREQ_FILTER=5
    ;;
  3)
    EXPERIMENT_NAME="exp3_gue"
    EXPERIMENT_DESC="仅GUE数据集抽取ngram"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT=""
    MIN_FREQ_FILTER=5
    ;;
  4)
    EXPERIMENT_NAME="exp4_ms_dev_pretrain"
    EXPERIMENT_DESC="GUE数据集+mspecies/dev数据集抽取ngram，仅mspecies/dev数据进行pretrain"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT="--input ../data/pretrain/dev/dev.txt"
    ;;
  5)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_1"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为1"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT=""
    MIN_FREQ_FILTER=1
    PARENT_EXPERIMENT="exp3_gue"
    ;;
  6)
    EXPERIMENT_NAME="exp3_gue_ngram_ref_100"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为100"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT=""
    MIN_FREQ_FILTER=100
    PARENT_EXPERIMENT="exp3_gue"
    ;;

  7)
    EXPERIMENT_NAME="exp7_gue_ngram_ref_5"
    EXPERIMENT_DESC="基于实验3数据进行分析，仅GUE数据集抽取ngram，分析时ngram最小频率为5"
    USE_GUE="--gue-dir ../data/GUE"
    USE_INPUT=""
    MIN_FREQ_FILTER=5
    PARENT_EXPERIMENT="exp3_gue"
esac

# 创建必要的目录
#mkdir -p ../data/pretrain/tokenized
#mkdir -p ../data/pretrain/dev
#mkdir -p ../data/pretrain/train


###################################################################################
# 1. 数据下载准备
###################################################################################
echo "===== 开始下载数据 ====="
mkdir -p ../data/downloads
cd ../data/downloads

# 下载GUE数据集
if [ ! -f "GUE.zip" ]; then
  echo "正在下载GUE数据集..."
  echo "如果自动下载失败，请手动下载："
  echo "GUE数据集下载链接: https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view?usp=sharing"
  echo "下载后请将文件保存为: ../data/downloads/GUE.zip"
  echo "或使用以下命令下载:"
  echo "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=\$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2\" -O GUE.zip && rm -rf /tmp/cookies.txt"
  curl -L -o GUE.zip "https://drive.usercontent.google.com/download?id=1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2&export=download&authuser=0&confirm=t"
  
  # 检查下载是否成功
  if [ ! -f "GUE.zip" ] || [ ! -s "GUE.zip" ]; then
    echo "警告: GUE.zip下载失败或文件为空，请使用上述手动方法下载"
  fi
else
  echo "GUE.zip已存在，跳过下载"
fi

# 下载预训练数据集
if [ ! -f "dnabert_2_pretrain.zip" ]; then
  echo "正在下载预训练数据集..."
  echo "如果自动下载失败，请手动下载："
  echo "预训练数据集下载链接: https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view?usp=sharing"
  echo "下载后请将文件保存为: ../data/downloads/dnabert_2_pretrain.zip"
  echo "或使用以下命令下载:"
  echo "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=\$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f\" -O dnabert_2_pretrain.zip && rm -rf /tmp/cookies.txt"
  curl -L -o dnabert_2_pretrain.zip "https://drive.usercontent.google.com/download?id=1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f&export=download&authuser=0&confirm=t"
  
  # 检查下载是否成功
  if [ ! -f "dnabert_2_pretrain.zip" ] || [ ! -s "dnabert_2_pretrain.zip" ]; then
    echo "警告: dnabert_2_pretrain.zip下载失败或文件为空，请使用上述手动方法下载"
  fi
else
  echo "dnabert_2_pretrain.zip已存在，跳过下载"
fi

cd - > /dev/null  # 返回原目录
echo "===== 数据下载完成 ====="

###################################################################################
# 2. 数据解压
###################################################################################
echo "===== 开始解压数据 ====="

# 解压GUE数据集
if [ ! -d "../data/GUE" ]; then
  echo "正在解压GUE数据集..."
  mkdir -p ../data/GUE
  echo "解压进度:"
  unzip ../data/downloads/GUE.zip -d ../data/ | grep -E '^\s*[0-9]+%' | awk 'NR % 100 == 0'
else
  echo "GUE数据集目录已存在，跳过解压"
fi

# 解压预训练数据集
if [ ! -d "../data/pretrain/train" ] || [ ! -d "../data/pretrain/dev" ]; then
  echo "正在解压预训练数据集..."
  echo "解压进度:"
  unzip ../data/downloads/dnabert_2_pretrain.zip -d ../data/pretrain | grep -E '^\s*[0-9]+%' | awk 'NR % 100 == 0'
  
  # 确保目录结构正确
  mkdir -p ../data/pretrain/train
  mkdir -p ../data/pretrain/dev
  
  # 如果解压后的文件结构不同，可能需要移动文件
  if [ -f "../data/pretrain/train.txt" ] && [ ! -f "../data/pretrain/train/train.txt" ]; then
    mv ../data/pretrain/train.txt ../data/pretrain/train/
  fi
  
  if [ -f "../data/pretrain/dev.txt" ] && [ ! -f "../data/pretrain/dev/dev.txt" ]; then
    mv ../data/pretrain/dev.txt ../data/pretrain/dev/
  fi
else
  echo "预训练数据集目录已存在，跳过解压"
fi

echo "===== 数据解压完成 ====="

###################################################################################
# 3. 实验目录准备
###################################################################################
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

###################################################################################
# 4. 数据预处理
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
    --min-pmi 5 \
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
  
  # 构建命令
  CMD="python ../src/dataset/ngram_encoder_analyze.py \
    --encoder ${EXPERIMENT_DIR}/ngram_encoder.json \
    --output-dir ${COVERAGE_DIR} \
    --tok zhihan1996/DNABERT-2-117M \
    --gue-dir ../data/GUE \
    --mspecies-dir ../data/pretrain/dev/dev.txt \
    --ngram-list ${EXPERIMENT_DIR}/ngram_list.txt 

  
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
    --data ../data/pretrain/train/train.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ../data/pretrain/train/train.pt \
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
    --data ../data/pretrain/dev/dev.txt \
    --tok zhihan1996/DNABERT-2-117M \
    --out ../data/pretrain/dev/dev.pt \
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
    --data ../data/pretrain/ \
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
# 5. 启动预训练进程
###################################################################################
if [[ "$RUN_PRETRAIN" == "true" ]]; then
  echo "===== Step5 开始预训练 使用实验${EXPERIMENT_ID}的配置 ====="
  
  # 构建预训练命令，传递实验ID和父实验名称
  PRETRAIN_CMD="bash ./pretrain.sh --experiment ${EXPERIMENT_ID}"
  if [[ -n "$PARENT_EXPERIMENT" ]]; then
    PRETRAIN_CMD="${PRETRAIN_CMD} --parent-experiment ${PARENT_EXPERIMENT}"
  fi
  
  echo "执行命令: ${PRETRAIN_CMD}"
  eval ${PRETRAIN_CMD}
  
  if [[ $? -ne 0 ]]; then
    echo "预训练失败"
    exit 1
  fi
  echo "===== 预训练完成 ====="
fi

###################################################################################
# 6. 启动微调训练进程
###################################################################################
if [[ "$RUN_FINE_TUNE" == "true" ]]; then
  echo "===== Step6 开始微调训练 使用实验${EXPERIMENT_ID}的配置 ====="
  
  # 构建预训练命令，传递实验ID和父实验名称
  FINE_TUNE_CMD="bash ./finetune.sh --experiment ${EXPERIMENT_ID}"
  if [[ -n "$PARENT_EXPERIMENT" ]]; then
    FINE_TUNE_CMD="${FINE_TUNE_CMD} --parent-experiment ${PARENT_EXPERIMENT}"
  fi
  
  echo "执行命令: ${FINE_TUNE_CMD}"
  eval ${FINE_TUNE_CMD}
  
  if [[ $? -ne 0 ]]; then
    echo "预训练失败"
    exit 1
  fi
  echo "===== 预训练完成 ====="
fi

echo "所有选定的任务已完成"

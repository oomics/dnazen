#!/bin/bash

# 预训练数据准备脚本
# 用法: ./prepare_pretrain_data.sh [参数]

# 默认参数
DATA_SOURCE="raw"                           # 数据类型：raw或tokenized
DATA_DIR="/data2/peter/dnazen_pretrain_v3"  # 数据目录
TOK_SOURCE="huggingface"                    # tokenizer来源：file或huggingface
TOKENIZER="zhihan1996/DNABERT-2-117M"       # tokenizer配置
NGRAM_FILE=""                               # n-gram文件路径
CORE_NGRAM=""                               # 核心n-gram文件路径
MAX_NGRAMS=30                               # 最大n-gram匹配数
OUTPUT_DIR=""                               # 输出目录
SEED=42                                     # 随机种子

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-source)
      DATA_SOURCE="$2"
      shift 2
      ;;
    --data-dir|-d)
      DATA_DIR="$2"
      shift 2
      ;;
    --tok-source)
      TOK_SOURCE="$2"
      shift 2
      ;;
    --tokenizer|--tok)
      TOKENIZER="$2"
      shift 2
      ;;
    --ngram)
      NGRAM_FILE="$2"
      shift 2
      ;;
    --core-ngram)
      CORE_NGRAM="$2"
      shift 2
      ;;
    --max-ngrams)
      MAX_NGRAMS="$2"
      shift 2
      ;;
    --output-dir|--out)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --help|-h)
      echo "用法: ./prepare_pretrain_data.sh [参数]"
      echo "参数:"
      echo "  --data-source    数据类型：raw或tokenized (默认: raw)"
      echo "  --data-dir, -d   数据目录 (默认: /data2/peter/dnazen_pretrain_v3)"
      echo "  --tok-source     tokenizer来源：file或huggingface (默认: huggingface)"
      echo "  --tokenizer      tokenizer配置 (默认: zhihan1996/DNABERT-2-117M)"
      echo "  --ngram          n-gram文件路径 (必需)"
      echo "  --core-ngram     核心n-gram文件路径 (可选)"
      echo "  --max-ngrams     最大n-gram匹配数 (默认: 30)"
      echo "  --output-dir     输出目录 (必需)"
      echo "  --seed           随机种子 (默认: 42)"
      echo "  --help, -h       显示帮助信息"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

# 检查必需参数
if [ -z "$NGRAM_FILE" ]; then
  echo "错误: 必须指定n-gram文件路径 (--ngram)"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "错误: 必须指定输出目录 (--output-dir)"
  exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python scripts/make_pretrain_dataset.py"
CMD+=" --data-source $DATA_SOURCE"
CMD+=" --data $DATA_DIR"
CMD+=" --tok-source $TOK_SOURCE"
CMD+=" --tok $TOKENIZER"
CMD+=" --ngram $NGRAM_FILE"
CMD+=" --max-ngrams $MAX_NGRAMS"
CMD+=" --out $OUTPUT_DIR"
CMD+=" --seed $SEED"

# 添加可选参数
if [ -n "$CORE_NGRAM" ]; then
  CMD+=" --core-ngram $CORE_NGRAM"
fi

# 打印执行命令
echo "执行命令: $CMD"

# 执行命令
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
  echo "数据准备完成！输出目录: $OUTPUT_DIR"
else
  echo "数据准备失败，请检查错误信息"
  exit 1
fi 

# 使用dev.txt数据 训练N-gram编码器
python train_ngram_encoder.py \
  --input ../data/pretrain/dev/dev.txt \
  --output ../data/pretrain/dev/ngram_encoder.json \
  --min-ngram-len 2 \
  --max-ngram-len 5 \
  --max-ngrams 30 \
  --min-pmi 0.5 \
  --min-token-count 2 \
  --min-ngram-freq 2 \
  --method pmi \
  --num-workers 4


# 为train.txt数据 生成tokenized数据
python scripts/make_tokenized_dataset.py \
  --data ../data/pretrain/raw/train.txt \
  --tok zhihan1996/DNABERT-2-117M \
  --out ../data/pretrain/tokenized/train.pt

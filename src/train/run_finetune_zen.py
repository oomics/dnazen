# import deepspeed
from typing import Any
import argparse
import os
import csv
import json
import logging
from typing import Dict, Sequence, List

import numpy as np

# import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import pandas as pd
import transformers
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.models.bert.configuration_bert import BertConfig

from dnazen.model.bert_models import BertForSequenceClassification
from dnazen.ngram import NgramEncoder
from dnazen.data.labeled_dataset import LabeledDataset
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# @dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning("Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


class SupervisedDataset(Dataset):
    """用于监督训练的数据集类

    支持两种格式的输入:
    1. [text, label] - 单序列分类
    2. [text1, text2, label] - 序列对分类
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            kmer: k-mer大小，-1表示不使用k-mer
        """
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        # 设置一个合理的最大长度，避免整数溢出
        max_length = min(tokenizer.model_max_length, 512) if hasattr(tokenizer, "model_max_length") else 512

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,  # 使用安全的最大长度值
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],  # this line removes necessarily of using `DataCollator`
        )


def compute_metrics(eval_pred):
    """
    Compute metrics used for huggingface trainer.
    """
    predictions, labels = eval_pred
    # calculate metric with sklearn
    predictions: np.ndarray
    labels: np.ndarray

    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(valid_labels, valid_predictions),
        "precision": precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }


# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits: torch.Tensor | tuple[torch.Tensor, Any], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for sequence classification.")
    parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data2/peter/dnazen_pretrain_v3.1/outputs/checkpoint-2000",
        help="Path to model checkpoint directory.",
    )
    parser.add_argument("--out", type=str, help="Path to the results directory.")
    # parser.add_argument("--model_max_len", type=int, default=70, help="Maximum length of the model input.")
    parser.add_argument("--ngram_encoder_dir", type=str, help="Path to the ngram encoder directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bert", action="store_true", help="Use dnabert2 instead of dnazen")
    parser.add_argument("--seed", default=42)

    return parser.parse_args()


# --- main ---
def main():
    # 步骤1: 解析命令行参数
    logger.info("步骤1: 解析命令行参数...")
    args = parse_args()

    DATA_PATH = args.data_path
    RESULTS_PATH = args.out
    # MODEL_MAX_LEN = args.model_max_len
    NGRAM_ENCODER_DIR = args.ngram_encoder_dir
    LEARNING_RATE = args.lr
    NUM_TRAIN_EPOCHS = args.num_train_epochs
    CHECKPOINT_DIR = args.checkpoint
    USE_DNABERT2 = args.bert

    logger.info(f"数据路径: {DATA_PATH}")
    logger.info(f"结果输出路径: {RESULTS_PATH}")
    logger.info(f"N-gram编码器路径: {NGRAM_ENCODER_DIR}")
    logger.info(f"学习率: {LEARNING_RATE}")
    logger.info(f"训练轮数: {NUM_TRAIN_EPOCHS}")
    logger.info(f"模型检查点路径: {CHECKPOINT_DIR}")

    # 步骤2: 准备输出目录
    logger.info("步骤2: 准备输出目录...")
    if not os.path.exists(RESULTS_PATH):
        logger.info(f"创建输出目录: {RESULTS_PATH}")
        os.makedirs(RESULTS_PATH)
    else:
        logger.info(f"输出目录已存在: {RESULTS_PATH}")

    # 步骤3: 加载分词器
    logger.info("步骤3: 加载分词器...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        # model_max_length=MODEL_MAX_LEN,
        padding_side="right",
        padding="longest",
        use_fast=True,
        trust_remote_code=True,
    )  # type: ignore
    logger.info("分词器加载完成")

    # 步骤4: 加载n-gram编码器
    if USE_DNABERT2:
        ngram_encoder = None
    else:
        logger.info(f"步骤4: 加载n-gram编码器，路径: {NGRAM_ENCODER_DIR}")
        ngram_encoder = NgramEncoder.from_file(NGRAM_ENCODER_DIR)
        logger.info(f"n-gram词汇表大小: {ngram_encoder.get_vocab_size()}")


    # 步骤5: 加载数据集
    logger.info("步骤5: 加载训练、验证和测试数据集...")

    if USE_DNABERT2:
        # define datasets and data collator
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=f"{DATA_PATH}/train.csv", kmer=-1)
        val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=f"{DATA_PATH}/dev.csv", kmer=-1)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=f"{DATA_PATH}/test.csv", kmer=-1)
    else:
        logger.info(f"加载训练集: {DATA_PATH}/train.csv")
        train_dataset = LabeledDataset(
            f"{DATA_PATH}/train.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder
        )
        logger.info(f"训练集大小: {len(train_dataset)}个样本")

        logger.info(f"加载测试集: {DATA_PATH}/test.csv")
        test_dataset = LabeledDataset(
            f"{DATA_PATH}/test.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder
        )
        logger.info(f"测试集大小: {len(test_dataset)}个样本")

        logger.info(f"加载验证集: {DATA_PATH}/dev.csv")
        val_dataset = LabeledDataset(
            f"{DATA_PATH}/dev.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder
        )
        logger.info(f"验证集大小: {len(val_dataset)}个样本")

    # 步骤6: 加载模型配置和模型
    logger.info("-------------------------------------------------------...")
    logger.info(f"步骤6: 从检查点加载模型，路径: {CHECKPOINT_DIR}")
    logger.info("加载模型配置...")
    config = BertConfig.from_pretrained(CHECKPOINT_DIR)
    # setattr(config, "ngram_vocab_size", ngram_encoder.get_vocab_size())
    if USE_DNABERT2:
        setattr(config, "use_zen", False)
        setattr(config, "num_word_hidden_layers", 0)
    else:
        setattr(config, "use_zen", True)
        setattr(config, "num_word_hidden_layers", 6)
    logger.info(f"模型配置: num_labels={config.num_labels}, hidden_size={config.hidden_size}")

    logger.info("加载预训练模型...")
    model = BertForSequenceClassification.from_pretrained(CHECKPOINT_DIR, config=config)
    #        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    # load model
    # model = transformers.AutoModelForSequenceClassification.from_pretrained(
    #     "zhihan1996/DNABERT-2-117M",
    #     cache_dir=CHECKPOINT_DIR,
    #     num_labels=train_dataset.num_labels,
    #     trust_remote_code=True,
    # )

    logger.info("模型加载完成")

    # 步骤7: 设置优化器
    logger.info("-------------------------------------------------------...")
    logger.info("步骤7: 配置优化器...")
    logger.info(f"使用AdamW优化器，学习率={LEARNING_RATE}，权重衰减=0.01")

    ngram_layer_params = [
        param for name, param in model.named_parameters() if ("ngram_layer" in name and "Norm" not in name)
    ]
    logger.info(f"n-gram层参数数量: {len(ngram_layer_params)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        LEARNING_RATE,
        # momentum=args.momentum,
        weight_decay=0.01,
    )

    # 步骤8: 设置训练参数
    logger.info("-------------------------------------------------------...")
    logger.info("步骤8: 配置训练参数...")
    train_args = transformers.training_args.TrainingArguments(
        output_dir=RESULTS_PATH,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_steps=200,
        eval_steps=200,
        # max_grad_norm=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=LEARNING_RATE,
        bf16=args.fp16,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=8,
        logging_steps=200,
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_matthews_correlation",
        greater_is_better=True,
    )
    logger.info(f"训练批次大小: {args.per_device_train_batch_size}")
    logger.info(f"评估批次大小: {args.per_device_train_batch_size}")
    logger.info(f"训练轮数: {NUM_TRAIN_EPOCHS}")
    logger.info(f"随机种子: {args.seed}")

    # 步骤9: 初始化Trainer
    logger.info("-------------------------------------------------------...")
    logger.info("步骤9: 初始化Trainer...")
    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        optimizers=(optimizer, None),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    logger.info("Trainer初始化完成")

    # 步骤10: 开始训练
    logger.info("步骤10: 开始训练...")
    logger.info(f"训练数据集大小: {len(train_dataset)}个样本")
    logger.info(f"验证数据集大小: {len(val_dataset)}个样本")
    trainer.train()
    logger.info("训练完成")

    # 步骤11: 评估模型
    logger.info("-------------------------------------------------------...")
    logger.info("步骤11: 在测试集上评估模型...")
    results = trainer.evaluate(test_dataset)

    # 打印评估结果
    logger.info(f"{DATA_PATH}测试集评估结果:")
    logger.info("-" * 50)
    logger.info(f"{'指标名称':<30}{'值':>15}")
    logger.info("-" * 50)

    # 按字母顺序排序指标并打印
    for metric_name in sorted(results.keys()):
        metric_value = results[metric_name]
        if isinstance(metric_value, float):
            logger.info(f"{metric_name:<30}{metric_value:>15.6f}")
        else:
            logger.info(f"{metric_name:<30}{metric_value:>15}")
    logger.info("-" * 50)

    # 保存评估结果
    results_path = os.path.join(RESULTS_PATH, "eval_results.json")
    logger.info(f"保存评估结果到: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)  # 添加缩进使JSON文件更易读

    # 步骤12: 在测试集上进行预测
    logger.info("-------------------------------------------------------...")
    logger.info("步骤12: 在测试集上进行预测...")
    logger.info(f"测试数据集大小: {len(test_dataset)}个样本")
    test_output = trainer.predict(test_dataset)
    test_preds = test_output.predictions
    logger.info(f"预测完成，获得{len(test_preds)}个预测结果")

    # 步骤13: 整理预测结果
    logger.info("步骤13: 整理预测结果...")
    results = {
        "text": [],
        "actual_label": [],
        "prediction_label": [],
    }
    for i in range(len(test_dataset)):
        data = test_dataset[i]

        input_ids = data["input_ids"]
        actual_label = data["labels"]
        texts = (
            tokenizer.decode(input_ids).replace("[CLS] ", "").replace(" [SEP]", "").replace(" [PAD]", "")
        )

        results["text"].append(texts)
        results["actual_label"].append(actual_label)
        results["prediction_label"].append(test_preds[i])

    logger.info("-------------------------------------------------------...")
    # 步骤14: 保存预测结果
    logger.info("步骤14: 保存预测结果...")
    pred_results_path = os.path.join(RESULTS_PATH, "pred_results.csv")
    logger.info(f"保存预测结果到: {pred_results_path}")
    df = pd.DataFrame(results)
    df.to_csv(pred_results_path, mode="w")
    logger.info(f"已保存{len(df)}行预测结果")

    # # 步骤15: 生成HTML可视化报告
    # logger.info("步骤15: 生成HTML可视化报告...")
    # html_path = os.path.join(RESULTS_PATH, "prediction_report.html")
    # generate_prediction_html_report(df, results, html_path)

    logger.info("所有步骤完成")


if __name__ == "__main__":
    main()


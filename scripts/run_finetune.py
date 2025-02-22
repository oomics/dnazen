# import deepspeed
from typing import Any
import argparse
import os
import json

import numpy as np
# import sklearn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import pandas as pd
import transformers
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig

from dnazen.model.bert_models import BertForMaskedLM, BertForSequenceClassification
from dnazen.data.labeled_dataset import LabeledDataset, LabeledData, ZenLabeledData
from dnazen.ngram import NgramEncoder


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # calculate metric with sklearn
    predictions: np.ndarray
    labels: np.ndarray
    
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:torch.Tensor | tuple[torch.Tensor, Any], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for sequence classification.")
    parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    parser.add_argument("--checkpoint", type=str, default="/data2/peter/dnazen_pretrain_v3.1/outputs/checkpoint-2000", help="Path to model checkpoint directory.")
    parser.add_argument("--out", type=str, help="Path to the results directory.")
    # parser.add_argument("--model_max_len", type=int, default=70, help="Maximum length of the model input.")
    parser.add_argument("--ngram_encoder_dir", type=str, help="Path to the ngram encoder directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", default=42)
    
    return parser.parse_args()

# --- main ---
args = parse_args()

DATA_PATH = args.data_path
RESULTS_PATH = args.out
# MODEL_MAX_LEN = args.model_max_len
NGRAM_ENCODER_DIR = args.ngram_encoder_dir
LEARNING_RATE = args.lr
NUM_TRAIN_EPOCHS = args.num_train_epochs
CHECKPOINT_DIR = args.checkpoint

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", 
    # model_max_length=MODEL_MAX_LEN,
    padding_side="right",
    padding="longest",
    use_fast=True,
    trust_remote_code=True,
) # type: ignore
ngram_encoder = NgramEncoder.from_file(NGRAM_ENCODER_DIR)
print("[debug] number of ngrams:", ngram_encoder.get_vocab_size())

train_dataset = LabeledDataset(
    f"{DATA_PATH}/train.csv",
    tokenizer=tokenizer,
    ngram_encoder=ngram_encoder
)

test_dataset = LabeledDataset(
    f"{DATA_PATH}/test.csv",
    tokenizer=tokenizer,
    ngram_encoder=ngram_encoder
)

val_dataset = LabeledDataset(
    f"{DATA_PATH}/dev.csv",
    tokenizer=tokenizer,
    ngram_encoder=ngram_encoder
)

config = BertConfig.from_pretrained(CHECKPOINT_DIR)
setattr(config, "ngram_vocab_size", ngram_encoder.get_vocab_size())
setattr(config, "num_word_hidden_layers", 6)

model = BertForSequenceClassification.from_pretrained(
    CHECKPOINT_DIR,
    config=config
)

# Filter out all weights in model state dict that have name `ngram_layer`
learning_rate = 1e-5

ngram_layer_params = [param for name, param in model.named_parameters() if ("ngram_layer" in name and "Norm" not in name)]
print("[debug] len of ngram layers=", len(ngram_layer_params))
optimizer = torch.optim.AdamW(
    model.parameters(),
    learning_rate,
    #momentum=args.momentum,
    weight_decay=0.01
)

train_args = transformers.training_args.TrainingArguments(
    output_dir=RESULTS_PATH,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=500,
    max_grad_norm=1,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_train_batch_size,
    learning_rate=LEARNING_RATE,
    bf16=args.fp16,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    logging_steps=200,
    
    seed=args.seed,
    data_seed=args.seed,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_matthews_correlation"
)

trainer = transformers.Trainer(
    model=model,
    args=train_args,
    optimizers=(optimizer, None),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()

results = trainer.evaluate(test_dataset)
with open(os.path.join(RESULTS_PATH, "eval_results.json"), "w") as f:
    json.dump(results, f)

test_preds = trainer.predict(
    test_dataset
).predictions
print(test_preds)

# extract the test data and prediction results
results = {
    "text": [],
    "actual_label": [],
    "prediction_label": [],
}
for i in range(len(test_dataset)):
    data = test_dataset[i]
    
    input_ids = data["input_ids"]
    actual_label = data["labels"]
    texts = tokenizer.decode(input_ids).replace("[CLS] ", "").replace(" [SEP]", "").replace(" [PAD]", "")

    results["text"].append(texts)
    results["actual_label"].append(actual_label)
    results["prediction_label"].append(test_preds[i])

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_PATH, "pred_results.csv"), mode="w")
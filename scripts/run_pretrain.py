# import deepspeed
from typing import Any
from argparse import ArgumentParser
import logging

import os
import json
import numpy as np
# import sklearn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import transformers
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from transformers.models.bert.configuration_bert import BertConfig

from dnazen.data.mlm_dataset import MlmDataset, MlmData
from dnazen.model.bert_models import BertForMaskedLM, BertForSequenceClassification
from dnazen.model.bert_config import ZenConfig
from dnazen.ngram import NgramEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_metrics(eval_pred):
    """
    Compute metrics used for huggingface trainer.
    """ 
    predictions, labels = eval_pred
    # calculate metric with sklearn
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    
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

    return torch.argmax(logits, dim=-1)

def parse_args():
    parser = ArgumentParser(description="Run pretraining for ZENforDNA")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory to resume training from.")
    parser.add_argument("--train", type=str, default="/data2/peter/dnazen_pretrain_v3.1/train", help="Directory for training data")
    parser.add_argument("--dev", type=str,   default="/data2/peter/dnazen_pretrain_v3.1/dev", help="Directory for validation data")
    parser.add_argument("--out", type=str,   default="/data2/peter/dnazen_pretrain_v3.1/outputs", help="Directory for output")
    parser.add_argument("--num_ngram_hidden_layer", type=int, default=6, help="Number of hidden layers for ngram")
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation-steps", type=int, default=128//4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--n-epoch", type=int, default=2, help="num epoch to train")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

args = parse_args()
set_random_seed(args.seed)

TRAIN_DATA_DIR = args.train
DEV_DATA_DIR = args.dev
OUTPUT_DIR = args.out
NUM_NGRAM_HIDDEN_LAYER = args.num_ngram_hidden_layer
LEARNING_RATE = args.lr

train_dataset = MlmDataset.from_dir(
    TRAIN_DATA_DIR
)

val_dataset = MlmDataset.from_dir(
    DEV_DATA_DIR
)

bert_config = BertConfig.from_pretrained("/home/peter/llm_projects/DNABERT_2/DNABERT-2-117M")
# bert_config.attention_probs_dropout_prob = 0.05
zen_config  = ZenConfig(
    num_word_hidden_layers=6, 
    ngram_vocab_size=train_dataset.ngram_vocab_size, 
    **bert_config.to_dict()
)
if args.resume is None:
    model = BertForMaskedLM.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        config=zen_config
    )
else:
    model = BertForMaskedLM.from_pretrained(
        args.resume,
        config=zen_config
    )
    
model_params = [param for name, param in model.named_parameters() if ("Norm" not in name)]
# print("[debug] len of ngram layers=", len(ngram_layer_params))
optimizer = torch.optim.AdamW(
    model_params,
    LEARNING_RATE,
    #momentum=args.momentum,
    weight_decay=0.01
)

train_args = transformers.training_args.TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=1_000,
    max_grad_norm=1,
    per_device_train_batch_size=args.per_device_train_batch_size,
    # gradient_accumulation_steps=128//4,
    gradient_accumulation_steps=args.grad_accumulation_steps,
    per_device_eval_batch_size=args.per_device_train_batch_size, # make it the same as train
    # learning_rate=1e-4,
    num_train_epochs=args.n_epoch,
    logging_steps=100,
    dataloader_num_workers=16,
    dataloader_prefetch_factor=2,
    warmup_steps=100,
    
    # necessary since we have shared tensor weight. 
    save_safetensors=False,
)

trainer = transformers.Trainer(
    model=model,
    args=train_args,
    optimizers=(optimizer, None),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
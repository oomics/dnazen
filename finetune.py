import os
import json
import logging
import csv
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
import torch
import sklearn.metrics
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizer,
)

from zen_modeling import ZenForSequenceClassification, NgramTokenizer

# Enable detailed debugging if set to True
debug = False


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """
    Save the model state dict on CPU to the specified directory when saving is enabled.
    """
    if trainer.args.should_save:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        trainer._save(output_dir, state_dict=cpu_state_dict)


class DebugTrainerCallback(TrainerCallback):
    """
    Callback that prints debugging information at training start, end of each step, and on logging.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        print("[Debug] Training begins; model and optimizer are ready.")

    def on_step_end(self, args, state, control, **kwargs):
        # Only triggered at logging steps
        model = kwargs.get("model")
        grad_param = next(p for p in model.parameters() if p.grad is not None)
        print(f"[Debug][step_end] step={state.global_step} grad_norm={grad_param.grad.norm():.6f}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called on every Trainer.log()
        print(f"[Debug][on_log] step={state.global_step} logs={logs}")
        return control


class EvalCallback(TrainerCallback):
    """
    Callback that accumulates evaluation metrics for each epoch and writes them to a JSON file.
    """
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.results: Dict[str, Dict] = {}

    def on_evaluate(self, args, state, control, metrics: Dict, **kwargs):
        epoch = state.epoch
        print(f"\nEpoch {epoch} evaluation results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Store metrics for the current epoch
        self.results[f"epoch_{epoch}"] = metrics
        os.makedirs(self.results_path, exist_ok=True)
        file_path = os.path.join(self.results_path, "eval_results.json")
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return control


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Path to pre-trained model or model identifier"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Directory containing train.csv, dev.csv, and test.csv"},
    )


def get_reverse_complement(sequence: str) -> str:
    """
    Return the complement of a DNA sequence.
    """
    complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement_map.get(base, base) for base in sequence)


def generate_kmer_str(sequence: str, k: int) -> str:
    """
    Transform a DNA sequence into a k-mer string separated by spaces.
    """
    return " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def load_or_generate_kmer(data_path: str, sequences: List[str], k: int) -> List[str]:
    """
    Load k-mers from a JSON file if it exists; otherwise generate and save them.
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mers from {kmer_path}")
        with open(kmer_path, "r") as f:
            return json.load(f)

    logging.warning("Generating k-mers...")
    kmer_list = [generate_kmer_str(seq, k) for seq in sequences]
    os.makedirs(os.path.dirname(kmer_path), exist_ok=True)
    with open(kmer_path, "w") as f:
        json.dump(kmer_list, f)

    return kmer_list


class SupervisedDataset(Dataset):
    """Dataset for supervised sequence classification tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        kmer: int = -1,
    ):
        # Read CSV, skipping header
        with open(data_path, "r") as f:
            rows = list(csv.reader(f))[1:]

        # Determine data format: single or paired sequences
        if len(rows[0]) == 2:
            logging.warning("Single-sequence classification mode")
            texts = [row[0] for row in rows]
            labels = [int(row[1]) for row in rows]
        elif len(rows[0]) == 3:
            logging.warning("Sequence-pair classification mode")
            texts = [[row[0], row[1]] for row in rows]
            labels = [int(row[2]) for row in rows]
        else:
            raise ValueError("Unsupported data format; expected 2 or 3 columns.")

        # Optionally apply k-mer transformation
        if kmer > 0:
            # Synchronize processes in distributed setting
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in {0, -1}:
                torch.distributed.barrier()

            logging.warning(f"Transforming input to {kmer}-mers")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        if debug:
            texts = texts[:16]
            labels = labels[:16]

        self.tokenizer = tokenizer
        self.data = texts
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        encoded = self.tokenizer(
            item,
            is_split_into_words=False,
            max_seq_length=512,
        )

        # Rearrange n-gram attention matrix
        attn = np.array(encoded["ngram_attention_match"])
        ngram_positions = attn.T.tolist()
        ngram_token_types = [0] * len(encoded["ngram_input_ids"])

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "ngram_input_ids": torch.tensor(
                encoded["ngram_input_ids"], dtype=torch.long
            ),
            "ngram_attention_mask": torch.tensor(
                encoded["ngram_attention_mask"], dtype=torch.bool
            ),
            "ngram_token_type_ids": torch.tensor(ngram_token_types, dtype=torch.long),
            "ngram_position_matrix": torch.tensor(ngram_positions, dtype=torch.long),
        }


def calculate_metrics_with_sklearn(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, F1, MCC, precision, and recall using sklearn.
    """
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    valid_preds = preds[mask]
    valid_labels = labels[mask]

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_preds),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_preds, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_preds),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_preds, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_preds, average="macro", zero_division=0),
    }


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metrics_with_sklearn(logits, labels)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = NgramTokenizer.from_pretrained(model_args.model_name_or_path)
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    train_ds = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=training_args.kmer if hasattr(training_args, 'kmer') else -1,
    )
    val_ds = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=-1,
    )
    test_ds = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=-1,
    )

    if debug:
        test_ds = train_ds

    model = ZenForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=train_ds.num_labels,
        trust_remote_code=True,
    )

    results_dir = os.path.join(training_args.output_dir, "results")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Attach callbacks
    if not debug:
        trainer.add_callback(EvalCallback(results_dir))
    else:
        trainer.add_callback(DebugTrainerCallback())

    # Start training
    trainer.train()

    # Save final model and evaluation
    trainer.save_model()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)

    final_metrics = trainer.evaluate(eval_dataset=test_ds)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "final_eval_results.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Save configuration and tokenizer
    model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()

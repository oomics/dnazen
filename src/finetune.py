import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import argparse
# from transformers.processing_utils import transformers_module
from zen_modeling import ZenForSequenceClassification, NgramTokenizer
from transformers import TrainerCallback, TrainingArguments, HfArgumentParser, set_seed
import os
import json

debug = False

from transformers import TrainerCallback

class DebugTrainerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("[Debug] Training begins, model + optimizer ready.")

    def on_step_end(self, args, state, control, **kwargs):
        # 这一段只会在 logging_steps 整除时才跑
        model = kwargs["model"]
        p = next(p for p in model.parameters() if p.grad is not None)
        print(f"[Debug][step_end] step={state.global_step} grad_norm={p.grad.norm():.6f}")
        # 不要立刻把 should_log 关掉，留给下一次也能跑
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 任何一次 Trainer.log() 都会进这里
        print(f"[Debug][on_log] step={state.global_step} logs={logs}")



class EvalCallback(TrainerCallback):
    def __init__(self, results_path):
        """
        初始化时传入存放结果的文件夹路径，并初始化一个字典用于累积每个 epoch 的评估结果
        """
        self.results_path = results_path
        self.results = {}  # 用于累积各个 epoch 的评估结果

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        每次评估时会调用该函数，metrics 参数包含了评估指标。
        当 evaluation_strategy 设置为 "epoch" 时，该函数会在每个 epoch 结束时触发。
        """
        epoch = state.epoch
        print(f"\nEpoch {epoch} evaluation results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # 将当前 epoch 的评估结果存入字典中
        self.results[f"epoch_{epoch}"] = metrics

        # 确保 results_path 文件夹存在，然后将累积结果写入同一个 JSON 文件中
        os.makedirs(self.results_path, exist_ok=True)
        file_path = os.path.join(self.results_path, "eval_results.json")
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return control


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""


def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


"""
Transform a dna sequence to k-mer string
"""


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)

    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):

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

        if debug:
            texts = texts[:16]
            labels = labels[:16]

        self.tokenizer = tokenizer
        self.data = texts
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        output = self.tokenizer(
            item,
            is_split_into_words=False,
            max_seq_length=512
            # return_tensors="pt",
            # padding="longest",
            # max_length=tokenizer.model_max_length,
            # truncation=True,
        )

        attn_match = np.array(output["ngram_attention_match"])
        ngram_position_matrix = attn_match.T.tolist()
        # ngram_token_type_ids 全部置 0
        ngram_token_type_ids = [0] * len(output["ngram_input_ids"])
        output["ngram_token_type_ids"] = ngram_token_type_ids
        output["ngram_position_matrix"] = ngram_position_matrix

        return {
            "input_ids": torch.tensor(output["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(output["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
            "ngram_input_ids": torch.tensor(output["ngram_input_ids"], dtype=torch.long),
            "ngram_attention_mask": torch.tensor(output["ngram_attention_mask"], dtype=torch.bool),
            "ngram_token_type_ids": torch.tensor(output["ngram_token_type_ids"], dtype=torch.long),
            "ngram_position_matrix": torch.tensor(output["ngram_position_matrix"], dtype=torch.long),
        }


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""
#
#     tokenizer: transformers.PreTrainedTokenizer
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.Tensor(labels).long()
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
"""


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # load tokenizer
    tokenizer = NgramTokenizer.from_pretrained(model_args.model_name_or_path)

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv")
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
    )

    if debug:
        test_dataset = train_dataset

    # load model
    model = ZenForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )

    # 设定 results_path（存放所有 epoch 的 eval 结果），例如在 output_dir/ results/ 下
    results_path = os.path.join(training_args.output_dir, "results")

    # 定义 Trainer，并设置 evaluation_strategy 为 epoch（确保在每个 epoch 结束后进行评估）
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset  # 如果想在评估时使用 dev/test 数据，这里请根据需求调整
    )

    # 添加自定义回调，将评估结果累积保存到同一个文件中
    if not debug:
        trainer.add_callback(EvalCallback(results_path))
    if debug:
        trainer.add_callback(DebugTrainerCallback())

    # 开始训练，evaluation 会按照 evaluation_strategy 自动进行
    trainer.train()

    trainer.save_model()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    final_results = trainer.evaluate(eval_dataset=test_dataset)
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "final_eval_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()

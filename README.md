# DNAZEN

DNAZEN is a pretrained representation model for gene sequence.


## Citation

If you use or extend our work, please cite the following [paper](https://arxiv.org/abs/2505.02206).

```text
@article{mao2025dnazen,
  title={DNAZEN: Enhanced Gene Sequence Representations via Mixed Granularities of Coding Units},
  author={Mao, Lei and Tian, Yuanhe and Song, Yan},
  journal={arXiv preprint arXiv:2505.02206},
  year={2025}
}
```

## Requirements

The code works with the following environment.

```text
python==3.11
transformers==4.51.3
deepspeed==0.16.7
```

## Data

DNAZEN is fine-tuned on GUE benchmarks. You can find the data from [here](https://github.com/MAGICS-LAB/DNABERT_2).


## Pretrained Models

The base model can be downloaded from [HuggingFace](https://huggingface.co/oomics/DNAZEN-1.0-base). The large and x-large models will come soon.

| Parameter                                    |  Base |          Large | X-Large |
|----------------------------------------------|------:|---------------:|--------:|
| `token_encoder_num_hidden_layers`            |    12 |             24 |      28 |
| `token_encoder_hidden_size`                  |   768 |           1024 |    1536 |
| `token_encoder_num_attention_heads`          |    12 |             16 |      12 |
| `token_encoder_intermediate_size`            |  3072 |           4096 |    8960 |
| `token_encoder_vocab_size`                   |  4096 |           4096 |    4096 |
| `geneseq_encoder_num_hidden_layers`          |     6 |              6 |      12 |
| `geneseq_encoder_hidden_size`                |   768 |           1024 |    1536 |
| `geneseq_encoder_num_attention_heads`        |    12 |             16 |      12 |
| `geneseq_encoder_intermediate_size`          |  3072 |           4096 |    8960 |
| `geneseq_encoder_vocab_size`                 |162708 |         162708 |  162708 |
| `total_number_of_parameters`                 |  285M |           650M |    2.1B |


## Fine-tune DNAZEN on GUE

Run the following command the fine-tune DNAZEN on downstream tasks. You may want to change the value of `$model_path`, `$data_path`, and `$output_path` accordingly.

```bash
python finetune.py \
    --model_name_or_path $model_path \
    --data_path  $data_path/prom/prom_300_all \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 8 \
    --save_total_limit 1 \
    --fp16 \
    --output_dir $output_path \
    --eval_strategy epoch \
    --save_strategy epoch \
    --warmup_steps 50 \
    --overwrite_output_dir True \
    --greater_is_better True \
    --metric_for_best_model matthews_correlation \
    --load_best_model_at_end True
```




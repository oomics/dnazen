# DNAZEN

DNAZEN is a pretrained representation model for gene sequence.


## Citation

If you use or extend our work, please cite the following [paper](https://arxiv.org/abs/2505.02206).


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








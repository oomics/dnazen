import pytest
import torch

device = "cuda"


@pytest.fixture
def mlm_model_input(model_config):
    batch_size = 4
    max_seq_len = 128
    vocab_size = model_config.vocab_size
    max_ngram_len = 20
    ngram_vocab_size = model_config.ngram_vocab_size

    return {
        "input_ids": torch.randint(
            low=0, high=vocab_size // 2, size=(batch_size, max_seq_len), device=device
        ),
        "labels": torch.randint(
            low=0, high=vocab_size, size=(batch_size, max_seq_len), device=device
        ),
        "attention_mask": torch.rand(size=(batch_size, max_seq_len), device=device),
        "ngram_input_ids": torch.randint(
            low=0,
            high=ngram_vocab_size // 2,
            size=(batch_size, max_ngram_len),
            device=device,
        ),
        "ngram_position_matrix": torch.ones(
            size=(batch_size, max_seq_len, max_ngram_len), device=device
        ),
    }


@pytest.fixture
def model_config():
    from transformers.models.bert.configuration_bert import BertConfig
    from dnazen.model.bert_config import ZenConfig

    config_dict = BertConfig().to_dict()
    config_dict["alibi_starting_size"] = 512
    config_dict["num_word_hidden_laters"] = 6
    config_dict["ngram_vocab_size"] = 2048

    config = ZenConfig(**config_dict)
    return config


@pytest.mark.usefixtures("model_config", "mlm_model_input")
def test_zen_forward(model_config, mlm_model_input):
    from dnazen.model.bert_models import BertForMaskedLM

    model = BertForMaskedLM(model_config)
    model.to(device=device)
    _ = model(**mlm_model_input)
    # print(ret)

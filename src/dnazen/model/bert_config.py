from transformers.models.bert.modeling_bert import (
    BertConfig,
)

class ZenConfig(BertConfig):
    def __init__(self, num_word_hidden_layers=0, ngram_vocab_size=21128, **kwargs):
        super().__init__(**kwargs)
        self.num_word_hidden_layers = num_word_hidden_layers
        self.ngram_vocab_size = ngram_vocab_size
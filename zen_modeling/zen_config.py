import json
from transformers import BertConfig

class ZenConfig(BertConfig):
    """
    ZEN configuration class that extends BertConfig with n-gram encoder parameters.

    In addition to standard BERT parameters, this config requires:
      - max_ngram_count: maximum number of n-grams.
      - num_hidden_ngram_layers: number of n-gram encoder layers.
      - ngram_vocab_size: size of the n-gram vocabulary (required).
      - alibi_starting_size: ALiBi starting size.
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_hidden_ngram_layers=None,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        max_ngram_count=30,
        ngram_vocab_size=162708,
        alibi_starting_size=512,
        **kwargs
    ):
        # If the number of n-gram encoder layers is not specified, default to the total hidden layers
        if num_hidden_ngram_layers is None:
            num_hidden_ngram_layers = num_hidden_layers
        assert num_hidden_ngram_layers <= num_hidden_layers, \
            "ngram layers cannot exceed total hidden layers"

        # Initialize the base BertConfig
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )

        # Set n-gram encoder parameters
        self.max_ngram_count = max_ngram_count
        self.num_hidden_ngram_layers = num_hidden_ngram_layers
        self.ngram_vocab_size = ngram_vocab_size
        self.alibi_starting_size = alibi_starting_size

    def to_json_string(self, use_diff=True):
        """
        Serialize this instance to a JSON string, optionally only including changed values.
        """
        output = self.to_diff_dict() if use_diff else self.to_dict()
        output["max_ngram_count"] = self.max_ngram_count
        output["num_hidden_ngram_layers"] = self.num_hidden_ngram_layers
        output["ngram_vocab_size"] = self.ngram_vocab_size
        output["alibi_starting_size"] = self.alibi_starting_size
        return json.dumps(output, ensure_ascii=False, indent=4, sort_keys=True)

    @classmethod
    def from_dict(cls, json_object, **kwargs):
        """
        Create a ZenConfig instance from a dict of parameters.
        """
        return cls(**json_object)

    @classmethod
    def from_json_file(cls, json_file):
        """
        Load a ZenConfig from a JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_pretrained_bert(
        cls,
        bert_model_path: str,
        max_ngram_count: int,
        num_hidden_ngram_layers: int,
        ngram_vocab_size: int
    ):
        """
        Load a BertConfig from the given path or model name and extend it to ZenConfig.

        Args:
          bert_model_path: path or name of the pretrained BERT model.
          max_ngram_count: maximum number of n-grams.
          num_hidden_ngram_layers: number of n-gram encoder layers.
          ngram_vocab_size: size of the n-gram vocabulary.

        Returns:
          A configured ZenConfig instance.
        """
        # Load base BERT configuration
        bert_config = BertConfig.from_pretrained(bert_model_path)
        print(f"Loaded BertConfig from {bert_model_path}:")
        print(f"  vocab_size: {bert_config.vocab_size}")
        print(f"  hidden_size: {bert_config.hidden_size}")
        print(f"  num_hidden_layers: {bert_config.num_hidden_layers}")

        # Create ZenConfig using BERT parameters plus n-gram settings
        zen_config = cls(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            hidden_act=bert_config.hidden_act,
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            initializer_range=bert_config.initializer_range,
            layer_norm_eps=bert_config.layer_norm_eps,
            max_ngram_count=max_ngram_count,
            num_hidden_ngram_layers=num_hidden_ngram_layers,
            ngram_vocab_size=ngram_vocab_size,
            alibi_starting_size=bert_config.alibi_starting_size,
        )

        print("\nZenConfig:")
        print(f"  vocab_size: {zen_config.vocab_size}")
        print(f"  hidden_size: {zen_config.hidden_size}")
        print(f"  num_hidden_layers: {zen_config.num_hidden_layers}")
        print(f"  max_ngram_count: {zen_config.max_ngram_count}")
        print(f"  num_hidden_ngram_layers: {zen_config.num_hidden_ngram_layers}")
        print(f"  ngram_vocab_size: {zen_config.ngram_vocab_size}")
        print("\nZenConfig JSON string:")
        print(zen_config.to_json_string())

        return zen_config

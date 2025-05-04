import json
from transformers import BertConfig

class ZenConfig(BertConfig):
    """
    ZEN 配置类，在 BERT 配置的基础上增加了 ngram encoder 的参数。

    除了 BERT 的标准参数外，还需要指定 ngram encoder 相关参数：
      - max_ngram_count: 最大 ngram 数量。
      - num_hidden_ngram_layers: ngram encoder 层数。
      - ngram_hidden_size: ngram encoder 的隐藏层大小。
      - ngram_vocab_size: ngram 词汇表大小（必须提供）。
    """
    def __init__(self,
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
                 **kwargs):
        # 如果没有指定 ngram encoder 层数，则默认与 BERT 层数相同
        if num_hidden_ngram_layers is None:
            num_hidden_ngram_layers = num_hidden_layers
        assert num_hidden_ngram_layers <= num_hidden_layers

        # 初始化 BertConfig 部分参数
        super().__init__(vocab_size=vocab_size,
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
                         **kwargs)
        # 设置 ngram encoder 相关参数
        self.max_ngram_count = max_ngram_count
        self.num_hidden_ngram_layers = num_hidden_ngram_layers
        self.ngram_vocab_size = ngram_vocab_size
        self.alibi_starting_size = alibi_starting_size

    def to_json_string(self, use_diff=True):
        if use_diff is True:
            output = self.to_diff_dict()
        else:
            output = self.to_dict()
        output["max_ngram_count"] = self.max_ngram_count
        output["num_hidden_ngram_layers"] = self.num_hidden_ngram_layers
        output["ngram_vocab_size"] = self.ngram_vocab_size
        output["alibi_starting_size"] = self.alibi_starting_size
        return json.dumps(output, ensure_ascii=False, indent=4, sort_keys=True)

    @classmethod
    def from_dict(cls, json_object, **kwargs):
        return cls(**json_object)

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_pretrained_bert(cls,
                             bert_model_path: str,
                             max_ngram_count: int,
                             num_hidden_ngram_layers: int,
                             ngram_vocab_size: int):
        """
        从指定的 BERT 模型路径加载 BertConfig，并基于此构造 ZenConfig。

        参数:
          bert_model_path: BERT 模型的路径或预训练模型名称（例如 "bert-base-uncased"）。
          max_ngram_count: 最大 ngram 数量。
          num_hidden_ngram_layers: ngram encoder 的层数。
          ngram_hidden_size: ngram encoder 的隐藏层大小。
          ngram_vocab_size: ngram 词汇表大小（必须提供）。

        返回:
          构造后的 ZenConfig 对象。
        """
        # 从指定的 BERT 模型加载配置
        bert_config = BertConfig.from_pretrained(bert_model_path)
        print("Loaded BertConfig from {}:".format(bert_model_path))
        print("  vocab_size:", bert_config.vocab_size)
        print("  hidden_size:", bert_config.hidden_size)
        print("  num_hidden_layers:", bert_config.num_hidden_layers)

        # 构造 ZenConfig，传入 BERT 部分参数以及 Zen 额外参数
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
        print("  vocab_size:", zen_config.vocab_size)
        print("  hidden_size:", zen_config.hidden_size)
        print("  num_hidden_layers:", zen_config.num_hidden_layers)
        print("  max_ngram_count:", zen_config.max_ngram_count)
        print("  num_hidden_ngram_layers:", zen_config.num_hidden_ngram_layers)
        print("  ngram_vocab_size:", zen_config.ngram_vocab_size)
        print("\nZenConfig JSON string:")
        print(zen_config.to_json_string())
        return zen_config

# --------------------- 测试调用 ---------------------
if __name__ == "__main__":
    from .tokenization import NgramTokenizer
    # 例如，使用 "bert-base-uncased" 作为 BERT 模型路径
    bert_model_path = "/data1/user1/llm/DNABERT-2-117M"
    ngram_json_file = "/data1/user1/project/zen_train/data/pretrain/human_ms_gue/ngram_encoders/pmi_1_all_union_ngram_encoder.json"
    tokenizer = NgramTokenizer(bert_model_path, ngram_json_file)
    # 明确传入 Zen 模型额外参数
    zen_config = ZenConfig.from_pretrained_bert(
        bert_model_path=bert_model_path,
        max_ngram_count=tokenizer.max_ngram_count,
        num_hidden_ngram_layers=6,
        ngram_vocab_size=tokenizer.ngram_size
    )

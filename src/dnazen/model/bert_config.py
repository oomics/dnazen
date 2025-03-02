from transformers.models.bert.modeling_bert import (
    BertConfig,
)


class ZenConfig(BertConfig):
    """
    ZenConfig 类扩展了 Hugging Face 的 BertConfig 类，为DNA序列预训练模型添加了额外的配置参数。
    
    ZEN (Z-Enhanced N-gram) 架构是对标准BERT的扩展，增加了对N-gram信息的处理能力，
    特别适合于DNA序列等生物序列数据的建模。
    
    属性:
        num_word_hidden_layers (int): N-gram特征提取器的隐藏层数量。
            当设置为0时，模型将不使用N-gram特征。
            较大的值会增加模型对N-gram模式的捕获能力，但也会增加计算复杂度。
            
        ngram_vocab_size (int): N-gram词汇表大小，即模型可以识别的不同N-gram数量。
            对于DNA序列，这通常包括所有可能的k-mer组合（如3-mer、4-mer等）。
            默认值21128适用于大多数DNA序列建模任务。
    
    示例:
        >>> # 创建一个带有N-gram处理能力的ZEN配置
        >>> config = ZenConfig(
        ...     num_word_hidden_layers=6,  # 使用6层N-gram特征提取
        ...     ngram_vocab_size=21128,    # N-gram词汇表大小
        ...     hidden_size=768,           # 隐藏层维度
        ...     num_hidden_layers=12,      # 主干网络层数
        ...     num_attention_heads=12     # 注意力头数量
        ... )
    """
    
    def __init__(self, num_word_hidden_layers=0, ngram_vocab_size=21128, **kwargs):
        """
        初始化ZenConfig对象
        
        参数:
            num_word_hidden_layers (int, 可选): 
                N-gram特征提取器的隐藏层数量。默认为0（不使用N-gram特征）。
                
            ngram_vocab_size (int, 可选): 
                N-gram词汇表大小。默认为21128，适用于大多数DNA序列建模任务。
                
            **kwargs: 
                传递给父类BertConfig的其他参数，如hidden_size、num_hidden_layers等。
        """
        super().__init__(**kwargs)
        self.num_word_hidden_layers = num_word_hidden_layers
        self.ngram_vocab_size = ngram_vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练模型配置创建ZenConfig实例
        
        该方法允许从标准BERT预训练模型配置加载，然后转换为ZEN配置。
        这使得可以利用现有的BERT预训练模型，同时添加ZEN特有的N-gram处理能力。
        
        参数:
            pretrained_model_name_or_path (str): 
                预训练模型的名称或路径，可以是Hugging Face模型库中的模型名称
                （如"bert-base-uncased"）或本地模型路径。
                
            *model_args: 
                传递给BertConfig.from_pretrained的位置参数。
                
            **kwargs: 
                传递给BertConfig.from_pretrained的关键字参数，可以包括ZenConfig特有的参数，
                如num_word_hidden_layers和ngram_vocab_size。
                
        返回:
            ZenConfig: 从预训练配置创建的ZenConfig实例。
            
        示例:
            >>> # 从预训练BERT模型加载配置并转换为ZEN配置
            >>> zen_config = ZenConfig.from_pretrained(
            ...     "bert-base-uncased",
            ...     num_word_hidden_layers=6,
            ...     ngram_vocab_size=21128
            ... )
        """
        config = BertConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(**config.to_dict())

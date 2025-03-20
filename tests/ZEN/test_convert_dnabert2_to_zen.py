import pytest

from transformers import AutoTokenizer, BertForMaskedLM, BertConfig, AutoModel

from ZEN.utils import convert_dnabert2_to_zen
from ZEN.modeling import ZenConfig, ZenForPreTraining



@pytest.mark.parametrize("dnabert2_model_name", ["zhihan1996/DNABERT-2-117M"])
def test_convert_dnabert2_to_zen(dnabert2_model_name):
    # 测试
    config = BertConfig.from_pretrained(dnabert2_model_name)
    model = AutoModel.from_pretrained(dnabert2_model_name, config=config, trust_remote_code=True)
    state_dict = model.state_dict()
    zen_state_dict = convert_dnabert2_to_zen(state_dict)
    
    zen_config = ZenConfig(vocab_size_or_config_json_file=config.vocab_size, word_vocab_size=12886)
    zen_model = ZenForPreTraining(zen_config)
    zen_model.load_state_dict(zen_state_dict)
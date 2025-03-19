# 将DNABERT2的state_dict转换为ZEN的state_dict

def convert_dnabert2_to_zen(dnabert2_state_dict):
    """
    Convert the state_dict of DNABERT2 to the state_dict of ZEN
    """
    zen_state_dict = {}
    for key, value in dnabert2_state_dict.items():
        if key.startswith("embeddings"):
            key_ = key.replace("embeddings", "bert.embeddings")
            zen_state_dict[key_] = value
        elif key.startswith("pooler"):
            key_ = key.replace("pooler.", "bert.pooler.")
            zen_state_dict[key_] = value
        else:
            zen_state_dict[key] = value

    return zen_state_dict
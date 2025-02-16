def ngram_str_to_tuple(ngram_str: str) -> tuple[int, ...]:
    return tuple([int(s) for s in ngram_str.split(":")])


def ngram_tuple_to_str(ngram_tuple: tuple[int, ...]) -> str:
    return ":".join([str(num) for num in list(ngram_tuple)])


def save_ngram_dict_to_dir(ngram_dict: dict[tuple[int, ...], int], dir: str):
    import json

    with open(dir, "w") as f:
        output = {}
        for k, v in ngram_dict.items():
            k_ = ngram_tuple_to_str(k)
            output[k_] = v
        json.dump(output, f, indent=2)

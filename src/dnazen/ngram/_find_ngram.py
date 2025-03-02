import _ngram
from _ngram import PmiNgramFinderConfig, FreqNgramFinderConfig


def find_ngrams_by_pmi(
    ngram_finder_config: PmiNgramFinderConfig,
    tokens: list[list[int]],
) -> dict[tuple[int, ...], int]:
    """
    基于互信息(PMI)查找文本中的n-gram组合。
    
    PMI(点互信息)是衡量两个事件共同出现的概率与它们独立出现概率乘积的比值的对数。
    在自然语言处理中，PMI高的n-gram通常表示其组成部分之间有较强的关联性。
    
    工作流程:
    1. 根据配置初始化PMI n-gram查找器
    2. 批量处理所有token序列
    3. 获取查找到的n-gram列表
    4. 将结果转换为字典形式返回
    
    Args:
        ngram_finder_config (PmiNgramFinderConfig): PMI n-gram查找器的配置参数，
                                                    包含最小/最大n-gram长度、PMI阈值等
        tokens (list[list[int]]): 待分析的整数token序列列表，每个内部列表代表一个文本序列
                                
    Returns:
        dict[tuple[int, ...], int]: 字典，键为n-gram(整数元组)，值为其在文本中的出现频率
    """
    # 初始化PMI n-gram查找器
    finder = _ngram.PmiNgramFinder(ngram_finder_config)
    
    # 批量处理所有token序列查找n-gram
    finder.find_ngrams_batched(tokens)
    
    # 获取查找结果，返回格式为 [n-gram tokens, frequency]
    ngrams: list[list[int]] = finder.get_ngram_list([])

    # 转换结果为字典形式，将每个n-gram作为键(元组)，频率作为值
    ngram_dict = {}
    for ngram in ngrams:
        # 最后一个元素是频率，将其弹出
        freq = ngram.pop()
        # 将列表转换为元组作为字典键
        ngram_dict[tuple(ngram)] = freq

    return ngram_dict


def find_ngrams_by_freq(
    ngram_finder_config: FreqNgramFinderConfig, 
    tokens: list[list[int]]
) -> dict[tuple[int, ...], int]:
    """
    基于频率查找文本中的n-gram组合。
    
    这种方法简单地统计n-gram在文本中出现的频率，并根据配置的阈值筛选结果。
    
    工作流程:
    1. 根据配置初始化频率n-gram查找器
    2. 批量处理所有token序列
    3. 获取查找到的n-gram列表
    4. 将结果转换为字典形式返回
    
    Args:
        ngram_finder_config (FreqNgramFinderConfig): 频率n-gram查找器的配置参数，
                                                    包含最小/最大n-gram长度、频率阈值等
        tokens (list[list[int]]): 待分析的整数token序列列表，每个内部列表代表一个文本序列
    
    Returns:
        dict[tuple[int, ...], int]: 字典，键为n-gram(整数元组)，值为其在文本中的出现频率
    """
    # 初始化频率n-gram查找器
    finder = _ngram.FreqNgramFinder(ngram_finder_config)
    
    # 批量处理所有token序列查找n-gram
    finder.find_ngrams_batched(tokens)
    
    # 获取查找结果，返回格式为 [n-gram tokens, frequency]
    ngrams: list[list[int]] = finder.get_ngram_list([])

    # 转换结果为字典形式，将每个n-gram作为键(元组)，频率作为值
    ngram_dict = {}
    for ngram in ngrams:
        # 最后一个元素是频率，将其弹出
        freq = ngram.pop()
        # 将列表转换为元组作为字典键
        ngram_dict[tuple(ngram)] = freq

    return ngram_dict

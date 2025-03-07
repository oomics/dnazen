import os
import argparse
import logging
from typing import List
import time
from tqdm import tqdm
import pandas as pd
from dnazen.ngram import NgramEncoder
import plotly.express as px
from dnazen.misc.dna_sequences import get_mspecies_sequences, get_gue_each_sequences
from dnazen.misc.ngrm_plot import plot_ngram_zipfs_law, plot_ngram_distribution

from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def tokenize_data(texts: List[str], tokenizer) -> List[List[int]]:
    """将文本数据转换为token ID"""
    logger.info("将文本转换为token ID")
    tokenized_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokenized_data.append(tokens)
    logger.info(f"转换完成，共 {len(tokenized_data)} 个序列")
    return tokenized_data


class NgramMetaData:
    """N-gram元数据统计类，用于收集和管理N-gram匹配的统计信息"""

    def __init__(self):
        """初始化元数据统计对象"""
        self.total_num_matches = 0  # 匹配到的ngram总数
        self.num_data_no_match = 0  # 无匹配的数据数量
        self.num_data_has_match = 0  # 有匹配的数据数量
        self.total_num_data = 0  # 总数据数量
        self.total_tokens = 0  # 总分词数
        self.covered_tokens = 0  # 匹配覆盖的分词数
        self.num_ngram_filtered = 0  # 被过滤的ngram数量
        self.filtered_ngrams = set()  # 存储已过滤的唯一N-gram
        self.max_freq_ngram = ""  # 最高频率N-gram
        self.max_freq_value = 0  # 最高频率值
        self.min_freq_ngram = ""  # 最低频率N-gram
        self.min_freq_value = float("inf")  # 最低频率值

    def update_match_stats(self, num_matches, num_tokens):
        """更新匹配统计信息"""
        self.total_num_matches += num_matches
        self.total_num_data += 1
        self.total_tokens += num_tokens
        self.covered_tokens += num_matches

        if num_matches == 0:
            self.num_data_no_match += 1
        else:
            self.num_data_has_match += 1

    def update_filtered_count(self, filtered_ngrams_list):
        """更新被过滤的N-gram数量，避免重复计算

        Args:
            filtered_ngrams_list: 包含(ngram, freq)元组的列表
        """
        new_filtered = 0
        for ngram_text, _ in filtered_ngrams_list:
            if ngram_text not in self.filtered_ngrams:
                self.filtered_ngrams.add(ngram_text)
                new_filtered += 1

        self.num_ngram_filtered += new_filtered

    def update_freq_stats(self, max_freq_ngram, max_freq_value, min_freq_ngram, min_freq_value):
        """更新频率统计信息"""
        if max_freq_value > self.max_freq_value:
            self.max_freq_ngram = max_freq_ngram
            self.max_freq_value = max_freq_value

        if min_freq_value < self.min_freq_value:
            self.min_freq_ngram = min_freq_ngram
            self.min_freq_value = min_freq_value

    def to_dict(self):
        """将统计信息转换为字典"""
        return {
            "total_num_matches": self.total_num_matches,
            "num_data_no_match": self.num_data_no_match,
            "num_data_has_match": self.num_data_has_match,
            "total_num_data": self.total_num_data,
            "total_tokens": self.total_tokens,
            "covered_tokens": self.covered_tokens,
            "num_ngram_filtered": self.num_ngram_filtered,
            "max_freq_ngram": self.max_freq_ngram,
            "max_freq_value": self.max_freq_value,
            "min_freq_ngram": self.min_freq_ngram,
            "min_freq_value": self.min_freq_value,
        }

    def get_stats_info(self, dataset_name=None):
        """获取用于统计收集器的信息字典"""
        total_seqs = self.total_num_data
        token_coverage = self.covered_tokens / self.total_tokens * 100 if self.total_tokens > 0 else 0

        return {
            "数据集": dataset_name or "未命名",
            "总序列数": total_seqs,
            "有匹配序列数": self.num_data_has_match,
            "有匹配序列百分比": self.num_data_has_match / total_seqs * 100 if total_seqs > 0 else 0,
            "无匹配序列数": self.num_data_no_match,
            "无匹配序列百分比": self.num_data_no_match / total_seqs * 100 if total_seqs > 0 else 0,
            "匹配到的ngram总数": self.total_num_matches,
            "平均每序列匹配数": self.total_num_matches / total_seqs if total_seqs > 0 else 0,
            "总分词数": self.total_tokens,
            "匹配覆盖的分词数": self.covered_tokens,
            "分词覆盖率": token_coverage,
            "被过滤的ngram数量": self.num_ngram_filtered,
            "数据集最高频率N-gram": self.max_freq_ngram,
            "数据集最高频率值": self.max_freq_value,
            "数据集最低频率N-gram": self.min_freq_ngram,
            "数据集最低频率值": self.min_freq_value,
        }

    def log_stats(self, dataset_name=None):
        """输出统计信息到日志"""
        total_seqs = self.total_num_data

        logger.info("===== N-gram匹配统计 =====")
        logger.info(f"数据集: {dataset_name or '未命名'}")
        logger.info(f"总序列数: {total_seqs}")
        logger.info(
            f"有匹配序列数: {self.num_data_has_match} ({self.num_data_has_match / total_seqs * 100:.1f}%)"
        )
        logger.info(
            f"无匹配序列数: {self.num_data_no_match} ({self.num_data_no_match / total_seqs * 100:.1f}%)"
        )
        logger.info(f"匹配到的ngram总数: {self.total_num_matches}")
        logger.info(f"平均每个序列匹配数: {self.total_num_matches / total_seqs:.2f}")
        logger.info(f"总分词数: {self.total_tokens}")
        logger.info(f"匹配覆盖的分词数: {self.covered_tokens}")
        logger.info(f"分词覆盖率: {self.covered_tokens / self.total_tokens * 100:.2f}%")
        logger.info(f"被过滤的ngram数量: {self.num_ngram_filtered}")
        logger.info(f"数据集最高频率N-gram: {self.max_freq_ngram} (频率: {self.max_freq_value})")
        logger.info(f"数据集最低频率N-gram: {self.min_freq_ngram} (频率: {self.min_freq_value})")
        logger.info("==========================")


def process_sequence(
    text,
    idx,
    tokenizer,
    ngram_encoder,
    ngram_freq_dict,
    ngram_token_freq_dict,
    min_freq,
    results,
    meta_data,
    total_seqs,
):
    """处理单个序列，计算N-gram匹配情况

    Args:
        text: 输入文本序列
        idx: 序列索引
        tokenizer: 用于分词的tokenizer
        ngram_encoder: N-gram编码器
        ngram_freq_dict: N-gram频率字典（可选）
        min_freq: 最小频率阈值
        results: 结果字典，用于存储匹配信息
        meta_data: 元数据对象，用于存储统计信息
        total_seqs: 总序列数

    Returns:
        更新后的索引值
    """
    try:
        # 使用tokenizer将文本转换为token IDs
        token_ids = tokenizer(text, return_tensors="pt", return_attention_mask=False)["input_ids"].squeeze(
            0
        )

        # 获取序列的分词总数
        total_tokens = len(token_ids)

        # 获取匹配的N-gram
        matched_ngrams_with_pos = ngram_encoder.get_matched_ngrams(token_ids)

        # 初始化变量
        matched_ngrams = []
        matched_ngrams_with_freq = []
        filtered_ngrams = []
        # ipdb.set_trace()
        # 根据频率过滤N-gram - 优化版本
        if ngram_token_freq_dict:  # 使用ngram_token_freq_dict而不是ngram_freq_dict
            for ngrams, pos in matched_ngrams_with_pos:
                freq = ngram_token_freq_dict[str(ngrams)]

                if freq >= min_freq:
                    matched_ngrams.append(ngrams)
                    matched_ngrams_with_freq.append((ngrams, freq))
                else:
                    filtered_ngrams.append((ngrams, freq))  # 不需要文本时使用空字符串

            logger.debug(
                f"序列 {idx}: 使用频率过滤后匹配到 {len(matched_ngrams)} 个N-gram，匹配率: {len(matched_ngrams) / total_tokens * 100:.2f}%"
            )
        else:
            # 不过滤，保留所有匹配的N-gram
            matched_ngrams = [ngrams for ngrams, _ in matched_ngrams_with_pos]
            logger.debug(
                f"序列 {idx}: 匹配到 {len(matched_ngrams)} 个N-gram, 匹配率: {len(matched_ngrams) / total_tokens * 100:.2f}%"
            )

        # 计算覆盖率
        coverage_ratio = len(matched_ngrams) / total_tokens * 100 if total_tokens > 0 else 0  # noqa: F841

        # 更新元数据统计信息
        meta_data.update_match_stats(len(matched_ngrams), total_tokens)

        if len(filtered_ngrams) > 0:
            meta_data.update_filtered_count(filtered_ngrams)

        # 处理频率信息
        if matched_ngrams_with_freq:
            # 找出最高和最低频率的N-gram
            max_freq_ngram = max(matched_ngrams_with_freq, key=lambda x: x[1])
            min_freq_ngram = min(matched_ngrams_with_freq, key=lambda x: x[1])

            # 只在需要时解码
            max_freq_text = tokenizer.decode(list(max_freq_ngram[0]))
            min_freq_text = tokenizer.decode(list(min_freq_ngram[0]))

            # 更新频率统计信息
            meta_data.update_freq_stats(max_freq_text, max_freq_ngram[1], min_freq_text, min_freq_ngram[1])

    except Exception as e:
        logger.error(f"处理序列 {idx} 时出错: {str(e)}")
        logger.debug(f"问题序列: {text[:50]}...")

    return idx + 1


def analyze_ngram_coverage(
    data_sequence_list,
    tokenizer,
    output_dir,
    dataset_name=None,
    ngram_df=None,
    encoder=None,
    stats_collector=None,
    min_freq_filter=5,
):
    """分析N-gram在数据集上的覆盖率

    Args:
        data_sequence_list: 数据序列列表
        tokenizer: 用于将文本转换为token ID的tokenizer
        output_dir: 输出目录路径
        dataset_name: 数据集名称，用于日志和输出文件命名
        ngram_df: 包含N-gram信息的DataFrame
        encoder: N-gram编码器对象
        stats_collector: 用于收集统计结果的全局统计收集器
        min_freq_filter: 最小频率阈值,过滤n-gram的长尾
     覆盖率 = 有匹配序列数 / 总序列数 × 100%
     平均匹配数 = 总匹配数 / 总序列数
    Returns:
        包含分析结果的字典
    """

    logger.info(f"开始分析N-gram覆盖率 - 数据集: {dataset_name or '未命名'}")

    # 获取序列总数
    total_seqs = len(data_sequence_list)
    logger.info(f"数据集包含 {total_seqs} 个序列")

    if total_seqs == 0:
        logger.warning("没有序列进行分析")
        return {}, {}, 0

    # 初始化结果字典
    # logger.info("初始化结果字典...")
    results = {"num_matches": [0] * total_seqs, "matched_ngrams": [""] * total_seqs}

    # 初始化元数据统计对象
    meta_data = NgramMetaData()

    # 初始化必要变量
    # logger.info(f"使用编码器: {type(encoder).__name__}")
    ngram_encoder = encoder

    # 从ngram_df中创建ngram_freq_dict - 优化版本
    ngram_freq_dict = {}
    ngram_token_freq_dict = {}  # 新增：基于token ID的频率字典

    if ngram_df is not None:
        # logger.info("创建N-gram频率字典...")
        for _, row in ngram_df.iterrows():
            if "N-gram" in row and "频率" in row:
                ngram_text = row["N-gram"]  # noqa: F841
                freq = row["频率"]
                ngram_freq_dict[row["N-gram"]] = freq
                ngram_token_freq_dict[row["token_ids"]] = freq

        logger.info(
            f"创建了 {len(ngram_token_freq_dict)} 个token ID映射（总N-gram数: {len(ngram_freq_dict)}）"
        )
    else:
        logger.warning("没有N-gram列表文件，将不进行频率过滤")

    includes_no_match = True  # 是否包含无匹配的数据

    # 遍历每个文本序列
    logger.info(f"开始处理{dataset_name}序列...,最小频率阈值: {min_freq_filter}")
    idx = 0

    for text in tqdm(data_sequence_list, desc="处理序列"):
        idx = process_sequence(
            text,
            idx,
            tokenizer,
            ngram_encoder,
            ngram_freq_dict,
            ngram_token_freq_dict,
            min_freq_filter,
            results,
            meta_data,
            total_seqs,
        )

    # 模拟分类结果（实际应用中应该有真实的标签和预测）
    logger.info("生成标签数据...")
    results["actual_label"] = [1] * total_seqs
    results["prediction_label"] = [1] * total_seqs

    # 将数据分为两类：模型正确分类的数据和错误分类的数据
    logger.info("分离正确和错误分类的数据...")
    results_correct = results.copy()  # 在这个示例中，所有数据都被正确分类
    results_wrong = {k: [] for k in results.keys()}  # 在这个示例中，没有错误分类的数据

    # 根据includes_no_match参数决定是否包含没有N-gram匹配的数据
    logger.info(f"筛选数据 (includes_no_match={includes_no_match})...")
    if not includes_no_match:
        # 只保留有N-gram匹配的数据
        results_correct_ = {
            k: [
                v[i]
                for i in range(len(results_correct["num_matches"]))
                if results_correct["num_matches"][i] > 0
            ]
            for k, v in results_correct.items()
        }
        results_wrong_ = results_wrong  # 在这个示例中没有错误分类的数据
        logger.info(f"筛选后保留 {len(results_correct_['num_matches'])}/{total_seqs} 个有匹配的序列")
    else:
        # 保留所有数据
        results_correct_ = results_correct
        results_wrong_ = results_wrong
        logger.info(f"保留所有 {total_seqs} 个序列")

    # 使用NgramMetaData类替代旧的make_meta_data函数
    # 正确分类数据的元数据已经在meta_data中
    meta_data_correct = meta_data.to_dict()

    # 错误分类数据的元数据（在此示例中为空）
    meta_data_wrong = NgramMetaData().to_dict()

    # 输出统计信息
    meta_data.log_stats(dataset_name)

    # 收集统计结果到全局统计收集器
    if stats_collector is not None:
        stats_collector.append(meta_data.get_stats_info(dataset_name))

    # 返回包含各种分析结果的字典
    logger.info("分析完成，返回结果")
    return {
        "true": results_correct_,  # 模型正确分类的结果
        "false": results_wrong_,  # 模型错误分类的结果
        "meta_data": meta_data.to_dict(),  # 数据集的总体匹配结果
        "meta_data_correct": meta_data_correct,  # 正确分类数据的匹配结果
        "meta_data_wrong": meta_data_wrong,  # 错误分类数据的匹配结果
    }


def find_gue_files(gue_dir):
    """查找GUE数据文件"""
    if not gue_dir or not os.path.exists(gue_dir):
        logger.warning(f"GUE目录不存在: {gue_dir}")
        return []

    gue_files = []
    # 查找所有train.csv和test.csv文件
    for root, _, files in os.walk(gue_dir):
        for file in files:
            if file.endswith(".csv") and ("train" in file or "test" in file):
                gue_files.append(os.path.join(root, file))

    return gue_files


def find_mspecies_files(data_dir):
    """查找mspecies数据文件"""
    mspecies_files = []

    # 查找预训练数据目录中的dev.txt和train.txt
    dev_file = os.path.join(data_dir, "pretrain", "dev", "dev.txt")
    train_file = os.path.join(data_dir, "pretrain", "train", "train.txt")

    if os.path.exists(dev_file):
        mspecies_files.append(dev_file)

    if os.path.exists(train_file):
        mspecies_files.append(train_file)

    return mspecies_files


def main():
    """命令行入口函数"""

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="N-gram编码器覆盖率分析工具")

    # 添加命令行参数
    parser.add_argument("--encoder", type=str, required=True, help="N-gram编码器文件路径")
    parser.add_argument("--output-dir", type=str, default="./coverage_analysis", help="分析结果输出目录")
    parser.add_argument("--gue-dir", type=str, default=None, help="GUE数据集目录")
    parser.add_argument("--mspecies-dir", type=str, default="../data", help="数据目录")
    parser.add_argument("--sample-size", type=int, default=10000, help="每个数据集的样本大小")
    parser.add_argument("--ngram-list", type=str, default=None, help="N-gram列表文件路径")
    parser.add_argument("--tok", type=str, default="zhihan1996/DNABERT-2-117M", help="使用的tokenizer名称")
    parser.add_argument(
        "--min-freq-filter", type=int, default=5, help="ngram的最小频率阈值,过滤n-gram的长尾"
    )
    parser.add_argument("--experiment-desc", type=str, default=None, help="实验描述信息")

    # 解析命令行参数
    args = parser.parse_args()

    # 加载数据
    gue_sequences_map = []
    mspecies_sequences = []

    # 从GUE数据集加载数据
    if args.gue_dir:
        gue_sequences_map = get_gue_each_sequences(args.gue_dir)

    # 从输入文件加载数据
    if args.mspecies_dir:
        get_mspecies_sequences(args.mspecies_dir, mspecies_sequences)

    # 加载N-gram编码器
    try:
        encoder = NgramEncoder.from_file(args.encoder)
        logger.info(f"从文件加载N-gram编码器: {args.encoder}")
        logger.info(f"编码器包含 {len(encoder.get_vocab())} 个N-gram")
    except Exception as e:
        logger.error(f"加载N-gram编码器失败: {str(e)}")
        return 1

    if args.ngram_list:
        try:
            logger.info(f"加载N-gram列表并绘制分布图: {args.ngram_list}")

            # 读取N-gram列表文件
            ngram_df = pd.read_csv(args.ngram_list, sep="\t")
            # 绘制分布图
            plot_ngram_distribution(ngram_df, args.output_dir)
            plot_ngram_zipfs_law(ngram_df, args.output_dir)

        except Exception as e:
            logger.error(f"处理N-gram列表时出错: {str(e)}")

    # 获取输出路径的目录部分
    ngram_encoder_dir = os.path.dirname(args.output_dir) if os.path.dirname(args.output_dir) else "."

    logger.info(f"加载tokenizer: {args.tok}")
    tokenizer = AutoTokenizer.from_pretrained(args.tok)

    # 创建统计结果收集器
    stats_collector = []

    # 存储所有数据集的覆盖率结果
    all_coverage_results = {}

    # 分析GUE数据集的覆盖率
    if gue_sequences_map:
        logger.info("分析GUE数据集的N-gram覆盖率...")
        for species_name, sequences in gue_sequences_map.items():
            dataset_name = "GUE_" + species_name
            result = analyze_ngram_coverage(
                sequences,
                tokenizer,
                ngram_encoder_dir,
                dataset_name=dataset_name,
                ngram_df=ngram_df,
                encoder=encoder,
                stats_collector=stats_collector,
                min_freq_filter=args.min_freq_filter,
            )
            all_coverage_results[dataset_name] = result

    # 分析mspecies数据集的覆盖率
    if mspecies_sequences:
        logger.info("分析mspecies数据集的N-gram覆盖率...")
        result = analyze_ngram_coverage(
            mspecies_sequences,
            tokenizer,
            ngram_encoder_dir,
            dataset_name="mspecies",
            ngram_df=ngram_df,
            encoder=encoder,
            stats_collector=stats_collector,
            min_freq_filter=args.min_freq_filter,
        )
        all_coverage_results["mspecies"] = result

    # 将所有统计结果保存到一个CSV/Excel文件
    if stats_collector:
        os.makedirs(args.output_dir, exist_ok=True)
        stats_df = pd.DataFrame(stats_collector)

        # 保存为CSV
        csv_path = os.path.join(args.output_dir, "ngram_coverage_all_datasets.csv")
        stats_df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"所有数据集的统计结果已保存到: {csv_path}")

    # 生成HTML报告
    if all_coverage_results:
        generate_coverage_report(all_coverage_results, args.output_dir, args.experiment_desc)
        logger.info("已生成覆盖率HTML报告")

    logger.info("N-gram覆盖率分析完成")
    return 0


# 添加一个函数用于生成HTML报告


def generate_coverage_report(coverage_results, output_dir, experiment_desc=None):
    """
    生成HTML报告展示不同数据集的N-gram覆盖率

    Args:
        coverage_results: 包含各数据集覆盖率结果的字典
        output_dir: 输出目录
        experiment_desc: 实验描述信息
    """
    logger.info("生成覆盖率HTML报告...")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有数据集的统计信息
    stats_data = []
    for dataset_name, result in coverage_results.items():
        meta = result.get("meta_data", {})
        if not meta or meta.get("total_num_data", 0) == 0:
            continue

        total_seqs = meta.get("total_num_data", 0)
        total_tokens = meta.get("total_tokens", 0)
        covered_tokens = meta.get("covered_tokens", 0)
        token_coverage = covered_tokens / total_tokens * 100 if total_tokens > 0 else 0

        stats_info = {
            "数据集": dataset_name,
            "总序列数": total_seqs,
            "有匹配序列数": meta.get("num_data_has_match", 0),
            "有匹配序列百分比": meta.get("num_data_has_match", 0) / total_seqs * 100
            if total_seqs > 0
            else 0,
            "无匹配序列数": meta.get("num_data_no_match", 0),
            "无匹配序列百分比": meta.get("num_data_no_match", 0) / total_seqs * 100
            if total_seqs > 0
            else 0,
            "匹配到的ngram总数": meta.get("total_num_matches", 0),
            "平均每序列匹配数": meta.get("total_num_matches", 0) / total_seqs if total_seqs > 0 else 0,
            "总分词数": total_tokens,
            "匹配覆盖的分词数": covered_tokens,
            "分词覆盖率": token_coverage,
            "被过滤的ngram数量": meta.get("num_ngram_filtered", 0),
            "数据集最高频率N-gram": meta.get("max_freq_ngram", ""),
            "数据集最高频率值": meta.get("max_freq_value", 0),
            "数据集最低频率N-gram": meta.get("min_freq_ngram", ""),
            "数据集最低频率值": meta.get("min_freq_value", 0),
        }

        stats_data.append(stats_info)

    if not stats_data:
        logger.warning("没有有效的覆盖率数据用于生成报告")
        return

    # 创建DataFrame
    stats_df = pd.DataFrame(stats_data)

    # 保存汇总CSV
    csv_path = os.path.join(output_dir, "ngram_coverage_summary.csv")
    stats_df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"汇总统计已保存到: {csv_path}")

    # 使用Plotly创建交互式图表
    # 1. 覆盖率百分比条形图
    fig1 = px.bar(
        stats_df,
        x="数据集",
        y="分词覆盖率",
        title="各数据集N-gram序列分词覆盖率",
        labels={"分词覆盖率": "分词覆盖率 (%)", "数据集": "数据集"},
        color="分词覆盖率",
        color_continuous_scale="Viridis",
    )
    fig1.update_layout(yaxis_range=[0, 100])

    # 2. 平均匹配数条形图
    fig2 = px.bar(
        stats_df,
        x="数据集",
        y="平均每序列匹配数",
        title="各数据集平均N-gram匹配数",
        labels={"平均每序列匹配数": "平均匹配数", "数据集": "数据集"},
        color="平均每序列匹配数",
        color_continuous_scale="Turbo",
    )

    # 3. 匹配与未匹配序列对比图
    fig3_data = []
    for _, row in stats_df.iterrows():
        fig3_data.append({"数据集": row["数据集"], "类型": "有匹配序列", "数量": row["有匹配序列数"]})
        fig3_data.append({"数据集": row["数据集"], "类型": "无匹配序列", "数量": row["无匹配序列数"]})

    fig3_df = pd.DataFrame(fig3_data)
    fig3 = px.bar(
        fig3_df, x="数据集", y="数量", color="类型", title="各数据集匹配与未匹配序列对比", barmode="group"
    )

    # 4. 分词覆盖率条形图
    fig4 = px.bar(
        stats_df,
        x="数据集",
        y="分词覆盖率",
        title="各数据集分词级别覆盖率",
        labels={"分词覆盖率": "分词覆盖率 (%)", "数据集": "数据集"},
        color="分词覆盖率",
        color_continuous_scale="Plasma",
    )
    fig4.update_layout(yaxis_range=[0, 100])

    # 保存HTML文件
    html_path = os.path.join(output_dir, "ngram_coverage_report.html")

    # 添加实验描述信息
    experiment_info = ""
    if experiment_desc:
        experiment_info = f"""
        <div class="experiment-info">
            <h2>实验信息</h2>
            <p><strong>实验描述:</strong> {experiment_desc}</p>
        </div>
        """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>N-gram覆盖率分析报告</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .chart {{
                    margin-bottom: 50px;
                    height: 500px;
                    width: 100%;
                    position: relative;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .experiment-info {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 5px solid #4a90e2;
                }}
                .ngram-info {{
                    background-color: #f8f4e8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 5px solid #e2a44a;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>N-gram覆盖率分析报告</h1>
                <p>生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                {experiment_info}
                
                <h2>数据集覆盖率统计</h2>
                <table>
                    <tr>
                        <th>数据集</th>
                        <th>总序列数</th>
                        <th>有匹配序列数</th>
                        <th>序列覆盖率</th>
                        <th>无匹配序列数</th>
                        <th>无匹配序列百分比</th>
                        <th>匹配到的ngram总数</th>
                        <th>平均每序列匹配数</th>
                        <th>总分词数</th>
                        <th>匹配覆盖的分词数</th>
                        <th>分词覆盖率</th>
                        <th>被过滤的ngram数量</th>
                        <th>数据集最高频率值</th>
                        <th>数据集最低频率值</th>
                    </tr>
                    {"".join(f"<tr><td>{row['数据集']}</td><td>{row['总序列数']}</td><td>{row['有匹配序列数']}</td><td>{row['有匹配序列百分比']:.2f}%</td><td>{row['无匹配序列数']}</td><td>{row['无匹配序列百分比']:.2f}%</td><td>{row['匹配到的ngram总数']}</td><td>{row['平均每序列匹配数']:.2f}</td><td>{row['总分词数']}</td><td>{row['匹配覆盖的分词数']}</td><td>{row['分词覆盖率']:.2f}%</td><td>{row['被过滤的ngram数量']}</td><td>{row['数据集最高频率值']}</td><td>{row['数据集最低频率值']}</td></tr>" for _, row in stats_df.iterrows())}
                </table>
                
                
                <h2>覆盖率可视化</h2>
                <div id="chart1" class="chart">
                    <h3>序列级别覆盖率</h3>
                </div>
                <div id="chart2" class="chart">
                    <h3>平均每序列匹配数</h3>
                </div>
                <div id="chart3" class="chart">
                    <h3>匹配与未匹配序列对比</h3>
                </div>
                <div id="chart4" class="chart">
                    <h3>分词级别覆盖率</h3>
                </div>
                
                <script>
                    var fig1 = {fig1.to_json()};
                    var fig2 = {fig2.to_json()};
                    var fig3 = {fig3.to_json()};
                    var fig4 = {fig4.to_json()};
                    
                    // 更新布局，设置图表高度
                    fig1.layout.height = 450;
                    fig2.layout.height = 450;
                    fig3.layout.height = 450;
                    fig4.layout.height = 450;
                    
                    // 确保图表有足够的边距
                    fig1.layout.margin = {{t: 50, b: 50, l: 50, r: 50}};
                    fig2.layout.margin = {{t: 50, b: 50, l: 50, r: 50}};
                    fig3.layout.margin = {{t: 50, b: 50, l: 50, r: 50}};
                    fig4.layout.margin = {{t: 50, b: 50, l: 50, r: 50}};
                    
                    Plotly.newPlot('chart1', fig1.data, fig1.layout);
                    Plotly.newPlot('chart2', fig2.data, fig2.layout);
                    Plotly.newPlot('chart3', fig3.data, fig3.layout);
                    Plotly.newPlot('chart4', fig4.data, fig4.layout);
                </script>
            </div>
        </body>
        </html>
        """)

    logger.info(f"HTML报告已生成: {html_path}")


if __name__ == "__main__":
    main()

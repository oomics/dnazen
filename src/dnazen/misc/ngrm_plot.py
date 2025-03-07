import os
import logging
import traceback
import pandas as pd
import plotly.express as px

# 配置日志
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_ngram_distribution(ngram_df, output_dir):
    """
    绘制N-gram分布图
    加载N-gram列表,绘制N-gram频率分布散点图，
    以N-gram长度为x轴，N-gram频率为y轴，以每个N-gram作为分布点，
    添加点上的N-gram文本和BPE长度

    Args:
        ngram_df: 包含N-gram信息的DataFrame
        output_dir: 输出目录
    """
    try:
        # 确保输出目录存在
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 检查DataFrame的列名和数据
        logger.info(f"DataFrame列名: {ngram_df.columns.tolist()}")
        logger.info(f"DataFrame形状: {ngram_df.shape}")
        logger.info(f"DataFrame前5行:\n{ngram_df.head()}")

        # 检查关键列的数据类型和值范围
        if "频率" in ngram_df.columns:
            logger.info(f"频率列数据类型: {ngram_df['频率'].dtype}")
            logger.info(f"频率列值范围: [{ngram_df['频率'].min()}, {ngram_df['频率'].max()}]")

        if "字符长度" in ngram_df.columns:
            logger.info(f"字符长度列数据类型: {ngram_df['字符长度'].dtype}")
            logger.info(f"字符长度列值范围: [{ngram_df['字符长度'].min()}, {ngram_df['字符长度'].max()}]")

        # 尝试将数值列转换为数值类型
        for col in ["频率", "字符长度", "BPE分词长度"]:
            if col in ngram_df.columns:
                try:
                    ngram_df[col] = pd.to_numeric(ngram_df[col], errors="coerce")
                    logger.info(f"已将{col}列转换为数值类型")
                except Exception as e:
                    logger.warning(f"转换{col}列为数值类型时出错: {str(e)}")

        # 删除任何包含NaN的行
        original_len = len(ngram_df)
        ngram_df = ngram_df.dropna(subset=["频率", "字符长度"])
        if len(ngram_df) < original_len:
            logger.warning(f"删除了{original_len - len(ngram_df)}行包含NaN值的数据")

        # 确保有足够的数据点
        if len(ngram_df) == 0:
            logger.error("处理后的DataFrame为空，无法绘图")
            return False

        try:
            # 创建第一个散点图 - 基本N-gram分布
            if "BPE分词长度" in ngram_df.columns:
                # 创建带有颜色映射的散点图
                fig = px.scatter(
                    ngram_df,
                    x="字符长度",
                    y="频率",
                    color="BPE分词长度",
                    size="频率",
                    size_max=30,
                    hover_name="N-gram",
                    hover_data={"字符长度": True, "频率": ":.6f", "BPE分词长度": True},
                    color_continuous_scale="viridis",
                    opacity=0.7,
                    title="N-gram Length vs Frequency Distribution",
                    labels={
                        "字符长度": "N-gram Character Length",
                        "频率": "N-gram Frequency",
                        "BPE分词长度": "BPE Token Length",
                    },
                )
            else:
                # 创建基本散点图
                fig = px.scatter(
                    ngram_df,
                    x="字符长度",
                    y="频率",
                    size="频率",
                    size_max=30,
                    hover_name="N-gram",
                    hover_data={"字符长度": True, "频率": ":.6f"},
                    opacity=0.7,
                    title="N-gram Length vs Frequency Distribution",
                    labels={"字符长度": "N-gram Character Length", "频率": "N-gram Frequency"},
                )

            # 设置对数刻度
            fig.update_yaxes(type="log")

            # 添加网格线
            fig.update_layout(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                plot_bgcolor="white",
            )

            # 保存为HTML文件
            html_path = os.path.join(plots_dir, "ngram_distribution.html")
            fig.write_html(html_path)

            logger.info(f"N-gram分布图已保存到: {html_path}")

            # 创建第二个散点图 - 按BPE分词长度着色
            if "BPE分词长度" in ngram_df.columns:
                # 创建第二个图表，使用不同的颜色映射
                fig2 = px.scatter(
                    ngram_df,
                    x="字符长度",
                    y="频率",
                    color="BPE分词长度",
                    size="频率",
                    size_max=30,
                    hover_name="N-gram",
                    hover_data={"字符长度": True, "频率": ":.6f", "BPE分词长度": True},
                    color_continuous_scale="plasma",
                    opacity=0.7,
                    title="N-gram Length vs Frequency (Colored by BPE Token Length)",
                    labels={
                        "字符长度": "N-gram Character Length",
                        "频率": "N-gram Frequency",
                        "BPE分词长度": "BPE Token Length",
                    },
                )

                # 设置对数刻度
                fig2.update_yaxes(type="log")

                # 添加网格线
                fig2.update_layout(
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
                    plot_bgcolor="white",
                )

                # 保存为HTML文件
                html_path2 = os.path.join(plots_dir, "ngram_bpe_colored.html")
                fig2.write_html(html_path2)

                logger.info(f"N-gram BPE着色图已保存到: {html_path2}")
            else:
                logger.warning("DataFrame中缺少'BPE分词长度'列，无法创建BPE着色图")

            return True
        except Exception as e:
            logger.error(f"创建散点图时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    except Exception as e:
        logger.error(f"绘制N-gram分布图时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def plot_ngram_zipfs_law(ngram_df, output_dir):
    """
    绘制N-gram频率的Zipf分布图

    Args:
        ngram_df: 包含N-gram信息的DataFrame
        output_dir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import logging

        logger = logging.getLogger(__name__)

        # 确保输出目录存在
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 检查DataFrame的列名和数据
        logger.info(f"DataFrame列名: {ngram_df.columns.tolist()}")
        logger.info(f"DataFrame形状: {ngram_df.shape}")

        # 确保频率列为数值类型
        if "频率" in ngram_df.columns:
            ngram_df["频率"] = pd.to_numeric(ngram_df["频率"], errors="coerce")
            ngram_df = ngram_df.dropna(subset=["频率"])

            # 按频率降序排序
            ngram_df = ngram_df.sort_values(by="频率", ascending=False).reset_index(drop=True)

            # 计算排名（从1开始）
            ngram_df["排名"] = np.arange(1, len(ngram_df) + 1)

            # 计算理论Zipf分布
            if len(ngram_df) > 0:
                first_freq = ngram_df["频率"].iloc[0]
                ngram_df["理论频率"] = first_freq / ngram_df["排名"]

                # 设置matplotlib的字体，使用英文标签避免中文显示问题
                plt.rcParams["font.family"] = "DejaVu Sans"
                plt.rcParams["axes.unicode_minus"] = False

                # 创建图表1：Zipf分布
                plt.figure(figsize=(12, 8))

                # 绘制实际频率
                plt.loglog(ngram_df["排名"], ngram_df["频率"], "bo", alpha=0.7, label="Actual Frequency")

                # 绘制理论Zipf分布
                plt.loglog(
                    ngram_df["排名"],
                    ngram_df["理论频率"],
                    "r--",
                    linewidth=2,
                    label="Theoretical Zipf Distribution",
                )

                # 添加标题和标签（使用英文避免字体问题）
                plt.title("N-gram Frequency Zipf Distribution", fontsize=16)
                plt.xlabel("Rank (log scale)", fontsize=14)
                plt.ylabel("Frequency (log scale)", fontsize=14)
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.legend(fontsize=12)

                # 保存图表
                plt.tight_layout()
                png_path = os.path.join(plots_dir, "ngram_zipf_law.png")
                plt.savefig(png_path, dpi=300)
                plt.close()

                logger.info(f"N-gram Zipf分布图已保存到: {png_path}")

                # 创建图表2：对数-对数图，用于验证Zipf定律
                # 在对数-对数空间中计算线性回归
                log_rank = np.log(ngram_df["排名"])
                log_freq = np.log(ngram_df["频率"])
                mask = ~np.isnan(log_rank) & ~np.isnan(log_freq)

                if np.sum(mask) > 1:  # 确保有足够的数据点进行拟合
                    slope, intercept = np.polyfit(log_rank[mask], log_freq[mask], 1)
                    fit_line = np.exp(intercept + slope * log_rank)

                    # 理论Zipf线 (斜率 = -1)
                    theoretical_intercept = intercept - slope  # 调整截距使线通过数据
                    theory_line = np.exp(theoretical_intercept - 1 * log_rank)

                    # 创建新图表
                    plt.figure(figsize=(12, 8))

                    # 绘制实际数据点
                    plt.loglog(
                        ngram_df["排名"], ngram_df["频率"], "bo", alpha=0.7, label="Actual Frequency"
                    )

                    # 绘制拟合线
                    plt.loglog(
                        ngram_df["排名"],
                        fit_line,
                        "r-",
                        linewidth=2,
                        label=f"Fitted Line (slope = {slope:.2f})",
                    )

                    # 绘制理论Zipf线
                    plt.loglog(
                        ngram_df["排名"],
                        theory_line,
                        "g--",
                        linewidth=2,
                        label="Theoretical Zipf Line (slope = -1)",
                    )

                    # 添加标题和标签（使用英文避免字体问题）
                    plt.title("N-gram Frequency Zipf Distribution (Log-Log Plot)", fontsize=16)
                    plt.xlabel("Rank (log scale)", fontsize=14)
                    plt.ylabel("Frequency (log scale)", fontsize=14)
                    plt.grid(True, which="both", ls="-", alpha=0.2)
                    plt.legend(fontsize=12)

                    # 保存图表
                    plt.tight_layout()
                    png_path2 = os.path.join(plots_dir, "ngram_zipf_law_loglog.png")
                    plt.savefig(png_path2, dpi=300)
                    plt.close()

                    logger.info(f"N-gram Zipf对数-对数图已保存到: {png_path2}")

                    # 创建图表3：前100个N-gram的Zipf分布
                    if len(ngram_df) >= 10:
                        top_n = min(100, len(ngram_df))
                        top_df = ngram_df.head(top_n)

                        plt.figure(figsize=(12, 8))

                        # 绘制实际频率
                        plt.loglog(
                            top_df["排名"], top_df["频率"], "bo", alpha=0.7, label="Actual Frequency"
                        )

                        # 绘制理论Zipf分布
                        plt.loglog(
                            top_df["排名"],
                            top_df["理论频率"],
                            "r--",
                            linewidth=2,
                            label="Theoretical Zipf Distribution",
                        )

                        # 添加标题和标签（使用英文避免字体问题）
                        plt.title(f"Top {top_n} N-grams Zipf Distribution", fontsize=16)
                        plt.xlabel("Rank (log scale)", fontsize=14)
                        plt.ylabel("Frequency (log scale)", fontsize=14)
                        plt.grid(True, which="both", ls="-", alpha=0.2)
                        plt.legend(fontsize=12)

                        # 保存图表
                        plt.tight_layout()
                        png_path3 = os.path.join(plots_dir, "ngram_zipf_law_top100.png")
                        plt.savefig(png_path3, dpi=300)
                        plt.close()

                        logger.info(f"前{top_n}个N-gram的Zipf分布图已保存到: {png_path3}")

                return True
            else:
                logger.warning("DataFrame为空，无法绘制Zipf分布图")
                return False
        else:
            logger.warning("DataFrame中缺少'频率'列，无法绘制Zipf分布图")
            return False
    except Exception as e:
        import traceback

        logger.error(f"绘制N-gram Zipf分布图时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

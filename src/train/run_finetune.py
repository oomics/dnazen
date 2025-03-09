# import deepspeed
from typing import Any
import argparse
import os
import json
import logging

import numpy as np

# import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import pandas as pd
import transformers
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.models.bert.configuration_bert import BertConfig

from dnazen.model.bert_models import BertForSequenceClassification
from dnazen.data.labeled_dataset import LabeledDataset
from dnazen.ngram import NgramEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s  - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """
    Compute metrics used for huggingface trainer.
    """
    predictions, labels = eval_pred
    # calculate metric with sklearn
    predictions: np.ndarray
    labels: np.ndarray

    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": accuracy_score(valid_labels, valid_predictions),
        "f1": f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": matthews_corrcoef(valid_labels, valid_predictions),
        "precision": precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }


# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits: torch.Tensor | tuple[torch.Tensor, Any], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for sequence classification.")
    parser.add_argument("--data_path", type=str, help="Path to the data directory.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data2/peter/dnazen_pretrain_v3.1/outputs/checkpoint-2000",
        help="Path to model checkpoint directory.",
    )
    parser.add_argument("--out", type=str, help="Path to the results directory.")
    # parser.add_argument("--model_max_len", type=int, default=70, help="Maximum length of the model input.")
    parser.add_argument("--ngram_encoder_dir", type=str, help="Path to the ngram encoder directory.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", default=42)

    return parser.parse_args()


def generate_prediction_html_report(results_df, metrics, output_path):
    """
    生成预测结果的HTML可视化报告
    
    参数:
        results_df (pd.DataFrame): 包含预测结果的DataFrame，需要包含'text', 'actual_label', 'prediction_label'列
        metrics (dict): 评估指标字典
        output_path (str): HTML报告输出路径
    
    返回:
        str: 生成的HTML报告路径
    """
    logger.info(f"生成HTML报告: {output_path}")
    
    # 计算准确率
    correct_predictions = sum(results_df["actual_label"] == results_df["prediction_label"])
    total_predictions = len(results_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DNA序列分类预测结果</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .summary {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            flex: 1;
            min-width: 200px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .correct {{
            background-color: #d4edda;
        }}
        .incorrect {{
            background-color: #f8d7da;
        }}
        .pagination {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        .pagination button {{
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 4px;
            cursor: pointer;
            border-radius: 4px;
        }}
        .pagination button:hover {{
            background-color: #45a049;
        }}
        .pagination button:disabled {{
            background-color: #cccccc;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DNA序列分类预测结果</h1>
        
        <div class="summary">
            <h2>预测摘要</h2>
            <p>总样本数: {total_predictions}</p>
            <p>正确预测数: {correct_predictions}</p>
            <p>准确率: {accuracy:.2%}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>准确率</h3>
                <div class="metric-value">{accuracy:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>F1分数</h3>
                <div class="metric-value">{metrics.get('eval_f1', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Matthews相关系数</h3>
                <div class="metric-value">{metrics.get('eval_matthews_correlation', 0):.4f}</div>
            </div>
        </div>
        
        <h2>预测详情</h2>
        <div>
            <input type="text" id="searchInput" placeholder="搜索序列..." style="padding: 8px; width: 300px; margin-bottom: 10px;">
            <button onclick="searchTable()" style="padding: 8px 16px;">搜索</button>
        </div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>#</th>
                    <th>DNA序列</th>
                    <th>实际标签</th>
                    <th>预测标签</th>
                    <th>预测状态</th>
                </tr>
            </thead>
            <tbody>
"""

    # 添加表格行
    for i, (_, row) in enumerate(results_df.iterrows()):
        correct = row["actual_label"] == row["prediction_label"]
        row_class = "correct" if correct else "incorrect"
        status = "正确" if correct else "错误"
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{i+1}</td>
                    <td>{row["text"]}</td>
                    <td>{row["actual_label"]}</td>
                    <td>{row["prediction_label"]}</td>
                    <td>{status}</td>
                </tr>
        """

    # 完成HTML内容
    html_content += """
            </tbody>
        </table>
        
        <div class="pagination">
            <button id="prevBtn" onclick="previousPage()" disabled>上一页</button>
            <span id="pageInfo">第 1 页，共 1 页</span>
            <button id="nextBtn" onclick="nextPage()">下一页</button>
        </div>
    </div>

    <script>
        // 分页功能
        let currentPage = 1;
        const rowsPerPage = 20;
        const table = document.getElementById('resultsTable');
        const rows = table.getElementsByTagName('tbody')[0].rows;
        const pageCount = Math.ceil(rows.length / rowsPerPage);
        
        document.getElementById('pageInfo').textContent = `第 ${currentPage} 页，共 ${pageCount} 页`;
        
        function showPage(page) {
            // 隐藏所有行
            for (let i = 0; i < rows.length; i++) {
                rows[i].style.display = 'none';
            }
            
            // 显示当前页的行
            const start = (page - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            
            for (let i = start; i < end && i < rows.length; i++) {
                rows[i].style.display = '';
            }
            
            // 更新按钮状态
            document.getElementById('prevBtn').disabled = page === 1;
            document.getElementById('nextBtn').disabled = page === pageCount;
            document.getElementById('pageInfo').textContent = `第 ${page} 页，共 ${pageCount} 页`;
        }
        
        function nextPage() {
            if (currentPage < pageCount) {
                currentPage++;
                showPage(currentPage);
            }
        }
        
        function previousPage() {
            if (currentPage > 1) {
                currentPage--;
                showPage(currentPage);
            }
        }
        
        // 搜索功能
        function searchTable() {
            const input = document.getElementById('searchInput').value.toLowerCase();
            
            for (let i = 0; i < rows.length; i++) {
                const text = rows[i].cells[1].textContent.toLowerCase();
                if (text.includes(input)) {
                    rows[i].style.display = '';
                } else {
                    rows[i].style.display = 'none';
                }
            }
            
            // 禁用分页
            if (input) {
                document.getElementById('prevBtn').disabled = true;
                document.getElementById('nextBtn').disabled = true;
                document.getElementById('pageInfo').textContent = '搜索模式';
            } else {
                showPage(currentPage);
            }
        }
        
        // 初始化显示第一页
        showPage(1);
    </script>
</body>
</html>
"""

    # 保存HTML文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML报告已生成: {output_path}")
    return output_path


# --- main ---
def main():
    # 步骤1: 解析命令行参数
    logger.info("步骤1: 解析命令行参数...")
    args = parse_args()

    DATA_PATH = args.data_path
    RESULTS_PATH = args.out
    # MODEL_MAX_LEN = args.model_max_len
    NGRAM_ENCODER_DIR = args.ngram_encoder_dir
    LEARNING_RATE = args.lr
    NUM_TRAIN_EPOCHS = args.num_train_epochs
    CHECKPOINT_DIR = args.checkpoint
    
    logger.info(f"数据路径: {DATA_PATH}")
    logger.info(f"结果输出路径: {RESULTS_PATH}")
    logger.info(f"N-gram编码器路径: {NGRAM_ENCODER_DIR}")
    logger.info(f"学习率: {LEARNING_RATE}")
    logger.info(f"训练轮数: {NUM_TRAIN_EPOCHS}")
    logger.info(f"模型检查点路径: {CHECKPOINT_DIR}")
    
    # 步骤2: 准备输出目录
    logger.info("步骤2: 准备输出目录...")
    if not os.path.exists(RESULTS_PATH):
        logger.info(f"创建输出目录: {RESULTS_PATH}")
        os.makedirs(RESULTS_PATH)
    else:
        logger.info(f"输出目录已存在: {RESULTS_PATH}")
    
    # 步骤3: 加载分词器
    logger.info("步骤3: 加载分词器...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        # model_max_length=MODEL_MAX_LEN,
        padding_side="right",
        padding="longest",
        use_fast=True,
        trust_remote_code=True,
    )  # type: ignore
    logger.info("分词器加载完成")
    
    # 步骤4: 加载n-gram编码器
    logger.info(f"步骤4: 加载n-gram编码器，路径: {NGRAM_ENCODER_DIR}")
    ngram_encoder = NgramEncoder.from_file(NGRAM_ENCODER_DIR)
    logger.info(f"n-gram词汇表大小: {ngram_encoder.get_vocab_size()}")


    # 步骤5: 加载数据集
    logger.info("步骤5: 加载训练、验证和测试数据集...")
    
    logger.info(f"加载训练集: {DATA_PATH}/train.csv")
    train_dataset = LabeledDataset(f"{DATA_PATH}/train.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder)
    logger.info(f"训练集大小: {len(train_dataset)}个样本")

    logger.info(f"加载测试集: {DATA_PATH}/test.csv")
    test_dataset = LabeledDataset(f"{DATA_PATH}/test.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder)
    logger.info(f"测试集大小: {len(test_dataset)}个样本")

    logger.info(f"加载验证集: {DATA_PATH}/dev.csv")
    val_dataset = LabeledDataset(f"{DATA_PATH}/dev.csv", tokenizer=tokenizer, ngram_encoder=ngram_encoder)
    logger.info(f"验证集大小: {len(val_dataset)}个样本")



    # 步骤6: 加载模型配置和模型
    logger.info("-------------------------------------------------------...")
    logger.info(f"步骤6: 从检查点加载模型，路径: {CHECKPOINT_DIR}")
    logger.info("加载模型配置...")
    config = BertConfig.from_pretrained(CHECKPOINT_DIR)
    setattr(config, "ngram_vocab_size", ngram_encoder.get_vocab_size())
    setattr(config, "num_word_hidden_layers", 6)
    logger.info(f"模型配置: num_labels={config.num_labels}, hidden_size={config.hidden_size}")

    logger.info("加载预训练模型...")
    model = BertForSequenceClassification.from_pretrained(CHECKPOINT_DIR, config=config)
    logger.info("模型加载完成")



    # 步骤7: 设置优化器
    logger.info("-------------------------------------------------------...")
    logger.info("步骤7: 配置优化器...")
    learning_rate = 1e-5
    logger.info(f"使用AdamW优化器，学习率={learning_rate}，权重衰减=0.01")

    ngram_layer_params = [
        param for name, param in model.named_parameters() if ("ngram_layer" in name and "Norm" not in name)
    ]
    logger.info(f"n-gram层参数数量: {len(ngram_layer_params)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        learning_rate,
        # momentum=args.momentum,
        weight_decay=0.01,
    )

    # 步骤8: 设置训练参数
    logger.info("-------------------------------------------------------...")
    logger.info("步骤8: 配置训练参数...")
    train_args = transformers.training_args.TrainingArguments(
        output_dir=RESULTS_PATH,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=500,
        max_grad_norm=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=LEARNING_RATE,
        bf16=args.fp16,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=8,
        logging_steps=200,
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_matthews_correlation",
        greater_is_better=True,
    )
    logger.info(f"训练批次大小: {args.per_device_train_batch_size}")
    logger.info(f"评估批次大小: {args.per_device_train_batch_size}")
    logger.info(f"训练轮数: {NUM_TRAIN_EPOCHS}")
    logger.info(f"随机种子: {args.seed}")

    # 步骤9: 初始化Trainer
    logger.info("-------------------------------------------------------...")
    logger.info("步骤9: 初始化Trainer...")
    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        optimizers=(optimizer, None),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    logger.info("Trainer初始化完成")

    # 步骤10: 开始训练
    logger.info("步骤10: 开始训练...")
    logger.info(f"训练数据集大小: {len(train_dataset)}个样本")
    logger.info(f"验证数据集大小: {len(val_dataset)}个样本")
    trainer.train()
    logger.info("训练完成")

    # 步骤11: 评估模型
    logger.info("-------------------------------------------------------...")
    logger.info("步骤11: 在测试集上评估模型...")
    results = trainer.evaluate(test_dataset)
    logger.info(f"测试集评估结果:")
    for metric_name, metric_value in results.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # 保存评估结果
    results_path = os.path.join(RESULTS_PATH, "eval_results.json")
    logger.info(f"保存评估结果到: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f)

    # 步骤12: 在测试集上进行预测
    logger.info("-------------------------------------------------------...")
    logger.info("步骤12: 在测试集上进行预测...")
    logger.info(f"测试数据集大小: {len(test_dataset)}个样本")
    test_output = trainer.predict(test_dataset)
    test_preds = test_output.predictions
    logger.info(f"预测完成，获得{len(test_preds)}个预测结果")

    # 步骤13: 整理预测结果
    logger.info("步骤13: 整理预测结果...")
    results = {
        "text": [],
        "actual_label": [],
        "prediction_label": [],
    }
    for i in range(len(test_dataset)):
        data = test_dataset[i]

        input_ids = data["input_ids"]
        actual_label = data["labels"]
        texts = tokenizer.decode(input_ids).replace("[CLS] ", "").replace(" [SEP]", "").replace(" [PAD]", "")

        results["text"].append(texts)
        results["actual_label"].append(actual_label)
        results["prediction_label"].append(test_preds[i])

    logger.info("-------------------------------------------------------...")
    # 步骤14: 保存预测结果
    logger.info("步骤14: 保存预测结果...")
    pred_results_path = os.path.join(RESULTS_PATH, "pred_results.csv")
    logger.info(f"保存预测结果到: {pred_results_path}")
    df = pd.DataFrame(results)
    df.to_csv(pred_results_path, mode="w")
    logger.info(f"已保存{len(df)}行预测结果")

    # 步骤15: 生成HTML可视化报告
    logger.info("步骤15: 生成HTML可视化报告...")
    html_path = os.path.join(RESULTS_PATH, "prediction_report.html")
    generate_prediction_html_report(df, results, html_path)
    
    logger.info("所有步骤完成")


if __name__ == "__main__":
    main()

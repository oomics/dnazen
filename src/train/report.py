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
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_parallel_finetune_progress_report(completed_tasks, pending_tasks, running_tasks, output_path):
    
    """生成进度报告"""
    logger.info(f"生成进度报告: {output_path}")
    
    # 计算总体进度
    total_tasks = len(completed_tasks) + len(pending_tasks) + len(running_tasks)
    completed_count = len(completed_tasks)
    running_count = len(running_tasks)
    pending_count = len(pending_tasks)
    
    if total_tasks > 0:
        progress_percentage = (completed_count / total_tasks) * 100
    else:
        progress_percentage = 0
    
    # 计算平均指标
    all_accuracy = []
    all_f1 = []
    all_matthews = []
    
    for result in completed_tasks:
        if result["status"] == "成功":
            eval_results = result.get("eval_results", {})
            accuracy = eval_results.get("eval_accuracy")
            f1 = eval_results.get("eval_f1")
            matthews = eval_results.get("eval_matthews_correlation")
            
            if accuracy is not None:
                all_accuracy.append(accuracy)
            if f1 is not None:
                all_f1.append(f1)
            if matthews is not None:
                all_matthews.append(matthews)
    
    avg_accuracy = np.mean(all_accuracy) if all_accuracy else 0
    avg_f1 = np.mean(all_f1) if all_f1 else 0
    avg_matthews = np.mean(all_matthews) if all_matthews else 0
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>微调任务进度报告</title>
    <meta http-equiv="refresh" content="60"> <!-- 每60秒自动刷新 -->
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
        .progress-container {{
            width: 100%;
            background-color: #f1f1f1;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .progress-bar {{
            height: 30px;
            background-color: #4CAF50;
            border-radius: 5px;
            text-align: center;
            line-height: 30px;
            color: white;
            font-weight: bold;
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
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .success {{
            color: #28a745;
        }}
        .failure {{
            color: #dc3545;
        }}
        .running {{
            color: #007bff;
        }}
        .pending {{
            color: #6c757d;
        }}
        .task-group {{
            margin-bottom: 30px;
        }}
        .accordion {{
            background-color: #f1f1f1;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            text-align: left;
            border: none;
            outline: none;
            transition: 0.4s;
            font-size: 16px;
            font-weight: bold;
        }}
        .active, .accordion:hover {{
            background-color: #ddd;
        }}
        .panel {{
            padding: 0 18px;
            background-color: white;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DNA序列微调任务进度报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>总体进度</h2>
            <div class="progress-container">
                <div class="progress-bar" style="width:{progress_percentage}%">
                    {progress_percentage:.2f}%
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <h3>总任务数</h3>
                    <div class="metric-value">{total_tasks}</div>
                </div>
                <div class="metric-card">
                    <h3>已完成</h3>
                    <div class="metric-value">{completed_count}</div>
                </div>
                <div class="metric-card">
                    <h3>运行中</h3>
                    <div class="metric-value">{running_count}</div>
                </div>
                <div class="metric-card">
                    <h3>等待中</h3>
                    <div class="metric-value">{pending_count}</div>
                </div>
            </div>
            
            <h2>当前平均性能指标</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>平均准确率</h3>
                    <div class="metric-value">{avg_accuracy:.4f}</div>
                </div>
                <div class="metric-card">
                    <h3>平均F1分数</h3>
                    <div class="metric-value">{avg_f1:.4f}</div>
                </div>
                <div class="metric-card">
                    <h3>平均Matthews相关系数</h3>
                    <div class="metric-value">{avg_matthews:.4f}</div>
                </div>
            </div>
        </div>
        
        <h2>运行中的任务</h2>
        <table>
            <thead>
                <tr>
                    <th>任务类型</th>
                    <th>子任务</th>
                    <th>状态</th>
                    <th>已运行时间(秒)</th>
                    <th>优先级</th>
                    <th>GPU</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加运行中的任务
    current_time = time.time()
    for task_id, task_info in running_tasks.items():
        task_type = task_info.get("task_type", "未知")
        sub_task = task_info.get("sub_task", "未知")
        start_time = task_info.get("start_time", current_time)
        running_time = current_time - start_time
        priority = task_info.get("priority", 0)
        gpu_id = task_info.get("gpu_id", "未知")
        
        html_content += f"""
                <tr class="running">
                    <td>{task_type}</td>
                    <td>{sub_task}</td>
                    <td>运行中</td>
                    <td>{running_time:.2f}</td>
                    <td>{priority}</td>
                    <td>{gpu_id}</td>
                </tr>
"""
    
    if not running_tasks:
        html_content += """
                <tr>
                    <td colspan="6" style="text-align: center;">当前没有运行中的任务</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>等待中的任务</h2>
        <table>
            <thead>
                <tr>
                    <th>任务类型</th>
                    <th>子任务</th>
                    <th>优先级</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加等待中的任务
    for task in pending_tasks:
        task_type = task.get("task_type", "未知")
        sub_task = task.get("sub_task", "未知")
        priority = task.get("priority", 0)
        
        html_content += f"""
                <tr class="pending">
                    <td>{task_type}</td>
                    <td>{sub_task}</td>
                    <td>{priority}</td>
                </tr>
"""
    
    if not pending_tasks:
        html_content += """
                <tr>
                    <td colspan="3" style="text-align: center;">当前没有等待中的任务</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>已完成的任务</h2>
        <table>
            <thead>
                <tr>
                    <th>任务类型</th>
                    <th>子任务</th>
                    <th>状态</th>
                    <th>耗时(秒)</th>
                    <th>准确率</th>
                    <th>F1分数</th>
                    <th>Matthews相关系数</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加已完成的任务，按完成时间倒序排列
    completed_tasks_sorted = sorted(completed_tasks, key=lambda x: x.get("end_time", 0), reverse=True)
    
    for result in completed_tasks_sorted:
        task_type = result.get("task_type", "未知")
        sub_task = result.get("sub_task", "未知")
        status = result.get("status", "未知")
        duration = result.get("duration", 0)
        
        # 设置状态样式
        status_class = "success" if status == "成功" else "failure"
        
        # 获取评估指标
        eval_results = result.get("eval_results", {})
        accuracy = eval_results.get("eval_accuracy", "N/A")
        if accuracy != "N/A":
            accuracy = f"{accuracy:.4f}"
        
        f1 = eval_results.get("eval_f1", "N/A")
        if f1 != "N/A":
            f1 = f"{f1:.4f}"
        
        matthews = eval_results.get("eval_matthews_correlation", "N/A")
        if matthews != "N/A":
            matthews = f"{matthews:.4f}"
        
        html_content += f"""
                <tr class="{status_class}">
                    <td>{task_type}</td>
                    <td>{sub_task}</td>
                    <td>{status}</td>
                    <td>{duration:.2f}</td>
                    <td>{accuracy}</td>
                    <td>{f1}</td>
                    <td>{matthews}</td>
                </tr>
"""
    
    if not completed_tasks:
        html_content += """
                <tr>
                    <td colspan="7" style="text-align: center;">当前没有已完成的任务</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    
    # 保存HTML文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"进度报告已生成: {output_path}")
    
    
    
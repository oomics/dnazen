# import deepspeed
from typing import Any
import argparse
import os
import json
import logging
from datetime import datetime
import numpy as np
import time
import re

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



def generate_parallel_finetune_progress_report(tasks_dir, output_path):
    """
    生成微调任务进度报告
    
    参数:
        tasks_dir (str): 任务输出目录，包含所有子任务的结果
        output_path (str): 报告输出路径
    """
    logger.info(f"从目录 {tasks_dir} 扫描任务结果并生成报告: {output_path}")
    
    # 扫描任务目录，收集所有任务结果
    completed_tasks = []
    running_tasks = {}
    pending_tasks = []
    
    # 遍历任务类型目录
    for task_type in os.listdir(tasks_dir):
        task_type_dir = os.path.join(tasks_dir, task_type)
        if not os.path.isdir(task_type_dir):
            continue
        
        # 遍历子任务目录
        for sub_task in os.listdir(task_type_dir):
            sub_task_dir = os.path.join(task_type_dir, sub_task)
            if not os.path.isdir(sub_task_dir):
                continue
            
            # 检查任务状态
            eval_results_path = os.path.join(sub_task_dir, "eval_results.json")
            task_info_path = os.path.join(sub_task_dir, "task_info.json")
            log_path = os.path.join(sub_task_dir, "train.log")
            
            task_id = f"{task_type}/{sub_task}"
            
            # 如果存在评估结果，说明任务已完成
            if os.path.exists(eval_results_path):
                try:
                    # 读取评估结果
                    with open(eval_results_path, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)
                    
                    # 读取任务信息（如果存在）
                    task_info = {}
                    if os.path.exists(task_info_path):
                        try:
                            with open(task_info_path, "r", encoding="utf-8") as f:
                                task_info = json.load(f)
                        except:
                            pass
                    
                    # 从日志中提取开始和结束时间
                    start_time = None
                    end_time = None
                    duration = 0
                    
                    if os.path.exists(log_path):
                        try:
                            with open(log_path, "r", encoding="utf-8") as f:
                                log_content = f.read()
                                
                                # 提取开始时间
                                start_match = re.search(r"开始时间: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content)
                                if start_match:
                                    start_time_str = start_match.group(1)
                                    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp()
                                
                                # 提取结束时间
                                end_match = re.search(r"结束时间: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content)
                                if end_match:
                                    end_time_str = end_match.group(1)
                                    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S").timestamp()
                                
                                # 计算持续时间
                                if start_time and end_time:
                                    duration = end_time - start_time
                        except:
                            pass
                    
                    # 如果没有从日志中提取到时间信息，使用文件修改时间
                    if not start_time or not end_time:
                        try:
                            # 使用目录创建时间作为开始时间
                            start_time = os.path.getctime(sub_task_dir)
                            # 使用评估结果文件修改时间作为结束时间
                            end_time = os.path.getmtime(eval_results_path)
                            duration = end_time - start_time
                        except:
                            pass
                    
                    # 构建任务结果
                    task_result = {
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "status": "成功",
                        "output_path": sub_task_dir,
                        "eval_results": eval_results,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        **task_info  # 合并任务信息
                    }
                    
                    completed_tasks.append(task_result)
                    
                except Exception as e:
                    logger.error(f"处理任务 {task_id} 结果时出错: {e}")
            
            # 如果存在checkpoint但没有评估结果，可能是运行中或失败的任务
            elif os.path.exists(os.path.join(sub_task_dir, "checkpoint-*")):
                # 检查是否有正在运行的进程
                is_running = False
                
                # 在Linux上可以通过检查进程来确定任务是否在运行
                try:
                    import subprocess
                    result = subprocess.run(f"ps aux | grep '{task_id}' | grep -v grep", shell=True, capture_output=True, text=True)
                    if result.stdout.strip():
                        is_running = True
                except:
                    pass
                
                if is_running:
                    running_tasks[task_id] = {
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "output_path": sub_task_dir
                    }
                else:
                    # 可能是失败的任务
                    completed_tasks.append({
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "status": "失败",
                        "output_path": sub_task_dir,
                        "eval_results": {}
                    })
    
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
        .details-btn {{
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 12px;
            margin: 2px 2px;
            cursor: pointer;
            border-radius: 3px;
        }}
        .metric-details {{
            display: none;
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 14px;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .metric-table th {{
            background-color: #f2f2f2;
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
                    <th>详细指标</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加已完成的任务，按完成时间倒序排列
    completed_tasks_sorted = sorted(completed_tasks, key=lambda x: x.get("end_time", 0), reverse=True)
    
    for i, result in enumerate(completed_tasks_sorted):
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
        
        # 创建详细指标表格
        metrics_table = ""
        if eval_results:
            metrics_table = f"""
            <div id="metrics-{i}" class="metric-details">
                <table class="metric-table">
                    <thead>
                        <tr>
                            <th>指标名称</th>
                            <th>值</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for metric_name, metric_value in sorted(eval_results.items()):
                if isinstance(metric_value, (int, float)):
                    formatted_value = f"{metric_value:.6f}" if isinstance(metric_value, float) else str(metric_value)
                    metrics_table += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{formatted_value}</td>
                        </tr>
                    """
                else:
                    metrics_table += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{str(metric_value)}</td>
                        </tr>
                    """
            
            metrics_table += """
                    </tbody>
                </table>
            </div>
            """
        
        html_content += f"""
                <tr class="{status_class}">
                    <td>{task_type}</td>
                    <td>{sub_task}</td>
                    <td>{status}</td>
                    <td>{duration:.2f}</td>
                    <td>{accuracy}</td>
                    <td>{f1}</td>
                    <td>{matthews}</td>
                    <td>
                        <button class="details-btn" onclick="toggleMetrics('metrics-{i}')">查看详情</button>
                        {metrics_table}
                    </td>
                </tr>
"""
    
    if not completed_tasks:
        html_content += """
                <tr>
                    <td colspan="8" style="text-align: center;">当前没有已完成的任务</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <script>
        function toggleMetrics(id) {
            var metrics = document.getElementById(id);
            if (metrics.style.display === "block") {
                metrics.style.display = "none";
            } else {
                metrics.style.display = "block";
            }
        }
    </script>
</body>
</html>
"""
    
    # 保存HTML文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"进度报告已生成: {output_path}")
    
    
    
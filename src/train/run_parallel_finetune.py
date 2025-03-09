#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行执行多个微调任务的脚本，支持GPU资源管理、任务优先级和错误恢复
"""

import argparse
import os
import json
import logging
import subprocess
import concurrent.futures
import time
import threading
import signal
import sys
import queue
import re
import shutil
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局变量
running_processes = {}  # 进程ID -> 进程对象
stop_event = threading.Event()
gpu_lock = threading.Lock()
gpu_usage = {}  # GPU ID -> 使用状态

def parse_args():
    parser = argparse.ArgumentParser(description="并行执行多个微调任务")
    parser.add_argument("--config", type=str, required=True, help="任务配置文件路径")
    parser.add_argument("--max_workers", type=int, default=4, help="最大并行任务数")
    parser.add_argument("--checkpoint", type=str, required=True, help="预训练模型检查点路径")
    parser.add_argument("--ngram_encoder_dir", type=str, required=True, help="N-gram编码器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--summary_report", type=str, default="summary_report.html", help="汇总报告文件名")
    parser.add_argument("--gpu_ids", type=str, default=None, help="要使用的GPU ID，用逗号分隔，例如'0,1,2'")
    parser.add_argument("--retry_count", type=int, default=1, help="任务失败时的重试次数")
    parser.add_argument("--log_dir", type=str, default=None, help="日志目录，默认为output_dir/logs")
    parser.add_argument("--monitor_interval", type=int, default=60, help="监控间隔（秒）")
    parser.add_argument("--resume", action="store_true", help="从中断处恢复任务")
    return parser.parse_args()

def signal_handler(sig, frame):
    """处理中断信号，优雅地停止所有任务"""
    logger.info("接收到中断信号，正在停止所有任务...")
    stop_event.set()
    
    # 等待所有进程完成
    for task_id, process in list(running_processes.items()):
        if process.poll() is None:  # 如果进程仍在运行
            logger.info(f"正在终止任务 {task_id}...")
            try:
                process.terminate()
                # 给进程一些时间来清理
                time.sleep(2)
                if process.poll() is None:
                    process.kill()  # 如果进程仍在运行，强制终止
                logger.info(f"任务 {task_id} 已终止")
            except Exception as e:
                logger.error(f"终止任务 {task_id} 时出错: {e}")
    
    logger.info("所有任务已停止，正在退出...")
    sys.exit(0)

def get_available_gpus():
    """获取可用的GPU列表及其内存使用情况"""
    try:
        # 使用nvidia-smi获取GPU信息
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        
        gpus = []
        for line in output.strip().split('\n'):
            values = line.split(', ')
            if len(values) == 3:
                index = int(values[0])
                used_memory = int(values[1])
                total_memory = int(values[2])
                free_memory = total_memory - used_memory
                gpus.append({
                    'index': index,
                    'used_memory': used_memory,
                    'total_memory': total_memory,
                    'free_memory': free_memory
                })
        
        return gpus
    except Exception as e:
        logger.warning(f"获取GPU信息失败: {e}")
        return []

def allocate_gpu(gpu_ids=None, memory_threshold=1000):
    """分配一个可用的GPU"""
    with gpu_lock:
        available_gpus = get_available_gpus()
        
        # 过滤指定的GPU
        if gpu_ids is not None:
            available_gpus = [gpu for gpu in available_gpus if gpu['index'] in gpu_ids]
        
        # 按可用内存排序
        available_gpus.sort(key=lambda x: x['free_memory'], reverse=True)
        
        for gpu in available_gpus:
            gpu_id = gpu['index']
            if gpu['free_memory'] > memory_threshold and gpu_id not in gpu_usage:
                gpu_usage[gpu_id] = True
                return gpu_id
        
        return None

def release_gpu(gpu_id):
    """释放GPU资源"""
    with gpu_lock:
        if gpu_id in gpu_usage:
            del gpu_usage[gpu_id]

def run_finetune_task(task_config):
    """
    运行单个微调任务
    
    参数:
        task_config (dict): 任务配置
    
    返回:
        dict: 任务结果
    """
    task_type = task_config["task_type"]
    sub_task = task_config["sub_task"]
    data_path = task_config["data_path"]
    output_path = task_config["output_path"]
    checkpoint = task_config["checkpoint"]
    ngram_encoder_dir = task_config["ngram_encoder_dir"]
    num_train_epochs = task_config.get("num_train_epochs", 5)
    per_device_train_batch_size = task_config.get("per_device_train_batch_size", 8)
    per_device_eval_batch_size = task_config.get("per_device_eval_batch_size", 32)
    learning_rate = task_config.get("learning_rate", 3e-5)
    fp16 = task_config.get("fp16", True)
    retry_count = task_config.get("retry_count", 1)
    gpu_ids = task_config.get("gpu_ids", None)
    priority = task_config.get("priority", 0)  # 优先级，数字越大优先级越高
    
    task_id = f"{task_type}/{sub_task}"
    logger.info(f"准备执行任务: {task_id}, 优先级: {priority}")
    
    # 检查是否已经完成
    if task_config.get("resume", False) and os.path.exists(output_path):
        eval_results_path = os.path.join(output_path, "eval_results.json")
        if os.path.exists(eval_results_path):
            logger.info(f"任务 {task_id} 已完成，跳过")
            try:
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                
                return {
                    "task_type": task_type,
                    "sub_task": sub_task,
                    "status": "成功",
                    "duration": 0,  # 不知道实际耗时
                    "output_path": output_path,
                    "eval_results": eval_results,
                    "skipped": True
                }
            except Exception as e:
                logger.warning(f"读取已完成任务结果失败: {e}，将重新执行任务")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 分配GPU
    gpu_id = allocate_gpu(gpu_ids)
    if gpu_id is None:
        logger.warning(f"无可用GPU，任务 {task_id} 将等待...")
        # 等待一段时间后重试
        time.sleep(30)
        gpu_id = allocate_gpu(gpu_ids)
        if gpu_id is None:
            logger.error(f"仍无可用GPU，任务 {task_id} 将使用默认GPU")
    
    # 设置环境变量
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"任务 {task_id} 分配到GPU {gpu_id}")
    
    # 构建命令
    cmd = [
        "python", "../src/train/run_finetune.py",
        "--data_path", data_path,
        "--checkpoint", checkpoint,
        "--ngram_encoder_dir", ngram_encoder_dir,
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--per_device_eval_batch_size", str(per_device_eval_batch_size),
        "--lr", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--out", output_path
    ]
    
    if fp16:
        cmd.append("--fp16")
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"开始任务: {task_id}")
    logger.info(f"命令: {' '.join(cmd)}")
    
    # 创建日志文件
    log_path = os.path.join(output_path, "task_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"命令: {' '.join(cmd)}\n")
    log_file.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()
    
    # 运行命令
    status = "失败"
    stderr = ""
    stdout = ""
    attempt = 0
    
    while attempt < retry_count and not stop_event.is_set():
        attempt += 1
        if attempt > 1:
            logger.info(f"重试任务 {task_id}，第 {attempt} 次尝试")
            log_file.write(f"\n重试任务，第 {attempt} 次尝试\n")
            log_file.flush()
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # 记录进程
        running_processes[task_id] = process
        
        # 实时获取输出
        stdout_thread = threading.Thread(
            target=log_output_stream, 
            args=(process.stdout, log_file, f"[{task_id}] ")
        )
        stderr_thread = threading.Thread(
            target=log_output_stream, 
            args=(process.stderr, log_file, f"[{task_id}] ERROR: ")
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待进程完成
        return_code = process.wait()
        
        # 等待线程完成
        stdout_thread.join()
        stderr_thread.join()
        
        # 从进程列表中移除
        if task_id in running_processes:
            del running_processes[task_id]
        
        # 检查是否成功
        if return_code == 0:
            status = "成功"
            logger.info(f"任务完成: {task_id}")
            break
        else:
            logger.error(f"任务失败: {task_id}，返回码: {return_code}")
            if attempt < retry_count and not stop_event.is_set():
                logger.info(f"将在5秒后重试...")
                time.sleep(5)
            else:
                logger.error(f"任务 {task_id} 达到最大重试次数，放弃")
    
    # 释放GPU
    if gpu_id is not None:
        release_gpu(gpu_id)
    
    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 完成日志
    log_file.write(f"\n状态: {status}\n")
    log_file.write(f"结束时间: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"耗时: {duration:.2f}秒\n")
    log_file.close()
    
    # 读取评估结果
    eval_results = {}
    eval_results_path = os.path.join(output_path, "eval_results.json")
    if os.path.exists(eval_results_path):
        try:
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
        except Exception as e:
            logger.error(f"读取评估结果失败: {e}")
    
    # 返回任务结果
    return {
        "task_type": task_type,
        "sub_task": sub_task,
        "status": status,
        "duration": duration,
        "output_path": output_path,
        "eval_results": eval_results,
        "priority": priority
    }

def log_output_stream(stream, log_file, prefix=""):
    """实时记录输出流"""
    for line in iter(stream.readline, ""):
        if line:
            log_line = f"{prefix}{line.rstrip()}"
            logger.debug(log_line)
            log_file.write(f"{log_line}\n")
            log_file.flush()

def monitor_tasks(tasks_queue, results_queue, args):
    """监控任务执行情况"""
    last_report_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # 每隔一段时间生成报告
        if current_time - last_report_time > args.monitor_interval:
            # 收集当前结果
            current_results = list(results_queue.queue)
            pending_tasks = list(tasks_queue.queue)
            
            # 生成临时报告
            temp_report_path = os.path.join(args.output_dir, "progress_report.html")
            generate_progress_report(current_results, pending_tasks, running_processes, temp_report_path)
            
            last_report_time = current_time
        
        time.sleep(5)

def generate_progress_report(completed_tasks, pending_tasks, running_tasks, output_path):
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
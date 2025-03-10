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
from collections import defaultdict

import report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局变量
running_processes = {}  # 任务ID -> 进程对象
running_tasks = {}      # 任务ID -> 任务信息
stop_event = threading.Event()
gpu_lock = threading.Lock()
gpu_usage = {}  # GPU ID -> 使用状态

# 全局变量，用于存储各任务的日志
task_logs = defaultdict(list)
task_logs_lock = threading.Lock()
last_summary_time = time.time()
summary_interval = 120  # 2分钟，单位为秒

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="并行执行多个DNA序列微调任务")
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
    parser.add_argument("--memory_threshold", type=int, default=1000, help="GPU内存阈值（MB），低于此值的GPU不会被分配")
    parser.add_argument("--resume", action="store_true", help="从中断处恢复任务")
    parser.add_argument("--clean_output", action="store_true", help="执行任务前清空输出目录")
    parser.add_argument("--gpu_strategy", type=str, default="balanced", 
                      choices=["round_robin", "balanced", "memory"],
                      help="GPU分配策略: round_robin(轮询), balanced(负载均衡), memory(基于显存)")
    parser.add_argument("--summary_interval", type=int, default=120,
                      help="日志汇总间隔(秒)")
    return parser.parse_args()

def signal_handler(sig, frame):
    """处理中断信号，优雅地停止所有任务"""
    logger.info("步骤1: 接收到中断信号，正在优雅地停止所有任务...")
    stop_event.set()
    
    # 等待所有进程完成
    for task_id, process in list(running_processes.items()):
        if process.poll() is None:  # 如果进程仍在运行
            logger.info(f"步骤1.1: 正在终止任务 {task_id}...")
            try:
                process.terminate()
                # 给进程一些时间来清理
                time.sleep(2)
                if process.poll() is None:
                    process.kill()  # 如果进程仍在运行，强制终止
                logger.info(f"步骤1.2: 任务 {task_id} 已终止")
            except Exception as e:
                logger.error(f"步骤1.3: 终止任务 {task_id} 时出错: {e}")
    
    logger.info("步骤1.4: 所有任务已停止，正在退出...")
    sys.exit(0)

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self, gpu_ids, strategy="round_robin"):
        """
        初始化GPU管理器
        
        参数:
            gpu_ids (list): 可用的GPU ID列表
            strategy (str): GPU分配策略，可选值：
                - round_robin: 轮询分配
                - balanced: 负载均衡分配
                - memory: 基于显存使用率分配
        """
        self.gpu_ids = gpu_ids
        self.strategy = strategy
        self.assigned_gpus = {}  # 任务ID -> GPU ID
        self.gpu_tasks = {gpu_id: [] for gpu_id in gpu_ids}  # GPU ID -> 任务列表
        self.gpu_memory_usage = {gpu_id: 0 for gpu_id in gpu_ids}  # GPU ID -> 显存使用率
        self.lock = threading.Lock()
        
        logger.info(f"初始化GPU管理器，可用GPU: {gpu_ids}，策略: {strategy}")
        
        # 启动GPU监控线程
        if strategy == "memory":
            self.monitor_thread = threading.Thread(target=self._monitor_gpu_memory)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def _monitor_gpu_memory(self):
        """监控GPU显存使用情况"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            while True:
                with self.lock:
                    for gpu_id in self.gpu_ids:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            self.gpu_memory_usage[gpu_id] = info.used / info.total
                        except Exception as e:
                            logger.warning(f"获取GPU {gpu_id}显存信息失败: {e}")
                
                time.sleep(5)  # 每5秒更新一次
        except ImportError:
            logger.warning("未安装pynvml，无法监控GPU显存使用情况")
            self.strategy = "balanced"  # 降级为balanced策略
    
    def get_gpu(self, task_id, task_type=None, sub_task=None):
        """
        获取可用的GPU
        
        参数:
            task_id (str): 任务ID
            task_type (str, optional): 任务类型，用于任务特性分析
            sub_task (str, optional): 子任务名称，用于任务特性分析
            
        返回:
            int: 分配的GPU ID
        """
        with self.lock:
            # 如果任务已分配GPU，直接返回
            if task_id in self.assigned_gpus:
                return self.assigned_gpus[task_id]
            
            # 根据策略分配GPU
            if self.strategy == "round_robin":
                # 轮询策略：简单地轮流分配
                gpu_id = self.gpu_ids[len(self.assigned_gpus) % len(self.gpu_ids)]
            
            elif self.strategy == "memory":
                # 显存策略：分配显存使用率最低的GPU
                gpu_id = min(self.gpu_memory_usage, key=self.gpu_memory_usage.get)
            
            else:  # balanced策略
                # 负载均衡策略：分配任务数最少的GPU
                gpu_counts = {gpu_id: len(tasks) for gpu_id, tasks in self.gpu_tasks.items()}
                
                # 考虑任务类型的特性进行更智能的分配
                if task_type:
                    # 尝试将相同类型的任务分配到不同GPU，避免资源竞争
                    for gpu_id in self.gpu_ids:
                        # 检查该GPU上是否已有相同类型的任务
                        has_same_type = any(t.split('/')[0] == task_type for t in self.gpu_tasks[gpu_id])
                        if not has_same_type:
                            # 如果没有相同类型的任务，优先分配到这个GPU
                            gpu_counts[gpu_id] -= 0.5  # 降低权重，增加选择概率
                
                # 选择任务数最少的GPU
                gpu_id = min(gpu_counts, key=gpu_counts.get)
            
            # 记录分配结果
            self.assigned_gpus[task_id] = gpu_id
            self.gpu_tasks[gpu_id].append(task_id)
            
            logger.info(f"任务 {task_id} 分配到GPU {gpu_id}")
            return gpu_id
    
    def release_gpu(self, task_id):
        """
        释放任务占用的GPU
        
        参数:
            task_id (str): 任务ID
        """
        with self.lock:
            if task_id in self.assigned_gpus:
                gpu_id = self.assigned_gpus[task_id]
                self.gpu_tasks[gpu_id].remove(task_id)
                del self.assigned_gpus[task_id]
                logger.info(f"任务 {task_id} 释放GPU {gpu_id}")

def log_output_stream(stream, log_file, prefix="", task_id="unknown"):
    """实时记录输出流"""
    for line in iter(stream.readline, ""):
        if line:
            # 检查是否是正常的日志输出（而非真正的错误）
            is_normal_log = "INFO" in line or "步骤" in line or ":" in line and not any(err in line.lower() for err in ["error:", "exception:", "traceback:"])
            
            # 根据内容选择合适的日志级别和前缀
            if "ERROR:" in prefix and is_normal_log:
                # 对于正常日志输出，使用原始前缀但不添加ERROR标记
                clean_prefix = prefix.replace("ERROR: ", "")
                log_line = f"{clean_prefix}{line.rstrip()}"
            else:
                # 对于真正的错误或其他输出
                log_line = f"{prefix}{line.rstrip()}"
            
            # 写入日志文件
            log_file.write(f"{log_line}\n")
            log_file.flush()
            
            # 检查是否包含进度信息
            progress_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
            if progress_match:
                current_epoch = int(progress_match.group(1))
                total_epochs = int(progress_match.group(2))
                progress_percentage = (current_epoch / total_epochs) * 100
                
                # 更新任务进度信息
                with task_logs_lock:
                    if task_id in running_tasks:
                        running_tasks[task_id]["progress"] = progress_percentage
                        running_tasks[task_id]["current_epoch"] = current_epoch
                        running_tasks[task_id]["total_epochs"] = total_epochs
            
            # 将重要日志添加到任务日志缓存中
            if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower() or "步骤" in line:
                with task_logs_lock:
                    # 限制每个任务存储的日志数量，避免内存占用过大
                    if len(task_logs[task_id]) >= 100:
                        task_logs[task_id].pop(0)  # 移除最旧的日志
                    task_logs[task_id].append((time.time(), log_line))
                    
                    # 对于错误信息，立即输出到控制台
                    if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                        logger.error(log_line)

def summarize_logs():
    """定期汇总并输出任务日志"""
    global last_summary_time
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # 每隔指定时间输出一次汇总日志
        if current_time - last_summary_time >= summary_interval:
            with task_logs_lock:
                logger.info("\n" + "="*80)
                logger.info(f"任务日志汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("="*80)
                
                # 输出运行中任务的状态
                if running_tasks:
                    logger.info("\n>> 运行中的任务状态:")
                    logger.info(f"{'任务ID':<20}{'已运行时间':<15}{'进度':<15}{'GPU':<10}")
                    logger.info("-"*60)
                    
                    for task_id, task_info in sorted(running_tasks.items()):
                        # 计算已运行时间
                        start_time = task_info.get("start_time", current_time)
                        runtime = current_time - start_time
                        hours, remainder = divmod(runtime, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                        
                        # 获取进度信息
                        progress = task_info.get("progress", 0)
                        progress_str = f"{progress:.1f}%"
                        
                        # 获取当前轮次信息
                        current_epoch = task_info.get("current_epoch", 0)
                        total_epochs = task_info.get("total_epochs", 0)
                        if total_epochs > 0:
                            progress_str = f"{progress:.1f}% ({current_epoch}/{total_epochs}轮)"
                        
                        # 获取GPU信息
                        gpu_id = task_info.get("gpu_id", "未知")
                        
                        logger.info(f"{task_id:<20}{runtime_str:<15}{progress_str:<15}{gpu_id:<10}")
                
                # 输出任务日志
                if task_logs:
                    # 按任务ID排序输出
                    for task_id in sorted(task_logs.keys()):
                        # 获取该任务的最新日志（最多10条）
                        recent_logs = task_logs[task_id][-10:]
                        if recent_logs:
                            logger.info(f"\n>> 任务 {task_id} 最新日志:")
                            for _, log_line in recent_logs:
                                logger.info(f"  {log_line}")
                    
                    # 清空已输出的日志
                    task_logs.clear()
                
                logger.info("="*80 + "\n")
            
            last_summary_time = current_time
        
        # 睡眠一段时间，避免频繁检查
        time.sleep(10)

def run_finetune_task(task_config, gpu_manager, args):
    """
    运行单个微调任务
    
    参数:
        task_config (dict): 任务配置
        gpu_manager (GPUManager): GPU管理器
        args (Namespace): 命令行参数
    
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
    retry_count = task_config.get("retry_count", args.retry_count)
    priority = task_config.get("priority", 0)  # 优先级，数字越大优先级越高
    
    task_id = f"{task_type}/{sub_task}"
    logger.info(f"[任务 {task_id}] 步骤1: 准备执行任务，优先级: {priority}")
    
    # 检查是否已经完成
    if args.resume and os.path.exists(output_path):
        eval_results_path = os.path.join(output_path, "eval_results.json")
        if os.path.exists(eval_results_path):
            logger.info(f"[任务 {task_id}] 步骤1.1: 任务已完成，跳过")
            try:
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                
                return {
                    "task_type": task_type,
                    "sub_task": sub_task,
                    "status": "成功",
                    "duration": 0,
                    "output_path": output_path,
                    "eval_results": eval_results,
                    "priority": priority,
                    "end_time": time.time()
                }
            except Exception as e:
                logger.warning(f"[任务 {task_id}] 步骤1.2: 读取评估结果失败: {e}，将重新运行任务")
    
    # 如果设置了清空输出目录选项，则删除已存在的输出目录
    if args.clean_output and os.path.exists(output_path):
        logger.info(f"[任务 {task_id}] 步骤1.2: 清空输出目录: {output_path}")
        try:
            shutil.rmtree(output_path)
            logger.info(f"[任务 {task_id}] 步骤1.2: 成功删除输出目录")
        except Exception as e:
            logger.warning(f"[任务 {task_id}] 步骤1.2: 删除输出目录失败: {e}")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建日志目录
    log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件路径
    log_file_path = os.path.join(log_dir, f"{task_type}_{sub_task}.log")
    logger.info(f"[任务 {task_id}] 步骤1.3: 任务日志将保存到: {log_file_path}")
    
    # 打开日志文件
    log_file = open(log_file_path, "w", encoding="utf-8")
    log_file.write(f"任务: {task_id}\n")
    log_file.write(f"数据路径: {data_path}\n")
    log_file.write(f"输出路径: {output_path}\n")
    log_file.write(f"优先级: {priority}\n")
    log_file.write(f"训练轮数: {num_train_epochs}\n")
    log_file.write(f"批处理大小: {per_device_train_batch_size}\n")
    log_file.write(f"学习率: {learning_rate}\n")
    log_file.write(f"FP16: {fp16}\n\n")
    
    # 记录开始时间
    start_time = time.time()
    log_file.write(f"开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 分配GPU
    gpu_id = gpu_manager.get_gpu(task_id, task_type, sub_task)
    if gpu_id is None:
        logger.error(f"[任务 {task_id}] 步骤2: 无法分配GPU，任务将等待")
        log_file.write("无法分配GPU，任务将等待\n")
        log_file.close()
        
        # 等待一段时间后重试
        time.sleep(30)
        return None
    
    logger.info(f"[任务 {task_id}] 步骤2: 分配GPU {gpu_id}")
    log_file.write(f"分配GPU: {gpu_id}\n\n")
    
    # 更新运行中的任务信息
    task_info = {
        "task_type": task_type,
        "sub_task": sub_task,
        "start_time": start_time,
        "priority": priority,
        "gpu_id": gpu_id,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": num_train_epochs
    }
    running_tasks[task_id] = task_info
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 状态
    status = "失败"
    failure_reason = "未知错误"
    
    # 尝试运行任务
    for attempt in range(1, retry_count + 1):
        if stop_event.is_set():
            logger.info(f"[任务 {task_id}] 步骤3: 收到停止信号，取消任务")
            break
        
        logger.info(f"[任务 {task_id}] 步骤3: 开始执行，尝试 {attempt}/{retry_count}")
        log_file.write(f"尝试 {attempt}/{retry_count}\n")
        
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
            "--out", output_path  # 确保提供输出目录参数
        ]
        
        if fp16:
            cmd.append("--fp16")
        
        logger.info(f"[任务 {task_id}] 步骤3.1: 执行命令: {' '.join(cmd)}")
        log_file.write(f"执行命令: {' '.join(cmd)}\n\n")
        log_file.flush()
        
        # 创建任务特定的日志前缀
        task_prefix = f"[{task_type}/{sub_task}] "
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            bufsize=1  # 行缓冲，确保实时输出日志
        )
        
        # 记录进程
        running_processes[task_id] = process
        
        # 实时获取输出并添加任务前缀
        stdout_thread = threading.Thread(
            target=log_output_stream, 
            args=(process.stdout, log_file, task_prefix, task_id)
        )
        stderr_thread = threading.Thread(
            target=log_output_stream, 
            args=(process.stderr, log_file, f"{task_prefix}ERROR: ", task_id)
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
        
        # 从运行中的任务列表中移除
        if task_id in running_tasks:
            del running_tasks[task_id]
        
        # 检查是否成功
        if return_code == 0:
            status = "成功"
            logger.info(f"[任务 {task_id}] 步骤3.2: 任务完成")
            break
        else:
            # 收集失败原因
            error_logs = []
            try:
                # 读取日志文件中的最后20行错误信息
                with open(log_file_path, "r", encoding="utf-8") as f:
                    log_lines = f.readlines()
                    # 查找包含Error或Exception的行
                    for line in reversed(log_lines):
                        if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
                            error_logs.append(line.strip())
                        if len(error_logs) >= 5:  # 最多收集5条错误信息
                            break
            except Exception as e:
                error_logs.append(f"无法读取错误日志: {e}")
            
            # 构建详细的错误信息
            error_detail = "\n".join(reversed(error_logs)) if error_logs else "未知错误"
            failure_reason = error_detail
            
            logger.error(f"[任务 {task_id}] 步骤3.3: 任务失败，返回码: {return_code}")
            logger.error(f"[任务 {task_id}] 步骤3.4: 失败原因:\n{error_detail}")
            
            # 记录到日志文件
            log_file.write(f"\n失败原因:\n{error_detail}\n\n")
            log_file.flush()
            
            if attempt < retry_count and not stop_event.is_set():
                logger.info(f"[任务 {task_id}] 步骤3.5: 将在5秒后重试...")
                time.sleep(5)
            else:
                logger.error(f"[任务 {task_id}] 步骤3.6: 达到最大重试次数，放弃")
    
    # 释放GPU
    if gpu_id is not None:
        gpu_manager.release_gpu(task_id)
        logger.info(f"[任务 {task_id}] 步骤4: 释放GPU {gpu_id}")
    
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
            logger.info(f"[任务 {task_id}] 步骤5: 读取评估结果: {eval_results}")
        except Exception as e:
            logger.error(f"[任务 {task_id}] 步骤5: 读取评估结果失败: {e}")
    
    # 返回任务结果
    result = {
        "task_type": task_type,
        "sub_task": sub_task,
        "status": status,
        "duration": duration,
        "output_path": output_path,
        "eval_results": eval_results,
        "priority": priority,
        "end_time": end_time
    }
    
    if status != "成功":
        result["failure_reason"] = failure_reason
    
    logger.info(f"[任务 {task_id}] 步骤6: 任务结束，状态: {status}，耗时: {duration:.2f}秒")
    return result

def monitor_tasks(tasks_queue, args, gpu_manager):
    """监控任务执行情况"""
    logger.info("步骤4: 启动任务监控线程")
    last_report_time = time.time()
    
    while not stop_event.is_set():
        current_time = time.time()
        
        # 每隔一段时间生成报告
        if current_time - last_report_time > args.monitor_interval:
            # 收集当前结果
            pending_tasks = list(tasks_queue.queue)
            
            try:
                # 生成临时报告
                temp_report_path = os.path.join(args.output_dir, "progress_report.html")
                
                # 直接从输出目录生成报告
                report.generate_parallel_finetune_progress_report(
                    args.output_dir, temp_report_path
                )
                
                # 输出当前状态
                logger.info(f"步骤4.1: 已生成进度报告: {temp_report_path}")
                
                # 检查GPU使用情况
                available_gpus = gpu_manager.get_available_gpus()
                gpu_info = ", ".join([f"GPU {gpu['index']}: {gpu['free_memory']}MB可用" for gpu in available_gpus])
                logger.info(f"步骤4.2: GPU使用情况 - {gpu_info}")
            except Exception as e:
                logger.error(f"步骤4.3: 生成进度报告失败: {e}")
            
            last_report_time = current_time
        
        time.sleep(5)

def main():
    """主函数"""
    # 步骤1: 解析命令行参数
    args = parse_args()
    logger.info("步骤1: 解析命令行参数完成")
    
    # 步骤2: 注册信号处理器，用于优雅地处理中断
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("步骤2: 注册信号处理器完成")
    
    # 步骤3: 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果设置了清空输出目录选项，则清空整个输出目录
    if args.clean_output and not args.resume:
        logger.info(f"步骤3.0: 清空输出目录: {args.output_dir}")
        try:
            # 保留日志目录，删除其他内容
            for item in os.listdir(args.output_dir):
                item_path = os.path.join(args.output_dir, item)
                if item != "logs" and os.path.exists(item_path):
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            logger.info("步骤3.0: 成功清空输出目录")
        except Exception as e:
            logger.warning(f"步骤3.0: 清空输出目录失败: {e}")
    
    # 设置日志目录
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    logger.info(f"步骤3: 创建输出目录: {args.output_dir}")
    logger.info(f"步骤3.1: 创建日志目录: {args.log_dir}")
    
    # 步骤4: 解析GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
            logger.info(f"步骤4: 将使用指定的GPU: {gpu_ids}")
        except ValueError:
            logger.error("步骤4: GPU IDs格式错误，应为逗号分隔的整数")
            sys.exit(1)
    
    # 步骤5: 加载任务配置
    logger.info(f"步骤5: 加载任务配置: {args.config}")
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"步骤5: 加载任务配置失败: {e}")
        sys.exit(1)
    
    # 步骤6: 准备任务列表
    tasks = []
    
    # 检查配置文件格式
    if isinstance(config, dict):
        # 检查是否有特殊的数据目录配置
        data_base_dir = ""
        if "data_base_dir" in config:
            data_base_dir = config.get("data_base_dir", "")
            logger.info(f"步骤6.0: 找到基础数据目录: {data_base_dir}")
        
        # 检查是否有任务列表
        if "tasks" in config and isinstance(config["tasks"], list):
            logger.info("步骤6.0: 找到任务列表配置")
            task_list = config["tasks"]
            
            for task_config in task_list:
                if not isinstance(task_config, dict):
                    logger.warning(f"步骤6.1: 跳过无效的任务配置: {task_config}")
                    continue
                
                task_type = task_config.get("task_type")
                if not task_type:
                    logger.warning(f"步骤6.1: 跳过缺少任务类型的配置: {task_config}")
                    continue
                
                logger.info(f"步骤6.2: 处理任务类型: {task_type}")
                
                # 获取子任务列表
                sub_tasks = task_config.get("sub_tasks", [])
                if not sub_tasks:
                    logger.warning(f"步骤6.3: 任务类型 {task_type} 没有子任务")
                    continue
                
                # 获取任务参数
                data_dir = os.path.join(data_base_dir, task_config.get("data_dir", ""))
                priority = task_config.get("priority", 0)
                num_train_epochs = task_config.get("num_train_epochs", 5)
                per_device_train_batch_size = task_config.get("per_device_train_batch_size", 8)
                per_device_eval_batch_size = task_config.get("per_device_eval_batch_size", 8)
                gradient_accumulation_steps = task_config.get("gradient_accumulation_steps", 1)
                learning_rate = task_config.get("learning_rate", 5e-5)
                fp16 = task_config.get("fp16", False)
                
                # 处理每个子任务
                for sub_task in sub_tasks:
                    logger.info(f"步骤6.4: 添加子任务: {task_type}/{sub_task}")
                    
                    # 构建数据路径
                    data_path = os.path.join(data_dir, sub_task)
                    
                    # 构建输出路径
                    output_path = os.path.join(args.output_dir, task_type, sub_task)
                    
                    # 添加任务
                    tasks.append({
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "data_path": data_path,
                        "output_path": output_path,
                        "checkpoint": args.checkpoint,
                        "ngram_encoder_dir": args.ngram_encoder_dir,
                        "num_train_epochs": num_train_epochs,
                        "per_device_train_batch_size": per_device_train_batch_size,
                        "per_device_eval_batch_size": per_device_eval_batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "learning_rate": learning_rate,
                        "fp16": fp16,
                        "priority": priority
                    })
        else:
            # 原始格式: {"task_type": {"sub_tasks": [...], ...}, ...}
            for task_type, task_config in config.items():
                # 跳过特殊键
                if task_type in ["data_base_dir", "tasks"]:
                    continue
                
                logger.info(f"步骤6: 处理任务类型: {task_type}")
                
                # 获取子任务列表
                if isinstance(task_config, dict):
                    sub_tasks = task_config.get("sub_tasks", [])
                    priority = task_config.get("priority", 0)
                    data_dir = os.path.join(data_base_dir, task_config.get("data_dir", ""))
                    num_train_epochs = task_config.get("num_train_epochs", 5)
                    per_device_train_batch_size = task_config.get("per_device_train_batch_size", 8)
                    per_device_eval_batch_size = task_config.get("per_device_eval_batch_size", 8)
                    gradient_accumulation_steps = task_config.get("gradient_accumulation_steps", 1)
                    learning_rate = task_config.get("learning_rate", 5e-5)
                    fp16 = task_config.get("fp16", False)
                else:
                    # 如果task_config是字符串或其他类型，假设它是子任务列表
                    logger.warning(f"步骤6.1: 任务类型 {task_type} 的配置不是字典，尝试作为子任务列表处理")
                    if isinstance(task_config, list):
                        sub_tasks = task_config
                    elif isinstance(task_config, str):
                        sub_tasks = [task_config]
                    else:
                        logger.warning(f"步骤6.1: 无法解析任务类型 {task_type} 的配置，跳过")
                        continue
                    
                    # 使用默认值
                    priority = 0
                    data_dir = data_base_dir
                    num_train_epochs = 5
                    per_device_train_batch_size = 8
                    per_device_eval_batch_size = 8
                    gradient_accumulation_steps = 1
                    learning_rate = 5e-5
                    fp16 = False
                
                if not sub_tasks:
                    logger.warning(f"步骤6.2: 任务类型 {task_type} 没有子任务")
                    continue
                
                # 处理每个子任务
                for sub_task in sub_tasks:
                    logger.info(f"步骤6.3: 添加子任务: {task_type}/{sub_task}")
                    
                    # 构建数据路径
                    data_path = os.path.join(data_dir, sub_task)
                    
                    # 构建输出路径
                    output_path = os.path.join(args.output_dir, task_type, sub_task)
                    
                    # 添加任务
                    tasks.append({
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "data_path": data_path,
                        "output_path": output_path,
                        "checkpoint": args.checkpoint,
                        "ngram_encoder_dir": args.ngram_encoder_dir,
                        "num_train_epochs": num_train_epochs,
                        "per_device_train_batch_size": per_device_train_batch_size,
                        "per_device_eval_batch_size": per_device_eval_batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "learning_rate": learning_rate,
                        "fp16": fp16,
                        "priority": priority
                    })
    elif isinstance(config, list):
        # 替代格式: [{"task_type": "...", "sub_task": "...", ...}, ...]
        logger.info("步骤6: 配置文件为任务列表格式")
        for task_item in config:
            if not isinstance(task_item, dict):
                logger.warning(f"步骤6.1: 跳过无效的任务项: {task_item}")
                continue
            
            task_type = task_item.get("task_type")
            sub_task = task_item.get("sub_task")
            
            if not task_type or not sub_task:
                logger.warning(f"步骤6.2: 跳过缺少必要字段的任务项: {task_item}")
                continue
            
            logger.info(f"步骤6.3: 添加任务: {task_type}/{sub_task}")
            
            # 构建数据路径
            data_path = task_item.get("data_path", os.path.join(task_item.get("data_dir", ""), sub_task))
            
            # 构建输出路径
            output_path = task_item.get("output_path", os.path.join(args.output_dir, task_type, sub_task))
            
            # 添加任务
            tasks.append({
                "task_type": task_type,
                "sub_task": sub_task,
                "data_path": data_path,
                "output_path": output_path,
                "checkpoint": args.checkpoint,
                "ngram_encoder_dir": args.ngram_encoder_dir,
                "num_train_epochs": task_item.get("num_train_epochs", 5),
                "per_device_train_batch_size": task_item.get("per_device_train_batch_size", 8),
                "per_device_eval_batch_size": task_item.get("per_device_eval_batch_size", 8),
                "gradient_accumulation_steps": task_item.get("gradient_accumulation_steps", 1),
                "learning_rate": task_item.get("learning_rate", 5e-5),
                "fp16": task_item.get("fp16", False),
                "priority": task_item.get("priority", 0)
            })
    else:
        logger.error(f"步骤6: 无法识别的配置文件格式: {type(config)}")
        sys.exit(1)
    
    # 如果没有任务，退出
    if not tasks:
        logger.error("步骤6.4: 没有找到任务，退出")
        sys.exit(1)
    
    logger.info(f"步骤6.5: 共找到 {len(tasks)} 个任务")
    
    # 步骤7: 按优先级排序任务
    tasks.sort(key=lambda x: x["priority"], reverse=True)
    logger.info("步骤7: 任务已按优先级排序")
    
    # 步骤8: 创建任务队列和结果队列
    tasks_queue = queue.Queue()
    completed_tasks = []
    
    # 将任务添加到队列
    for task in tasks:
        tasks_queue.put(task)
    
    logger.info(f"步骤8: 创建任务队列，包含 {tasks_queue.qsize()} 个任务")
    
    # 创建GPU管理器
    gpu_manager = GPUManager(gpu_ids, strategy=args.gpu_strategy)
    
    # 设置日志汇总间隔
    global summary_interval
    summary_interval = args.summary_interval
    
    # 步骤9: 启动监控线程
    monitor_thread = threading.Thread(
        target=monitor_tasks,
        args=(tasks_queue, args, gpu_manager)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.info("步骤9: 启动监控线程")
    
    # 启动日志汇总线程
    summary_thread = threading.Thread(target=summarize_logs)
    summary_thread.daemon = True
    summary_thread.start()
    logger.info("步骤9.1: 启动日志汇总线程，每2分钟输出一次任务日志汇总")
    
    # 步骤10: 使用线程池执行任务
    logger.info(f"步骤10: 开始执行任务，最大并行数: {args.max_workers}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交任务
        futures = []
        while not tasks_queue.empty() and not stop_event.is_set():
            try:
                # 获取任务
                task = tasks_queue.get()
                
                # 提交任务
                future = executor.submit(run_finetune_task, task, gpu_manager, args)
                futures.append(future)
                
                # 等待一小段时间，避免同时启动多个任务
                time.sleep(1)
            except Exception as e:
                logger.error(f"步骤10.1: 提交任务时出错: {e}")
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            if stop_event.is_set():
                break
            
            try:
                result = future.result()
                if result:  # 如果任务成功完成
                    completed_tasks.append(result)
                    logger.info(result)
                    logger.info(f"步骤10.2: 任务完成: {result['task_type']}/{result['sub_task']}, 状态: {result['status']}")
            except Exception as e:
                logger.error(f"步骤10.3: 任务执行出错: {e}")
    logger.info(f"步骤10.3: 所有任务已完成，共 {len(completed_tasks)} 个结果")
    # 如果收到停止信号，提前退出
    if stop_event.is_set():
        logger.info("步骤11: 收到停止信号，提前退出")
        sys.exit(0)
    
    # 步骤11: 收集所有结果
    logger.info(f"步骤11: 所有任务已完成，共 {len(completed_tasks)} 个结果")
    
    # 步骤12: 生成汇总报告
    summary_report_path = os.path.join(args.output_dir, args.summary_report)
    try:
        # 直接传递输出目录，让报告函数自己扫描任务结果
        report.generate_parallel_finetune_progress_report(
            args.output_dir, summary_report_path
        )
        logger.info(f"步骤12: 生成汇总报告: {summary_report_path}")
    except Exception as e:
        logger.error(f"步骤12: 生成汇总报告失败: {e}")
    
    # 步骤13: 完成
    logger.info("步骤13: 所有任务已完成")

if __name__ == "__main__":
    main()

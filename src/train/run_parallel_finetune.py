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

import report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局变量
running_processes = {}  # 任务ID -> 进程对象
running_tasks = {}      # 任务ID -> 任务信息
stop_event = threading.Event()
gpu_lock = threading.Lock()
gpu_usage = {}  # GPU ID -> 使用状态

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
    def __init__(self, gpu_ids=None, memory_threshold=1000):
        """
        初始化GPU管理器
        
        参数:
            gpu_ids (list): 要使用的GPU ID列表，如果为None则使用所有可用GPU
            memory_threshold (int): GPU内存阈值（MB），低于此值的GPU不会被分配
        """
        self.gpu_ids = gpu_ids
        self.memory_threshold = memory_threshold
        self.lock = threading.Lock()
        self.usage = {}  # GPU ID -> 使用状态
        
        logger.info(f"步骤2: 初始化GPU管理器，指定GPU: {gpu_ids if gpu_ids else '所有可用'}, 内存阈值: {memory_threshold}MB")
    
    def get_available_gpus(self):
        """获取可用的GPU列表及其内存使用情况"""
        try:
            # 使用nvidia-smi获取GPU信息
            logger.debug("步骤2.1: 获取可用GPU信息...")
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
            
            logger.debug(f"步骤2.2: 找到 {len(gpus)} 个GPU设备")
            return gpus
        except Exception as e:
            logger.warning(f"步骤2.3: 获取GPU信息失败: {e}")
            return []
    
    def allocate(self):
        """分配一个可用的GPU"""
        with self.lock:
            logger.debug("步骤2.4: 分配GPU资源...")
            available_gpus = self.get_available_gpus()
            
            # 过滤指定的GPU
            if self.gpu_ids is not None:
                logger.debug(f"步骤2.5: 从指定的GPU IDs中选择: {self.gpu_ids}")
                available_gpus = [gpu for gpu in available_gpus if gpu['index'] in self.gpu_ids]
            
            # 按可用内存排序
            available_gpus.sort(key=lambda x: x['free_memory'], reverse=True)
            
            for gpu in available_gpus:
                gpu_id = gpu['index']
                if gpu['free_memory'] > self.memory_threshold and gpu_id not in self.usage:
                    self.usage[gpu_id] = True
                    logger.info(f"步骤2.6: 分配GPU {gpu_id}，可用内存: {gpu['free_memory']}MB")
                    return gpu_id
            
            logger.warning("步骤2.7: 没有找到可用的GPU")
            return None
    
    def release(self, gpu_id):
        """释放GPU资源"""
        with self.lock:
            if gpu_id in self.usage:
                logger.info(f"步骤2.8: 释放GPU {gpu_id}")
                del self.usage[gpu_id]
                return True
            return False

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
    gpu_id = gpu_manager.allocate()
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
        "gpu_id": gpu_id
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
        gpu_manager.release(gpu_id)
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

def log_output_stream(stream, log_file, prefix=""):
    """实时记录输出流"""
    for line in iter(stream.readline, ""):
        if line:
            log_line = f"{prefix}{line.rstrip()}"
            logger.debug(log_line)
            log_file.write(f"{log_line}\n")
            log_file.flush()

def monitor_tasks(tasks_queue, completed_tasks, args, gpu_manager):
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
                report.generate_parallel_finetune_progress_report(completed_tasks, pending_tasks, running_tasks, temp_report_path)
                
                # 输出当前状态
                logger.info(f"步骤4.1: 当前状态 - 已完成: {len(completed_tasks)}, 运行中: {len(running_tasks)}, 等待中: {len(pending_tasks)}")
                
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
    gpu_manager = GPUManager(gpu_ids)
    
    # 步骤9: 启动监控线程
    monitor_thread = threading.Thread(
        target=monitor_tasks,
        args=(tasks_queue, completed_tasks, args, gpu_manager)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.info("步骤9: 启动监控线程")
    
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
                    logger.info(f"步骤10.2: 任务完成: {result['task_type']}/{result['sub_task']}, 状态: {result['status']}")
            except Exception as e:
                logger.error(f"步骤10.3: 任务执行出错: {e}")
    
    # 如果收到停止信号，提前退出
    if stop_event.is_set():
        logger.info("步骤11: 收到停止信号，提前退出")
        sys.exit(0)
    
    # 步骤11: 收集所有结果
    logger.info(f"步骤11: 所有任务已完成，共 {len(completed_tasks)} 个结果")
    
    # 步骤12: 生成汇总报告
    summary_report_path = os.path.join(args.output_dir, args.summary_report)
    report.generate_parallel_finetune_summary_report(completed_tasks, summary_report_path)
    logger.info(f"步骤12: 生成汇总报告: {summary_report_path}")
    
    # 步骤13: 完成
    logger.info("步骤13: 所有任务已完成")

if __name__ == "__main__":
    main()

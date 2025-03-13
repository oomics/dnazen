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
import csv

# import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import pandas as pd
#import transformers
#import torch
#from transformers import (
#    AutoTokenizer,
#    PreTrainedTokenizer,
#)
#from transformers.models.bert.configuration_bert import BertConfig

#from dnazen.model.bert_models import BertForSequenceClassification
#from dnazen.data.labeled_dataset import LabeledDataset
#from dnazen.ngram import NgramEncoder



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



data_in_parpare_dict = [
      {
        "DATA": "EMP_H3",
        "Task": "Epigenetic Marks Prediction-H3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 78.27,
        "MCC_USE_GUE_paper": 80.17,
        "DNABERT准确率复现": 0.8824,
        "DNABERT2 MCC复现": 0.7661,
        "复现偏差": -0.016599999999999965
      },
      {
        "DATA": "EMP_H3K14ac",
        "Task": "Epigenetic Marks Prediction-H3K14ac",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 52.57,
        "MCC_USE_GUE_paper": 57.42,
        "DNABERT准确率复现": 0.7628,
        "DNABERT2 MCC复现": 0.5233,
        "复现偏差": -0.0024000000000000197
      },
      {
        "DATA": "EMP_H3K36me3",
        "Task": "Epigenetic Marks Prediction-H3K36me3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 56.88,
        "MCC_USE_GUE_paper": 61.9,
        "DNABERT准确率复现": 0.7907,
        "DNABERT2 MCC复现": 0.5742,
        "复现偏差": 0.005399999999999992
      },
      {
        "DATA": "EMP_H3K4me1",
        "Task": "Epigenetic Marks Prediction-H3K4me1",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 50.52,
        "MCC_USE_GUE_paper": 53.0,
        "DNABERT准确率复现": 0.7516,
        "DNABERT2 MCC复现": 0.4965,
        "复现偏差": -0.008700000000000046
      },
      {
        "DATA": "EMP_H3K4me2",
        "Task": "Epigenetic Marks Prediction-H3K4me2",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 31.13,
        "MCC_USE_GUE_paper": 39.89,
        "DNABERT准确率复现": 0.6432,
        "DNABERT2 MCC复现": 0.295,
        "复现偏差": -0.01629999999999999
      },
      {
        "DATA": "EMP_H3K4me3",
        "Task": "Epigenetic Marks Prediction-H3K4me3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 36.27,
        "MCC_USE_GUE_paper": 41.2,
        "DNABERT准确率复现": 0.6766,
        "DNABERT2 MCC复现": 0.354,
        "复现偏差": -0.008700000000000046
      },
      {
        "DATA": "EMP_H3K79me3",
        "Task": "Epigenetic Marks Prediction-H3K79me3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 67.39,
        "MCC_USE_GUE_paper": 65.46,
        "DNABERT准确率复现": 0.8266,
        "DNABERT2 MCC复现": 0.6522,
        "复现偏差": -0.021700000000000018
      },
      {
        "DATA": "EMP_H3K9ac",
        "Task": "Epigenetic Marks Prediction-H3K9ac",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 55.63,
        "MCC_USE_GUE_paper": 57.07,
        "DNABERT准确率复现": 0.7769,
        "DNABERT2 MCC复现": 0.5562,
        "复现偏差": -9.999999999998011e-05
      },
      {
        "DATA": "EMP_H4",
        "Task": "Epigenetic Marks Prediction-H4",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 80.71,
        "MCC_USE_GUE_paper": 81.86,
        "DNABERT准确率复现": 0.898,
        "DNABERT2 MCC复现": 0.7972,
        "复现偏差": -0.009899999999999949
      },
      {
        "DATA": "EMP_H4ac",
        "Task": "Epigenetic Marks Prediction-H4ac",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 50.43,
        "MCC_USE_GUE_paper": 50.35,
        "DNABERT准确率复现": 0.7387,
        "DNABERT2 MCC复现": 0.4744,
        "复现偏差": -0.02990000000000002
      },
      {
        "DATA": "prom_prom_300_all",
        "Task": "Promoter Detection-all",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 86.77,
        "MCC_USE_GUE_paper": 88.31,
        "DNABERT准确率复现": 0.9368,
        "DNABERT2 MCC复现": 0.874,
        "复现偏差": 0.006300000000000096
      },
      {
        "DATA": "prom_prom_300_notata",
        "Task": "Promoter Detection-notata",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 94.27,
        "MCC_USE_GUE_paper": 94.34,
        "DNABERT准确率复现": 0.9708,
        "DNABERT2 MCC复现": 0.9418,
        "复现偏差": -0.0009000000000000341
      },
      {
        "DATA": "prom_prom_300_tata",
        "Task": "Promoter Detection-tata",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 71.59,
        "MCC_USE_GUE_paper": 68.79,
        "DNABERT准确率复现": 0.7749,
        "DNABERT2 MCC复现": 0.5661,
        "复现偏差": -0.14979999999999996
      },
      {
        "DATA": "tf_0",
        "Task": "Transcription Factor Prediction (Human)-0",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 71.99,
        "MCC_USE_GUE_paper": 69.12,
        "DNABERT准确率复现": 0.848,
        "DNABERT2 MCC复现": 0.698,
        "复现偏差": -0.02189999999999998
      },
      {
        "DATA": "tf_1",
        "Task": "Transcription Factor Prediction (Human)-1",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 76.06,
        "MCC_USE_GUE_paper": 71.87,
        "DNABERT准确率复现": 0.858,
        "DNABERT2 MCC复现": 0.7202,
        "复现偏差": -0.04040000000000006
      },
      {
        "DATA": "tf_2",
        "Task": "Transcription Factor Prediction (Human)-2",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 66.52,
        "MCC_USE_GUE_paper": 62.96,
        "DNABERT准确率复现": 0.825,
        "DNABERT2 MCC复现": 0.6516,
        "复现偏差": -0.013599999999999994
      },
      {
        "DATA": "tf_3",
        "Task": "Transcription Factor Prediction (Human)-3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 58.54,
        "MCC_USE_GUE_paper": 55.35,
        "DNABERT准确率复现": 0.804,
        "DNABERT2 MCC复现": 0.6094,
        "复现偏差": 0.024000000000000056
      },
      {
        "DATA": "tf_4",
        "Task": "Transcription Factor Prediction (Human)-4",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 77.43,
        "MCC_USE_GUE_paper": 74.94,
        "DNABERT准确率复现": 0.879,
        "DNABERT2 MCC复现": 0.7589,
        "复现偏差": -0.015400000000000063
      },
      {
        "DATA": "prom_prom_core_all",
        "Task": "Core Promoter Detection-all",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 69.37,
        "MCC_USE_GUE_paper": 67.5,
        "DNABERT准确率复现": 0.8287,
        "DNABERT2 MCC复现": 0.6581,
        "复现偏差": -0.03560000000000002
      },
      {
        "DATA": "prom_prom_core_notata",
        "Task": "Core Promoter Detection-notata",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 68.04,
        "MCC_USE_GUE_paper": 69.53,
        "DNABERT准确率复现": 0.8444,
        "DNABERT2 MCC复现": 0.6888,
        "复现偏差": 0.008399999999999892
      },
      {
        "DATA": "prom_prom_core_tata",
        "Task": "Core Promoter Detection-tata",
        "Metric": "mcc",
        "Train": 4094.0,
        "Dev": 613.0,
        "Test": 613.0,
        "MCC_paper": 74.17,
        "MCC_USE_GUE_paper": 76.18,
        "DNABERT准确率复现": 0.8711,
        "DNABERT2 MCC复现": 0.7424,
        "复现偏差": 0.0006999999999999318
      },
      {
        "DATA": "mouse_0",
        "Task": "Transcription Factor prediction (Mouse)-0",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 56.76,
        "MCC_USE_GUE_paper": 64.23,
        "DNABERT准确率复现": 0.7716,
        "DNABERT2 MCC复现": 0.545,
        "复现偏差": -0.02259999999999991
      },
      {
        "DATA": "mouse_1",
        "Task": "Transcription Factor prediction (Mouse)-1",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 84.77,
        "MCC_USE_GUE_paper": 86.28,
        "DNABERT准确率复现": 0.9272,
        "DNABERT2 MCC复现": 0.8548,
        "复现偏差": 0.007100000000000079
      },
      {
        "DATA": "mouse_2",
        "Task": "Transcription Factor prediction (Mouse)-2",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 79.32,
        "MCC_USE_GUE_paper": 81.28,
        "DNABERT准确率复现": 0.9146,
        "DNABERT2 MCC复现": 0.8293,
        "复现偏差": 0.03610000000000014
      },
      {
        "DATA": "mouse_3",
        "Task": "Transcription Factor prediction (Mouse)-3",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 66.47,
        "MCC_USE_GUE_paper": 73.49,
        "DNABERT准确率复现": 0.8452,
        "DNABERT2 MCC复现": 0.7015,
        "复现偏差": 0.03680000000000007
      },
      {
        "DATA": "mouse_4",
        "Task": "Transcription Factor prediction (Mouse)-4",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 52.66,
        "MCC_USE_GUE_paper": 50.8,
        "DNABERT准确率复现": 0.7573,
        "DNABERT2 MCC复现": 0.5152,
        "复现偏差": -0.011400000000000006
      },
      {
        "DATA": "virus_covid",
        "Task": "Covid Variant Classification",
        "Metric": "f1",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 71.02,
        "MCC_USE_GUE_paper": 68.49,
        "DNABERT准确率复现": None,
        "DNABERT2 MCC复现": None,
        "复现偏差": None
      },
      {
        "DATA": "splice_reconstructed",
        "Task": "Species Classification",
        "Metric": "mcc",
        "Train": None,
        "Dev": None,
        "Test": None,
        "MCC_paper": 84.99,
        "MCC_USE_GUE_paper": 85.93,
        "DNABERT准确率复现": 0.9165,
        "DNABERT2 MCC复现": 0.8565,
        "复现偏差": 0.006600000000000108
      }
    ]



class ReportData:
    """和论文对比的统计信息"""

    def __init__(self, eval_results=None, data=None):
        """初始化数据统计对象"""
        # 确保eval_results不为None
        self.eval_results = eval_results if eval_results is not None else {}
        
        if eval_results is None:
            # 从eval_results中提取指标
            self.accuracy = self.eval_results.get("eval_accuracy")
            self.f1 = self.eval_results.get("eval_f1")
            self.matthews = self.eval_results.get("eval_matthews_correlation")
            self.mcc = self.eval_results.get("eval_matthews_correlation")
        else:
            self.accuracy = None
            self.f1 = None
            self.matthews = None
            self.mcc = None
        
        # 确保data参数存在
        if data:
            self.data_name = data.get("DATA")
            self.task = data.get("Task")
            self.metric = data.get("Metric")
            self.train = data.get("Train")
            self.dev = data.get("Dev")
            self.test = data.get("Test")
            self.mcc_paper = data.get("MCC_paper")
            self.mcc_gue_paper = data.get("MCC_USE_GUE_paper")
            self.dnabert_accuracy = data.get("DNABERT准确率复现")
            self.dnabert2_mcc = data.get("DNABERT2 MCC复现")
            
            # 安全处理复现偏差字段
            bias_value = data.get("复现偏差")
            if bias_value is not None:
                self.bias = f"{round(bias_value*100, 2)}%"
            else:
                self.bias = None
        else:
            self.data_name = None
            self.task = None
            self.metric = None
            self.train = None
            self.dev = None
            self.test = None
            self.mcc_paper = None
            self.mcc_gue_paper = None
            self.dnabert_accuracy = None
            self.dnabert2_mcc = None
            self.bias = None


def save_report_data_to_csv(report_data_list, output_path):
    """
    将报告数据列表保存为CSV文件
    
    参数:
        report_data_list (list): ReportData对象列表
        output_path (str): 输出CSV文件路径
    """
    logger.info(f"将报告数据保存到CSV文件: {output_path}")
    logger.info(f"report_data_list: len{len(report_data_list)}")
    
    # 定义CSV表头
    headers = [
        "数据集", "任务", "指标", "训练集大小", "验证集大小", "测试集大小", 
        "论文MCC", "论文MCC(GUE)", "DNABERT准确率", "DNABERT2 MCC", "复现偏差",
        "实验MCC", "MCC差异", "MCC(GUE)差异"
    ]
    
    # 准备CSV数据
    csv_data = []
    for report_data in report_data_list:
        # 获取实验MCC和差异值

        experiment_mcc = 0.0
        mcc_diff = ""
        mcc_gue_diff = ""
        
        if hasattr(report_data, "eval_results") and report_data.eval_results:
            if "eval_matthews_correlation" in report_data.eval_results:
                # 将MCC值乘以100并保留2位小数
                experiment_mcc = round(report_data.eval_results["eval_matthews_correlation"] * 100, 2)
                
                if report_data.mcc_paper is not None:
                    # 计算差异百分比并格式化为字符串
                    diff_value = round((experiment_mcc - report_data.mcc_paper)/report_data.mcc_paper * 100, 2)
                    mcc_diff = f"{diff_value}%"
                
                if report_data.mcc_gue_paper is not None:
                    # 计算差异百分比并格式化为字符串
                    diff_value = round((experiment_mcc - report_data.mcc_gue_paper)/report_data.mcc_gue_paper * 100, 2)
                    mcc_gue_diff = f"{diff_value}%"
        
        # 构建CSV行数据
        row = [
            report_data.data_name,
            report_data.task,
            report_data.metric,
            report_data.train,
            report_data.dev,
            report_data.test,
            report_data.mcc_paper,
            report_data.mcc_gue_paper,
            report_data.dnabert_accuracy,
            report_data.dnabert2_mcc,
            report_data.bias,
            experiment_mcc,
            mcc_diff,
            mcc_gue_diff
        ]
        logger.info(f"row: {row}")
        csv_data.append(row)
    
    # 写入CSV文件
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)
        logger.info(f"成功将{len(csv_data)}条报告数据保存到: {output_path}")
        return True
    except Exception as e:
        logger.error(f"保存CSV文件时出错: {e}")
        return False



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
            logger.info(f"处理任务: {task_type}/{sub_task}")
            if not os.path.isdir(sub_task_dir):
                continue
            
            logger.info(f"任务文件包括: ")

            # 检查任务状态
            eval_results_path = os.path.join(sub_task_dir, "eval_results.json")
            task_info_path = os.path.join(sub_task_dir, "task_info.json")
            log_path = os.path.join(sub_task_dir, "train.log")
            
            task_id = f"{task_type}/{sub_task}"
           
           
            # 打印子任务目录下的文件列表    
            for file in os.listdir(sub_task_dir):
                file_path = os.path.join(sub_task_dir, file)
                file_type = "目录" if os.path.isdir(file_path) else "文件"
                logger.info(f"  - {file} ({file_type})")
                if file == "eval_results.json":
                    logger.info(f"任务可解析文件包括============: {file}")
                    eval_results_path = os.path.join(sub_task_dir, file)
                if file == "task_info.json":
                    logger.info(f"任务可解析文件包括============: {file}")
                    task_info_path = os.path.join(sub_task_dir, file)
                if file == "train.log":
                    logger.info(f"任务可解析文件包括============: {file}")
                    log_path = os.path.join(sub_task_dir, file)

            
            # 如果存在评估结果，说明任务已完成
            if os.path.exists(eval_results_path):
                try:
                    # 读取评估结果
                    with open(eval_results_path, "r", encoding="utf-8") as f:
                        logger.info(f"读取任务 {task_id} 评估结果: {eval_results_path}")
                        eval_results = json.load(f)
                        logger.info(f"评估结果: {eval_results}")
                    
                    # 读取任务信息（如果存在）
                    task_info = {}
                    if os.path.exists(task_info_path):
                        try:
                            with open(task_info_path, "r", encoding="utf-8") as f:
                                logger.info(f"读取任务 {task_id} 任务信息: {task_info_path}")
                                task_info = json.load(f)
                                logger.info(f"任务信息: {task_info}")
                        except:
                            pass
                    
                    
                    # 构建任务结果
                    task_result = {
                        "task_name": task_type+"_"+sub_task,
                        "task_type": task_type,
                        "sub_task": sub_task,
                        "status": "成功",
                        "output_path": sub_task_dir,
                        "eval_results": eval_results,
                        **task_info  # 合并任务信息
                    }
                    
                    completed_tasks.append(task_result)
                    
                except Exception as e:
                    logger.error(f"处理任务 {task_id} 结果时出错: {e}")
            
           
    
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
    
    report_data_list = []

    
    for item in data_in_parpare_dict:
        report_data = ReportData(data=item)
        match_task = None
        for task in completed_tasks:
            if report_data.data_name == task["task_name"]:
                match_task=task
                logger.info(f"找到匹配任务: {match_task}")
        if match_task is not None:
            report_data.eval_results = match_task.get("eval_results", {})
            report_data.accuracy = match_task.get("eval_accuracy")
            report_data.f1 = match_task.get("eval_f1")
            report_data.matthews = match_task.get("eval_matthews_correlation")


        report_data_list.append(report_data)

    save_report_data_to_csv(report_data_list, output_path)



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成微调任务报告")
    parser.add_argument("--tasks_dir", type=str, required=True, help="任务输出目录")
    parser.add_argument("--output", type=str, required=True, help="报告输出路径")
    
    args = parser.parse_args()
    
    generate_parallel_finetune_progress_report(args.tasks_dir, args.output)

     
    
#python ../src/train/report.py --tasks_dir ../data/output/finetune/output/ --output ./report.html
if __name__ == "__main__":
    main()

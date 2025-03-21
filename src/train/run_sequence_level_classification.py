# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run sequence level classification task on ZEN model."""

from __future__ import absolute_import, division, print_function

import argparse
import sys
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import datetime

from tensorboardX import SummaryWriter

from utils_sequence_level_task import processors, convert_examples_to_features, compute_metrics
from ZEN import BertTokenizer, BertAdam, WarmupLinearSchedule
from ZEN import ZenForSequenceClassification, ZenNgramDict
from ZEN import WEIGHTS_NAME, CONFIG_NAME, NGRAM_DICT_NAME
from transformers import AutoTokenizer

# 导入自定义工具和模块
from dnazen.ngram import NgramEncoder  # n-gram编码器模块，用于处理n-gram特征

 # 计算Matthews相关系数(MCC)
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def load_examples(args, tokenizer, ngram_dict, processor, label_list, mode):
    logger.info(f"开始加载{mode}数据集...")
    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
        logger.info(f"从{args.data_dir}加载了{len(examples)}个训练样本")
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
        logger.info(f"从{args.data_dir}加载了{len(examples)}个测试样本")

    logger.info("开始将样本转换为特征...")
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, ngram_dict)
    logger.info(f"特征转换完成，共生成{len(features)}个特征")

    logger.info("开始构建张量数据集...")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ngram_ids,
                              all_ngram_positions, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks)
    logger.info(f"数据集构建完成，形状: {len(dataset)}")
    return dataset

def save_zen_model(save_zen_model_path, model, tokenizer, ngram_dict, args):
    logger.info(f"开始保存模型到: {save_zen_model_path}")
    if not os.path.exists(save_zen_model_path):
        os.makedirs(save_zen_model_path)
    
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    logger.info("准备保存模型文件...")
    
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    output_ngram_dict_file = os.path.join(save_zen_model_path, NGRAM_DICT_NAME)
    
    logger.info(f"保存模型权重到: {output_model_file}")
    torch.save(model_to_save.state_dict(), output_model_file)
    
    logger.info(f"保存模型配置到: {output_config_file}")
    model_to_save.config.to_json_file(output_config_file)
    
    logger.info(f"保存tokenizer到: {save_zen_model_path}")
    #tokenizer.save_vocabulary(save_zen_model_path)
    tokenizer.save_pretrained(save_zen_model_path)
    logger.info("tokenizer保存完成，model_save_dir: " + str(save_zen_model_path))
        
        
    logger.info(f"保存N-gram字典到: {output_ngram_dict_file}")
    ngram_dict.save(output_ngram_dict_file)
    
    output_args_file = os.path.join(save_zen_model_path, 'training_args.bin')
    logger.info(f"保存训练参数到: {output_args_file}")
    torch.save(args, output_args_file)
    
    logger.info("模型保存完成")





###################################
# 评估模型
###################################
def evaluate(args, model, tokenizer, ngram_dict, processor, label_list):
    logger.info("开始准备评估数据...")
    eval_dataset = load_examples(args, tokenizer, ngram_dict, processor, label_list, mode="test")
    
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_dataset)
        logger.info("使用顺序采样器进行评估")
    else:
        eval_sampler = DistributedSampler(eval_dataset)
        logger.info("使用分布式采样器进行评估")
    
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** 开始评估 *****")
    logger.info(f"  评估样本数量: {len(eval_dataset)}")
    logger.info(f"  评估批次大小: {args.eval_batch_size}")

    model.eval()
    preds = []  # 存储模型预测结果
    out_label_ids = None  # 存储真实标签
    total_eval_loss = 0  # 总评估损失
    nb_eval_steps = 0  # 评估步数
    avg_eval_loss =0
    logger.info("开始模型推理...")
    #for batch in eval_dataloader:
    for batch in tqdm(eval_dataloader, mininterval=20, desc="Evaluating loss={:.4f}".format(avg_eval_loss)):
        # 将数据移动到指定设备（CPU/GPU）
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_ngram_ids, ngram_position_matrix, \
        ngram_lengths, ngram_seg_ids, ngram_masks = batch
        # 在评估模式下进行前向传播，不计算梯度
        with torch.no_grad():
            # 使用关键字参数调用模型
            logits = model(
                input_ids=input_ids,
                input_ngram_ids=input_ngram_ids,
                ngram_position_matrix=ngram_position_matrix,
                labels=None, 
                head_mask=None
                )
            
            # 计算损失
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids.view(-1))
            
            # 累加评估损失和步数                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            total_eval_loss += loss.item()
            nb_eval_steps += 1

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            #import pdb; pdb.set_trace()
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
            
            # 将logits转换为预测标签（取最大概率的类别）
            #predsx = np.argmax(preds[0], axis=1)
            
            # 计算平均评估损失
            avg_eval_loss = total_eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
            logger.info(f"当前评估平均损失: {avg_eval_loss:.4f}")
    

    # 将logits转换为预测标签（取最大概率的类别）
    preds = np.argmax(preds[0], axis=1)
    
    # 计算平均评估损失
    avg_eval_loss = total_eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    logger.info(f"评估完成，平均损失: {avg_eval_loss:.4f}")
    

    # MCC计算公式：MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # 其中：TP(真阳性), TN(真阴性), FP(假阳性), FN(假阴性)
    mcc = matthews_corrcoef(out_label_ids, preds)
    logger.info(f"Matthews相关系数 (MCC): {mcc:.4f}")
    
    # 计算其他评估指标（如准确率、精确率、召回率等）
    result = compute_metrics("DNAZEN", preds, out_label_ids)
    logger.info("评估指标:")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {result[key]}")
    
    # 将MCC添加到评估结果字典中
    result['mcc2'] = mcc
    
    return result


###################################
# 训练模型
###################################
def train(args, model, tokenizer, ngram_dict, processor, label_list):
    # 检查输出目录是否存在且不为空，且未设置覆盖标志
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        logger.warning(f"输出目录 {args.output_dir} 已存在且不为空，且未设置覆盖标志，跳过训练")
        return
    
    global_step = 0

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
        logger.info("初始化TensorBoard写入器")

    logger.info("开始准备训练数据...")
    train_dataset = load_examples(args, tokenizer, ngram_dict, processor, label_list, mode="train")
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        logger.info("使用随机采样器")
    else:
        train_sampler = DistributedSampler(train_dataset)
        logger.info("使用分布式采样器")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer
    logger.info("配置优化器参数...")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)
    logger.info(f"优化器配置完成，学习率: {args.learning_rate}, 预热比例: {args.warmup_proportion}")

    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量: {len(train_dataset)}")
    logger.info(f"  批次大小: {args.train_batch_size}")
    logger.info(f"  总步数: {num_train_optimization_steps}")
    logger.info(f"  训练轮数: {args.num_train_epochs}")

    for epoch_num in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info(f"开始第 {epoch_num + 1} 轮训练")
        avg_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration loss={:.4f}".format(avg_loss),mininterval=20, disable=args.local_rank not in [-1, 0])):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, ngram_ids, ngram_positions, \
            ngram_lengths, ngram_seg_ids, ngram_masks = batch

            loss = model(input_ids,
                         ngram_ids,
                         ngram_positions,
                         labels=label_ids)
            #logger.info("loss: " + str(loss))
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            avg_loss = tr_loss/nb_tr_steps
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                #if args.local_rank in [-1, 0]:
                if True:
                    tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', loss.item(), global_step)
                    if global_step % 100 == 0:
                        logger.info(f"全局步数: {global_step}, 当前损失: {loss.item():.4f}, 学习率: {optimizer.get_lr()[0]:.2e}")
                
                #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if False:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info(f"保存模型检查点到: {output_dir}")
                    save_zen_model(output_dir, model, tokenizer, ngram_dict, args)
        
        logger.info(f"全局步数: {global_step}, 当前损失: {loss.item():.4f}, 学习率: {optimizer.get_lr()[0]:.2e}")
        output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}-{}".format(epoch_num, global_step, tr_loss/nb_tr_steps))
        save_zen_model(output_dir, model, tokenizer, ngram_dict, args)
        logger.info(f"第 {epoch_num + 1} 轮训练完成，平均损失: {tr_loss/nb_tr_steps:.4f}")


def save_evaluate_results(args, results):
    # 保存评估结果
    results_path = os.path.join(args.output_dir, "eval_results.json")
    logger.info(f"保存评估结果到: {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)  # 添加缩进使JSON文件更易读

    

def main():
    parser = argparse.ArgumentParser()
    logger.info("开始解析命令行参数...")

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='./results/result-seqlevel-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')),
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--multift",
                        action='store_true',
                        help="True for multi-task fine tune")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--ngram_list_dir", type=str, default=None,
                        help="Path to the n-gram list file.")
    args = parser.parse_args()
    logger.info("命令行参数解析完成")

    args.task_name = args.task_name.lower()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        filemode='w',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info(f"使用设备: {args.device}, GPU数量: {args.n_gpu}")
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    logger.info("设置随机种子...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"随机种子设置为: {args.seed}")

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        logger.warning(f"输出目录 {args.output_dir} 已存在且不为空")
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        logger.info(f"创建输出目录: {args.output_dir}")


    processor = processors["DNAZEN"]() 
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(f"标签列表: {label_list}, 标签数量: {num_labels}")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    logger.info(f"加载预训练模型: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, from_tf=True)


    logger.info(f"加载N-gram字典: {args.ngram_list_dir}")
    ngram_dict = ZenNgramDict(args.ngram_list_dir, tokenizer=tokenizer)
    logger.info(f"N-gram字典加载完成，包含 {len(ngram_dict.ngram_to_id_dict)} 个N-gram")

    logger.info("加载分类模型...")
    model = ZenForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=num_labels,
        multift=args.multift,
        from_tf=True
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
        logger.info("使用FP16精度")
    model.to(args.device)
    model = torch.nn.DataParallel(model)
  

    if args.do_train:
        logger.info("开始训练流程...")
        train(args, model, tokenizer, ngram_dict, processor, label_list)
        
        logger.info("训练完成")

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("开始评估流程...")
        result = evaluate(args, model, tokenizer, ngram_dict, processor, label_list)
        save_evaluate_results(args, result)
        logger.info("评估完成")

if __name__ == "__main__":
    main()
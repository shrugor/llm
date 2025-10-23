import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
import deepspeed
from typing import Optional, List
import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    '''
    关于模型的参数
    '''
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":(
                "后训练使用，为预训练模型参数地址"
            )
        },
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "预训练使用Config文件地址"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "预训练Tokenizer地址"})
    torch_dtype: Optional[str] = field(default=None, metadata={"help":("模型训练使用的数据类型，推荐bfloat16"), "choices": ['auto', 'bfloat16', 'float16', 'float32']})


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """
    train_files: Optional[List[str]] = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(default=None, metadata={"help":("设置的文本块长度")},)
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "预处理使用的线程数"})


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

logger = logging.getLogger(__name__)
# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# 将日志级别设置为INFO
transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


# 训练整体情况记录
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"输出路径({training_args.output_dir}) 非空"
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"从{last_checkpoint} 恢复训练"
        )

# 设置随机数种子
set_seed(training_args.seed)

# 初始化模型
if model_args.config_name is not None:
    config = AutoConfig.from_pretrained(model_args.config_name)
    logger.warning("你正在从零初始化一个模型")
    logger.info(f"模型参数配置地址：{model_args.config_name}")
    logger.info(f"模型参数：{config}")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"预训练一个新模型 - Total size={n_params/2**20:.2f} M params")
elif model_args.model_name_or_path is not None:
    logger.warning("你正在初始化一个预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f} M params")
else:
    logger.error("config_name 和model_name_or_path 不能均为空")
    raise ValueError("config_name 和model_name_or_path 不能均为空")

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
logger.info("完成tokenizer加载")
logger.info(f"tokenizer配置地址：{model_args.tokenizer_name}")

# 加载预训练数据
ds = load_dataset('json', data_files=data_args.train_files)
logger.info("完成训练集加载")
logger.info(f"训练集地址：{data_args.train_files}")
logger.info(f"训练文件总数：{len(ds["train"])}")

# 文本tokenize
column_names = list(ds["train"].features)
logger.info("训练特征：", column_names)
text_column_name = "text" if "text" in column_names else column_names

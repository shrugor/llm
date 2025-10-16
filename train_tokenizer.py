import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from train_tokenizer import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tolenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

def read_text_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f'Missing text field in line {line_num}')
                yield data['text']
            except json.JSONDecodeError:
                print('f"Error decoding JSON in line {line_num}')
                continue
            except KeyError as e:
                print(e)
                continue

def create_tokenizer_config(save_dir: str) -> None:
    '''创建完整的tokenizer配置文件'''
    config = {
        
    }


if __name__ == "__main__":
    pass

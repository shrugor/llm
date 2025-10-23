import os
from tqdm import tqdm
import json


# # 处理预训练数据
# def split_text(text, chunk_size=512):
#     '''将文本按指定长度切分成块'''
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# input_file = '/home/zhanghudong/llm/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl'


# save_data_p = 'seq_monkey.jsonl'

# if not os.path.isfile(save_data_p):

#     with open(save_data_p, 'a', encoding='utf-8') as pretrain:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             data = f.readlines()
#             for line in tqdm(data, desc=f"Processing lines in {input_file}", leave=False):  # 添加行级别的进度条
#                 line = json.loads(line)
#                 text = line['text']
#                 chunks = split_text(text)
#                 for chunk in chunks:
#                     pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 处理sft数据
def convert_message(data):
    '''将原始数据转换为标准格式'''
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user','content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role':'assistant', 'content': item['value']})
    return message

save_data_p = "BelleGroup_sft.jsonl"
input_file = r'/home/zhanghudong/llm/dataset/BelleGroup/train_3.5M_CN.json'

with open(save_data_p, 'a', encoding='utf-8') as sft:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in tqdm(data, desc='Processing', unit='lines'):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')    

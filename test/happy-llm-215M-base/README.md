---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- text-generation
---

# Happy-LLM-215M-Base

<div align='center'>
    <img src="./images/head.jpg" alt="alt text" width="80%">
    <h1>Happy-LLM</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/github/stars/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/datawhalechina/happy-llm?style=flat&logo=github" alt="GitHub forks"/>
  <img src="https://img.shields.io/badge/language-Chinese-brightgreen?style=flat" alt="Language"/>
  <a href="https://github.com/datawhalechina/happy-llm"><img src="https://img.shields.io/badge/GitHub-Project-blue?style=flat&logo=github" alt="GitHub Project"></a>
  <a href="https://swanlab.cn/@kmno4/Happy-LLM/overview"><img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>
</div>

&emsp;&emsp;本模型为 Happy-LLM 215M 基础预训练模型，基于 Pytorch 框架，模型大小为 215M，训练数据为[出门问问序列猴子开源数据集](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey/files)，总量大概在 10B Token 左右。

> 注：本模型仅用于教学目的，请勿将本模型应用于任何生产环境。

> 注：模型在 8卡4090 平台上进行训练，batch_size 设置为 64，训练总 step 为 453110。

## 模型下载

&emsp;&emsp;可以使用如下方式下载模型

```bash
#安装ModelScope
pip install modelscope
```
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('kmno4zx/happy-llm-215M-base')
```

## 模型使用

```bash
pip install -r requirements.txt
```

&emsp;&emsp;运行以下代码即可：

```python
import os
import pickle
from contextlib import nullcontext
import torch
from k_model import ModelConfig, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class TextGenerator:
    def __init__(self, 
                 checkpoint='./pretrain_1024_18_6144.pth',  # 模型检查点路径
                 tokenizer_model_path='./tokenizer_k/',  # 分词器模型路径
                 seed=42,  # 随机种子，确保可重复性
                 device=None,  # 设备，优先使用 CUDA，如果没有可用的 CUDA，则使用 CPU
                 dtype="bfloat16"):  # 数据类型，默认为 float32，可以选择 float16 或 bfloat16
        """
        初始化 TextGenerator 类，加载模型、设置设备和分词器等。
        """
        # 模型加载配置
        self.checkpoint = checkpoint  # 保存的模型检查点路径
        self.tokenizer_model_path = tokenizer_model_path  # 分词器模型文件路径
        self.seed = seed  # 随机数种子，用于生成的可重复性
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')  # 根据硬件条件选择设备
        self.dtype = dtype  # 模型的浮点数类型
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # 判断当前设备是否为 CUDA
        
        # 设置随机种子，确保生成的可重复性
        torch.manual_seed(seed)  # 设置 CPU 随机种子
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许 CUDA 使用 TF32 精度进行矩阵乘法运算
        torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 精度加速
        
        # 根据 dtype 选择适当的自动混合精度上下文
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        # 加载模型检查点文件
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)  # 加载模型参数 # 初始化模型参数
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))  # 实例化 Transformer 模型
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)
        
        # 计算模型参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        # 设置模型为评估模式（evaluation mode），防止训练模式下的 dropout 等操作影响结果
        self.model.eval()
        # 将模型放置到正确的设备上（GPU 或 CPU）
        self.model.to(self.device)
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)  # 根据指定的路径加载分词器

    def pretrain_sample(self, 
               start="Hello!",  # 生成文本的起始提示词，可以是任意字符串
               num_samples=3,  # 生成样本的数量，默认生成 3 个样本
               max_new_tokens=256,  # 每个样本生成的最大 token 数，默认最多生成 256 个 token
               temperature=0.7,  # 控制生成的随机性，1.0 为标准，值越大越随机
               top_k=300):  # 保留概率最高的 top_k 个 token，限制生成时的选择范围
        """
        根据给定的起始文本生成样本。
        
        :param start: 生成文本的起始提示词
        :param num_samples: 要生成的文本样本数
        :param max_new_tokens: 每个样本生成的最大 token 数
        :param temperature: 控制生成的随机性，值越小生成越确定，值越大生成越随机
        :param top_k: 限制生成时选择的 token 范围
        :return: 生成的文本样本列表
        """
        # 如果 start 是以 'FILE:' 开头，表示从文件中读取起始文本
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()  # 读取文件内容作为起始文本
        
        # 将起始文本编码为 token id 序列
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # 将编码后的 token id 转为 PyTorch 张量
        # print(x.shape)
        generated_texts = []  # 用于保存生成的文本样本
        with torch.no_grad():  # 禁用梯度计算，提升效率
            with self.ctx:  # 进入自动混合精度的上下文（如果是 GPU 并使用 float16 时）
                for k in range(num_samples):  # 循环生成指定数量的样本
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)  # 生成文本
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # 解码生成的 token 序列为可读文本
        
        return generated_texts  # 返回生成的文本样本
    
if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是',
        '<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院',
    ]

    generator = TextGenerator(checkpoint='./base_model_215M/pretrain_1024_18_6144.pth')  # 初始化生成器
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i+1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-'*20}")  # 打印生成的样本并用分隔线分割
```

## 🙏 致谢

### 核心贡献者
- [宋志学-项目负责人](https://github.com/KMnO4-zx) (Datawhale成员-中国矿业大学(北京))
- [邹雨衡-项目负责人](https://github.com/logan-zou) (Datawhale成员-对外经济贸易大学)
- [朱信忠-指导专家](https://xinzhongzhu.github.io/)（Datawhale首席科学家-浙江师范大学杭州人工智能研究院教授）

### 特别感谢
- 感谢 [@Sm1les](https://github.com/Sm1les) 对本项目的帮助与支持
- 感谢所有为本项目做出贡献的开发者们 ❤️

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/happy-llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/happy-llm" />
  </a>
</div>

## Star History

<div align='center'>
    <img src="./images/star-history-2025612.png" alt="Datawhale" width="90%">
</div>

<div align="center">
  <p>⭐ 如果这个项目对你有帮助，请给我们一个 Star！</p>
</div>

## 关于 Datawhale

<div align='center'>
    <img src="./images/datawhale.png" alt="Datawhale" width="30%">
    <p>扫描二维码关注 Datawhale 公众号，获取更多优质开源内容</p>
</div>

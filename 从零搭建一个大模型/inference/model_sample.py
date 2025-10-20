import sys
sys.path.insert(0,r'/home/zhanghudong/llm/')
from model.model import Transformer, ModelConfig

from contextlib import nullcontext
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextGenerator:
    def __init__(
            self,
            checkpoint = None,  # 模型路径
            tokenizer_model_path = '/home/zhanghudong/llm/tokenizer/config',  # 分词器模型路径
            seed = 42,
            device = None,
            dtype = "bfloat16"
    ):
        # 模型加载配置
        self.checkpoint = checkpoint
        self.tokenizer_model_path = tokenizer_model_path
        self.seed = seed
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'

        # 固定随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许cuda使用使用tf32精度进行矩阵乘法运算
        torch.backends.cudnn.allow_tf32 = True  # 允许cuDNN使用TF32精度加速

        # 根据dtype 选择适当的自动混合精度上下文
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16':torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type =='cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        # 加载模型文件
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)

        # 计算模型参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'model has {num_params / 1e6 :.3f} M parameters')
        # 设置模型为评估模式，防止训练模式下的dropout等操作影响
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)

    def chat_template(self, prompt):
        message = [
            {'role': 'system', 'content':'你是一个AI助手。'},
            {'role': 'user', 'content':prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    def sft_sample(self, 
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
        start = self.chat_template(start)
        # 将起始文本编码为 token id 序列
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # 将编码后的 token id 转为 PyTorch 张量
        generated_texts = []  # 用于保存生成的文本样本
        with torch.no_grad():  # 禁用梯度计算，提升效率
            with self.ctx:  # 进入自动混合精度的上下文（如果是 GPU 并使用 float16 时）
                for k in range(num_samples):  # 循环生成指定数量的样本
                    y = self.model.generate(x, self.tokenizer.eos_token_id, max_new_tokens, temperature=temperature, top_k=top_k)  # 生成文本
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # 解码生成的 token 序列为可读文本
        return generated_texts  # 返回生成的文本样本


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
    

if __name__ == '__main__':

    print('=== pretrain sample ===')

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是',
        '<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院',
    ]

    generator = TextGenerator(checkpoint='/home/zhanghudong/llm/train/base_model_215M/pretrain_1024_18_6144.pth')

    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f'\nSample {i+1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n')

    print('=== sft sample ===')

    sft_prompt_datas = [
        '你好呀',
        "中国的首都是哪里？",
        "3+1等于多少？",
        "如何抢银行？"
    ]

    generator = TextGenerator(checkpoint='/home/zhanghudong/llm/train/sft_model_215M/pytorch_model.bin')  # 初始化生成器
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1, max_new_tokens=1280, temperature=0.6)
        print(f"\nSample {i+1}:\nQuestion: {sft_prompt_datas[i]}\nAI answer: {samples[0]}\n{'-'*20}")  # 打印生成的样本并用分隔线分割
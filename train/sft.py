import sys
sys.path.insert(0,r'/home/zhanghudong/llm/')
import os
import time
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import ModelConfig, Transformer
from dataset.load_data import SFTDataset

import swanlab


class Args:
    def __init__(self):
        self.out_dir = "sft_model_215M"  # 模型输出目录
        self.epochs = 1  # 训练轮数
        self.batch_size = 64  # 轮次大小
        self.learning_rate = 2e-4  # 学习率
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 训练设备
        self.dtype = "bfloat16"  # 数据类型
        self.use_swanlab = False  # 是否使用SwanLab进行实验跟踪
        self.num_workers = 8  # 数据加载的工作进程数
        self.data_path = "/home/zhanghudong/llm/dataset/BelleGroup_sft.jsonl"  # 数据训练路径
        self.accumulation_steps = 8  # 梯度累计步数
        self.grad_clip = 1.0  # 梯度裁剪阈值
        self.warmup_iters = 0  # 学习率预热迭代次数
        self.log_interval = 100  # 日志间隔
        self.save_interval = 1000000000  # 保存间隔
        self.gpus = '0,1,2,3,4,5,6,7'  # 多GPU训练参数


def Logger(content):
    """日志记录器"""
    print(content)

def get_lr(it, all):
    """获取学习率"""
    # 1) linear warmup for warmup_iters steps
    # 1) 预热迭代的线性预热
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) 如果迭代次数超过学习率衰减迭代次数，则返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) 在两者之间，使用余弦衰减至最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """训练一个epoch"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取学习率并更新优化器
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 反向传播
        scaler.scale(loss).backward()

        # 更新权重
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # 保存模型
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # 处理多卡保存
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # 定期保存模型
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """初始化模型"""
    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/zhanghudong/llm/tokenizer")

    # 初始化模型
    model = Transformer(lm_config)

    # 加载预训练权重
    ckp = '/home/zhanghudong/llm/train/base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    # 多卡初始化
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":

    args = Args()

    # 设置可见GPU
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # 初始化swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="First-llm",
            experiment_name="SFT-215M",
            config=args,
        )

    # 模型配置
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')

    # 初始化模型和分词器
    model, tokenizer = init_model()
    
    # 创建数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 缩放器和优化器
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)

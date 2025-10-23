import sys
sys.path.insert(0,r'/home/zhanghudong/llm/')
from model.model import Transformer, ModelConfig
import torch
import os
import swanlab
from contextlib import nullcontext
from transformers import AutoTokenizer
from dataset.load_data import PretrainDataset
from torch.utils.data import DataLoader
from torch import optim
import time
import math


class Args:
    def __init__(self):
        self.out_dir = "base_model_215M"  # 模型输出目录
        self.epochs = 1  # 训练轮数
        self.batch_size = 64  # 轮次大小
        self.learning_rate = 2e-4  # 学习率
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 训练设备
        self.dtype = "bfloat16"  # 数据类型
        self.use_swanlab = False  # 是否使用SwanLab进行实验跟踪
        self.num_workers = 8  # 数据加载的工作进程数
        self.data_path = "/home/zhanghudong/llm/dataset/seq_monkey.jsonl"  # 数据训练路径
        self.accumulation_steps = 8  # 梯度累计步数
        self.grad_clip = 1.0  # 梯度裁剪阈值
        self.warmup_iters = 0  # 学习率预热迭代次数
        self.log_interval = 100  # 日志间隔
        self.save_interval = 1000  # 保存间隔
        self.gpus = '0,1,2,3,4,5,6,7'  # 多GPU训练参数


def Logger(content):
    """
    简单的日志记录函数
    
    Args:
        content (str): 要打印的内容
    """
    print(content)


def init_model():
    '''
    初始化模型和分词器

    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并行训练
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量

    Reruen:
        tuple: (model, tokenizer) 初始化后的模型和分词器

    '''
    def count_parameters(model):
        """
        统计模型中可训练参数的数量

        Args:
            model: PyTorch模型
        Return:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/zhanghudong/llm/tokenizer")

    # 根据配置创建Transformer模型
    model = Transformer(lm_config)

    # 多卡初始化：检查可用GPU的数量并设置DataParallel
    num_gpus= torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # 使用DataParallel包装模型以支持多GPU训练
        model = torch.nn.DataParallel(model)  # todo: 尝试 torch.nn.DistributedDataParallel
    
    # 将模型移动到指定设备
    model  = model.to(args.device)

    # 计算并打印模型参数量
    Logger(f"LLM 总参数量：{count_parameters(model) / 1e6 : .3f} 百万")
    return model, tokenizer

    
def get_lr(it, all):
    '''
    计算当前迭代的学习率, 使用余弦退火调整策略

    1. Warm up阶段: 学习率从0线性增长到目标学习率
    2. 余弦退火阶段: 学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后: 保持最小学习率

    Args:
        it(int): 当前迭代步数
        all(int): 总迭代步数

    Returns:
        float: 当前步数对应的学习率
    '''
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率，为初始学习率的1/10

    # Warmup阶段：线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr

    # 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    """
    训练一个epoch的函数

    实现了完整的训练循环, 包括:
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累计和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存

    Arg:
        epoch (int): 当前epoch 

    """
    start_time = time.time()  # 记录开始时间

    # 遍历数据加载器中的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转到指定设备 GPU
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略padding token

        # 计算当前步骤的学习率
        lr =  get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)

        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            out = model(X, Y)
            # 计算损失并除以累计步数
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask 展平为一维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（忽略padding位置）
            loss = torch.sum(loss*loss_mask) / loss_mask.sum()
        
        # 使用scaler进行混合精度的反向传播
        scaler.scale(loss).backward()

        # 每accumulation_steps步执行一次优化器更新

        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步骤
            scaler.step(optimizer)
            # 更新scaler的缩放因子
            scaler.update()

            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
        
        # 每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            # 如果启用Swanlab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })
        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'
            # 处理多卡保存：如果是DataParall模型，需要访问.module属性
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


if __name__ == '__main__':

    args = Args()

    # === GPU环境设置 ===
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个可用GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"
    
    #  === 实验跟踪初始化 ===
    if args.use_swanlab:
        # 注意：使用前需要登录 
        run = swanlab.init(
            project = "First-llm",  # 项目名称
            experiment = "Pretrain-215M",  # 实验名称
            config = args,  # 保存所有的超参数
        )
    
    # === 模型配置 ===
    lm_config = ModelConfig(
        dim=1024,  # 模型维度
        n_layers=18,  # Transformer层数 
    )

    # === 训练环境设置 ===
    max_seq_len = lm_config.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存目录

    # 创建必要的目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 随机种子
    torch.manual_seed(42)

    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置混合精度训练的上下文管理器
    # CPU训练时使用nullcontext， GPU训练时使用autocast
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda')

    # === 模型和数据初始化 ===
    # 初始化模型和分词器
    model, tokenizer = init_model()

    # 创建训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,  # 批次大小
        pin_memory=True,  # 将数据加载到固定内存中，加速GPU传输
        drop_last=False,  # 不丢弃最后一个不完整的批次
        shuffle=True,  # 随机打乱数据
        num_workers=args.num_workers  # 数据加载的并行工作进程数 
    )

    # === 优化器和训练组件初始化 ===
    # 初始化混合精度训练的梯度缩放器
    # 只有在使用float16或bfloat16时才启用
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype in ['float16', 'bfloat16']))

    # 初始化Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # === 开始训练 ===
    # 计算每个epoch的迭代数
    iter_per_epoch = len(train_loader)
    print(iter_per_epoch)

    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch)
    
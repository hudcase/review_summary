import torch
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from utils import *


def initialize_model(model_path: str, use_lora: bool = True):
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用bfloat16混合精度
        device_map="auto"            # 自动设备分配
    )

    # 配置LoRA适配器
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[  # 覆盖关键注意力模块
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            inference_mode=False,
            r=8,                  # LoRA秩（平衡效果与效率）
            lora_alpha=32,         # 缩放因子（一般=2r 或 4r）
            lora_dropout=0.1,       # 正则化比率
        )
        model = get_peft_model(model, lora_config)

    # 启用梯度检查点（显存优化）
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print(model.print_trainable_parameters()
)
    return model, tokenizer

def preprocess_data(file_path, sample_size=None, DEFAULT_SEED=42):
    raw_dataset = load_dataset('json', data_files=file_path)['train']
    '''
    Dataset({
        features: ['review_body', 'review_title'],
        num_rows: 15000
    })
    (Pdb) raw_dataset[0]
    {'review_body': 'These are AWFUL. They are see through, the fabric feels like tablecloth, and they fit like children’s clothing. Customer service did seem to be nice though, but I regret missing my return date for these. I wouldn’t even donate them because the quality is so poor.', 'review_title': 'Don’t waste your time!'}
    '''
    # 数据采样与随机化
    if sample_size:
        raw_dataset = raw_dataset.shuffle(seed=DEFAULT_SEED).select(range(sample_size))
    return raw_dataset


def tokenize_function(tokenizer, examples):
    """
    将文本数据转换为模型所需的输入格式
    
    处理逻辑:
    1. 构建系统提示和用户指令
    2. 将标签转换为文本形式
    3. 拼接指令和响应生成完整序列
    4. 处理长度限制和特殊token
    
    参数:
        examples (dict): 包含'review_body'和'review_title'的样本字典
        例子： {'review_body': '刚收到货，烧水时有嗡嗡的声音，不知道是不是产品缺陷。', 'review_title': '烧水时发出较大嗡嗡的声音'}
        
    返回:
        dict: 包含input_ids, attention_mask, labels的字典
    """
    MAX_LENGTH = 1024  # 模型最大上下文长度
    
    # ref https://github.com/QwenLM/Qwen/blob/main/finetune.py
    # 构建指令部分
    instruction = tokenizer(
        generate_review_title_prompt_tmp0(examples['review_body']),
        add_special_tokens=False  # 禁用自动添加特殊token
    )
    '''
    (Pdb) apply_chat_template(system_prompt, examples['review_body'])
    '<|im_start|>system\n你将看到一个来自电商平台的商品评论, 请根据评论内容，生成简短总结。<|im_end|>\n<|im_start|>user\n刚收到货，烧水时有嗡嗡的声音，不知道是不是产品缺陷。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    (Pdb) instruction
    {'input_ids': [151644, 8948, 198, 56568, 44063, 101038, 46944, 101919, 110514, 109348, 85641, 11, 220, 14880, 100345, 85641, 43815, 3837, 43959, 220, 16, 4142, 17, 26853, 98, 9370, 98237, 99534, 102050, 90395, 100662, 57218, 52129, 85641, 107668, 102064, 9909, 104811, 5373, 8903, 16744, 57191, 105205, 74276, 31196, 45943, 85641, 5122, 151645, 198, 151644, 872, 198, 99900, 104381, 81668, 3837, 100228, 52510, 13343, 18830, 117377, 117377, 104668, 3837, 103950, 104074, 82700, 106136, 1773, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    '''
    # 构建响应部分（真实标签）
    response = tokenizer(
        f"{examples['review_title']}", 
        add_special_tokens=False
    )
    '''
    (Pdb) response
    {'input_ids': [100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
    '''
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    '''
    (Pdb) input_ids
    [151644, 8948, 198, 56568, 44063, 101038, 46944, 101919, 110514, 109348, 85641, 11, 220, 14880, 100345, 85641, 43815, 3837, 43959, 220, 16, 4142, 17, 26853, 98, 9370, 98237, 99534, 102050, 90395, 100662, 57218, 52129, 85641, 107668, 102064, 9909, 104811, 5373, 8903, 16744, 57191, 105205, 74276, 31196, 45943, 85641, 5122, 151645, 198, 151644, 872, 198, 99900, 104381, 81668, 3837, 100228, 52510, 13343, 18830, 117377, 117377, 104668, 3837, 103950, 104074, 82700, 106136, 1773, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668, 151643]
    '''
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    '''
    (Pdb) attention_mask
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100]*len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    '''
    (Pdb) labels
    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 100228, 52510, 13343, 104351, 104590, 117377, 117377, 104668, 151643]
    '''
    # 长度截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # import pdb;pdb.set_trace()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def train(model, tokenizer, dataset, args):
    """
    参数：
        model: 待训练模型
        tokenizer: 文本处理器
        dataset: 训练数据集
        args: argparse解析后的训练超参数
    """
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,       # 根据显存调整
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积（模拟更大batch）
        logging_steps=args.logging_steps,                  # 监控训练状态
        num_train_epochs=args.epochs,                      # 训练轮数
        save_steps=args.save_steps,                        # 检查点间隔
        learning_rate=args.learning_rate,                  # 学习率（LoRA典型值）
        save_on_each_node=True,                            # 分布式训练设置
        gradient_checkpointing=True,                       # 启用梯度检查点
        bf16=True,                                         # 使用bfloat16混合精度训练
        remove_unused_columns=False,                       # 保留未使用列（用于标签处理）
        report_to="none",
        logging_dir="./logs"
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # 启动训练流程
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--model_path", type=str, default="./model_weights/Qwen/Qwen3-0.6B", help="预训练模型路径")
    parser.add_argument("--data_path", type=str, default="./dataset/train.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=8, help="每个设备的训练batch大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存检查点间隔")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--sample_size", type=int, default=None, help="数据采样大小，可选")
    parser.add_argument("--use_lora", type=bool, default=True, help="是否启用LoRA微调")
    args = parser.parse_args()


    # 模型 和 tokenizer
    model, tokenizer = initialize_model(args.model_path, use_lora=args.use_lora)

    # 加载数据
    dataset = preprocess_data(args.data_path, sample_size=args.sample_size)

    # 应用分词处理
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(tokenizer, examples), remove_columns=dataset.column_names)

    # 启动训练
    train(model, tokenizer, tokenized_dataset, args)

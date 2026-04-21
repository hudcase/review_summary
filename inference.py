import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Qwen3 商品评论总结生成器")
    parser.add_argument("--model_name", type=str, default="./model_weights/Qwen/Qwen3-0.6B", help="模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重路径")
    parser.add_argument("--review_body", type=str, default="键盘还可以，比较舒适，但是鼠标里面两节电池让人很难受，因为鼠标得经常移动，重量比较敏感，这个设计比较差。", help="商品评论内容")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="生成的最大token数")
    parser.add_argument("--do_sample", action="store_true", help="是否启用随机采样")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus采样概率")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成文本创造性控制")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")
    parser.add_argument("--device", type=str, default="auto", help="模型加载设备，如 'cuda', 'cpu', 'auto'")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device
    ).eval()

    # LoRA适配器加载
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path)

    # 构造输入文本
    text = generate_review_title_prompt_tmp0(args.review_body)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成策略配置
    generate_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_config)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

    # 解码与输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("生成总结:", response)


if __name__ == "__main__":
    main()
# train_grpo.py
import os
import re
import argparse
import torch
import numpy as np
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

prompt_think="""
人得到一个问题，首先是根据问题的深度，难度，复杂度，慢慢地思考问题应该如何解决。在思考过程中运用自己的各种能力，对于问题进行逐步逐步推导解决，在一步一步地推导解决过程中得到结果。得到结果之后，人停止思考并且整合思考过程中的内容，进行大量简化后，得到问题的整个解决办法和结果，最后输出问题的解决办法和结果。
任务描述：
question:{question}
你的任务是考虑问题在人思考过程中所花费的令牌数量的范围。具体根据问题可能涉及的深度、复杂性和长度，估算出思考过程总共花费的令牌数量的范围。
例如:
1估计的令牌数量范围在200，则真实思考过程中所花费的令牌数量基本在(200*25,201*25-1)
2估计的令牌数量范围在320，则真实思考过程中所花费的令牌数量基本在(320*25,321*25-1)
要求：仅输出估计的令牌数量范围，不生成其他的内容。
"""

def reward(prompts, completions, labels, **kwargs):
    def extract_number(text):
        if not isinstance(text, str):
            return None
        # 严格匹配整数（避免匹配到小数部分）
        match = re.search(r'-?\b\d+\b', text)
        if match:
            try:
                return int(match.group())
            except ValueError:
                return None
        return None

    rewards = []
    for comp, label in zip(completions, labels):
        pred = extract_number(comp)
        truth = extract_number(label)

        if truth is None:
            rewards.append(0.0)
        elif pred is None:
            rewards.append(0.0)
        else:
            diff = abs(truth - pred)
            if diff == 0:
                r = 12.0
            elif diff == 1:
                r = 6.0
            elif diff == 2:
                r = 3.0
            else:
                r = 0.0
            rewards.append(r)

    return torch.tensor(rewards, dtype=torch.float32, device="cuda")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (args 定义保持不变) ...
    parser.add_argument('--model_path', type=str, default=r"../model/think/..")
    parser.add_argument('--train_data_path', type=str, default=r"../data/think/")
    parser.add_argument('--valid_data_path', type=str, default=r"../data/think/..")
    parser.add_argument('--logging_steps', type=int, help="the logging frequency.", default=10)
    parser.add_argument('--save_steps', type=int, help="the saving frequency.", default=13000)
    parser.add_argument('--num_train_epochs', type=int, help="The number of training epochs for the reward model.",
                        default=8)
    parser.add_argument("--max_completion_length", type=int,
                        help="Maximum length of the completion. This argument is required if you want to use the default data "
                             "collator and your model is an encoder-decoder.", default=20)

    parser.add_argument('--learning_rate', type=float, help="The initial learning rate for [`AdamW`] optimizer.",
                        default=5e-6)
    parser.add_argument('--adam_beta1', type=float, help="The beta1 hyperparameter for the [`AdamW`] optimizer.",
                        default=0.9)
    parser.add_argument('--adam_beta2', type=float, help="The beta2 hyperparameter for the [`AdamW`] optimizer.",
                        default=0.99)
    parser.add_argument('--weight_decay', type=float, help="Weight decay for AdamW if we apply some.", default=0.1)
    parser.add_argument('--warmup_ratio', type=float, help="Linear warmup over warmup_ratio fraction of total steps.",
                        default=0.1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)  # ← 改为 1
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  # ← 用梯度累积模拟大 batch
    parser.add_argument("--num_generations", type=int,
                        help="Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value.",
                        default=8)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--max_prompt_length', type=int,
                        help="Maximum length of the completion. This argument is required if you want to use the default data "
                             "collator and your model is an encoder-decoder.", default=1024)
    parser.add_argument('--output_path', type=str, help="Output path for the trained model",
                        default=r"/mnt/f/home_fix/1/GRPO/model")

    args = parser.parse_args()

    max_seq_length = 1044
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=r"/mnt/d/home/home/picture_core/total/1/Qwen2-0.5B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        model_type="qwen2"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0.00,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407
    )

    # === 构建 prompt 字符串 ===
    def build_prompt(question):
        return prompt_think.format(question=question).strip()

    dataset_1 = load_from_disk(args.train_data_path)
    train_dataset = dataset_1.map(lambda x: {
        "prompt": build_prompt(x["prompt"]),
        "labels": x["class_name"]
    })

    dataset_2 = load_from_disk(args.valid_data_path)
    eval_dataset = dataset_2.map(lambda x: {
        "prompt": build_prompt(x["prompt"]),
        "labels": x["class_name"]
    })

    # 小样本调试
    # train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
    # eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))

    training_args = GRPOConfig(
        overwrite_output_dir=True,
        output_dir=args.output_path,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        max_completion_length=args.max_completion_length,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        save_steps=args.save_steps,
        report_to="tensorboard",
        num_generations=args.num_generations,
        generation_kwargs={
            "do_sample": True,
            "temperature": 1.0,  # ← 鼓励探索错误答案
            "top_p": 0.95,
            "max_new_tokens": 20,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer  # ✅ 关键修正
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

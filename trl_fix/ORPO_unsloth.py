# train_dpo.py
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from unsloth import FastLanguageModel

from trl import DPOConfig, DPOTrainer
from trl import apply_chat_template

prompt_think = """
人得到一个问题，首先是根据问题的深度，难度，复杂度，慢慢地思考问题应该如何解决。在思考过程中运用自己的各种能力，对于问题进行逐步逐步推导解决，在一步一步地推导解决过程中得到结果。得到结果之后，人停止思考并且整合思考过程中的内容，进行大量简化后，得到问题的整个解决办法和结果，最后输出问题的解决办法和结果。
任务描述：
question:{question}
你的任务是考虑问题在人思考过程中所花费的令牌数量的范围。具体根据问题可能涉及的深度、复杂性和长度，估算出思考过程总共花费的令牌数量的范围。
例如:
1估计的令牌数量范围在200，则真实思考过程中所花费的令牌数量基本在(200*25,201*25-1)
2估计的令牌数量范围在320，则真实思考过程中所花费的令牌数量基本在(320*25,321*25-1)
要求：仅输出估计的令牌数量范围，不生成其他的内容。
"""

if __name__ == '__main__':
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_from_disk
    from trl import ORPOTrainer, ORPOConfig
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r"../model/think/..")
    parser.add_argument('--train_data_path', type=str, default=r"../data/think/data1/train_data")
    parser.add_argument('--valid_data_path', type=str, default=r"../data/think/data1/valid_data")
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--num_train_epochs', type=int, default=6)
    parser.add_argument('--output_path', type=str, help="",default=r"../model/think/ORPO/data1/model")
    args = parser.parse_args()


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=1088,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )

    # Load datasets
    train_dataset = load_from_disk(args.train_data_path)
    eval_dataset = load_from_disk(args.valid_data_path)
    print("train_dataset:",train_dataset.column_names)
    print("eval_dataset:",eval_dataset.column_names)
    #
    #
    # def debug_dataset(dataset, num_samples=3):
    #     print("\n" + "=" * 50)
    #     print("数据集调试信息")
    #     print("=" * 50)
    #
    #     for i in range(min(num_samples, len(dataset))):
    #         example = dataset[i]
    #         print(f"\n样本 {i}:")
    #         print(f"chosen 类型: {type(example['chosen'])}")
    #         print(f"rejected 类型: {type(example['rejected'])}")
    #
    #         # 如果是字符串，显示部分内容
    #         if isinstance(example['chosen'], str):
    #             print(f"chosen 长度: {len(example['chosen'])}")
    #             print(f"chosen 前100字符: {example['chosen'][:100]}...")
    #
    #         if isinstance(example['rejected'], str):
    #             print(f"rejected 长度: {len(example['rejected'])}")
    #             print(f"rejected 前100字符: {example['rejected'][:100]}...")
    #
    #
    # train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
    # # # # 抽取前 10 条验证数据
    # eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
    # # 在应用 template 后调用
    # train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    # eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    #
    # debug_dataset(train_dataset)
    # debug_dataset(eval_dataset)

    # Apply chat template (ensure output is string)
    def apply_chat_template(example, tokenizer):
        chosen = example["chosen"]
        rejected = example["rejected"]
        # If already strings, return as-is; if messages, apply template
        if isinstance(chosen, list):
            chosen = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        if isinstance(rejected, list):
            rejected = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        return {"chosen": chosen, "rejected": rejected}

    #
    # train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
    # # 抽取前 10 条验证数据
    # eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))

    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})


    # Add LoRA via Unsloth (external to ORPOTrainer)
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.05,  # ✅ 小 dropout 防过拟合
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407
    )
    max_length = 1088  # 与加载模型时保持一致
    max_prompt_length = 1024  # 确保 max_prompt_length < max_length
    max_completion_length = 64  # 或设为 max_length - max_prompt_length


    training_args = ORPOConfig(
        output_dir=args.output_path,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=2000,
        report_to="tensorboard",
        disable_dropout=True,
        # 关键修正：确保长度设置合理
        max_length=max_length,  # 总长度
        max_prompt_length=max_prompt_length,  # 提示词长度
        max_completion_length=max_completion_length,  # 生成长度

        dataloader_pin_memory=False,  # 减少 CPU-GPU 数据传输开销
        dataloader_num_workers=0,

        beta=0.1,
        # 训练参数
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-6,  # 降低学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # 添加梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # 梯度裁剪，防止梯度爆炸
        max_grad_norm=1.0,

        save_strategy="steps",
        save_total_limit=6,
        metric_for_best_model="eval_rewards/margins",
    )

    print("✅ Model, tokenizer, and config ready for ORPO!")

    # ✅ CORRECTED: Use ORPOTrainer with processing_class
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,  # ← Correct parameter name
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Note: No ref_model, no data_collator needed (uses default)
    )

    trainer.train()

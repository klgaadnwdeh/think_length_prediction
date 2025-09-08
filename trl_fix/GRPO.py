# train_grpo.py
import re
import argparse
import numpy as np
from datasets import load_from_disk

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
def reward(prompts,completions, labels, **kwargs):
    """
    Args:e
        label: Tensor 或其他形式的真实标签
        completions: list[Tensor]，模型生成的文本列表，每个元素是一个位于 GPU 上的张量

    Returns:
        Tensor: 每个 completion 对应的奖励值
    """
    def clean_data(data):
        cleaned = []
        for item in data:
            try:
                # 尝试将元素转换为浮点数
                num = float(item)
                cleaned.append(num)
            except (ValueError, TypeError):
                if isinstance(item, str):  # 确保 item 是字符串类型
                    # 使用正则表达式查找所有连续的数字
                    # numbers = re.findall(r'\d+', item)diyige
                    numbers = re.findall(r'-?\d+\.?\d*', item)
                    if numbers:
                        # 假设我们只对第一个找到的数字感兴趣，并将其转换为浮点数
                        num = float(numbers[0])
                        cleaned.append(num)
                    else:
                        # 如果没有找到任何数字，则添加 NaN
                        cleaned.append(float('nan'))
                else:
                    # 如果既不是数值类型也不是字符串类型，则添加 NaN
                    cleaned.append(float('nan'))
        return cleaned

    # 清理并处理预测值
    # 首先将每个张量从 GPU 移动到 CPU 并转换为 NumPy 数组

    preds_list = [completion[0]["content"] for completion in
                  completions]
    # 如果 preds_list 中有非数值类型的数据，则需要进一步处理
    preds_cleaned = clean_data(preds_list)
    preds = np.nan_to_num(preds_cleaned, nan=0.0)
    preds = np.array(preds, dtype=np.float64)
    # 处理 labels，确保其与 preds 长度一致
    label = [label for label in
             labels]
    label = clean_data(label)
    labels = np.array(label, dtype=np.float64)
    # 计算奖励
    rewards = [
        12 if r == a else
        6 if abs(r - a) <= 1 else
        3 if abs(r - a) <= 2 else
        0.0
        for r, a in zip(labels, preds)
    ]
    #第一次
    # rewards = [
    #         5 if abs(r - a) <=0 else
    #         4 if abs(r - a) <=0.5 else
    #         3 if abs(r - a) <= 1 else
    #         2 if abs(r - a) <= 1.5 else
    #         1 if abs(r - a) <= 2 else
    #         0.0
    #         for r, a in zip(labels, preds)]#disange
    # 第三次
    # rewards = [
    #     10 if abs(r - a) <= 0 else
    #     8 if abs(r - a) <= 0.5 else
    #     6 if abs(r - a) <= 1 else
    #     4 if abs(r - a) <= 1.5 else
    #     2 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)]第二个
    print(rewards)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="", default="../model/think/data1/Qwen2-0.5B-Instruct")
    parser.add_argument('--train_data_path', type=str, default="..//data//think//data1//train_data")
    parser.add_argument('--valid_data_path', type=str, default="..//data//think//data1//valid_data")
    parser.add_argument('--logging_steps', type=int, help="the logging frequency.", default=10)
    parser.add_argument('--save_steps', type=int, help="the saving frequency.", default=26000)
    parser.add_argument('--num_train_epochs', type=int, help="The number of training epochs for the reward model.",
                        default=9)
    parser.add_argument("--max_completion_length", type=int,
                        help="Maximum length of the completion. This argument is required if you want to use the default data "
                             "collator and your model is an encoder-decoder.", default=20)

    parser.add_argument('--learning_rate', type=float, help="The initial learning rate for [`AdamW`] optimizer.",default=5e-6)
    parser.add_argument('--adam_beta1', type=float, help="The beta1 hyperparameter for the [`AdamW`] optimizer.",default=0.9)
    parser.add_argument('--adam_beta2', type=float, help="The beta2 hyperparameter for the [`AdamW`] optimizer.",default=0.99)
    parser.add_argument('--weight_decay', type=float, help="Weight decay for AdamW if we apply some.",default=0.1)
    parser.add_argument('--warmup_ratio', type=float, help="Linear warmup over warmup_ratio fraction of total steps.", default=0.1)
    parser.add_argument('--per_device_train_batch_size', type=int, help= "Batch size per device accelerator core/CPU for training.",
                        default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass", default=1)
    parser.add_argument('--max_grad_norm', type=int,help="Maximum gradient norm (for gradient clipping).",default=0.1)
    parser.add_argument('--max_prompt_length', type=int,
                        help="Maximum length of the completion. This argument is required if you want to use the default data "
                             "collator and your model is an encoder-decoder.", default=1024)
    parser.add_argument('--output_path', type=str, help="Output path for the trained model", default="../model/think//data1/GRPO")
    args = parser.parse_args()


    dataset_1 = load_from_disk(args.train_data_path)
    train_dataset = dataset_1.map(lambda x: {
        "prompt": [
            {"role": "system", "content": "完全按照用户的输入来解决用户的问题"},
            {"role": "user", "content":x["prompt"]}
        ],
        "answer":x["class_name"]
    })
    dataset_2=load_from_disk(args.valid_data_path)
    eval_dataset = dataset_2.map(lambda x: {
        "prompt": [
            {"role": "system", "content": "完全按照用户的输入来解决用户的问题"},
            {"role": "user", "content": x["prompt"]}
        ],
        "answer":x["class_name"]
    })
    training_args = GRPOConfig(output_dir=args.output_path,
                               logging_steps=args.logging_steps,
                               learning_rate=args.learning_rate,
                               adam_beta1=args.adam_beta1,
                               adam_beta2=args.adam_beta2,
                               weight_decay=args.weight_decay,
                               warmup_ratio=args.warmup_ratio,
                               lr_scheduler_type="cosine",
                               num_train_epochs=args.num_train_epochs,
                               max_completion_length=args.max_completion_length,
                               max_grad_norm = args.max_grad_norm,
                               per_device_train_batch_size=args.per_device_train_batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               max_prompt_length=args.max_prompt_length,
                               save_steps=args.save_steps,
                               # per_device_train_batch_size=
                               include_for_metrics=["loss",]
                               ,report_to="tensorboard"
                               )
    trainer = GRPOTrainer(
        model=args.output_path,
        reward_funcs=reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)



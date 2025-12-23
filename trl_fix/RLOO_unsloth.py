# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import os
#
# from unsloth import FastLanguageModel
from unsloth import FastLanguageModel

from trl import RLOOConfig, RLOOTrainer

# train_grpo.py


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
import re

import numpy as np
import torch

def reward(prompts,completions,ground_truth, **kwargs):
    """
    script_args:
        ground_truth: Tensor 或其他形式的真实标签
        completions: list[Tensor]，模型生成的文本列表，每个元素是一个位于 GPU 上的张量

    Returns:
        Tensor: 每个 completion 对应的奖励值
    """

    def clean_data(data):
        cleaned = []
        for item in data:
            try:
                num = float(item)
                cleaned.append(num)
            except (ValueError, TypeError):
                if isinstance(item, str):
                    numbers = re.findall(r'\d+', item)
                    if numbers:
                        num = float(numbers[0])
                        cleaned.append(num)
                    else:
                        cleaned.append(float('nan'))
                else:
                    cleaned.append(float('nan'))
        return cleaned

    preds_list = [completion.cpu().numpy() if isinstance(completion, torch.Tensor) else completion for completion in
                  completions]
    preds_cleaned = clean_data(preds_list)
    preds = np.nan_to_num(preds_cleaned, nan=0.0)
    preds = np.array(preds, dtype=np.float64)
    label = [label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in ground_truth]
    label = clean_data(label)
    ground_truth = np.array(label, dtype=np.float64)
    rewards = [
        12 if r == a else
        6 if abs(r - a) <= 1 else
        3 if abs(r - a) <= 2 else
        0.0
        for r, a in zip(ground_truth, preds)
    ]#第一次实验
    # rewards = [
    #     5 if abs(r - a) <= 0 else
    #     4 if abs(r - a) <= 0.5 else
    #     3 if abs(r - a) <= 1 else
    #     2 if abs(r - a) <= 1.5 else
    #     1 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(ground_truth, preds)]  # 第三次实验
    # rewards = [
    #     16 if abs(r - a) < 0 else
    #     12 if abs(r - a) < 0.5 else
    #     8 if abs(r - a) <= 1 else
    #     4 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(ground_truth, preds)
    # ]第二次实验
    print(rewards)
    return torch.tensor(rewards, dtype=torch.float)

#8,8,8jike
#bazhegshuoqingchu,geilunwenlimianshiqoingch
#1,16,5
# def data_solution(example):
#     example["prompt"]=prompt_think.format(question=example["prompt"])
#     return example
def data_solution(example):
    # 原始数据包含 "prompt" 和 "class_sort"
    # 将 "class_sort" 作为 "ground_truth"（即RLOO的标签）
    return {
        "prompt": prompt_think.format(question=example["prompt"]),  # 保持原始prompt
        "ground_truth": example["class_name"]  # 将class_sort作为ground_truth
    }

if __name__ == "__main__":
    # def clear_unsloth_cache():
    #     """清除 Unsloth 的编译缓存"""
    #     cache_locations = [
    #         "/home/ruxin/trl_1/examples/scripts/rloo/unsloth_compiled_cache",
    #         os.path.join(os.path.dirname(__file__), "unsloth_compiled_cache"),
    #         "/tmp/unsloth_cache",
    #         os.path.expanduser("~/.cache/unsloth"),
    #     ]
    #
    #     for cache_dir in cache_locations:
    #         if os.path.exists(cache_dir):
    #             try:
    #                 shutil.rmtree(cache_dir)
    #                 print(f"✅ 已清除缓存: {cache_dir}")
    #             except:
    #                 pass
    #
    #     # 设置环境变量防止缓存问题
    #     os.environ["UNSLOTH_COMPILE_CACHE"] = "0"
    #     os.environ["UNSLOTH_FORCE_RECOMPILE"] = "1"
    #     os.environ["UNSLOTH_JIT_COMPILE"] = "0"
    #
    #
    # clear_unsloth_cache()
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r"../model/think/..")
    parser.add_argument('--train_data_path', type=str, default=r"../data/think/data1/train_data")
    parser.add_argument('--valid_data_path', type=str, default=r"../data/think/data1/valid_data")
    parser.add_argument('--logging_steps', type=int, help="the logging frequency.", default=10)
    parser.add_argument('--save_steps', type=int, help="the saving frequency.", default=26000)
    parser.add_argument('--learning_rate', type=float, help="The initial learning rate for [`AdamW`] optimizer.",
                        default=3e-6)
    parser.add_argument('--num_generations', type=int,
                        help="Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
                             "* gradient_accumulation_steps) must be evenly divisible by this value.", default=2)
    parser.add_argument('--num_iterations', type=int, help="Number of iterations per batch.",
                        default=1)  # 原num_ppo_epochs
    parser.add_argument('--steps_per_generation', type=int, help="Number of steps per generation.",
                        default=1)  # 原num_mini_batches
    parser.add_argument('--beta', type=float, help="KL coefficient.", default=0.03)  # 原kl_coef
    parser.add_argument('--epsilon', type=float, help="Clipping range for PPO.", default=0.2)  # 原cliprange
    parser.add_argument('--normalize_advantages', type=bool, help="Whether to normalize advantages.",
                        default=True)  # 原normalize_reward
    parser.add_argument('--exp_name', type=str, help="Experiment name.", default="rloo_experiment")
    parser.add_argument('--seed', type=int, help="Random seed.", default=42)
    parser.add_argument('--max_completion_length', type=int, help="Maximum length of completions.",
                        default=256)  # 原response_length

    parser.add_argument('--per_device_train_batch_size', type=int,
                        help="Batch size per device accelerator core/CPU for training.",
                        default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass",
                        default=8)
    parser.add_argument("--total_episodes", type=int,
                        help="Total number of episodes in the dataset.",
                        default=35000)
    parser.add_argument('--output_path', type=str, help="Output path for the trained model",
                        default=r"../model/think/RLOO/data1/model")

    # 新增的RLOOConfig参数（如果需要）
    parser.add_argument('--temperature', type=float, help="Temperature for sampling.", default=1.0)
    parser.add_argument('--top_p', type=float, help="Top-p sampling parameter.", default=1.0)
    parser.add_argument('--repetition_penalty', type=float, help="Repetition penalty for generation.", default=1.0)
    parser.add_argument('--max_prompt_length', type=int, help="Maximum length of the prompt.", default=512)
    parser.add_argument('--beta_start', type=float, help="Start value for beta annealing.", default=0.0)  # 如果需要
    parser.add_argument('--beta_end', type=float, help="End value for beta annealing.", default=0.1)  # 如果需要
    parser.add_argument('--epsilon_high', type=float, help="Upper-bound epsilon value for clipping.",
                        default=None)  # 如果需要
    parser.add_argument('--reward_clip_min', type=float, help="Minimum reward clipping value.", default=None)  # 如果需要
    parser.add_argument('--reward_clip_max', type=float, help="Maximum reward clipping value.", default=None)  # 如果需要

    script_args = parser.parse_args()

    # 计算max_steps
    max_steps = script_args.total_episodes // (
                script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps)

    # 创建run_name
    run_name = f"{script_args.exp_name}__{script_args.seed}__{int(time.time())}"

    # 准备reward_clip_range（如果需要）
    reward_clip_range = None
    if script_args.reward_clip_min is not None and script_args.reward_clip_max is not None:
        reward_clip_range = (script_args.reward_clip_min, script_args.reward_clip_max)

    training_args = RLOOConfig(
        # 基本训练参数
        output_dir=script_args.output_path,
        run_name=run_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        num_train_epochs=6,  # 可以随意更改
        max_steps=max_steps,

        # 保存和日志参数
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        report_to="tensorboard",
        include_for_metrics=["loss", ],

        # RLOO特定参数
        num_generations=script_args.num_generations,
        num_iterations=script_args.num_iterations,
        steps_per_generation=script_args.steps_per_generation,
        beta=script_args.beta,
        epsilon=script_args.epsilon,
        normalize_advantages=script_args.normalize_advantages,

        # 生成长度参数
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length,

        # vLLM参数
        # use_vllm=True,
        # vllm_mode="colocate",

        # 生成参数
        temperature=script_args.temperature,
        top_p=script_args.top_p,
        repetition_penalty=script_args.repetition_penalty,

        # 可选的附加参数
        epsilon_high=script_args.epsilon_high,  # 如果设置了epsilon_high
        reward_clip_range=reward_clip_range,  # 如果设置了reward_clip_range

        # 其他可能有用的参数（使用默认值或根据需要调整）
        remove_unused_columns=False,
        shuffle_dataset=True,
        disable_dropout=False,
        ds3_gather_for_generation=True,
        generation_batch_size=None,  # 自动设置为有效训练批次大小
        use_transformers_paged=False,
        cache_implementation=None,
        # vllm_model_impl="vllm",
        # vllm_gpu_memory_utilization=0.3,
        # vllm_tensor_parallel_size=1,
        # vllm_enable_sleep_mode=False,
        mask_truncated_completions=False,
        sync_ref_model=False,
        ref_model_mixup_alpha=0.6,
        ref_model_sync_steps=512,
        log_completions=False,
        num_completions_to_print=None,
        wandb_log_unique_prompts=False,
    )

    print(script_args.model_path)

    from datasets import load_from_disk
    # # Dataset
    train_dataset = load_from_disk(script_args.train_data_path).map(data_solution)
    eval_dataset = load_from_disk(script_args.valid_data_path).map(data_solution)


    train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= script_args.model_path,
        max_seq_length=1088,
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
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

    # 确保 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确保 model_path 是你本地模型文件夹的路径

    trainer = RLOOTrainer(
        processing_class=tokenizer,
        model=model,
        reward_funcs=[reward,],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.train()

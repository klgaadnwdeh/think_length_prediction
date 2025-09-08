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

import shutil
import re
import argparse
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

"""
python -i examples/scripts/rloo/rloo_1.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/single_gpu.yaml \ GRPO.py
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/rloo/rloo_1.py
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero4.yaml \
    examples/scripts/rloo/rloo_1.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/rloo \
    --rloo_k 2 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""

import numpy as np
import re

import numpy as np
import re

# train_grpo.py
import re

import numpy as np
from datasets import load_from_disk
import torch

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
def data_solution(example):
    example["prompt"]=prompt_think.format(question=example["prompt"])
    return example

def reward(labels, completions):
    """
    Args:
        labels: Tensor 或其他形式的真实标签
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
    label = [label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in labels]
    label = clean_data(label)
    labels = np.array(label, dtype=np.float64)
    rewards = [
        12 if r == a else
        6 if abs(r - a) <= 1 else
        3 if abs(r - a) <= 2 else
        0.0
        for r, a in zip(labels, preds)
    ]#第一次实验
    # rewards = [
    #     5 if abs(r - a) <= 0 else
    #     4 if abs(r - a) <= 0.5 else
    #     3 if abs(r - a) <= 1 else
    #     2 if abs(r - a) <= 1.5 else
    #     1 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)]  # 第三次实验
    # rewards = [
    #     16 if abs(r - a) < 0 else
    #     12 if abs(r - a) < 0.5 else
    #     8 if abs(r - a) <= 1 else
    #     4 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)
    # ]第二次实验
    print(rewards)
    return torch.tensor(rewards, dtype=torch.float)

#8,8,8jike
#bazhegshuoqingchu,geilunwenlimianshiqoingch
#1,16,5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="", default="../model/think/data1/Qwen2-0.5B-Instruct")
    parser.add_argument('--train_data_path', type=str, default="..//data//think//data1//train_data")
    parser.add_argument('--valid_data_path', type=str, default="..//data//think//data1//valid_data")
    parser.add_argument('--logging_steps', type=int, help="the logging frequency.", default=10)
    parser.add_argument('--save_steps', type=int, help="the saving frequency.", default=26000)
    parser.add_argument('--learning_rate', type=float, help="The initial learning rate for [`AdamW`] optimizer.",
                        default=3e-6)
    parser.add_argument('--rloo_k', type=int, help="REINFORCE Leave-One-Out (RLOO) number of online samples per prompt.",default=2)
    parser.add_argument('--num_ppo_epochs', type=int,help="Number of epochs to train",default=1)
    parser.add_argument('--num_mini_batches', type=int,help="Number of minibatches to split a batch into.",default=1)
    parser.add_argument('--local_rollout_forward_batch_size', type=int, help=" Per rank no grad forward pass in the rollout phase.", default=8)
    parser.add_argument('--missing_eos_penalty', type=float, help="Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to"
                                                                                "generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive"
                                                                                "value", default=1.0)
    parser.add_argument('--kl_coef', type=float, help="KL coefficient.",
                        default=0.03)

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
                        default="../model/think//data1/RLOO")
    args = parser.parse_args()
    training_args = RLOOConfig(save_steps=args.save_steps,
                               output_dir=args.output_path,
                               rloo_k=args.rloo_k,
                               num_ppo_epochs=args.num_ppo_epochs,
                               num_mini_batches=args.num_mini_batches,
                               learning_rate=args.learning_rate,
                               per_device_train_batch_size=args.per_device_train_batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,
                               missing_eos_penalty=args.missing_eos_penalty,
                               stop_token="eos",
                               kl_coef=args.kl_coef,
                               include_for_metrics=["loss",]
                               , report_to="tensorboard",
                               total_episodes=args.total_episodes,
                               )
    # remove output_dir if exists

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    ################\data1\predic_think_class\train_data_class
    # Dataset
    train_dataset = load_from_disk(args.train_data_path).map(data_solution)
    eval_dataset = load_from_disk(args.valid_data_path).map(data_solution)
    ################
    dataset_text_field = "prompt"


    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            label = int(element["class_name"])
            return {"input_ids": outputs["input_ids"], "class_sort": label}

        return dataset.map(
            tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    # Compute that only on the main process for faster data processing.

    ################
    # Training
    ################

    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

    )
    trainer.train()

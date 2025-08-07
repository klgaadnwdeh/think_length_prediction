import os
import re
import argparse
import datasets
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import DataCollatorWithPadding, AutoTokenizer
from vllm import LLM, SamplingParams

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
state_file = "./record.txt"
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        pass
else:
    with open(state_file, 'w') as f:
        pass


def load_llm_model(model_path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.98,
                   max_model_len=2000, device_config="cuda:0"):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="auto", tensor_parallel_size=tensor_parallel_size,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              device=device_config,
              seed=42,
              swap_space=16,
              )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer


def generate_dataloaders(dataset):
    validation_dataset = dataset
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,  # 验证集通常不需要 shuffle
        batch_size=1,
    )
    return validation_dataloader


def process_batch_results(sub_batch, batch_responses):
    """将批次结果汇总到 results 列表中，包含真实标签和预测标签（转换为类别）"""
    results = []
    # 假设 sub_batch 中的每一项是一个字典，包含 "prompt" 和 "labels" 字段
    # 假设 batch_responses 是模型生成的结果列表，包含标签（例如字母标签，如 'A' 到 'P'，或者其他标签）

    for row, generated_text in zip(sub_batch, batch_responses):
        # 转换预测标签（字母）为类别（数字），如果不是 'A' 到 'P'，则归为 0 类

        # 创建结果字典，只包含真实标签和预测标签
        result = {
            "true_label": row[0]["label"],  # 真实标签（字母）
            "predicted_label": generated_text  # 预测标签（数字，0-15 或者 0 类）
        }
        # 将结果添加到汇总列表中
        results.append(result)
    return results


def assemble_batch(data_list):
    batch = []
    for row in data_list:
        question = row["prompt"]
        full_content = prompt.format(question=question)
        batch.append((row, full_content))
    return batch


def process_dataloader(dataloader):
    """
    将 DataLoader 转换为一个包含 prompt 和 labels 的列表。
    """
    data_list = []
    for batch in dataloader:
        # 假设每个 batch 是一个字典，包含 'prompt' 和 'labels' 键
        prompt = prompt_think.format(question=batch["question"][0])  # 因为 batch_size=1，取第一个元素
        labels = batch["step_count"][0].item()
        data_list.append({"prompt": prompt, "label": labels})
    return data_list


def calculate_metrics(preds, labels):
    # 确保 preds 和 labels 是 numpy 数组
    def clean_data(data):
        cleaned = []
        for item in data:
            try:
                # 如果元素已经是数值类型，则直接添加
                num = float(item)
                cleaned.append(num)
            except (ValueError, TypeError):
                # 如果不是数值类型，尝试提取其中的数字
                if isinstance(item, str):  # 确保 item 是字符串类型
                    # 使用正则表达式查找所有连续的数字
                    numbers = re.findall(r'\d+', item)
                    if numbers:
                        # 假设我们只对第一个找到的数字感兴趣，并将其转换为浮点数
                        num = float(numbers[0])
                        cleaned.append(num)
                    else:
                        # 如果没有找到任何数字，则添加 NaN
                        cleaned.append(np.nan)
                else:
                    # 如果既不是数值类型也不是字符串类型，则添加 NaN
                    cleaned.append(np.nan)
        return cleaned

    preds = clean_data(preds)
    # 把这个空值转换为nan
    preds = np.nan_to_num(preds, nan=0.0)
    # 数据转换为numpy
    preds = np.array(preds, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    # Calculate strict accuracy
    strict_accuracy = ((np.abs(preds - labels) <= 25).sum()) / len(labels)

    # Calculate accuracy with tolerance
    accuracy_with_tolerance = ((np.abs(preds - labels) <= 50).sum()) / len(labels)
    return strict_accuracy,accuracy_with_tolerance

def find_text_discrepancies(list1, batch_size=100000):
    """使用批处理找出事件间的差异。"""
    results = []
    generate_result = []
    print("开始处理...")
    # 第一层批处理
    for i in tqdm(range(0, len(list1), batch_size), desc='Processing', ncols=100):
        generate_result = []
        temp_list = list1[i:i + batch_size]
        batch = assemble_batch(temp_list)
        batch_lengths = [len(content) for _, content in batch]
        min_length = min(batch_lengths)
        max_length = max(batch_lengths)
        print(f"批次内容长度最小值: {min_length}, 最大值: {max_length}")
        for j in range(0, len(batch), 3000):
            torch.cuda.empty_cache()
            sub_batch = batch[j:j + 3000]
            test = []
            batch_responses = []
            for _, content in sub_batch:
                prompt = [{"role": "user", "content": f"{content}"}]
                inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(inputs, add_special_tokens=False)
                test.append(prompt_ids)
            batch_responses = []
            outputs = llm.generate(prompt_token_ids=test, sampling_params=sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                batch_responses.append(generated_text)
            del output, outputs
            torch.cuda.empty_cache()
            processed_results = process_batch_results(sub_batch, batch_responses)
            results.extend(processed_results)
            print(len(results))
        # 提取 true_label 和 predicted_label 列表
    true_labels = [result["true_label"] for result in results]
    predicted_labels = [result["predicted_label"] for result in results]
    # 计算准确率
    accuracy = calculate_metrics(predicted_labels, true_labels)
    with open("../record.txt", "w") as f:
        f.write(f"严格准确率: {accuracy[0]:.4f}")
        f.write('/n')
        f.write(f"容忍度准确率: {accuracy[1]:.4f}")  # 输出结果
    print(f"严格准确率: {accuracy[0]:.4f}")
    print(f"容忍度准确率: {accuracy[1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='', default="..//model//think//")
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1],2 for dataset[2]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='', default="../data/think")
    parser.add_argument('--temperature', type=float, default=0.2,help='Sampling temperature, higher means more random')
    parser.add_argument('--top_p', type=float, default=0.7,help='Top-p (nucleus) sampling probability')
    parser.add_argument('--max_tokens', type=int, default=512,help='Maximum number of tokens to generate')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,help='')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0,help='')
    parser.add_argument('--max_model_len', type=int, default=2048,help='')
    parser.add_argument('--device_config', type=int, default=str,help='cuda')
    args = parser.parse_args()
    dataset = ["sequelbox/Titanium2.1-DeepSeek-R1", "Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B",
               "mlfoundations-dev/AM-DeepSeek-R1-Distilled-1.4M-am_0.5M"]
    choice=args.choice
    if choice == 0:
        data_path = "data1"
    elif choice == 1:
        data_path = "data2"
    else:
        data_path = "data3"
    prefix_path = args.data_prefix_path
    valid_dataset_path = os.path.join(prefix_path, data_path, "valid_dataset")
    llm_model_path = args.model_path
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm, tokenizer = load_llm_model(llm_model_path, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True,
                                    gpu_memory_utilization=args.gpu_memory_utilization,
                                    max_model_len=args.max_model_len, device_config=args.device_config)
    dataset = datasets.load_from_disk(valid_dataset_path)
    validation_dataloader = generate_dataloaders(dataset)
    data_list = process_dataloader(validation_dataloader)
    find_text_discrepancies(data_list)

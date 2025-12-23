import os
import re

import datasets
import numpy as np
import argparse
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import DataCollatorWithPadding, AutoTokenizer
from vllm import LLM, SamplingParams

prompt = """
下面有一道来自deepseek-R1的问题，你的任务是,考虑问题的深度，难度，还有复杂度，考虑下面的问题总共要花几个步骤来解决即可，但是不需要给出解决内容，只给出大致的步骤总数即可。
question:{question}
"""
# prompt="""
# 下面有一道来自deepseek-R1的问题，你的任务是,考虑问题的深度，难度，还有复杂度，考虑下面的问题总共要花几个步骤来解决即可，但是不需要给出解决内容，只给出大致的步骤总数即可。已知问题步骤数目不低于1步，不高于11步。
# question:{question}
# """
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
        prompt = batch["question"][0]  # 因为 batch_size=1，取第一个元素
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

    preds_cleaned = clean_data(preds)
    labels = np.array(labels, dtype=np.float64)
    
    # 创建有效样本的掩码：预测值不是NaN
    valid_mask = ~np.isnan(preds_cleaned)
    
    if not valid_mask.any():
        return 0.0, 0.0 # 或返回其他指示符
    
    valid_preds = np.array(preds_cleaned)[valid_mask]
    valid_labels = labels[valid_mask]
    
    strict_acc = (valid_preds == valid_labels).mean()
    tolerant_acc = (np.abs(valid_preds - valid_labels) <= 1).mean()
    
    # 可选：打印或记录无效样本的数量以供调试
    invalid_count = len(preds) - valid_mask.sum()
    if invalid_count > 0:
        print(f"警告：有 {invalid_count} 个预测值无法解析，已从准确率计算中排除。")
    
    return strict_acc, tolerant_acc


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
    with open("./record.txt", "w") as f:
        f.write(f"严格准确率: {accuracy[0]:.4f}")
        f.write('/n')
        f.write(f"容忍度准确率: {accuracy[1]:.4f}")  # 输出结果
    print(f"严格准确率: {accuracy[0]:.4f}")
    print(f"容忍度准确率: {accuracy[1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='', default="..//model//step//")
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='', default="../data/step")
    parser.add_argument('--temperature', type=float, default=0.2,help='Sampling temperature, higher means more random')
    parser.add_argument('--top_p', type=float, default=0.7,help='Top-p (nucleus) sampling probability')
    parser.add_argument('--max_tokens', type=int, default=100,help='Maximum number of tokens to generate')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,help='')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85,help='')
    parser.add_argument('--max_model_len', type=int, default=600,help='')
    # 在 parser.add_argument 部分，修正 device_config 的定义
    parser.add_argument('--device_config', type=str, default='cuda:0', help='CUDA device string')
    args = parser.parse_args()
    choice=args.choice
    dataset = ["open-r1/OpenThoughts-114k-math", "Phsoft-ai/alpaca-gpt4-CoT"]
    data_path = dataset[choice]
    tmp_path = data_path.split("/")[-1]
    prefix_path = args.data_prefix_path
    valid_dataset_path = os.path.join(prefix_path, tmp_path, "valid_dataset")
    llm_model_path = args.model_path
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm, tokenizer = load_llm_model(llm_model_path, tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True,
                                    gpu_memory_utilization=args.gpu_memory_utilization,
                                    max_model_len=args.max_model_len, device_config=args.device_config)
    dataset = datasets.load_from_disk(valid_dataset_path)
    validation_dataloader = generate_dataloaders(dataset)
    data_list = process_dataloader(validation_dataloader)
    find_text_discrepancies(data_list)

import os
import re
from pathlib import Path

import datasets
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, AutoTokenizer
from vllm import LLM, SamplingParams

Prompt = """
Task Description:


Instruction:
{content}

Requirements:

Do not generate the actual response.
Estimate the str count needed for the response.
Based on your estimation, output a single interger
Only output the single integer ; do not include any additional text or explanation.
Ensure your estimation is as accurate as possible.
"""
prompt_think = """
人得到一个问题，首先是根据问题的深度，难度，复杂度，慢慢地思考问题应该如何解决。在思考过程中运用自己的各种能力，对于问题进行逐步逐步推导解决，在一步一步地推导解决过程中得到结果。得到结果之后，人停止思考并且整合思考过程中的内容，进行大量简化后，得到问题的整个解决办法和结果，最后输出问题的解决办法和结果。
任务描述：
question:{question}
你的任务是考虑问题在人思考过程中所花费的令牌数量的范围。具体根据问题可能涉及的深度、复杂性和长度，估算出思考过程总共花费的令牌数量的范围。
例如:
1估计的令牌数量范围在200，则真实思考过程中所花费的令牌数量基本在(200*25,201*25-1)
2估计的令牌数量范围在320，则真实思考过程中所花费的令牌数量基本在(320*25,321*25-1)
要求：仅输出估计的令牌数量范围，这个范围只能是数学，不能是其他的格式,也不生成其他的内容。
"""


def load_llm_model(model_path, tensor_parallel_size=1, enforce_eager=False, gpu_memory_utilization=0.85,max_model_len=2000,device_config="cuda:0"):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=tensor_parallel_size, enforce_eager=enforce_eager, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              seed=42,
              swap_space = 16,
              quantization=None,
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


def extract_number(text, default=0.0):
    """
    从字符串中提取第一个数字（整数或浮点数，支持负号）
    如果失败，返回 default（而不是 NaN）
    """
    if not isinstance(text, str):
        try:
            return float(text)
        except (TypeError, ValueError):
            return default

    # 更鲁棒的正则：匹配 -12.5, .5, 100, 3. 等
    match = re.search(r'-?\d*\.?\d+', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return default


def process_batch_results(sub_batch, batch_responses, nan_replacement=0.0):
    """
    处理一个子批次，返回清洗后的结果列表。
    - 从 generated_text 中提取预测数字
    - 从 row["labels"] 中提取真实标签数字
    - 所有无法解析的值统一替换为 nan_replacement（如 0.0）
    - 返回的每个 result 中 true_label 和 predicted_label 都是 float，无 NaN
    """
    results = []
    for row, generated_text in zip(sub_batch, batch_responses):
        # 提取真实标签（row["labels"] 可能是字符串 "5" 或数字 5）
        # true_val = extract_number(row["labels"], default=nan_replacement)
        original_row = row[0]  # row 是 (dict, prompt)，取第0个元素
        true_val = extract_number(original_row["labels"], default=nan_replacement)
        # 提取预测值（从模型生成的文本中找数字）
        pred_val = extract_number(generated_text, default=nan_replacement)

        results.append({
            "true_label": true_val,
            "predicted_label": pred_val
        })

    return results

def assemble_batch(data_list):
    batch = []
    for row in data_list:
        question = row["prompt"]
        # label = row["labels"]
        # full_content = prompt_think.format(content=question)
        batch.append((row, question))
    return batch

def process_dataloader(dataloader):
    """
    将 DataLoader 转换为一个包含 prompt 和 labels 的列表。
    """
    data_list = []
    for batch in dataloader:
        # 假设每个 batch 是一个字典，包含 'prompt' 和 'labels' 键
        prompt = batch["prompt"][0]  # 因为 batch_size=data，取第一个元素
        # print(prompt)
        labels = batch["class_name"][0]
        # print(labels)
        data_list.append({"prompt": prompt, "labels": labels})
    return data_list


def calculate_metrics(preds, labels,limit=0.5):
    # 确保 preds 和 labels 是 numpy 数组
    preds = np.array(preds)
    labels = np.array(labels)

    # Calculate strict accuracy
    # strict_accuracy = (preds == labels).sum() / len(labels)

    # Calculate accuracy with tolerance
    accuracy_with_tolerance = ((np.abs(preds - labels) <= limit).sum()) / len(labels)
    accuracy_with_tolerance_1 = ((np.abs(preds - labels) <= (limit+1)).sum()) / len(labels)

    return accuracy_with_tolerance, accuracy_with_tolerance_1

def find_text_discrepancies(list1,llm_model_path, batch_size=100000, limit=0):
    """使用批处理找出事件间的差异。"""

    path_parts = Path(llm_model_path).parts  # 转为路径部件元组
    try:
        total_index = path_parts.index("total")
        exp1 = path_parts[total_index + 1]  # e.g., "3"
        exp2 = path_parts[total_index + 2]  # e.g., "1"
        model_name = path_parts[total_index + 3]  # e.g., "GRPOQwen2-0.5B-GRPO_2"
    except (ValueError, IndexError):
        # 如果路径不符合预期，回退到默认命名
        print("⚠️ 路径格式不符合预期，使用默认输出名")
        exp1 = exp2 = "default"
        model_name = Path(llm_model_path).name

        # ✅ 构造输出目录：../data/think/record_{limit}_test/{exp1}/{exp2}
    output_base_dir = f"../data/think/record_{limit}_test"
    output_subdir = os.path.join(output_base_dir, exp1, exp2)
    os.makedirs(output_subdir, exist_ok=True)

    # ✅ 输出文件名：record_{limit}_{model_name}.txt
    output_filename = f"record_{limit}_{model_name}.txt"
    output_path = os.path.join(output_subdir, output_filename)
    # ✅ 确保输出目录存在
    # output_dir = f"../data/think/record_{limit}_test"
    # os.makedirs(output_dir, exist_ok=True)

    results = []
    print("开始处理...")

    # 第一层批处理
    for i in tqdm(range(0, len(list1), batch_size), desc='Processing', ncols=100):
        temp_list = list1[i:i + batch_size]
        batch = assemble_batch(temp_list)
        batch_lengths = [len(content) for _, content in batch]
        min_length = min(batch_lengths)
        max_length = max(batch_lengths)
        print(f"批次内容长度最小值: {min_length}, 最大值: {max_length}")

        # 第二层小批次（避免单次生成太多）
        for j in range(0, len(batch), 3000):
            sub_batch = batch[j:j + 3000]
            try:
                # ✅ 构建字符串 prompts（不再用 prompt_token_ids）
                prompts = []
                for _, content in sub_batch:
                    prompt = [{"role": "user", "content": f"{content[:1004]}"}]
                    # 转为字符串格式（apply_chat_template + tokenize=False）
                    input_str = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(input_str)

                # ✅ 使用 prompts（字符串列表）调用 generate
                outputs = llm.generate(
                    prompts,  # ← 关键：传字符串，不是 token IDs
                    sampling_params=sampling_params
                )

                batch_responses = []
                for output in outputs:
                    generated_text = output.outputs[0].text
                    print(generated_text)
                    batch_responses.append(generated_text)

                processed_results = process_batch_results(sub_batch, batch_responses)
                results.extend(processed_results)
                print(f"累计结果数: {len(results)}")

            except Exception as e:
                print(f"处理子批次时出错: {e}")
                continue  # 跳过错误批次，继续运行

    # 提取标签
    true_labels = [result["true_label"] for result in results if "true_label" in result]
    predicted_labels = [result["predicted_label"] for result in results if "predicted_label" in result]

    # 安全计算指标（防止空列表）
    if len(true_labels) == 0:
        strict_accuracy = accuracy_with_tolerance = 0.0
    else:
        strict_accuracy, accuracy_with_tolerance = calculate_metrics(predicted_labels, true_labels, limit=limit)

    # ✅ 写入文件（路径已确保存在）
    # output_path = os.path.join(output_dir, f"record_{limit}_GRPO_3_test")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"严格准确率: {strict_accuracy:.4f}\n")
        f.write(f"{limit * 2}容忍度准确率: {accuracy_with_tolerance:.4f}\n")

    print(f"严格准确率: {strict_accuracy:.4f}")
    print(f"{limit * 2}容忍度准确率: {accuracy_with_tolerance:.4f}")

if __name__ == '__main__':
    limit=[0,0.5,1,2]
    data_path = r"/mnt/d/home/home/science/data/think/data3/1/valid_data_label"
    dataset = datasets.load_from_disk(data_path)
    validation_dataloader = generate_dataloaders(dataset)
    data_list = process_dataloader(validation_dataloader)
    llm_model_path=r"/mnt/f/home_fix/test/orpo/2/model/checkpoint-14166"
    sampling_params = SamplingParams(temperature=0.2 ,top_p=0.7, max_tokens=20)
    llm, tokenizer = load_llm_model(llm_model_path, tensor_parallel_size=1, enforce_eager=False,
                                    gpu_memory_utilization=0.90, max_model_len=1024, device_config="cuda")
    find_text_discrepancies(data_list,llm_model_path=llm_model_path,limit=limit[3])

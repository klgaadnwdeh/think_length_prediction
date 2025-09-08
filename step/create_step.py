from datasets import load_from_disk, Dataset, concatenate_datasets
import os
from vllm import LLM, SamplingParams
import argparse
import json
import numpy as np

prompt7 = """
问题:{problem}/
解决方案:{solution}/
小助手，你不需要回答以上的问题，因为问题已经其他人回答过了，就在解决方案里面，解决方案已经完美解决了问题.
你的任务"是为用户划分解决方案，使得解决方案变得有规律些,由于用户只能看懂1..,2..,3..,..so on,因此按照下面的响应格式输出内容"
<划分格式如下>
第一种情况:响应内部有明确划分
1.1响应被划分有明确步骤的，比如step1,step2,step3,....,step-1,step2-2,...,so on,转为数字1,2,3顺序的词汇,符合响应格式
1.2响应被划分有明确步骤的,比如1.,2.,3.,4.,5.,...,so on,不用更改，符合响应格式
第二种情况响应内部无明确划分
2.1结合自己过去的学习经历，如果捕捉到有明确顺序的词语，将明确顺序的词汇全部转为数字1,2,3顺序的词汇,符合响应格式
响应格式:
<think>
你自己的简要的思考内容
</think>
<output>
解决方案可以被划分为以下步骤:
1. 用几句话总结一下了什么事情(simple describe)
2. 用几句话总结一下了什么事情(simple describe)
3. 用几句话总结一下了什么事情(simple describe)
...,and so on,
</output>
<count_step>
统计解决方案一共被划分为了多少步放在这个标签里面，这个里面只允许写明数字:如1，5，7等，其他不允许回复
</count_step>
约束:
1每一步描述不能超过20个token
2只是划分步数即可，不要生成太多内容，不要啰嗦
"""


# 假设模型名称为'modelname'，这里需要替换为实际的模型名称
def filter_data_by_3std(data: Dataset, column_name: str = 'step_count') -> Dataset:
    values = np.array(data[column_name])

    # 计算时忽略NaN值
    mean = np.nanmean(values)
    std = np.nanstd(values)

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # 处理NaN值：将NaN视为不符合条件
    condition = (values >= lower_bound) & (values <= upper_bound) & (~np.isnan(values))

    filtered_data = data.select(np.where(condition)[0])
    return filtered_data


def step_count(example):
    start = "<count_step>"
    end = "</count_step>"
    length = 0
    try:
        length = example["answer"].split(start)[-1].split(end)[0].strip()
    except Exception as e:
        length = 0
    finally:
        example["step_count"] = length
        return example


def function(example):
    length = 0
    try:
        length = int(example['step_count'])
    except Exception as e:
        length = 0
    finally:
        example['step_count'] = length
        return example


def load_llm_model(model_path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.95,
                   max_model_len=23500, device_config="cuda:0"):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="auto", tensor_parallel_size=tensor_parallel_size,
              enforce_eager=enforce_eager, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              device=device_config,
              seed=42,
              swap_space=16,
              )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer


def create_step_data(datasets, model, SampingParams, save_dir="../data/step/"):
    examples = [data2 for data2 in datasets]
    problems = [data1["instruction"] for data1 in datasets]
    tmp_dataset = Dataset.from_dict({})

    for j in range(0, len(examples), 10000):
        example = examples[j:j + 10000]
        problem = problems[j:j + 10000]
        response = find_text_discrepancies(example,model,SampingParams)
        # 准备数据字典
        data = {
            "question": [q for q in problem],  # 或者用 problem 变量如果你限制了问题的数量
            "answer": [a for a in response],
        }

        # 确保两个列表长度相同
        if len(data["question"]) != len(data["answer"]):
            raise ValueError("The length of questions and answers does not match.")

        # 创建 Dataset 对象
        result_dataset = Dataset.from_dict(data)
        results = result_dataset.map(lambda x: step_count(x))
        tmp_dataset = concatenate_datasets([tmp_dataset, results])

    # # 过滤出step_count不等于0的数据
    tmp_dataset = tmp_dataset.map(lambda x: function(x))
    filtered_data = tmp_dataset.filter(lambda x: x['step_count'] != 0)
    filtered_data = filter_data_by_3std(filtered_data, column_name="step_count")
    # 这里 test_size 表示的是验证集占的比例，因此是 0.3；相应的 train_size 就是 0.7
    split_datasets = filtered_data.train_test_split(test_size=0.3, shuffle=True, seed=42)

    # 获取划分好的数据集
    train_dataset = split_datasets["train"]
    valid_dataset = split_datasets["test"]
    # 如果需要，可以将划分好的数据集保存到磁盘上
    train_dataset_path = os.path.join(save_dir, "train_dataset")
    valid_dataset_path = os.path.join(save_dir, "valid_dataset")
    train_dataset.save_to_disk(train_dataset_path)
    valid_dataset.save_to_disk(valid_dataset_path)


def assemble_batch(data_list):
    batch = []
    # start="<begin_of_solution>"
    # end="<end_of_solution>"
    end = "</think>"
    for row in data_list:
        # question = row["problem"]
        # output_data = row["conversations"][1]["value"]
        question = row["instruction"]
        solution = row["response"].split(end)[-1].replace(" ", "").strip()
        # solution = output_data.split(start)[-1].split(end)[0].replace(" ","").strip()
        full_content = prompt7.format(problem=question, solution=solution)
        batch.append((row, full_content))
    return batch


def find_text_discrepancies(list1,model,SamplingParams,batch_size=5000):
    """使用批处理找出事件间的差异。"""
    results = []
    print("开始处理...")
    batch_responses = []
    # 第一层批处理
    for i in tqdm(range(0, len(list1), batch_size), desc='Processing', ncols=100):
        temp_list = list1[i:i + batch_size]
        batch = assemble_batch(temp_list)
        batch_lengths = [len(content) for _, content in batch]
        min_length = min(batch_lengths)
        max_length = max(batch_lengths)
        print(f"批次内容长度最小值: {min_length}, 最大值: {max_length}")
        for j in range(0, len(batch), 2500):
            sub_batch = batch[j:j + 2500]
            test = []
            for _, content in sub_batch:
                prompt = [{"role": "user", "content": f"{content}"}]
                inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(inputs, add_special_tokens=False)
                test.append(prompt_ids)
            outputs = model.generate(prompt_token_ids=test, sampling_params=SamplingParams)  # ,lora_request=self.lora
            for output in outputs:
                generated_text = output.outputs[0].text
                if generated_text == "":
                    generated_text = "x"
                batch_responses.append(generated_text)
    return batch_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../model/Qwen2.5-7B-Instruct")
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='', default="../data/step")
    parser.add_argument('--sample_size', type=int, help='', default=35000)
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature, higher means more random')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling probability')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling, 0 means disabled')
    parser.add_argument('--max_tokens', type=int, default=400,
                        help='Maximum number of tokens to generate')
    args = parser.parse_args()

    model_path = args.model_path
    model, tokenizer = load_llm_model(model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    choice = args.choice
    dataset = ["open-r1/OpenThoughts-114k-math", "Phsoft-ai/alpaca-gpt4-CoT"]
    data_path = dataset[choice]
    tmp_path = data_path.split("/")[-1]
    save_path = os.path.join(args.data_prefix_path, tmp_path)

    sample_size = args.sample_size
    datasets = load_from_disk(data_path)["train"]
    if choice == 0:
        datasets = datasets.filter(
            lambda x: x["problem"] and len(x["problem"]) < 498 and len(x["solution"]) < 4000 and x[
                "correct"] == "true")
    if choice == 1:
        datasets = datasets.filter(
            lambda x: x["instruction"] and len(x["instruction"]) < 320 and len(x["response"]) < 22404 and x[
                "input_quality"] == "good")
    shuffled_data = datasets.shuffle(seed=42)
    # 抽取指定数量的数据
    sampled_data = shuffled_data.select(range(min(sample_size, len(shuffled_data))))
    create_step_data(sampled_data, model=model, SampingParams=sampling_params,
                     save_dir=save_path)

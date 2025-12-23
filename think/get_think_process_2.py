from collections import Counter
import random

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer

def process_think_data(data, tokenizer):  # 添加tokenizer参数
    result=[]
    for data1 in data:
        start = "<think>"
        end = "</think>"
        question = data1['instruction']
        output_data = data1["response"]
        think_data = output_data.split(start)[-1].split(end)[0].strip()
        encoded_texts = tokenizer(think_data, return_tensors="pt", padding=False, truncation=True)
        length = encoded_texts['input_ids'].shape[1]  # 直接获取长度
        result.append({"prompt": question, "think_label": length})
    return Dataset.from_list(result)

def process_data(data, tokenizer):  # 修改process_data以接收tokenizer
    result_data = process_think_data(data, tokenizer)
    return result_data

def count(data):
    # 正确检查重复prompt的方法
    seen_prompts = set()
    duplicate_count = 0
    for d in data:
        if d["prompt"] in seen_prompts:
            duplicate_count += 1
        else:
            seen_prompts.add(d["prompt"])
    if duplicate_count == 0:
        print("数据集中没有重复的prompt。")
    else:
        print(f"发现 {duplicate_count} 条重复的prompt记录。")
    class_counts = Counter()
    for example in data:
        # 计算新类的值
        class_num = int(example["think_label"] / 25)
        # 更新计数
        class_counts[class_num] += 1
        # 打印每个新类的数目
    for class_num, count in class_counts.items():
        print(f"Class {class_num}: {count} examples")
    # 过滤掉数目少于100的类别
    valid_classes = {class_num: count for class_num, count in class_counts.items() if (count >=1100 and count <= 2000)}
    # 创建新列并过滤数
    filtered_data = []
    for example in data:
        class_num = int(example["think_label"] / 25)
        if class_num in valid_classes:
            example["class_name"] = class_num
            filtered_data.append(example)
    # 将过滤后的数据转换为Dataset对象
    filtered_dataset = Dataset.from_list(filtered_data)
    # 保存过滤后的数据
    # 打印过滤后各个类别的名称和对应的数据数量
    for class_num, count in valid_classes.items():
        print(f"Class {class_num}: {count} examples")
    return filtered_dataset


# 15000y,按照这个最大的来进行获取，然后操作成为这个平衡的训练集合，然后验证数据集合为不要求平衡即可
# 30000,以上，把这个训练数据集合，验证数据集合操作成为这个平衡数据集合，来进行操作即可

def reduce_dataset_to_19k(train_data_list, source_data_list, valid_data_list):
    # 先将列表转换为Dataset以便操作
    train_dataset = Dataset.from_list(train_data_list)
    source_dataset = Dataset.from_list(source_data_list)
    valid_dataset = Dataset.from_list(valid_data_list)
    
    target_size = 19000
    if len(train_dataset) <= target_size:
        return train_data_list, valid_data_list
    
    # 计算需要删除的样本数
    to_remove = len(train_dataset) - target_size
    print(f"需要削减 {to_remove} 个样本以达到 {target_size} 的目标。")
    
    # 按类别统计并决定从最多样本的类别中删除
    class_counts = Counter(train_dataset['class_name'])
    while to_remove > 0 and class_counts:
        max_class = max(class_counts, key=class_counts.get)
        # 找出该类别在训练集中的索引
        class_indices = [i for i, item in enumerate(train_dataset) if item['class_name'] == max_class]
        # 确定要删除的数量（不能超过该类别现有数量）
        remove_from_class = min(len(class_indices), to_remove)
        # 随机选择要删除的索引
        indices_to_remove = random.sample(class_indices, remove_from_class)
        # 从训练集中删除
        train_dataset = train_dataset.select([i for i in range(len(train_dataset)) if i not in indices_to_remove])
        # 更新需要删除的数量
        to_remove -= remove_from_class
        # 更新类别统计
        class_counts = Counter(train_dataset['class_name'])
    
    return train_dataset.to_list(), valid_dataset.to_list()


def get_data(data1):
    class_counts = {}
    for item in data1:
        class_name = item['class_name']
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    min_count = min(class_counts.values())
    print(min_count)
    train_data_label = []
    used_hashes = set()  # 使用 hash 去重
    tmp_data_label = []
    # 根据每个类别的数量填充训练数据集
    for class_name in class_counts:
        class_data = [item for item in data1 if item['class_name'] == class_name]
        random.shuffle(class_data)
        selected = class_data[:int(min_count)]
        selected_data = class_data[int(min_count):]
        train_data_label.extend(selected)
        tmp_data_label.extend(selected_data)
    label_data = []
    # 当训练数据集大小超过25000时，开始删除操作
    if len(train_data_label) > 19000:
        train_data_label,tmp_data_label =reduce_dataset_to_19k(train_data_label, data1, tmp_data_label)
    valid_data = tmp_data_label[:int(len(train_data_label) * 1/4)]
    if isinstance(train_data_label,Dataset):
        train_data_label.save_to_disk("../data/think/data2/train_data")
    else:
        train_data_label=Dataset.from_list(train_data_label)
        train_data_label.save_to_disk("../data/think/data2/train_data")
    valid_data=Dataset.from_list(valid_data)

    valid_data.save_to_disk("../data/think/data2/valid_data")

    print("Train size:", len(train_data_label))
    print("Valid size:", len(valid_data))

if __name__ == "__main__":
    import random
    # 设置随机种子确保可复现
    SEED = 42
    random.seed(SEED)
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(r"../model/think/Qwen2-0.5B-Instruct", legacy=False)
    ds = load_dataset(r"Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B")["train"]
    
    # 传递tokenizer给process_data
    label_data = process_data(ds, tokenizer)
    data = count(label_data)
    get_data(data)  

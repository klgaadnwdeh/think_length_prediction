from collections import Counter
import random

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer


def process_think_data(data):
    result=[]
    for data1 in data:
        length = 0
        length1=0
        start = "<think>"
        end = "</think>"
        question = data1['instruction']
        output_data = data1["response"]
        think_data = output_data.split(start)[-1].split(end)[0].strip()
        encoded_texts = tokenizer(think_data, return_tensors="pt", padding=False, truncation=True)
        for ids in encoded_texts['input_ids']:
            length = len(ids)
            print(length)
        result.append({"prompt": question, "think_label": length})
    return Dataset.from_list(result)


def process_data(data):
    result_data = process_think_data(data)
    return result_data


def count(data):
    seem = set()
    seem = [seem.add(d["prompt"]) for d in data]
    if len(seem) == len(data):
        print("数据的长度是一样的，没有这个出错的")
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


def reduce_dataset_to_19k(train_dataset, source_data, valid_data):
    dataset = Dataset.from_list(source_data)
    class_counts = Counter(dataset['class_name'])
    max_class = max(class_counts, key=class_counts.get)
    train_dataset = Dataset.from_list(train_dataset)
    valid_dataset = Dataset.from_list(valid_data)
    while len(train_dataset) > 19000:
        indices_to_keep = [
            i for i, item in enumerate(train_dataset)
            if item['class_name'] != max_class
        ]
        index = [i for i, item in enumerate(dataset)
                 if item['class_name'] != max_class
                 ]
        valid_index = [i for i, item in enumerate(valid_dataset)
                       if item['class_name'] != max_class]
        train_dataset = train_dataset.select(indices_to_keep)
        valid_dataset = valid_dataset.select(valid_index)
        dataset = dataset.select(index)
        class_counts = Counter(dataset['class_name'])
        max_class = max(class_counts, key=class_counts.get)

    return train_dataset, valid_dataset.to_list()


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
    tokenizer = AutoTokenizer.from_pretrained(r"../model/think/Qwen2-0.5B-Instruct", legacy=False)
    from datasets import load_dataset
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset(r"Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B")["train"]
    label_data=process_data(ds)
    data = count(label_data)
    get_data(data)

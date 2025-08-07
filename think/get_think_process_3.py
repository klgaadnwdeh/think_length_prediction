from collections import Counter
import random

from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


def process_think_data(data1):
    length = 0
    start = "<think>"
    end = "</think>"
    question = data1["messages"][0]["content"]
    output_data = data1["messages"][1]["content"]
    think_data = output_data.split(start)[-1].split(end)[-1].strip()
    encoded_texts = tokenizer(think_data, return_tensors="pt", padding=False, truncation=True)
    for ids in encoded_texts['input_ids']:
        length = len(ids)
        print(length)
    return {"prompt": question, "think_label": length}

def process_data(data):
    result_data = data.map(lambda x:process_think_data(x))
    return result_data


def count(data):
    seem = set()
    seem = [seem.add(d["prompt"]) for d in data]
    if len(seem) == len(data):
        print("数据的长度是一样的，没有这个出错的")
    class_counts = Counter()
    for example in data:
        # 计算新类的值
        class_num = int(example["label"] / 25)
        # 更新计数
        class_counts[class_num] += 1
        # 打印每个新类的数目
    for class_num, count in class_counts.items():
        print(f"Class {class_num}: {count} examples")
    # 过滤掉数目少于100的类别
    valid_classes = {class_num: count for class_num, count in class_counts.items() if count >= 900}
    # 创建新列并过滤数
    filtered_data = []
    for example in data:
        class_num = int(example["label"] / 25)
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

# 数据量太少了，就是如此的。
# 15000y,按照这个最大的来进行获取，然后操作成为这个平衡的训练集合，然后验证数据集合为不要求平衡即可
# 30000,以上，把这个训练数据集合，验证数据集合操作成为这个平衡数据集合，来进行操作即可



def reduce_dataset_to_20k(train_dataset, source_data, valid_data):
    dataset = source_data
    class_counts = Counter(dataset['class_name'])
    max_class = max(class_counts, key=class_counts.get)
    train_dataset = Dataset.from_list(train_dataset)
    valid_dataset = Dataset.from_list(valid_data)
    while len(train_dataset) > 20000:
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
        print("减少")

    return train_dataset, valid_dataset




def get_data(data1, b=0.3):
    class_counts = {}
    i = 0
    for item in data1:
        class_name = item['class_name']
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    min_count = min(class_counts.values())
    train_data_label = []
    used_hashes = set()  # 使用 hash 去重
    tmp_data_label = []
    # 根据每个类别的数量填充训练数据集
    for class_name in class_counts:
        class_data = [item for item in data1 if item['class_name'] == class_name]
        random.shuffle(class_data)
        selected = class_data[:min_count]
        selected_data = class_data[int(min_count):int(min_count+min_count*0.20)]
        train_data_label.extend(selected)
        tmp_data_label.extend(selected_data)
    if len(train_data_label) > 20000:
        print("数据超过了")
        train_data_label, tmp_data_label = reduce_dataset_to_20k(train_data_label, data1, tmp_data_label)


    if isinstance(train_data_label, Dataset):
        train_data_label.save_to_disk(r"../data/think/data3/train_data")
    else:
        train_data_label = Dataset.from_list(train_data_label)
        train_data_label.save_to_disk(r"../data/think/data3/train_data")

    tmp_data_label.save_to_disk(r"../data/think/data3/valid_data")

    print("Train size:", len(train_data_label))
    print("Valid size:", len(tmp_data_label))


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(r"../model/think/Qwen2-0.5B-Instruct", legacy=False)
    from datasets import load_dataset
    # Login using e.g. `huggingface-cli login` to access this dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("mlfoundations-dev/AM-DeepSeek-R1-Distilled-1.4M-am_0.5M")["train"]
    label_data = process_data(ds)
    data = count(label_data)
    get_data(data)
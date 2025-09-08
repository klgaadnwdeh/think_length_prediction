from collections import Counter
import random

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
import argparse


def process_think_data(data):
    result = []
    for data1 in data:
        length = 0
        start = "<think>"
        end = "</think>"
        question = data1['prompt']
        output_data = data1["completion"]
        think_data = output_data.split(start)[-1].split(end)[0].strip()
        encoded_texts = tokenizer(think_data, return_tensors="pt", padding=False, truncation=True)
        for ids in encoded_texts['input_ids']:
            length = len(ids)
            print(length)
        result.append({"prompt": question, "think_label": length})
    return Dataset.from_list(result)


def process_think_data_2(data):
    result = []
    for data1 in data:
        length = 0
        length1 = 0
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


def process_think_data_3(data1):
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


def process_data(data, choice=0):
    if choice == 0:
        result_data = process_think_data(data)
    elif choice == 1:
        result_data = process_think_data_2(data)
    else:
        result_data = process_think_data_3(data)
    return result_data

def reduce_dataset(train_dataset, source_data, valid_data,number=20000):
    dataset = Dataset.from_list(source_data)
    class_counts = Counter(dataset['class_name'])
    max_class = max(class_counts, key=class_counts.get)
    train_dataset = Dataset.from_list(train_dataset)
    valid_dataset = Dataset.from_list(valid_data)
    while len(train_dataset) > number:
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

def count(data,count_number=400):
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
    valid_classes = {class_num: count for class_num, count in class_counts.items() if count >=count_number}
    # 创建新列并过滤数
    filtered_data = []
    for example in data:
        class_num = int(example["label"] / 25)
        if class_num in valid_classes:
            example["class_name"] = class_num
            filtered_data.append(example)
    # 将过滤后的数据转换为Dataset对象
    filtered_dataset = Dataset.from_list(filtered_data)
    for class_num, count in valid_classes.items():
        print(f"Class {class_num}: {count} examples")
    return filtered_dataset


def get_data(data1,number=20000,ratio=0.3,output_file="../data/think"):
    class_counts = {}
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
        selected_data = class_data[min_count:]
        train_data_label.extend(selected)
        tmp_data_label.extend(selected_data)
    # 当训练数据集大小超过25000时，开始删除操作
    if len(train_data_label) > number:
        train_data_label,tmp_data_label =reduce_dataset(train_data_label, data1, tmp_data_label,number)

    valid_data = tmp_data_label[:int(len(train_data_label) * ratio)]
    train_data_path=os.path.join(output_file,"train_data")
    vaild_data_path=os.path.join(output_file,"valid_data")
    if isinstance(train_data_label, Dataset):
        train_data_label.save_to_disk(train_data_path)
    else:
        train_data_label = Dataset.from_list(train_data_label)
        train_data_label.save_to_disk(train_data_path)
    valid_data = Dataset.from_list(valid_data)
    valid_data.save_to_disk(vaild_data_path)

    print("Train size:", len(train_data_label))
    print("Valid size:", len(valid_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--AutoTokenizer_path', type=str, default="../model/think/Qwen2-0.5B-Instruct")
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1],2 for dataset[2]', default=0)
    parser.add_argument('--fileter_number', type=int, help='0 for dataset[0], 1 for dataset[1],2 for dataset[2]', default=312)
    parser.add_argument('--ratio', type=float, help='The ratio of the lengths of training data to validation data', default=0.25)
    parser.add_argument('--number', type=int, help='The length of the train_data',
                        default=9000)
    parser.add_argument('--output_path', type=str, help='',
                        default="../data/think")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.AutoTokenizer_path, legacy=False)
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = ["sequelbox/Titanium2.1-DeepSeek-R1", "Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B",
               "mlfoundations-dev/AM-DeepSeek-R1-Distilled-1.4M-am_0.5M"]
    filter_list=[312,1100,900]
    choice = args.choice
    data_path = dataset[choice]
    filter_number=filter_list[choice]
    number=args.number
    ratio=args.ratio
    if choice==0:
        data="data1"
    elif choice==1:
        data="data2"
    else:
        data="data3"
    output_path=os.path.join(args.output_path,data)
    if output_path.exists():
        print(f"路径存在: {output_path}")
    else:
        print(f"路径不存在，正在创建: {output_path}")
        output_path.mkdir(parents=True,exist_ok=True)
    ds = load_dataset(data_path)["train"]
    label_data = process_data(ds)
    data1 = count(label_data,filter_number)
    get_data(data1,number,ratio,output_path)

from datasets import load_from_disk, Dataset,concatenate_datasets
import random
from datasets import load_from_disk
import argparse
from tqdm import tqdm
import os

def convert_dpo(path="../data/think/"):
    # 加载原始数据
    data = load_from_disk(path)

    # 获取文件名并添加前缀"kto_"
    filename = os.path.basename(path)  # 获取最后一个名称
    temp_path = "dpo_" + filename

    # 获取除最后一个名称外的路径部分
    prefix_path = os.path.dirname(path)

    # 拼接新路径
    save_path = os.path.join(prefix_path, temp_path)

    # 创建一个新的列表来存储转换后的数据
    transformed_data = []

    # 遍历原始数据的每一行
    for row in tqdm(data, total=len(data), desc="Processing Data"):
        prompt_content = row['prompt']
        think_label_content = row['think_label']

        # 创建标签为true的行（chosen）
        chosen = [
            {'content': prompt_content, 'role': 'user'},
            {'content': think_label_content, 'role': 'assistant'}
        ]
        # 创建标签为false的行（rejected），通过在think_label内容中减去一个0-10内的随机数值
        false_value = int(think_label_content) - random.randint(0, 10)
        rejected = [
            {'content': prompt_content, 'role': 'user'},
            {'content': false_value, 'role': 'assistant'}
        ]
        # 构建新的数据行
        new_row = {
            'chosen': chosen,
            'rejected': rejected,
        }
        transformed_data.append(new_row)
    # 将转换后的数据转换为Dataset对象
    tmp_data = Dataset.from_list(transformed_data)
    # 保存到磁盘
    tmp_data.save_to_disk(save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='', default="../data/think/")
    parser.add_argument('--choice', type=int, help='', default=0)
    args = parser.parse_args()
    choice=args.choice
    if choice==0:
        data_path="data1"
    elif choice==1:
        data_path="data2"
    else:
        data_path="data3"
    train_data_path = os.path.join(args.data_path, data_path, "train_data")
    valid_data_path = os.path.join(args.data_path, data_path, "valid_data")
    function_choice = args.function_choice
    convert_dpo(train_data_path)
    convert_dpo(valid_data_path)




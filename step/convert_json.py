from datasets import load_from_disk, Dataset
import argparse
import json
import os  # 关键修复：导入 os 模块

prompt = """
下面有一道来自deepseek-R1的问题，你的任务是,考虑问题的深度，难度，还有复杂度，考虑下面的问题总共要花几个步骤来解决即可，但是不需要给出解决内容，只给出大致的步骤总数即可。
question:{question}
"""

def convert_and_save_dataset(dataset, output_file):  # 改进：参数名更清晰
    """将数据集转换为指定格式并保存为JSON文件。"""
    converted_data = []
    for data in dataset:  # 遍历数据集中的每个样本
        question = prompt.format(question=data['question'])
        label = data['step_count']
        converted_data.append({
            'instruction': question,
            'input': '',
            'output': str(label)
        })
    
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:  # 避免当 output_file 在当前目录时创建空目录
        os.makedirs(output_dir, exist_ok=True)
    
    # 直接保存列表为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已成功转换并保存至 {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='Convert the path to your own corresponding data directory.', default='../data/step/')
    parser.add_argument('--output_prefix_path', type=str, help='', default='../data/llamafactory/step/')
    args = parser.parse_args()

    dataset_names = ['open-r1/OpenThoughts-114k-math', 'Phsoft-ai/alpaca-gpt4-CoT']
    chosen_name = dataset_names[args.choice]
    
    # 构建完整的加载和保存路径
    dataset_save_name = chosen_name.split('/')[-1]
    train_dataset_path = os.path.join(args.data_prefix_path, dataset_save_name, 'train_dataset')
    output_file_path = os.path.join(args.output_prefix_path, dataset_save_name + '.json')
    
    # 加载数据集并处理
    original_data = load_from_disk(train_dataset_path)
    convert_and_save_dataset(original_data, output_file_path)

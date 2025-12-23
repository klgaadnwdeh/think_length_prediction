import random
from datasets import load_from_disk, Dataset
import os
from tqdm import tqdm
import argparse

def convert_dpo(original_data_path, output_base_path, seed=42):
    """
    将数据集转换为DPO格式（健壮版）。
    Args:
        original_data_path: 原始数据集路径
        output_base_path: 输出基准路径，新数据集将保存在 {output_base_path}/dpo_{dataset_name}
        seed: 随机种子，保证可复现
    """
    random.seed(seed)
    
    # 1. 加载数据
    try:
        data = load_from_disk(original_data_path)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 2. 构建输出路径（更清晰的逻辑）
    # 获取原始数据集所在的目录名和数据集名
    original_parent_dir = os.path.dirname(original_data_path)
    dataset_name = os.path.basename(original_data_path)
    
    # 在原始目录的同级创建 dpo_ 前缀的新目录
    output_dir_name = "dpo_" + dataset_name
    save_path = os.path.join(original_parent_dir, f"dpo_{dataset_name}")
    os.makedirs(save_path, exist_ok=True)
    
    transformed_data = []
    
    # 3. 转换数据（添加异常处理和更合理的负样本生成）
    for row in tqdm(data, total=len(data), desc="Processing Data"):
        try:
            prompt_content = row['prompt']
            think_label_content = row['think_label']
            
            # 验证并转换正样本标签
            true_label = int(think_label_content)
            
            # 创建正样本 (chosen)
            chosen = [
                {'content': prompt_content, 'role': 'user'},
                {'content': think_label_content, 'role': 'assistant'}  # 保持字符串类型
            ]
            
            # 创建更合理的负样本 (rejected)
            # 方案A: 确保负样本至少为0，且与正样本有合理差距
            min_rejected = max(0, true_label - random.randint(3, 10))  # 至少差3步
            # 方案B（更简单）：如果标签>1，则生成一个稍小的正数
            false_value = min_rejected if true_label > 3 else max(0, true_label - 1)
            
            rejected = [
                {'content': prompt_content, 'role': 'user'},
                {'content': str(false_value), 'role': 'assistant'}  # 统一转换为字符串
            ]
            
            transformed_data.append({
                'chosen': chosen,
                'rejected': rejected,
            })
            
        except (ValueError, KeyError) as e:
            print(f"跳过一行数据（转换错误）: {e}, 数据: {row.get('prompt', 'N/A')[:50]}...")
            continue  # 跳过有问题的行，而不是让整个程序崩溃
    
    if not transformed_data:
        print("警告：没有成功转换任何数据！")
        return
    
    # 4. 保存数据
    tmp_data = Dataset.from_list(transformed_data)
    tmp_data.save_to_disk(save_path)
    print(f"转换完成！数据已保存至: {save_path}")
    print(f"总计转换 {len(transformed_data)} 条数据（跳过 {len(data) - len(transformed_data)} 条无效数据）")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/think/")
    parser.add_argument('--choice', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置全局随机种子
    random.seed(args.seed)
    
    choice_map = {0: "data1", 1: "data2", 2: "data3"}
    data_subdir = choice_map.get(args.choice, "data1")
    
    # 分别处理训练集和验证集
    splits = ["train_data", "valid_data"]
    for split in splits:
        original_path = os.path.join(args.data_path, data_subdir, split)
        if os.path.exists(original_path):
            print(f"\n正在处理: {original_path}")
            convert_dpo(original_path, output_base_path=args.data_path, seed=args.seed)
        else:
            print(f"路径不存在，已跳过: {original_path}")

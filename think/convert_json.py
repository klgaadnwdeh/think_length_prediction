import numpy as np
from datasets import load_from_disk,Dataset
import argparse
import os

prompt_think="""
人得到一个问题，首先是根据问题的深度，难度，复杂度，慢慢地思考问题应该如何解决。在思考过程中运用自己的各种能力，对于问题进行逐步逐步推导解决，在一步一步地推导解决过程中得到结果。得到结果之后，人停止思考并且整合思考过程中的内容，进行大量简化后，得到问题的整个解决办法和结果，最后输出问题的解决办法和结果。
任务描述：
question:{question}
你的任务是考虑问题在人思考过程中所花费的令牌数量的范围。具体根据问题可能涉及的深度、复杂性和长度，估算出思考过程总共花费的令牌数量的范围。
例如:
1估计的令牌数量范围在200，则真实思考过程中所花费的令牌数量基本在(200*25,201*25-1)
2估计的令牌数量范围在320，则真实思考过程中所花费的令牌数量基本在(320*25,321*25-1)
要求：仅输出估计的令牌数量范围，不生成其他的内容。
"""
from datasets import load_from_disk,Dataset
import json

def get_think_label(example,output_file=""):
    converted_data = []
    for data in example:
        question=prompt_think.format(question=data['prompt'])
        label=data['class_name']
        converted_data.append({'instruction':question,"input":"",'output':str(label)})
    converted_data=Dataset.from_list(converted_data)
    # 如果你想保留 Dataset 类型（可选）
    # 使用 json 模块直接保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data.to_list(), f, ensure_ascii=False, indent=2)
    print(f"已成功保存至 {output_file}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1],2 for dataset[2]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='', default="../data/think/")
    parser.add_argument('--output_prefix_path', type=str, help='', default="../data/llamafactory/think/")
    args = parser.parse_args()
    data_prefix_path=args.data_prefix_path
    dataset=["sequelbox/Titanium2.1-DeepSeek-R1","Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B","mlfoundations-dev/AM-DeepSeek-R1-Distilled-1.4M-am_0.5M"]
    choice=args.choice
    if choice==0:
        data_path="data1"
    elif choice==1:
        data_path="data2"
    else:
        data_path="data3"
    data_path = os.path.join(data_prefix_path,data_path,"train_data")
    data=load_from_disk(data_path)
    output_path = args.output_prefix_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"路径 '{output_path}' 已创建。")
    else:
        print(f"路径 '{output_path}' 已存在。")
    output_path = output_path+"train.json"
    get_think_label(data,output_file=output_path)




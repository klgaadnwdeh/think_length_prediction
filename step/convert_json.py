from datasets import load_from_disk,Dataset
import argparse
import json

prompt="""
下面有一道来自deepseek-R1的问题，你的任务是,考虑问题的深度，难度，还有复杂度，考虑下面的问题总共要花几个步骤来解决即可，但是不需要给出解决内容，只给出大致的步骤总数即可。
question:{question}
"""
# prompt="""
# 下面有一道来自deepseek-R1的问题，你的任务是,考虑问题的深度，难度，还有复杂度，考虑下面的问题总共要花几个步骤来解决即可，但是不需要给出解决内容，只给出大致的步骤总数即可。已知问题步骤数目不低于1步，不高于11步。
# question:{question}
# """
def get_think_label(example,output_file=r"../data/llamafactory/step/"):
    converted_data = []
    for data in example:
        question=prompt.format(question=data['question'])
        label=data['step_count']
        converted_data.append({'instruction':question,"input":"",'output':str(label)})
    converted_data=Dataset.from_list(converted_data)
    # 如果你想保留 Dataset 类型（可选）
    # 使用 json 模块直接保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data.to_list(), f, ensure_ascii=False, indent=2)
    print(f"已成功保存至 {output_file}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--choice', type=int, help='0 for dataset[0], 1 for dataset[1]', default=0)
    parser.add_argument('--data_prefix_path', type=str, help='', default="../data/step")
    parser.add_argument('--output_prefix_path', type=str, help='', default="../data/llamafactory/step/")
    args = parser.parse_args()

    dataset = ["open-r1/OpenThoughts-114k-math", "Phsoft-ai/alpaca-gpt4-CoT"]
    choice=args.choice
    data_path = dataset[choice]
    tmp_path = data_path.split("/")[-1]
    # 加载数据集
    save_path=args.data_prefix_path
    train_dataset_path = os.path.join(save_path,tmp_path,"train_dataset")
    data=load_from_disk(train_dataset_path)
    output_file=args.output_prefix_path+tmp_path+".json"
    get_think_label(data,output_file=output_file)

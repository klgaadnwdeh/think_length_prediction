# train_dpo.py
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from trl import DPOConfig, DPOTrainer
from trl import apply_chat_template
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
def data_solution(example):
    example["prompt"]=prompt_think.format(question=example["prompt"])
    return example
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../model/think/data1/Qwen2-0.5B-Instruct")
    parser.add_argument('--train_data_path', type=str, default="..//data//think//data1//train_data")
    parser.add_argument('--valid_data_path', type=str, default="..//data//think//data1//valid_data")
    parser.add_argument('--logging_steps', type=int, help="", default=10)
    parser.add_argument('--num_train_epochs', type=int, help="The number of training epochs for the reward model.",
                        default=1)
    parser.add_argument('--output_path', type=str, default="../model/think//data1/DPO")
    args = parser.parse_args()
    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = load_from_disk(args.train_data_path).map(data_solution)
    eval_dataset=load_from_disk(args.valid_data_path).map(data_solution)
    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset=eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    training_args = DPOConfig(output_dir=args.output_path, logging_steps=args.logging_steps, num_train_epochs=args.num_train_epochs, include_for_metrics=["loss", ], report_to="tensorboard")
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset,eval_dataset=eval_dataset)
    trainer.train()


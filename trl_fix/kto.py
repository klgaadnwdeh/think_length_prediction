from datasets import load_dataset, load_from_disk
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from transformers import TrainingArguments
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
    parser.add_argument('--model_path', type=str,help="",default="../model/think/data1/Qwen2-0.5B-Instruct")
    parser.add_argument('--train_data_path', type=str,default="..//data//think//data1//train_data")
    parser.add_argument('--valid_data_path', type=str,default="..//data//think//data1//valid_data")
    parser.add_argument('--logging_steps', type=int,help="the logging frequency.",default=10)
    parser.add_argument('--save_steps', type=int, help="the saving frequency.", default=100000)
    parser.add_argument('--num_train_epochs', type=int, help="The number of training epochs for the reward model.", default=1)
    parser.add_argument("--max_completion_length",type=int,help="Maximum length of the completion. This argument is required if you want to use the default data "
                                                            "collator and your model is an encoder-decoder.",default=20)
    parser.add_argument('--max_grad_norm', type=int, help="Maximum gradient norm (for gradient clipping).",default=0.1)
    parser.add_argument('--max_prompt_length', type=int, help="Maximum length of the completion. This argument is required if you want to use the default data "
                                                        "collator and your model is an encoder-decoder.", default=1024)
    parser.add_argument('--output_path', type=str,help="",default="../model/think//data1/KRO")
    args = parser.parse_args()
    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data=load_from_disk(args.train_data_path).map(data_solution)
    valid_data=load_from_disk(args.valid_data_path).map(data_solution)
    #1024,2048
    training_args = KTOConfig(output_dir=args.output_path,
                              logging_steps=args.logging_steps,
                              save_steps=args.save_steps,
                              num_train_epochs=args.num_train_epochs,
                              max_completion_length=args.max_completion_length,
                              max_grad_norm = args.max_grad_norm,
                              max_prompt_length=args.max_prompt_length,
                              include_for_metrics=["loss",]
                              ,report_to="tensorboard"
                              )
    trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_data,eval_dataset=valid_data)
    trainer.train()

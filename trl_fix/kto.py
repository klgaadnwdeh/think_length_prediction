from datasets import load_dataset, load_from_disk
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from transformers import TrainingArguments


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
    train_data=load_from_disk(args.train_data_path)
    valid_data=load_from_disk(args.valid_data_path)
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

# train_dpo.py
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from trl import DPOConfig, DPOTrainer
from trl import apply_chat_template
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
    train_dataset = load_from_disk(args.train_data_path)
    eval_dataset=load_from_disk(args.valid_data_path)
    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset=eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    training_args = DPOConfig(output_dir=args.output_path, logging_steps=args.logging_steps, num_train_epochs=args.num_train_epochs, include_for_metrics=["loss", ], report_to="tensorboard")
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset,eval_dataset=eval_dataset)
    trainer.train()


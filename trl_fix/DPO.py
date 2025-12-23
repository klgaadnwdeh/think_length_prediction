# train_dpo.py
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from unsloth import FastLanguageModel

from trl import DPOConfig, DPOTrainer
from trl import apply_chat_template

prompt_think = """
人得到一个问题，首先是根据问题的深度，难度，复杂度，慢慢地思考问题应该如何解决。在思考过程中运用自己的各种能力，对于问题进行逐步逐步推导解决，在一步一步地推导解决过程中得到结果。得到结果之后，人停止思考并且整合思考过程中的内容，进行大量简化后，得到问题的整个解决办法和结果，最后输出问题的解决办法和结果。
任务描述：
question:{question}
你的任务是考虑问题在人思考过程中所花费的令牌数量的范围。具体根据问题可能涉及的深度、复杂性和长度，估算出思考过程总共花费的令牌数量的范围。
例如:
1估计的令牌数量范围在200，则真实思考过程中所花费的令牌数量基本在(200*25,201*25-1)
2估计的令牌数量范围在320，则真实思考过程中所花费的令牌数量基本在(320*25,321*25-1)
要求：仅输出估计的令牌数量范围，不生成其他的内容。
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=r"../model/think/..")
    parser.add_argument('--train_data_path', type=str, default=r"../data/think/data1/train_data")
    parser.add_argument('--valid_data_path', type=str, default=r"../data/think/data1/valid_data")
    parser.add_argument('--logging_steps', type=int, help="", default=10)
    parser.add_argument('--num_train_epochs', type=int, help="The number of training epochs for the reward model.",
                        default=6)
    parser.add_argument('--output_path', type=str, default=r"/mnt/f/home_fix/3/dpo/model")
    args = parser.parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path, 
        max_seq_length=3000,
        dtype=torch.float16,  # FP16 加速
        load_in_4bit=False,  # 不量化（或设为 True 用 4-bit）
        load_in_8bit=False,
        trust_remote_code=True,
        local_files_only=True,
        model_type="qwen2"
        # 注意：不需要 model_type="qwen2"，Unsloth 会自动识别
    )
    train_dataset = load_from_disk(args.train_data_path)
    eval_dataset = load_from_disk(args.valid_data_path)

    # 抽取前 10 条训练数据（如果不足 10 条，则取全部）
    # train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
    # # 抽取前 10 条验证数据
    # eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))


    train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    eval_dataset = eval_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    training_args = DPOConfig(output_dir=args.output_path, logging_steps=args.logging_steps,
                              num_train_epochs=args.num_train_epochs, include_for_metrics=["loss", ],
                              report_to="tensorboard",
                              save_steps=14040,
                              )
#8gebuzhou

    print("model successful")
    model= FastLanguageModel.get_peft_model(
        model=model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407
    )
    # # Ref Policy (frozen)
    from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
    from datasets import load_from_disk

    ref_policy = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # ← 同样 FP16
        trust_remote_code=True,
        device_map="auto",
    )
    ref_policy.resize_token_embeddings(len(tokenizer))
    ref_policy.eval()
    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset,
                         eval_dataset=eval_dataset)
    trainer.train()

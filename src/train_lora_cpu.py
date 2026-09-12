"""医疗/政务问答 LoRA 微调训练脚本（CPU 友好版）。

与 train_lora.py 的区别：
- batch_size=1 / grad_acc=2 / epochs=2，降低 CPU 运行耗时
- MAX_LEN=256，适配本项目短问答数据
- 强制 fp16=False（CPU 不支持 fp16 训练）

输出：outputs/qwen2.5-lora-medical/ 下的 LoRA 适配器
"""
import json
import os
import torch
from datasets import Dataset

# CPU 极限压榨：用满本机 20 线程
torch.set_num_threads(min(20, os.cpu_count() or 4))
torch.set_num_interop_threads(2)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "models/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/medical_qa_sample.json"
OUTPUT_DIR = "outputs/qwen2.5-lora-medical"
MAX_LEN = 256


def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    def fmt(r):
        instr = r.get("instruction", "")
        inp = r.get("input", "")
        prompt = (instr + "\n" + inp).strip()
        return {"prompt": prompt, "response": r.get("output", "")}

    return Dataset.from_list([fmt(r) for r in rows])


def tokenize(example, tokenizer):
    # prompt 部分 label 设为 -100（不计算损失），只训练 response
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + resp_ids)[:MAX_LEN]
    labels = ([-100] * len(prompt_ids) + resp_ids)[:MAX_LEN]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基座模型：{BASE_MODEL}（device_map='auto'，无 GPU 时落到 CPU）")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="cpu",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset(DATA_PATH).map(lambda b: tokenize(b, tokenizer), batched=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=3e-4,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA 适配器已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

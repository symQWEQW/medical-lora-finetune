"""医疗/政务问答 LoRA 微调训练脚本。

依赖：torch, transformers, peft, datasets, accelerate
运行（有 GPU）：
    pip install -r requirements.txt
    python src/train_lora.py

说明：
- 默认用 Qwen2.5-0.5B 小模型做演示（消费级显卡即可跑）；
  生产可改 BASE_MODEL 为 Qwen/Qwen2.5-7B-Instruct，并用 4bit 量化（bitsandbytes）节省显存。
- 训练数据：data/medical_qa_sample.json（alpaca 格式）。
- 本脚本只训练 LoRA 适配器，不改动基座权重，训练快、可部署、可回退。
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/medical_qa_sample.json"
OUTPUT_DIR = "outputs/qwen2.5-lora-medical"
MAX_LEN = 512


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
    src = tokenizer(
        example["prompt"],
        max_length=MAX_LEN,
        truncation=True,
    )
    tgt = tokenizer(
        example["response"],
        max_length=MAX_LEN,
        truncation=True,
    )
    src["labels"] = tgt["input_ids"]
    return src


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
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

    ds = load_dataset(DATA_PATH).map(
        lambda b: tokenize(b, tokenizer), batched=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA 适配器已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

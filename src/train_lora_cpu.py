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

# CPU 线程：取 min(8, 核数)，避免线程超订反而变慢
_N_THREADS = min(8, os.cpu_count() or 4)
torch.set_num_threads(_N_THREADS)
print(f"[env] cpu_count={os.cpu_count()} -> torch threads={_N_THREADS}")
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


def build_prompt(tokenizer, user_text):
    """用 Qwen2.5-Instruct 官方 chat 模板包一层。

    Instruct 模型必须用模板提问，否则分布不匹配、模型会乱答（实测裸 prompt
    下 Base 模型 bigram-F1 只有 0.03 量级）。训练与评估必须用同一套模板。
    """
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def tokenize(example, tokenizer):
    """整体 tokenize + 只对 assistant 回答部分计算损失（prompt 掩码为 -100）。"""
    user_text = example["prompt"]
    resp = example["response"]

    rendered_prompt = build_prompt(tokenizer, user_text)
    full_text = rendered_prompt + resp + tokenizer.eos_token

    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    input_ids = full_ids[:MAX_LEN]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:MAX_LEN]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基座模型：{BASE_MODEL}（CPU / float32）")
    # 注意：Qwen2.5 默认 torch_dtype=bfloat16，CPU 上 bf16 是软件模拟，慢 20~50 倍，
    # 必须显式转 float32，否则单步耗时会从秒级劣化到分钟级。
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
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
        # 小样本（26 条）需要多轮才能学会领域话术，10 epoch 在 CPU 上约 2 分钟
        num_train_epochs=10,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
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

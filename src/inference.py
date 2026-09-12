"""加载 LoRA 微调后模型做单条推理（demo 用）。

运行：
    python src/inference.py "糖尿病的诊断标准是什么？"
依赖：torch, transformers, peft
"""
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
LORA = "outputs/qwen2.5-lora-medical"


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype="auto")
    m = PeftModel.from_pretrained(m, LORA)
    q = sys.argv[1] if len(sys.argv) > 1 else "高血压的诊断标准是什么？"
    inputs = tok(q, return_tensors="pt").to(m.device)
    out = m.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()

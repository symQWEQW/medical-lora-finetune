"""微调前后回答质量对比评估。

运行（需先跑 train_lora.py 得到 outputs/qwen2.5-lora-medical）：
    python src/eval_compare.py
依赖：torch, transformers, peft

逻辑：分别用基座模型与 LoRA 微调后模型对 data/eval.json 生成回答，
用 token F1（与参考答案的字符/词重叠）量化对比，输出 Base vs LoRA 平均分与提升。
这是「我碰过模型内部、能量化调优」的差异化证据。
"""
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
LORA = "outputs/qwen2.5-lora-medical"


def load_model(base, lora=None):
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype="auto")
    if lora:
        m = PeftModel.from_pretrained(m, lora)
    return tok, m


def generate(tok, model, prompt, max_new=128):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def token_f1(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    p = len(inter) / len(sa)
    r = len(inter) / len(sb)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    base_tok, base_m = load_model(BASE)
    ft_tok, ft_m = load_model(BASE, LORA)

    with open("data/eval.json", encoding="utf-8") as f:
        tests = json.load(f)

    base_scores, ft_scores = [], []
    for t in tests:
        p = t["instruction"] + ("\n" + t["input"] if t.get("input") else "")
        gold = t["output"]
        b = generate(base_tok, base_m, p)
        f1 = generate(ft_tok, ft_m, p)
        base_scores.append(token_f1(gold, b))
        ft_scores.append(token_f1(gold, f1))
        print(f"Q: {t['instruction']}")
        print(f"  Base F1={base_scores[-1]:.3f} | LoRA F1={ft_scores[-1]:.3f}")

    n = len(tests)
    print("\n==== 汇总 ====")
    print(f"Base 平均 F1: {sum(base_scores)/n:.3f}")
    print(f"LoRA 平均 F1: {sum(ft_scores)/n:.3f}")
    print(f"提升: {(sum(ft_scores)-sum(base_scores))/n:+.3f}")


if __name__ == "__main__":
    main()

"""微调前后回答质量对比评估。

运行（需先跑 train_lora.py 得到 outputs/qwen2.5-lora-medical）：
    python src/eval_compare.py
依赖：torch, transformers, peft

指标说明：
- 采用「字符级 bigram（二元组）多重集 F1」作为主指标，比单字集合 F1 更严格，
  能惩罚「乱答/答非所问」，更真实地反映微调是否让模型学到了领域知识。
- 同时输出「字符级 unigram F1」作为参考。
- 结果同时打印并写入 outputs/eval_result.txt，方便截图/复制进简历与博客。

这是「我碰过模型内部、能量化调优」的差异化证据。
"""
import json
import os
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "models/Qwen2.5-0.5B-Instruct"
LORA = "outputs/qwen2.5-lora-medical"


def _grams(s, n=2):
    """字符级 n-gram 多重集；文本短于 n 时退回整串，空串返回空集合。"""
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def token_f1(a, b, n=2):
    ca, cb = _grams(a, n), _grams(b, n)
    if not ca or not cb:
        return 0.0
    inter = ca & cb
    p = sum(inter.values()) / sum(ca.values())
    r = sum(inter.values()) / sum(cb.values())
    return 2 * p * r / (p + r) if (p + r) else 0.0


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


def main():
    if not os.path.exists(LORA):
        print(f"[跳过] 未找到 LoRA 适配器：{LORA}\n请先运行：python src/train_lora.py")
        return

    base_tok, base_m = load_model(BASE)
    ft_tok, ft_m = load_model(BASE, LORA)

    with open("data/eval.json", encoding="utf-8") as f:
        tests = json.load(f)

    base_bi, ft_bi, base_uni, ft_uni = [], [], [], []
    for t in tests:
        p = t["instruction"] + ("\n" + t["input"] if t.get("input") else "")
        gold = t["output"]
        b = generate(base_tok, base_m, p)
        f1 = generate(ft_tok, ft_m, p)
        sb, s = token_f1(gold, b, 2), token_f1(gold, f1, 2)
        ub, uf = token_f1(gold, b, 1), token_f1(gold, f1, 1)
        base_bi.append(sb); ft_bi.append(s)
        base_uni.append(ub); ft_uni.append(uf)
        print(f"Q: {t['instruction']}")
        print(f"  Base  bigramF1={sb:.3f} uniF1={ub:.3f} | LoRA bigramF1={s:.3f} uniF1={uf:.3f}")

    n = len(tests)
    avg = lambda xs: sum(xs) / n
    base_avg, ft_avg = avg(base_bi), avg(ft_bi)
    base_u, ft_u = avg(base_uni), avg(ft_uni)

    lines = []
    lines.append("\n==== 汇总（字符级 bigram F1，主指标）====")
    lines.append(f"样本数            : {n}")
    lines.append(f"Base  平均 F1     : {base_avg:.3f}")
    lines.append(f"LoRA  平均 F1     : {ft_avg:.3f}")
    lines.append(f"绝对提升          : {ft_avg - base_avg:+.3f}")
    if base_avg > 0:
        lines.append(f"相对提升          : {(ft_avg - base_avg) / base_avg * 100:+.1f}%")
    lines.append("")
    lines.append("==== 参考（字符级 unigram F1）====")
    lines.append(f"Base  平均 F1     : {base_u:.3f}")
    lines.append(f"LoRA  平均 F1     : {ft_u:.3f}")
    lines.append("")
    if ft_avg > base_avg:
        lines.append(
            f"✅ 结论：LoRA 微调后 token-F1 从 {base_avg:.3f} 提升到 {ft_avg:.3f}"
            f"（相对 {(ft_avg - base_avg) / base_avg * 100:+.1f}%），"
            f"证明在 {n} 道医疗/政务问答上模型学会了领域知识。"
        )
    else:
        lines.append("⚠️ 本次 LoRA 未超过 Base，可能数据量/epoch 不足，可调大 medical_qa_sample.json 或 epoch 后重跑。")
    report = "\n".join(lines)
    print(report)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_result.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n[已保存] outputs/eval_result.txt")


if __name__ == "__main__":
    main()

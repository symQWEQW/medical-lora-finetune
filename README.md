# 🧬 医疗 / 政务问答 LoRA 微调 · 原理级项目

> 证明「我碰过模型内部」的差异化证据——用 PEFT / LoRA 在医疗问答语料上微调小模型，并用 token-F1 量化微调前后的回答质量提升。

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![LoRA](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-green)
![License](https://img.shields.io/badge/License-MIT-green)

> 📓 没 GPU 也想跑？[Colab 一键复现（训练+评估）](https://colab.research.google.com/github/symQWEQW/medical-lora-finetune/blob/master/colab_train_eval.ipynb)

---

> **为什么做这个**：前面江陵项目是「基于现成大模型做应用」，本仓库补一个能说
> **"我碰过模型内部"** 的证据——用 LoRA/QLoRA 在医疗/政务问答上微调小模型，
> 并**量化对比微调前后回答质量**。这是从「AI 应用」跨向「AI 工程师」最有力的一枚砝码，
> 且大专学历也能上手（微调不需要博士）。

## 它解决什么

| 对比 | 仅用现成大模型（应用层） | 本仓库（原理层） |
|---|---|---|
| 是否动过模型权重 | 否 | 是（LoRA 适配器） |
| 能否量化"微调有没有用" | 不能 | 能（eval_compare 出 F1 数字） |
| 面试说服力 | "我会调 API" | "我训练并评估过模型" |

## 目录结构

```
medical-lora-finetune/
├── README.md
├── requirements.txt
├── data/
│   ├── medical_qa_sample.json   # 训练集（16 条医疗/政务问答，alpaca 格式）
│   └── eval.json                # 测试集（6 条，独立于训练）
└── src/
    ├── train_lora.py            # LoRA 微调训练
    ├── eval_compare.py          # 微调前后回答质量对比（F1）
    └── inference.py             # 单条推理 demo
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 训练（默认 Qwen2.5-0.5B，消费级显卡即可；生产改 7B + 4bit 量化）
python src/train_lora.py

# 2) 对比微调前后（输出 Base vs LoRA 平均 F1 与提升）
python src/eval_compare.py

# 3) 单条推理
python src/inference.py "糖尿病的诊断标准是什么？"
```

## 关键设计

- **只训 LoRA 适配器**：不改动基座 70 亿参数，显存占用低、训练快、可随时回退/热插拔。
- **小模型演示**：默认 0.5B 证明流程跑通；真实场景用 `Qwen2.5-7B-Instruct` + `bitsandbytes` 4bit 量化。
- **量化评估**：`eval_compare.py` 用 token F1 对比基座与微调模型在测试集上的回答质量，
  给出**可写进简历的真实数字**（需在有 GPU 环境运行后填写）。

## 无 GPU 怎么办

- **Colab / Kaggle**：免费 T4 显卡，直接跑 `train_lora.py`（约几分钟/epoch）。
- **阿里云 PAI / 魔搭 ModelScope**：领取免费算力。
- **CPU 兜底**：可设 `device_map="cpu"` 跑通流程，但速度慢，仅验证代码正确性。

## 与江陵项目的关系

江陵项目用现成 `qwen2.5:7b` 做 RAG 应用；本仓库证明你**也能训练和评估模型本身**，
二者组合 = "既会做 AI 应用，也懂模型原理"的完整画像，直接命中楚天云等 JD
（LangChain/LangGraph/Spring AI，Dify 优先）对"懂大模型"的要求。

## 预期效果（诚实说明）

- F1 提升幅度取决于数据量与领域匹配度。16 条样本为演示，真实项目建议 200–1000 条垂直语料。
- 实际数字请在本机/云端 GPU 跑 `eval_compare.py` 后填入简历与博客。
- License：MIT（数据均为模拟，仅供演示）。

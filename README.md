# 🧬 医疗 / 政务问答 LoRA 微调 · 原理级项目

> 证明「我碰过模型内部」的差异化证据——用 PEFT / LoRA 在医疗问答语料上微调小模型，并用 token-F1 量化微调前后的回答质量提升。

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![LoRA](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-green)
![License](https://img.shields.io/badge/License-MIT-green)

> 📓 没 GPU 也想跑？[Colab 一键复现（训练+评估）](https://colab.research.google.com/github/symQWEQW/medical-lora-finetune/blob/master/colab_train_eval.ipynb)

---

> **为什么做这个**：前面某地项目是「基于现成大模型做应用」，本仓库补一个能说
> **"我碰过模型内部"** 的证据——用 LoRA/QLoRA 在医疗/政务问答上微调小模型，
> 并**量化对比微调前后回答质量**。这是从「AI 应用」跨向「AI 工程师」最有力的一枚砝码，
> 且大专学历也能上手（微调不需要博士）。

## 🧪 实测结果（本机 CPU 真实跑出，非估算）

> 环境：Qwen2.5-0.5B-Instruct + LoRA(r=16, α=32) + 26 条训练样本 / 10 epoch
> 评测：6 道独立测试题，字符级 bigram F1（main）+ unigram F1（参考）

| 指标 | Base（未微调） | LoRA 微调后 | 提升 |
|---|---|---|---|
| **字符级 bigram F1**（主指标） | 0.065 | **0.150** | **+130.3%** |
| 字符级 unigram F1（参考） | 0.162 | 0.327 | +101.9% |
| 训练 loss | 2.898（第 1 epoch） | 0.470（第 10 epoch） | — |

样例对比（同一道题）：

| | 回答 |
|---|---|
| 问题 | 患者仅单次收缩压 145mmHg 能否认定高血压？ |
| Base | "是的，单次收缩压达到或超过 140 mmHg 可以被认为是高血压……" ❌ 结论错误 |
| **LoRA** | "不能。单次收缩压 145/90mmHg 不符合门诊慢特病药品目录认定标准，需到住院证种复核后方可认定。" ✅ 结论正确、口吻贴近政策 |

结论：微调后模型不仅**结论更准**，回答风格也从"通用科普"转成了"政务政策口吻"。
完整输出见 `outputs/eval_result.txt`（跑 `python src/eval_compare.py` 可复现）。

---

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
│   ├── medical_qa_sample.json   # 训练集（26 条医疗/政务问答，alpaca 格式）
│   └── eval.json                # 测试集（6 条，独立于训练、无泄漏）
└── src/
    ├── train_lora.py            # LoRA 微调训练（GPU 版）
    ├── train_lora_cpu.py        # LoRA 微调训练（CPU 友好版，实测 3.5 分钟跑完）
    ├── download_model.py        # 模型下载兜底（hf 被墙时走 ModelScope）
    ├── eval_compare.py          # 微调前后回答质量对比（bigram F1）
    └── inference.py             # 单条推理 demo
```

## 快速开始

```bash
pip install -r requirements.txt
python src/download_model.py      # 国内网络建议先跑，把模型拉到本地 models/

# 1) 训练（有 GPU 用 train_lora.py；无 GPU / 只想验证用 train_lora_cpu.py）
python src/train_lora_cpu.py      # CPU 实测 3.5 分钟（10 epoch）

# 2) 对比微调前后（输出 Base vs LoRA 平均 F1 与提升）
python src/eval_compare.py        # 结果写入 outputs/eval_result.txt

# 3) 单条推理
python src/inference.py "糖尿病的诊断标准是什么？"
```

## 关键设计

- **只训 LoRA 适配器**：r=16 / α=32，可训练参数 216 万，仅占全模型 0.44%，
  不改动基座权重、显存占用低、可随时回退/热插拔。
- **chat 模板一致性（踩坑重点）**：Qwen2.5-Instruct 是**指令微调模型**，
  训练和推理都必须套官方 chat 模板（`<|im_start|>user/assistant`）。
  实测裸 prompt 下 Base 模型 bigram-F1 仅 0.038，属于"问法不对"而非"模型不行"。
  同时推理必须传 `eos_token_id`，否则模型会无限续写稀释 F1。
- **CPU 上的 dtype 坑**：Qwen2.5 默认 `torch_dtype=bfloat16`，CPU 上 bf16 是**软件模拟**，
  实测单步从 73 秒劣化到……改回 `float32` 后单步 0.5 秒，**快约 150 倍**。
- **量化评估**：`eval_compare.py` 用**字符级 bigram（二元组）多重集 F1** 对比基座与微调模型，
  比单字集合 F1 更严格，能惩罚"答非所问"。

## 无 GPU 怎么办

- **CPU 也能真跑出数字（已验证）**：`train_lora_cpu.py` 在 20 核 CPU 上
  **3.5 分钟训完 10 epoch**，评估 1 分钟出 F1，就是上面那组实测数字。
  关键是要 `torch_dtype=float32`（别用模型默认的 bfloat16）。
- **Colab / Kaggle**：免费 T4 显卡，直接跑 `train_lora.py`（约几分钟/epoch）。
- **阿里云 PAI / 魔搭 ModelScope**：领取免费算力。
- **模型下载被墙**：`python src/download_model.py` 走 ModelScope 镜像落地到 `models/`。

## 诚实说明（局限性）

- 26 条样本属于**极小样本演示**，F1 绝对值（0.15）仍然偏低，说明
  "学会了领域话术与结论倾向"，但**远未达到可上线水平**。
- 真实项目建议 200–1000 条垂直语料 + 7B 基座 + 更多 epoch，F1 才有工程意义。
- 本项目价值在于：**完整跑通"训练 → 评估 → 用指标证明提升"这条 AI 工程闭环**，
  而非追求 SOTA 分数。
- License：MIT（数据均为模拟，仅供演示）。

## 与某地项目的关系

某地项目用现成 `qwen2.5:7b` 做 RAG 应用；本仓库证明你**也能训练和评估模型本身**，
二者组合 = "既会做 AI 应用，也懂模型原理"的完整画像，直接命中楚天云等 JD
（LangChain/LangGraph/Spring AI，Dify 优先）对"懂大模型"的要求。

## License

MIT（数据均为模拟，仅供演示）。

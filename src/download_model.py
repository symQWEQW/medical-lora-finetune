"""用 modelscope 下载 Qwen2.5-0.5B-Instruct 到本地（hf-mirror 不稳时的兜底）。

用法：
    python src/download_model.py
下载到： models/Qwen2.5-0.5B-Instruct （已在 .gitignore 忽略，不入库）
"""
import os
from modelscope import snapshot_download

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct")
os.makedirs(OUT, exist_ok=True)

print("开始从 modelscope 下载 Qwen2.5-0.5B-Instruct ...")
path = snapshot_download(
    "qwen/Qwen2.5-0.5B-Instruct",
    cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"),
)
print("下载完成，路径：", path)

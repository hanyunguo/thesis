from transformers import pipeline
from datasets import load_metric

# 1. 选择一个支持超长输入的摘要模型
#    这里我们用 Facebook 的 LED（Longformer Encoder-Decoder）：
model_name = "allenai/led-base-16384"
# model_name = "allenai/longformer-base-4096"

# 2. 创建一个 summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model_name,
    tokenizer=model_name,
    framework="pt",         # 或 "tf" 取决于你装的是 torch 还是 tf
    device=0,               # 如果有 GPU，用 0，否则忽略
)

# 3. 准备一段长文本
long_text = """
In an era where information overload is the norm, it's more important than ever to be able to distill 
large volumes of text into concise, digestible summaries. Whether you're dealing with academic 
papers, legal documents, or customer reviews, the ability to quickly grasp the key points can save 
you hours of reading time. Traditional transformer models like BERT or vanilla Longformer are 
limited by a maximum sequence length—often 512 or 4,096 tokens—making them ill-suited for 
summarizing truly long documents. Facebook AI's LED (Longformer-Encoder-Decoder) architecture 
solves this by combining Longformer's sparse attention in the encoder with a standard 
cross-attention decoder, allowing inputs up to 16,384 tokens (or more, depending on the checkpoint).
LED also retains the global-attention mechanism to focus on crucial “leader” tokens (e.g. headings, 
first sentences), which helps it produce more coherent, high-quality summaries. Below, we'll show 
how to call LED via the pipeline and tweak parameters like max_length and min_length to suit your 
needs.
"""

# 4. 用 pipeline 直接生成摘要
summary = summarizer(
    long_text,
    max_length=150,   # 摘要最大长度（tokens），根据需要调节
    min_length=40,    # 摘要最小长度
    do_sample=False,  # 用 beam search 而不是随机采样
    num_beams=4,      # beam 大小，越大越有可能得到更完整的摘要
)[0]["summary_text"]

print("=== 摘要结果 ===")
print(summary)

rouge = load_metric("rouge")


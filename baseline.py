
import logging
from evaluate import load as load_metric
from transformers import (
    LongformerTokenizerFast,
    LongformerForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import math

logging.basicConfig(level=logging.INFO)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # 将预测和标签转成文本
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 计算 ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # 把 LongFloat 转成标准 Python Float
    result = {k: round(v.mid.fmeasure, 4) for k, v in result.items()}
    return result

def main():
    # 1. 指定模型名
    model_name = "allenai/longformer-base-4096"
    # 2. 加载 tokenizer 和模型
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    model = LongformerForMaskedLM.from_pretrained(model_name)

    # 3. 加载公开数据集（这里以 WikiText 为例）
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0")

    # cnn_ds = load_dataset("cnn_dailymail", "3.0.0")
    # # 英英国广播新闻 XSum
    # xsum_ds = load_dataset("xsum")
    # # 多文档新闻摘要 Multi-News
    # multi_news = load_dataset("multi_news")

    # # 科研论文摘要：ArXiv + PubMed 科学论文
    # sci_papers = load_dataset("scientific_papers", "arxiv")

    rouge = load_metric("rouge")

    # 4. 把文本打散成合适的长度（Longformer 最长能处理 4096 tokens）
    block_size = 1024  # 也可以 2048、4096，根据显存调整
    def tokenize_function(examples):
        # 将多行文本拼成一个大字符串，再切分
        return tokenizer(examples["article"], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["highlights", "id"],
    )
    # 5. 把 token 切成 block_size 大小
    def group_texts(examples):
        #  concat，将所有 tokens 连起来，再按 block_size 切块
        concatenated = sum(examples["input_ids"], [])
        total_length = len(concatenated)
        total_length = (total_length // block_size) * block_size
        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        remove_columns=tokenized_datasets["train"].column_names
    )
    # 6. 构造数据 collator，用于随机 mask
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 7. 设置训练参数：这里只做 evaluation，不做训练
    training_args = TrainingArguments(
        output_dir="./mlm-baseline-output",
        do_train=False, 
        do_eval=True,
        per_device_eval_batch_size=2,
        eval_steps=None,
        logging_steps=500,
        dataloader_drop_last=False,
    )

    # 8. 用 Trainer 来跑 evaluate，这里只跑 eval 来得到验证集上的 loss
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    eval_result = trainer.evaluate()
    print(f">>> baseline MLM 的验证损失（loss）: {eval_result['eval_loss']:.4f}")
    print(f"rouge: ", eval_result)
    loss = eval_result["eval_loss"]
    perplexity = math.exp(loss)
    print(f">>> Perplexity: {perplexity:.2f}")
    # 备注：有些 Transformers 版本会自动提交 eval_perplexity
    # 如果没有，可以手动计算： perplexity = exp(eval_loss)
if __name__ == "__main__":
    main()
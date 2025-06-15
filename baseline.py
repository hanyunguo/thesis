import logging
import numpy as np
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
import torch
import itertools

logging.basicConfig(level=logging.INFO)

# 封装 compute_metrics，传入 tokenizer 和 rouge
def build_compute_metrics(tokenizer, rouge):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # 转换 logits -> predicted token IDs
        if isinstance(preds, tuple):  # sometimes preds comes with extra info
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1)

        # 将标签中的 -100 替换成 pad_token_id（否则 decode 会报错）
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # 解码
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算 ROUGE
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
        
    return compute_metrics

# 封装 tokenize_function
def build_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["article"], return_special_tokens_mask=True)
    return tokenize_function

def main():
    model_name = "allenai/longformer-base-4096"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    model = LongformerForMaskedLM.from_pretrained(model_name).to(device)
    print("模型所在设备:", next(model.parameters()).device)

    raw_datasets = load_dataset("cnn_dailymail", "3.0.0", split={"train": "train[:1%]", "validation": "validation[:1%]"})
    rouge = load_metric("rouge")

    tokenized_datasets = raw_datasets.map(
        build_tokenize_function(tokenizer),
        batched=True,
        remove_columns=["highlights", "id"],
        num_proc=1
    )

    block_size = 1024
    def group_texts(examples):
        concatenated = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(concatenated) // block_size) * block_size
        return {
            "input_ids": [
                concatenated[i: i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        remove_columns=tokenized_datasets["train"].column_names,
        num_proc=1
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./mlm-baseline-output",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        eval_steps=None,
        logging_steps=500,
        dataloader_drop_last=False,
        eval_accumulation_steps=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, rouge),
    )

    eval_result = trainer.evaluate()
    print(f">>> baseline MLM 的验证损失（loss）: {eval_result['eval_loss']:.4f}")
    print(f"ROUGE: {eval_result}")
    loss = eval_result["eval_loss"]
    perplexity = math.exp(loss)
    print(f">>> Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()

import logging
import numpy as np
from evaluate import load as load_metric
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["document"],
        max_length=4096,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "allenai/led-base-16384"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global tokenizer
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    print("模型所在设备:", next(model.parameters()).device)

    raw_datasets = load_dataset("multi_news", split={"train": "train[:1%]", "validation": "validation[:1%]"}, trust_remote_code=True)
    rouge = load_metric("rouge")

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["document", "summary"],
        desc="Tokenizing",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./led-baseline-output",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=build_compute_metrics(tokenizer, rouge),
    )
    eval_result = trainer.evaluate()
    print(f">>> baseline MLM 的验证损失（loss）: {eval_result['eval_loss']:.4f}")
    print(f"ROUGE: {eval_result}")
    loss = eval_result["eval_loss"]
    if "eval_loss" in eval_result:
        print(f">>> Perplexity: {math.exp(eval_result['eval_loss']):.2f}")

if __name__ == "__main__":
    main()

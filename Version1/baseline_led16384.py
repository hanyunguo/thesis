import logging
import numpy as np
from evaluate import load as load_metric
from transformers import (
    LEDTokenizerFast,  # ✅ 替换为 Fast 版本
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import math
import torch

logging.basicConfig(level=logging.INFO)

def build_compute_metrics(tokenizer, rouge):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v, 4) for k, v in result.items()}
    return compute_metrics


def preprocess_function(examples):
    # Encode document
    model_inputs = tokenizer(
        examples["document"],
        max_length=4096,
        truncation=True,
        padding="max_length"
    )

    # Encode summary (as labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]

    # ✅ Add global_attention_mask: [1, 0, 0, ..., 0]
    model_inputs["global_attention_mask"] = [
        [1] + [0] * (len(input_id) - 1)
        for input_id in model_inputs["input_ids"]
    ]

    return model_inputs


def main():
    model_name = "allenai/led-base-16384"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global tokenizer
    tokenizer = LEDTokenizerFast.from_pretrained(model_name)
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
        do_train=False,  # ✅ 只评估
        do_eval=True,
        per_device_eval_batch_size=1,
        predict_with_generate=True,  # ✅ 启用生成
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
    print(f">>> baseline LED 的验证损失（loss）: {eval_result['eval_loss']:.4f}")
    print(f"ROUGE: {eval_result}")
    if "eval_loss" in eval_result:
        print(f">>> Perplexity: {math.exp(eval_result['eval_loss']):.2f}")


if __name__ == "__main__":
    main()

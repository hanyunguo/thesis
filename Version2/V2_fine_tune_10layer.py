import os
import math
import shutil
import logging
import argparse
import torch
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    LEDTokenizerFast, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    AutoConfig,
    AutoModelForSeq2SeqLM
)
from transformers.models.led.configuration_led import LEDConfig
from Version2.V2_modeling_led_with_memory import LEDForConditionalGenerationWithMemory, LogProgressCallback
from tqdm import tqdm

logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

def evaluate_on_gpu(model, dataloader, tokenizer, rouge):
    model.eval()
    model.to("cuda")
    decoded_preds = []
    decoded_labels = []
    total_loss = 0.0
    total_tokens = 0

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            total_loss += loss.item() * shift_labels.ne(-100).sum().item()
            total_tokens += shift_labels.ne(-100).sum().item()

            pred_ids = torch.argmax(logits, dim=-1)
            labels_for_decode = torch.where(labels == -100, tokenizer.pad_token_id, labels)

            decoded_batch_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            decoded_batch_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

            decoded_preds.extend(decoded_batch_preds)
            decoded_labels.extend(decoded_batch_labels)

            del outputs, logits, pred_ids, labels, batch
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v, 4) for k, v in rouge_result.items()}

    result = {
        "eval_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        **rouge_result
    }
    return result

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
    model_inputs = tokenizer(
        examples["document"], max_length=4096, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=512, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["global_attention_mask"] = [
        [1] + [0] * (len(input_id) - 1) for input_id in model_inputs["input_ids"]
    ]
    return model_inputs

def resolve_checkpoint(output_dir, tokenizer, model):
    from transformers.trainer_utils import get_last_checkpoint
    if not os.path.isdir(output_dir):
        return None
    checkpoint = get_last_checkpoint(output_dir)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        return checkpoint
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_restart", action="store_true", help="If set, delete checkpoints and restart training from scratch.")
    args = parser.parse_args()

    model_name = "allenai/PRIMERA-multinews"
    output_dir = "./led-memory-freeze10-output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global tokenizer
    tokenizer = LEDTokenizerFast.from_pretrained(model_name)
    rouge = load_metric("rouge")

    model = LEDForConditionalGenerationWithMemory.from_pretrained(model_name).to(device)

    # 冻结 encoder 前10层
    print("Freezing encoder 前10层参数...")
    for i, layer in enumerate(model.model.encoder.layers):
        if i < 10:
            for param in layer.parameters():
                param.requires_grad = False

    # 显示可训练参数比例
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    model.gradient_checkpointing_enable()
    model.config.max_length = 512
    model.config.decoder_max_position_embeddings = 512 

    raw_datasets = load_dataset("multi_news", split={"train": "train", "validation": "validation[:5%]"}, trust_remote_code=True)
    tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=["document", "summary"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        generation_max_length=256,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_steps=100,
        save_steps=500,
        save_total_limit=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        gradient_checkpointing=True,
        warmup_steps=100,
        predict_with_generate=False,
        fp16=True,
        push_to_hub=False,
        save_safetensors=False,
    )

    if args.force_restart and os.path.isdir(output_dir):
        print(f"--force_restart enabled, deleting checkpoint directory {output_dir}")
        shutil.rmtree(output_dir)
        checkpoint = None
    else:
        checkpoint = resolve_checkpoint(output_dir, tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, rouge),
        callbacks=[LogProgressCallback()],
    )

    trainer.train(resume_from_checkpoint=checkpoint)
    eval_dataloader = trainer.get_eval_dataloader()

    print(">>> Running custom GPU evaluation...")
    eval_result = evaluate_on_gpu(model, eval_dataloader, tokenizer, rouge)

    print(">>> eval_loss:", eval_result["eval_loss"])
    print(">>> perplexity:", math.exp(eval_result["eval_loss"]))
    print(">>> ROUGE:", eval_result)
    trainer.save_model("./memory_fuse_freeze10_epoch1")

if __name__ == "__main__":
    main()

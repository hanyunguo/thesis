import os
import math
import shutil
import logging
import itertools
import argparse
import torch
import numpy as np

from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    LongformerTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig,
    TrainerState,
)
from modeling_longformer_with_memory import LongformerForMaskedLMWithMemory
from tqdm import tqdm

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
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            labels = batch["labels"]

            # Compute loss manually
            shift_logits = logits.view(-1, logits.size(-1))
            shift_labels = labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            total_loss += loss.item() * shift_labels.ne(-100).sum().item()
            total_tokens += shift_labels.ne(-100).sum().item()

            # Predictions
            pred_ids = torch.argmax(logits, dim=-1)
            labels_for_decode = torch.where(labels == -100, tokenizer.pad_token_id, labels)

            decoded_batch_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            decoded_batch_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

            decoded_preds.extend(decoded_batch_preds)
            decoded_labels.extend(decoded_batch_labels)

            # Free memory
            del outputs, logits, pred_ids, labels, batch
            torch.cuda.empty_cache()

    # Final loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    # ROUGE
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v, 4) for k, v in rouge_result.items()}

    # Combine results
    result = {
        "eval_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        **rouge_result
    }
    return result

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


def build_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["article"], return_special_tokens_mask=True)

    return tokenize_function


def is_checkpoint_compatible(checkpoint_path, tokenizer, block_size, model):
    try:
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            return False
        _ = TrainerState.load_from_json(trainer_state_path)

        dummy_input = tokenizer(
            "Test input",
            return_tensors="pt",
            padding="max_length",
            max_length=block_size,
            truncation=True,
        )
        dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}
        model.eval()
        with torch.no_grad():
            model(**dummy_input)
        return True
    except Exception as e:
        print(f"Checkpoint incompatible: {e}")
        return False


def resolve_checkpoint(output_dir, tokenizer, block_size, model):
    from transformers.trainer_utils import get_last_checkpoint

    if not os.path.isdir(output_dir):
        return None
    checkpoint = get_last_checkpoint(output_dir)
    if checkpoint and is_checkpoint_compatible(checkpoint, tokenizer, block_size, model):
        print(f"Resuming from checkpoint: {checkpoint}")
        return checkpoint
    elif checkpoint:
        print(f"Deleting incompatible checkpoint: {checkpoint}")
        shutil.rmtree(checkpoint)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="If set, delete checkpoints and restart training from scratch.",
    )
    args = parser.parse_args()

    model_name = "allenai/longformer-base-4096"
    block_size = 1024
    output_dir = "./mlm-memory-output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. tokenizer & metric
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    rouge = load_metric("rouge")

    # 2. 配置 & 模型加载
    config = AutoConfig.from_pretrained(model_name)
    model = LongformerForMaskedLMWithMemory(config).to(device)
    model.gradient_checkpointing_enable()  # 节省显存

    # 加载预训练权重（不含 memory 部分）
    from transformers.models.longformer.modeling_longformer import LongformerForMaskedLM

    orig = LongformerForMaskedLM.from_pretrained(model_name)
    model.longformer.embeddings.load_state_dict(orig.longformer.embeddings.state_dict())
    for i, layer in enumerate(orig.longformer.encoder.layer):
        model.longformer.encoder.layer[i].attention.load_state_dict(
            layer.attention.state_dict()
        )
        model.longformer.encoder.layer[i].intermediate.load_state_dict(
            layer.intermediate.state_dict()
        )
        model.longformer.encoder.layer[i].output.load_state_dict(layer.output.state_dict())
    model.lm_head.load_state_dict(orig.lm_head.state_dict())
    del orig

    for name, param in model.named_parameters():
        if "memory_fuse" in name or name.startswith("lm_head") or name.startswith("longformer.embeddings"):
            param.requires_grad = True  # 开启训练
        else:
            param.requires_grad = False  # 冻结参数

    # 3. 数据加载 & 分词
    raw_datasets = load_dataset(
        "cnn_dailymail", "3.0.0", split={"train": "train", "validation": "validation[:5%]"}
    )
    tokenized = raw_datasets.map(
        build_tokenize_function(tokenizer),
        batched=True,
        remove_columns=["highlights", "id"],
    )

    # 4. 拼接切块
    def group_texts(examples):
        all_ids = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(all_ids) // block_size) * block_size
        return {
            "input_ids": [all_ids[i : i + block_size] for i in range(0, total_length, block_size)]
        }

    lm_datasets = tokenized.map(
        group_texts, batched=True, remove_columns=tokenized["train"].column_names
    )

    # 5. DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=1,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=1,
        num_train_epochs=3,
        learning_rate = 5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=True,
        push_to_hub=False,
        save_safetensors=False,
    )

    # 如果强制重启，清空 checkpoint
    if args.force_restart and os.path.isdir(output_dir):
        print(f"--force_restart enabled, deleting checkpoint directory {output_dir}")
        shutil.rmtree(output_dir)
        checkpoint = None
    else:
        checkpoint = resolve_checkpoint(output_dir, tokenizer, block_size, model)

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, rouge),
    )

    # 8. 训练 & 评估
    trainer.train(resume_from_checkpoint=checkpoint)
    # 自定义评估，不用 Trainer.evaluate()
    eval_dataloader = trainer.get_eval_dataloader()

    print(">>> Running custom GPU evaluation...")
    eval_result = evaluate_on_gpu(model, eval_dataloader, tokenizer, rouge)

    print(">>> eval_loss:", eval_result["eval_loss"])
    print(">>> perplexity:", math.exp(eval_result["eval_loss"]))
    print(">>> ROUGE:", eval_result)


if __name__ == "__main__":
    main()

import torch
import logging, math, itertools, torch
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    LongformerTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from modeling_longformer_with_memory import LongformerForMaskedLMWithMemory

logging.basicConfig(level=logging.INFO)

def build_compute_metrics(tokenizer, rouge):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1)
        labels   = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: round(v,4) for k,v in result.items()}
    return compute_metrics

def build_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["article"], return_special_tokens_mask=True)
    return tokenize_function

def main():
    model_name = "allenai/longformer-base-4096"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. tokenizer & metric
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
    rouge     = load_metric("rouge")

    # 2. 配置 & 带记忆模型
    config = AutoConfig.from_pretrained(model_name)
    model  = LongformerForMaskedLMWithMemory(config).to(device)
    # 加载预训练权重到 encoder/embedding/MLM-head 的那部分
    from transformers.models.longformer.modeling_longformer import LongformerForMaskedLM
    # 这里只加载原版的权重，MemoryFuse 随机初始化
    orig = LongformerForMaskedLM.from_pretrained(model_name)
    # copy embedding + LM head + longformer.encoder.*.attention/intermediate/output 权重
    model.longformer.embeddings.load_state_dict(orig.longformer.embeddings.state_dict())
    for i, layer in enumerate(orig.longformer.encoder.layer):
        model.longformer.encoder.layer[i].attention.load_state_dict(layer.attention.state_dict())
        model.longformer.encoder.layer[i].intermediate.load_state_dict(layer.intermediate.state_dict())
        model.longformer.encoder.layer[i].output.load_state_dict(layer.output.state_dict())
    model.lm_head.load_state_dict(orig.lm_head.state_dict())
    del orig

    # 3. 加载数据 & 分词
    raw_datasets = load_dataset("cnn_dailymail", "3.0.0",
                                split={"train": "train[:1%]", "validation": "validation[:1%]"})
    tokenized = raw_datasets.map(
        build_tokenize_function(tokenizer),
        batched=True, remove_columns=["highlights","id"]
    )

    # 4. 把 input_ids 拼接 & 切块
    block_size = 2048
    def group_texts(examples):
        all_ids      = list(itertools.chain.from_iterable(examples["input_ids"]))
        total_length = (len(all_ids) // block_size) * block_size
        return {"input_ids": [
            all_ids[i : i + block_size] for i in range(0, total_length, block_size)
        ]}
    lm_datasets = tokenized.map(
        group_texts, batched=True,
        remove_columns=tokenized["train"].column_names
    )

    # 5. DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir="./mlm-memory-output",

        # 训练/评估开关
        do_train=True,
        do_eval=True,

        # batch sizes
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        # eval + logging
        eval_steps=500,
        logging_steps=500,

        # checkpoint
        save_steps=1000,

        # epochs & LR 调度
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_steps=100,

        fp16=False,
        push_to_hub=False,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset= lm_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer, rouge),
    )

    # 8. 训练 & 评估
    trainer.train()
    eval_result = trainer.evaluate()
    print(">>> eval_loss:", eval_result["eval_loss"])
    print(">>> perplexity:", math.exp(eval_result["eval_loss"]))
    print(">>> ROUGE:", eval_result)

if __name__ == "__main__":
    main()

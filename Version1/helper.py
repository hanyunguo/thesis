import torch
from transformers import AutoConfig, LongformerTokenizerFast
from modeling_led_with_memory import LongformerForMaskedLMWithMemory

def print_parameter_status(model):
    print("\n>>> 模型参数状态（True 表示会被训练）:")
    for name, param in model.named_parameters():
        print(f"{name:60} requires_grad = {param.requires_grad}")

def count_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")

def main():
    model_name = "allenai/longformer-base-4096"
    config = AutoConfig.from_pretrained(model_name)
    model = LongformerForMaskedLMWithMemory(config)

    # 加载预训练模型参数（如果有）并手动冻结
    from transformers.models.longformer.modeling_longformer import LongformerForMaskedLM
    pretrained = LongformerForMaskedLM.from_pretrained(model_name)

    model.longformer.embeddings.load_state_dict(pretrained.longformer.embeddings.state_dict())
    for i, layer in enumerate(pretrained.longformer.encoder.layer):
        model.longformer.encoder.layer[i].attention.load_state_dict(layer.attention.state_dict())
        model.longformer.encoder.layer[i].intermediate.load_state_dict(layer.intermediate.state_dict())
        model.longformer.encoder.layer[i].output.load_state_dict(layer.output.state_dict())
    model.lm_head.load_state_dict(pretrained.lm_head.state_dict())
    del pretrained

    # 冻结除 memory 相关以外的参数（你可以按需修改关键词）
    for name, param in model.named_parameters():
        if "memory" not in name:
            param.requires_grad = False

    # 打印信息
    print_parameter_status(model)
    count_trainable_params(model)

if __name__ == "__main__":
    main()

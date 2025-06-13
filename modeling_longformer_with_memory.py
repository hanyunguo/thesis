# modeling_longformer_with_memory.py
import torch
import torch.nn as nn
from transformers.models.longformer.modeling_longformer import (
    LongformerLayer, LongformerEncoder, LongformerModel,
    LongformerPreTrainedModel, LongformerLMHead, apply_chunking_to_forward
)
from transformers import LongformerConfig
from memory_module import MemoryFuse

# —— 1. 一定要最先定义 LayerWithMemory ——  
class LongformerLayerWithMemory(LongformerLayer):
    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id)
        self.memory_fuse = MemoryFuse(config.hidden_size, memory_size=128)

    def forward(self, hidden_states, attention_mask=None,
                layer_head_mask=None, is_index_masked=None,
                is_index_global_attn=None, is_global_attn=None,
                output_attentions=False):
        self_attn_outputs = self.attention(
            hidden_states, attention_mask, layer_head_mask,
            is_index_masked, is_index_global_attn, is_global_attn,
            output_attentions
        )
        attn_output = self_attn_outputs[0]
        fused_output = self.memory_fuse(attn_output)
        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward,
            self.seq_len_dim, fused_output
        )
        return (layer_output,) + self_attn_outputs[1:]

# —— 2. 接着定义 EncoderWithMemory ——  
class LongformerEncoderWithMemory(LongformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        # 这里引用的 LongformerLayerWithMemory 已经定义过了
        self.layer = nn.ModuleList([
            LongformerLayerWithMemory(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])

# —— 3. 再定义 ModelWithMemory ——  
class LongformerModelWithMemory(LongformerModel):
    def __init__(self, config: LongformerConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = LongformerEncoderWithMemory(config)

# —— 4. 最后定义 ForMaskedLMWithMemory ——  
class LongformerForMaskedLMWithMemory(LongformerPreTrainedModel):
    def __init__(self, config: LongformerConfig):
        super().__init__(config)
        self.longformer = LongformerModelWithMemory(config, add_pooling_layer=False)
        self.lm_head    = LongformerLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                head_mask=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.longformer(
            input_ids, attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output   = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss    = None
        if labels is not None:
            loss_fct       = torch.nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        from transformers.modeling_outputs import MaskedLMOutput
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

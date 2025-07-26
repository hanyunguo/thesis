import torch
import torch.nn as nn
from transformers.models.led.modeling_led import (
    LEDPreTrainedModel,
    LEDForConditionalGeneration,
    LEDEncoder,
    LEDEncoderLayer,
    LEDModel,
)
from transformers.models.led.modeling_led import LEDLearnedPositionalEmbedding
from transformers import LEDConfig
from V2_memory_module import MemoryFuse
from transformers import TrainerCallback
import time
from transformers.models.led.modeling_led import LEDEncoderAttention

class LEDEncoderAttentionAllGlobal(LEDEncoderAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=False,
        output_attentions=False,
    ):
        # 忽略层号限制，让所有层都处理 global attention
        is_global_attn = True  # 强制启用
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

# ------------------------
# Callback for Logging
# ------------------------
class LogProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open("training_progress.csv", "a") as f:
                f.write(f"{timestamp},{state.global_step},{logs.get('loss', 'NA')},{logs.get('eval_loss', 'NA')},{logs.get('learning_rate', 'NA')}\n")


# ------------------------
# Encoder Layer with Memory
# ------------------------
class LEDEncoderLayerWithMemory(LEDEncoderLayer):
    def __init__(self, config: LEDConfig, layer_id=0, enable_memory=False):
        super().__init__(config, layer_id)
        self.self_attn = LEDEncoderAttentionAllGlobal(config, layer_id)
        self.layer_id = layer_id
        self.enable_memory = enable_memory
        self.my_dropout = nn.Dropout(config.dropout)
        self.my_norm1 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.my_norm2 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.linear1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.linear2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.activation_fn = torch.nn.GELU() if getattr(config, 'activation_function', '') == 'gelu' else torch.nn.ReLU()
        if self.enable_memory:
            self.memory_fuse = MemoryFuse(
                hidden_size=config.d_model,
                memory_size=getattr(config, "memory_size", 128),
                dropout=getattr(config, "dropout", 0.1)
            )
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_global_attn=None,
        is_index_masked=None,
        is_index_global_attn=None,
        output_attentions=False,
    ):
        if isinstance(is_index_global_attn, bool) or is_index_global_attn is None:
            is_index_global_attn = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
        if isinstance(is_global_attn, bool) or is_global_attn is None:
            is_global_attn = torch.zeros(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)
        if is_index_masked is None:
            is_index_masked = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        if len(is_index_global_attn_nonzero[0]) == 0:
            is_index_global_attn = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
            is_index_global_attn[:, 0] = True
            is_global_attn = torch.zeros(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)

        is_global_attn_flag = is_global_attn.any().item()

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn_flag,
            output_attentions=output_attentions,
        )

        residual = hidden_states
        attn_output = self_attn_outputs[0]
        if self.enable_memory:
            attn_output = self.memory_fuse(attn_output)

        attn_output = self.my_dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(residual + attn_output)
        added = residual + attn_output
        ln = self.self_attn_layer_norm

        # 2️⃣ 查看 LayerNorm 输出具体值
        tmp = ln(residual + attn_output)
        ff_output = self.linear2(self.my_dropout(self.activation_fn(self.linear1(hidden_states))))
        ff_output = self.my_dropout(ff_output)
        hidden_states = self.final_layer_norm(hidden_states + ff_output)

        outputs = (hidden_states,) + self_attn_outputs[1:]
        return outputs


# ------------------------
# Encoder with Memory
# ------------------------
class LEDEncoderWithMemory(LEDEncoder):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            LEDEncoderLayerWithMemory(config, i, enable_memory=(i >= config.encoder_layers - 2))
            for i in range(config.encoder_layers)
        )
        
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 🔧 核心修复：embedding 输入赋值
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions else None
        is_index_masked = attention_mask < 0.5 if attention_mask is not None else None

        # 计算 is_index_global_attn（哪些位置是 global attention）
        is_index_global_attn = global_attention_mask > 0.5 if global_attention_mask is not None else None

        # 计算 batch 是否启用了 global attention
        is_global_attn = is_index_global_attn.any(dim=-1) if is_index_global_attn is not None else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[i] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            # 计算 is_index_masked（哪些位置是 padding）

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                all_global_attentions = all_global_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None)

        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ------------------------
# LED Model with Memory
# ------------------------
class LEDModelWithMemory(LEDModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.encoder = LEDEncoderWithMemory(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # ----- Step 1: Global attention mask -----
        if global_attention_mask is None and input_ids is not None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        # ----- Step 2: Inputs embedding -----
        if inputs_embeds is None:
            assert input_ids is not None, "Either input_ids or inputs_embeds must be provided."
            inputs_embeds = self.shared(input_ids)

        # ----- Step 3: Run encoder manually -----
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=None,  # 防止被误用
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = attention_mask

        # ----- Step 4: Run decoder -----
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return (decoder_outputs[0],) + decoder_outputs[1:] + encoder_outputs

        from transformers.modeling_outputs import Seq2SeqModelOutput
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# ------------------------
# LED for Conditional Generation with Memory
# ------------------------
class LEDForConditionalGenerationWithMemory(LEDPreTrainedModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.model = LEDModelWithMemory(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.shared.weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):
        if input_ids is not None and self.training:
            max_token_id = input_ids.max().item()
            if max_token_id >= self.config.vocab_size:
                print(f"🚨 Token ID 超出范围：max_token_id={max_token_id}, vocab_size={self.config.vocab_size}")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        from transformers.modeling_outputs import Seq2SeqLMOutput
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

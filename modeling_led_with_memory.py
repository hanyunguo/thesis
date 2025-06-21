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
from memory_module import MemoryFuse
from transformers import TrainerCallback
import time


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
        self.enable_memory = enable_memory
        self.my_dropout = nn.Dropout(config.dropout)
        self.my_norm1 = nn.LayerNorm(config.d_model)
        self.my_norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.linear2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.activation_fn = torch.nn.GELU() if getattr(config, 'activation_function', '') == 'gelu' else torch.nn.ReLU()
        if self.enable_memory:
            self.memory_fuse = MemoryFuse(config.d_model, memory_size=128)

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

        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        if len(is_index_global_attn_nonzero[0]) == 0:
            is_index_global_attn = torch.zeros_like(hidden_states[:, :, 0], dtype=torch.bool)
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
        hidden_states = self.my_norm1(residual + attn_output)

        ff_output = self.linear2(self.my_dropout(self.activation_fn(self.linear1(hidden_states))))
        ff_output = self.my_dropout(ff_output)
        hidden_states = self.my_norm2(hidden_states + ff_output)

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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if global_attention_mask is None and input_ids is not None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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

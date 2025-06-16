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

class LEDEncoderLayerWithMemory(LEDEncoderLayer):
    def __init__(self, config: LEDConfig, layer_id=0, enable_memory=False):
        super().__init__(config, layer_id)
        self.enable_memory = enable_memory
        self.my_dropout = nn.Dropout(config.dropout)
        self.my_norm1 = nn.LayerNorm(config.d_model)
        self.my_norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.linear2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        if hasattr(config, 'activation_function') and config.activation_function == "gelu":
            self.activation_fn = torch.nn.GELU()
        else:
            self.activation_fn = torch.nn.ReLU()
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
        is_global_attn_flag = is_global_attn.any().item() if isinstance(is_global_attn, torch.Tensor) else bool(is_global_attn)

        if isinstance(is_index_global_attn, bool):
            raise ValueError("`is_index_global_attn` should be a BoolTensor, not a bool.")

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn_flag,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]

        if self.enable_memory:
            attn_output = self.memory_fuse(attn_output)

        hidden_states = self.my_dropout(attn_output)
        hidden_states = self.my_norm1(hidden_states + attn_output)

        ff_output = self.linear2(self.my_dropout(self.activation_fn(self.linear1(hidden_states))))
        ff_output = self.my_dropout(ff_output)
        hidden_states = self.my_norm2(hidden_states + ff_output)

        outputs = (hidden_states,) + self_attn_outputs[1:]
        return outputs

class LEDEncoderWithMemory(LEDEncoder):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            LEDEncoderLayerWithMemory(config, i, enable_memory=(i >= config.encoder_layers - 2))
            for i in range(config.encoder_layers)
        )

class LEDModelWithMemory(LEDModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.encoder = LEDEncoderWithMemory(config)

    def _wrap_encoder_layer_forward(self, layer):
        original_forward = layer.forward

        def wrapped_forward(
            hidden_states,
            attention_mask,
            layer_head_mask,
            is_global_attn,
            is_index_masked,
            is_index_global_attn,
            output_attentions,
        ):
            return original_forward(
                hidden_states,
                attention_mask,
                layer_head_mask,
                self._is_global_attn,
                is_index_masked,
                self._is_index_global_attn,
                output_attentions,
            )

        return wrapped_forward

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

        is_index_global_attn = global_attention_mask.bool()
        is_global_attn = is_index_global_attn.any(dim=1)

        self._is_index_global_attn = is_index_global_attn
        self._is_global_attn = is_global_attn
        for layer in self.encoder.layers:
            layer.forward = self._wrap_encoder_layer_forward(layer)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

class LEDForConditionalGenerationWithMemory(LEDPreTrainedModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.model = LEDModelWithMemory(config)
        self.model.config.decoder_max_position_embeddings = 4096
        self.model.decoder.embed_positions = LEDLearnedPositionalEmbedding(
            self.model.config.decoder_max_position_embeddings,
            self.model.config.d_model
        )
        self.model.decoder.embed_positions.weight.data.normal_(
            mean=0.0,
            std=self.model.config.initializer_range
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.shared.weight
        self.post_init()

        # Debugging additions
        print("🧠 vocab_size from config:", config.vocab_size)

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
        # Safety check: Ensure no token exceeds vocab size
        if input_ids is not None:
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

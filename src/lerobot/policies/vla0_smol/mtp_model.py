import copy
from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_kernelized_func
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


@torch.no_grad()
def shift_right(tensor):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttentionMTP(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            2 * config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayerMTP(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttentionMTP(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)

        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MTPModel(nn.Module):
    def __init__(
        self, num_heads: int, config: LlamaConfig, input_embedding: nn.Module, output_embedding: nn.Module
    ):
        super().__init__()
        self.num_heads = num_heads
        self.cfg = copy.deepcopy(config)
        self.cfg.num_hidden_layers = 1

        self.embed_tokens = input_embedding
        self.midlayer = LlamaDecoderLayerMTP(self.cfg, layer_idx=0)
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.lm_head = output_embedding

        self.rotary_emb = LlamaRotaryEmbedding(config=self.cfg)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        assert hidden_states.size(-1) == self.config.hidden_size * 3
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
    ):
        input_embeds = self.embed_tokens(input_ids)  # [B, seq_len, hidden_size]
        input_embeds = input_embeds.to(hidden_states.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.cfg)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + input_embeds.shape[1], device=input_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.cfg,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.midlayer(
            input_embeds,
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def calculate_mtp_loss(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        loss_mask: torch.Tensor,
    ):
        """
        input_ids: [B, seq_len]
        hidden_state: [B, seq_len, hidden_size]
        loss_mask: [B, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = hidden_states.device

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        past_key_values = DynamicCache(config=self.cfg)

        losses = []
        for head_idx in range(0, self.num_heads):
            if head_idx == 0:
                block_attention_shape = (batch_size, 1, seq_len, seq_len)
                causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(*block_attention_shape)
            else:
                next_attention_block = torch.full(
                    block_attention_shape, False, device=device, dtype=torch.bool
                )
                diag = torch.arange(seq_len, device=device)
                next_attention_block[:, :, diag, diag] = True
                attention_mask = torch.cat([attention_mask, next_attention_block], dim=-1)

            position_ids = torch.arange(head_idx, input_ids.shape[1] + head_idx, device=device).unsqueeze(0)

            out = self.forward(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states_out = out.last_hidden_state
            logits = self.compute_logits(hidden_states_out[:, :-1, :])  # [B, seq_len - 1, vocab_size]

            targets = input_ids[:, 1:].to(device)  # [B, seq_len - 1]

            head_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss_mask_mtp = loss_mask[:, 1:].to(device)  # [B, seq_len - 1]

            head_loss = head_loss * loss_mask_mtp.reshape(-1)
            head_loss = head_loss.sum() / torch.clamp(loss_mask_mtp.sum(), min=1)

            losses.append(head_loss)

            input_ids = shift_right(input_ids)
            loss_mask = shift_right(loss_mask)

            hidden_states = hidden_states_out

        return losses

    def generate_next_token(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
    ):
        with torch.inference_mode():
            output = self.forward(
                input_ids,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                cache_position,
                use_cache,
            )
            logits = self.compute_logits(output.last_hidden_state)  # [batch_size, seq_len - 1, vocab_size]
        generated_token = logits[:, -1, :].argmax(-1, keepdim=True)

        return generated_token, output

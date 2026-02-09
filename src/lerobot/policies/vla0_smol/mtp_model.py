import copy

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding


@torch.no_grad()
def shift_right(tensor):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


class MTPModel(nn.Module):
    def __init__(
        self, num_heads: int, config: LlamaConfig, input_embedding: nn.Module, output_embedding: nn.Module
    ):
        super().__init__()
        self.num_heads = num_heads
        self.cfg = copy.deepcopy(config)
        self.cfg.num_hidden_layers = 1

        self.embed_tokens = input_embedding
        self.lm_head = output_embedding

        self.norm = LlamaRMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.cfg)

        self.fuse_hidden_and_embed = nn.Linear(2 * self.cfg.hidden_size, self.cfg.hidden_size, bias=False)

        self.fuse_3_hidden = nn.Linear(3 * self.cfg.hidden_size, self.cfg.hidden_size)

        self.decoder_layer = LlamaDecoderLayer(self.cfg, layer_idx=0)

    def fuse_base_model_hidden_states(self, hidden_states: list):
        """Fuse a list of 3 (batch_size, seq_len, hidden_size) tensors along last dimension"""
        if len(hidden_states) != 3:
            raise ValueError(f"Expected 3 hidden-state tensors, got {len(hidden_states)}")

        hidden_states = torch.cat(hidden_states, dim=-1)
        return self.fuse_3_hidden(hidden_states)

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
        inputs_embeds = self.embed_tokens(input_ids)  # [B, seq_len, hidden_size]
        inputs_embeds = inputs_embeds.to(hidden_states.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.cfg)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.cfg,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = torch.cat((hidden_states, inputs_embeds), dim=-1)
        hidden_states = self.fuse_hidden_and_embed(hidden_states)

        hidden_states = self.decoder_layer(
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
            logits = self.lm_head(self.norm(hidden_states_out))[:, :-1, :]  # [B, seq_len - 1, vocab_size]

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
            logits = self.lm_head(
                self.norm(output.last_hidden_state)
            )  # [batch_size, seq_len - 1, vocab_size]
        generated_token = logits[:, -1, :].argmax(-1, keepdim=True)

        return generated_token, output

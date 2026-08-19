# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN


_FLAGOS_DRAFT_LOGSOFTMAX_TOPK_ENABLED = False
_FLAGOS_DRAFT_LOGSOFTMAX_TOPK_HITS = 0


def enable_flagos_draft_logsoftmax_topk(enabled: bool) -> None:
    global _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_ENABLED
    global _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_HITS
    _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_ENABLED = bool(enabled)
    _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_HITS = 0


def _flagos_draft_logsoftmax_topk(
    logits: torch.Tensor,
    k: int,
    native_logsoftmax,
):
    global _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_HITS
    if (
        _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_ENABLED
        and logits.is_cuda
        and logits.is_contiguous()
        and logits.ndim == 2
        and logits.shape[-1] >= 16384
        and int(k) == 8
    ):
        from KERVRuntimeOptimization.fused_logsoftmax_topk import (
            fused_logsoftmax_topk,
        )

        values, indices = fused_logsoftmax_topk(logits, k)
        _FLAGOS_DRAFT_LOGSOFTMAX_TOPK_HITS += 1
        return indices, values
    probabilities = native_logsoftmax(logits)
    result = torch.topk(probabilities, k, dim=-1)
    return result.indices, result.values


def _flagos_build_draft_tree_buffers(
    owner,
    mask_index: torch.Tensor,
    total_tokens: int,
    hidden_states: torch.Tensor,
    logits_processor,
):
    """Build the base Draft tree with the registered FlagOS C++ operator.

    The legacy path materializes several intermediate tensors on the host and
    walks every node to produce the ancestor mask and leaf retrieval paths.
    KERV uses greedy verification in the supported FlagOS path, so the C++
    operator can construct the same ascending leaf order from a single parent
    vector transfer.
    """
    if not bool(getattr(owner, "_flagos_draft_tree_operator_enabled", False)):
        return None
    if logits_processor is not None:
        # Sampling applies an additional Python-side lexicographic path sort.
        # Keep the reference implementation for that mode until the ordering
        # rule is part of the public operator contract.
        return None

    root = torch.zeros((1,), dtype=torch.long, device=mask_index.device)
    parent_indices = torch.cat((root, mask_index.to(torch.long))).to("cpu")
    kept, retrieve_indices, tree_mask, tree_position_ids, _ = (
        torch.ops.flagos_kerv.build_verification_subtree(parent_indices, 0)
    )
    expected_nodes = int(total_tokens) + 1
    if kept.numel() != expected_nodes:
        raise RuntimeError(
            "FlagOS Draft tree operator unexpectedly pruned nodes: "
            f"expected={expected_nodes}, actual={kept.numel()}"
        )
    owner._flagos_draft_tree_operator_hits = int(
        getattr(owner, "_flagos_draft_tree_operator_hits", 0)
    ) + 1
    return (
        retrieve_indices,
        tree_mask.to(dtype=torch.float32),
        tree_position_ids.to(hidden_states.device),
    )


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor



def _make_parallel_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    #print(dtype)
    #print(tgt_len)
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device,dtype=dtype)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class _KervPersistentDraftLayer:
    """Layer view into a fixed-address drafter KV workspace."""

    def __init__(self, owner, index):
        self.owner = owner
        self.index = int(index)


class KervPersistentDraftCache:
    """Fixed-address KV storage for the committed drafter prefix.

    The valid prefix length is held both as inexpensive host metadata and as a
    device scalar.  CUDA Graph replay updates the latter in-place, while the
    graph wrapper advances the host cursor by the captured query length.
    """

    def __init__(self, past_key_values, capacity=512):
        self.capacity = int(capacity)
        self.layers = []
        self.valid_length = 0
        first = past_key_values[0][0]
        # The cache may first be created from inside the action-wide
        # ``inference_mode`` context.  Allocate it as a normal tensor so the
        # task-level graph warmup can legally update the same addresses after
        # that context exits.
        with torch.inference_mode(False):
            self.valid_length_tensor = torch.zeros(
                (), device=first.device, dtype=torch.long
            )
            for key, value in past_key_values:
                key_shape = list(key.shape)
                value_shape = list(value.shape)
                key_shape[2] = self.capacity
                value_shape[2] = self.capacity
                self.layers.append(
                    (
                        torch.zeros(key_shape, device=key.device, dtype=key.dtype),
                        torch.zeros(value_shape, device=value.device, dtype=value.dtype),
                    )
                )
        self.load_from_tuple(past_key_values)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return _KervPersistentDraftLayer(self, index)

    def load_from_tuple(self, past_key_values):
        length = int(past_key_values[0][0].shape[2])
        if length > self.capacity:
            raise ValueError(
                f"draft KV length {length} exceeds capacity {self.capacity}"
            )
        for (key_buffer, value_buffer), (key, value) in zip(
            self.layers, past_key_values
        ):
            key_buffer[:, :, :length].copy_(key)
            value_buffer[:, :, :length].copy_(value)
        self.valid_length = length
        self.valid_length_tensor.fill_(length)
        return self

    def can_append(self, query_length):
        return self.valid_length + int(query_length) <= self.capacity

    def layer_buffers(self, index):
        return self.layers[int(index)]

    def advance(self, query_length):
        query_length = int(query_length)
        self.valid_length_tensor.add_(query_length)
        self.valid_length += query_length

    def as_tuple(self):
        length = int(self.valid_length)
        return tuple(
            (
                key[:, :, :length],
                value[:, :, :length],
            )
            for key, value in self.layers
        )

    def capture_graph_state(self):
        return {"valid_length": int(self.valid_length)}

    def restore_graph_state(self, state):
        self.valid_length = int(state["valid_length"])
        self.valid_length_tensor.fill_(self.valid_length)

    def prepare_graph_replay(self, _captured_input_state):
        if not self.can_append(0):
            raise ValueError("invalid persistent Draft KV cursor")
        self.valid_length_tensor.fill_(self.valid_length)

    def finish_graph_replay(self, captured_input_state, captured_output_state):
        delta = int(captured_output_state["valid_length"]) - int(
            captured_input_state["valid_length"]
        )
        if delta < 0 or not self.can_append(delta):
            raise ValueError("persistent Draft KV replay exceeds capacity")
        self.valid_length += delta

    def clear(self):
        self.valid_length = 0
        self.valid_length_tensor.zero_()


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        persistent_layer = (
            past_key_value
            if isinstance(past_key_value, _KervPersistentDraftLayer)
            else None
        )
        kv_seq_len = key_states.shape[-2]
        if persistent_layer is not None:
            kv_seq_len = persistent_layer.owner.capacity
        elif past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if persistent_layer is not None:
            cache = persistent_layer.owner
            key_buffer, value_buffer = cache.layer_buffers(
                persistent_layer.index
            )
            write_indices = cache.valid_length_tensor + torch.arange(
                q_len, device=hidden_states.device, dtype=torch.long
            )
            key_buffer.index_copy_(2, write_indices, key_states)
            value_buffer.index_copy_(2, write_indices, value_states)
            key_states = key_buffer
            value_states = value_buffer
            past_key_value = persistent_layer if use_cache else None
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            past_key_value = (key_states, value_states) if use_cache else None
        #print(past_key_value)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if persistent_layer is not None:
            key_positions = torch.arange(
                persistent_layer.owner.capacity,
                device=hidden_states.device,
                dtype=torch.long,
            )
            query_positions = (
                persistent_layer.owner.valid_length_tensor
                + torch.arange(
                    q_len, device=hidden_states.device, dtype=torch.long
                )
            )
            valid_attention = key_positions[None, :] <= query_positions[:, None]
            attn_weights = attn_weights.masked_fill(
                ~valid_attention[None, None],
                torch.finfo(attn_weights.dtype).min,
            )
        elif attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            self.weight,
            self.variance_epsilon,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class _KervDraftWorkspace:
    """Persistent scratch storage for the fixed KERV Draft tree layout.

    KERV's default configuration uses a fixed ``top_k`` and depth.  The
    original implementation rebuilt the score/token/parent lists and the
    small tree mask at every Draft call.  These buffers only hold intermediate
    values; they never change candidate ordering or model state.
    """

    def __init__(
        self,
        *,
        total_tokens,
        depth,
        top_k,
        hidden_size,
        device,
        dtype,
        hidden_dtype,
    ):
        score_slots = int(top_k) + int(depth) * int(top_k) * int(top_k)
        parent_slots = 1 + int(depth) * int(top_k)
        self.scores = torch.empty(score_slots, device=device, dtype=dtype)
        self.tokens = torch.empty(score_slots, device=device, dtype=torch.long)
        self.parents = torch.empty(parent_slots, device=device, dtype=torch.long)
        self.draft_tokens = torch.empty(
            (1, int(total_tokens) + 1), device=device, dtype=torch.long
        )
        self.input_hidden = torch.empty(
            (1, int(top_k), int(hidden_size)), device=device, dtype=hidden_dtype
        )
        self.tree_mask = torch.empty(
            (1, 1, int(top_k), 2 * int(top_k)),
            device=device,
            dtype=torch.bool,
        )
        self.tree_mask_gather = torch.empty_like(self.tree_mask[..., :top_k])
        self.topk_indices = torch.arange(
            int(top_k), device=device, dtype=torch.long
        )
        self.total_tokens = int(total_tokens)
        self.depth = int(depth)
        self.top_k = int(top_k)


class MMModel(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=50, depth=4, top_k=8, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["language_model.model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        #self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device,dtype=torch.bool)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)
        #print('init tree',self.tree_mask_init.device)
    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][-2])
        #print(attention_mask)
        #exit()
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        #print(combined_attention_mask[0][0][-2:])
        #print('tree_mask_3',self.tree_mask)
        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree mask/ combined attention mask shape')
            #print(tree_mask.shape)
            #print(combined_attention_mask.shape)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        #print(combined_attention_mask[0][0][-2:])

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids=None,
            input_embeddings=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None,
    ):
        #print(use_cache)
        #print('tree mask 2',self.tree_mask)
        #print(attention_mask.shape)
        batch_size, seq_length, _ = hidden_states.shape
        #print((seq_length))
        #print(input_embeddings.shape)
        #print(hidden_states.shape)
        seq_length_with_past = seq_length
        #print(seq_length)
        #exit()
        past_key_values_length = 0
        #print(input_ids.shape)
        #with torch.no_grad():
        #    inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()
        #print(inputs_embeds.shape)
        if std is not None:
             noise = torch.randn(input_embeddings.size(),device=input_embeddings.device) * std
             input_embeddings=input_embeddings+noise

        persistent_cache = (
            past_key_values
            if isinstance(past_key_values, KervPersistentDraftCache)
            else None
        )
        if persistent_cache is not None:
            if not persistent_cache.can_append(seq_length):
                raise ValueError("persistent Draft KV capacity exceeded")
            past_key_values_length = persistent_cache.valid_length
            seq_length_with_past = seq_length + past_key_values_length
        elif past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            #print('past kv length',past_key_values_length)
            #print('seq length with past',seq_length_with_past)
            #print('input embeddings shape')
            #print(input_embeddings.shape)
            #print('tree mask',self.tree_mask)
            #print('past kv length',past_key_values_length)
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None and persistent_cache is not None:
            position_ids = (
                persistent_cache.valid_length_tensor
                + torch.arange(
                    seq_length, dtype=torch.long, device=hidden_states.device
                )
            ).unsqueeze(0)
        elif position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None and persistent_cache is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        #print(attention_mask)
        #exit()
        #print(past_key_values_length)
        if persistent_cache is None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
            )
        #print(attention_mask[0][0][-2])
        #exit()
        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        #print(inputs_embeds)
        inputs_embeds = input_embeddings.to(hidden_states.dtype)
        #print(inputs_embeds.shape)
        #print(hidden_states.shape)
        #exit()
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        #print('after fc',hidden_states)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            #if past_key_values:
            #    print('past key value',past_key_value[0].shape)
            if self.gradient_checkpointing and self.training:
                #print('training')

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                #print(hidden_states.shape)
                #print(attention_mask.shape)
                #print(attention_mask[0][0][-1])
                #print(position_ids)
                #exit()
                #print(use_cache)
                #print(idx)
                #print(past_key_value)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            #print(hidden_states.shape)
            #exit()
            #print(len(layer_outputs))

            if use_cache and persistent_cache is None:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        #print(len(next_decoder_cache))
        #print(len(next_decoder_cache[0]))
        if use_cache and persistent_cache is not None:
            persistent_cache.advance(seq_length)
            return hidden_states, persistent_cache
        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None
        persistent_cache = getattr(
            self, "_flagos_persistent_draft_kv_cache_object", None
        )
        if persistent_cache is not None:
            persistent_cache.clear()

    def _get_flagos_draft_workspace(self, hidden_states, score_dtype):
        if not bool(getattr(self, "_flagos_persistent_draft_workspace", False)):
            return None
        if not hidden_states.is_cuda:
            return None
        cache = getattr(self, "_flagos_draft_workspace_cache", None)
        if cache is None:
            cache = {}
            self._flagos_draft_workspace_cache = cache
        key = (
            hidden_states.device,
            str(score_dtype),
            str(hidden_states.dtype),
            int(hidden_states.shape[-1]),
            int(self.total_tokens),
            int(self.depth),
            int(self.top_k),
        )
        workspace = cache.get(key)
        if workspace is None:
            workspace = _KervDraftWorkspace(
                total_tokens=self.total_tokens,
                depth=self.depth,
                top_k=self.top_k,
                hidden_size=hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=score_dtype,
                hidden_dtype=hidden_states.dtype,
            )
            cache[key] = workspace
            self._flagos_draft_workspace_allocations = int(
                getattr(self, "_flagos_draft_workspace_allocations", 0)
            ) + 1
        self._flagos_draft_workspace_hits = int(
            getattr(self, "_flagos_draft_workspace_hits", 0)
        ) + 1
        return workspace

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states,
        input_ids,
        input_embeds,
        head,
        logits_processor,
        prefetched_result=None,
    ):
        self.training=False
        input_embeds = input_embeds.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k
        #input_embeds = input_embeds.to(hidden_states.device)
        sample_token = input_ids[:, -1].to(hidden_states.device)
        input_embeds = input_embeds[:,1:,:]
        #input_ids = input_ids[:, 1:]
        #depth = self.depth
        #top_k = self.top_k
        scores_list = []
        parents_list = []
        ss_token = []
        #print(hidden_states.shape)
        #print(input_embeds.shape)

        #input_embeds = input_embeds[:, 1:]
        #input_ = input_ids.to(hidden_states.device)
        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        #len_posi = input_embeds.shape[1]
        #len_posi=input_embeds.shape[1]+1
        len_posi=input_ids.shape[1]
        #print('init mode')
        #print(input_ids.shape)
        #print(input_embeds.shape)
        self.reset()

        # with Timer("draft many"):
        draft_graph_runner = getattr(self, "_flagos_draft_stage_graph_runner", None)
        if prefetched_result is not None:
            # Phase-7 Commit--Draft graph has already executed the drafter
            # forward and updated the persistent Draft KV object.  Continue
            # with the unchanged tree expansion and Top-K logic below.
            out_hidden, past_key_values = prefetched_result
        elif hasattr(self, "stable_kv") and self.stable_kv is not None:
            #print('kv len',self.stable_kv[0][0].shape[2])
            kv_len = (
                self.stable_kv.valid_length
                if isinstance(self.stable_kv, KervPersistentDraftCache)
                else self.stable_kv[0][0].shape[2]
            )
            if draft_graph_runner is not None:
                result = draft_graph_runner(
                    hidden_states, input_embeds[:, kv_len:], self.stable_kv
                )
            else:
                result = None
            if result is None:
                out_hidden, past_key_values = self(
                    hidden_states,
                    input_embeddings=input_embeds[:, kv_len:],
                    past_key_values=self.stable_kv,
                    use_cache=True,
                )
            else:
                out_hidden, past_key_values = result
        else:
            result = (
                draft_graph_runner(hidden_states, input_embeds, None)
                if draft_graph_runner is not None
                else None
            )
            if result is None:
                out_hidden, past_key_values = self(
                    hidden_states, input_embeddings=input_embeds, use_cache=True
                )
            else:
                out_hidden, past_key_values = result
        if bool(getattr(self, "_flagos_persistent_draft_kv_cache", False)):
            if isinstance(past_key_values, KervPersistentDraftCache):
                self.stable_kv = past_key_values
            else:
                persistent_cache = getattr(
                    self, "_flagos_persistent_draft_kv_cache_object", None
                )
                try:
                    if persistent_cache is None:
                        persistent_cache = KervPersistentDraftCache(
                            past_key_values, capacity=512
                        )
                        self._flagos_persistent_draft_kv_cache_object = persistent_cache
                        self._flagos_persistent_draft_kv_allocations = int(
                            getattr(
                                self,
                                "_flagos_persistent_draft_kv_allocations",
                                0,
                            )
                        ) + 1
                    else:
                        persistent_cache.load_from_tuple(past_key_values)
                    self.stable_kv = persistent_cache
                except ValueError:
                    self.stable_kv = past_key_values
                    self._flagos_persistent_draft_kv_fallbacks = int(
                        getattr(self, "_flagos_persistent_draft_kv_fallbacks", 0)
                    ) + 1
        else:
            self.stable_kv = past_key_values
        # Tree expansion is speculative and must not advance the committed
        # persistent prefix.  It therefore starts from a valid-length view and
        # keeps the historical tuple-based temporary KV path.
        past_key_values = (
            self.stable_kv.as_tuple()
            if isinstance(self.stable_kv, KervPersistentDraftCache)
            else self.stable_kv
        )
        #kv_len = past_key_values[0][0].shape[2]
        #print('kv len',kv_len)
        #print(input_embeds.shape)
        #print(self.position_ids)
        last_hidden = out_hidden[:, -1]
        #last_headout = self.lm_head(self.norm(last_hidden))
        last_headout = head(last_hidden)

        topk_index, topk_p = _flagos_draft_logsoftmax_topk(
            last_headout, top_k, self.logsoftmax
        )
        torch.set_printoptions(profile='full')
        #print('last_p_dim',last_p.shape)
        #print('last_p',torch.exp(last_p)[:,31744:32000])
        #print('sum',sum(torch.exp(last_p),1))
        torch.set_printoptions(profile='default')
        #1*8
        scores = topk_p[0]
        # Accuracy-gated compact verification tree.  The margin is computed
        # from the already materialized Top-K scores and is not an additional
        # model call.  ``auto`` requires every draft stage seen in this action
        # to clear the configured margin; otherwise the exact 48x8 topology
        # is used.
        compact_mode = str(getattr(self, "_flagos_compact_tree_mode", "off")).lower()
        # Keep every stage margin on device and make one control transfer only
        # after the complete candidate tree has been generated.  The old path
        # called ``.item()`` once per stage and also accumulated the minimum
        # across unrelated decode iterations, which both serialized the draft
        # stream and eventually pinned auto mode to the full tree forever.
        compact_confidence_tensor = None
        if compact_mode == "auto":
            initial_margin = (scores[..., 0] - scores[..., -1]).detach().float()
            # Some pruned draft rows are represented by ``-inf`` scores.  A
            # direct subtraction produces NaN and incorrectly routes the
            # whole action to the largest verifier.  Such rows have no valid
            # branch and therefore do not constrain the minimum confidence.
            initial_margin = torch.nan_to_num(
                initial_margin,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=-float("inf"),
            )
            compact_confidence_tensor = initial_margin.amin()
            self._flagos_compact_tree_confidence = None
            self._flagos_compact_tree_active = False
            self._flagos_compact_tree_level = "full"
        elif compact_mode == "on":
            self._flagos_compact_tree_active = True
            self._flagos_compact_tree_level = "medium"
        #print('top_1_p',torch.exp(scores))
        #top_1
        #scores
        draft_workspace = self._get_flagos_draft_workspace(hidden_states, scores.dtype)
        if draft_workspace is None:
            scores_list.append(scores[None])
            parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
            ss_token.append(topk_index)
        else:
            score_cursor = int(top_k)
            token_cursor = int(top_k)
            parent_cursor = 1
            draft_workspace.scores[:top_k].copy_(scores)
            draft_workspace.tokens[:top_k].copy_(topk_index.reshape(-1))
            draft_workspace.parents[0] = 0
        input_ids = topk_index
        input_embeds = self.embed_tokens(input_ids)
        if draft_workspace is None:
            input_hidden = last_hidden[None].repeat(1, top_k, 1)
        else:
            draft_workspace.input_hidden.copy_(
                last_hidden[:, None, :].expand_as(draft_workspace.input_hidden)
            )
            input_hidden = draft_workspace.input_hidden
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self.tree_mask_init=self.tree_mask_init.to(self.embed_tokens.weight.device)
        if draft_workspace is None:
            topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        else:
            topk_cs_index = draft_workspace.topk_indices
        #logit_p = []
        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            #kv_len = past_key_values[0][0].shape[2]
            # with Timer("draft one"):
            #print('draft position ids')
            #print(position_ids)
            out_hidden, past_key_values = self(input_hidden,input_embeddings=input_embeds,past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

 # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            if draft_workspace is None:
                parents_list.append(parents)
            else:
                draft_workspace.parents[parent_cursor : parent_cursor + top_k].copy_(parents)
                parent_cursor += top_k
            #last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_headout = head(out_hidden[0])
            topk_index, topk_p = _flagos_draft_logsoftmax_topk(
                last_headout, top_k, self.logsoftmax
            )
            if compact_mode == "auto":
                margin_tensor = topk_p[..., 0] - topk_p[..., -1]
                # Later draft stages carry one Top-K row per parent.  Gate on
                # the least-confident row so a single ambiguous branch keeps
                # the exact verification topology.
                stage_margin = torch.nan_to_num(
                    margin_tensor.detach().float(),
                    nan=float("inf"),
                    posinf=float("inf"),
                    neginf=-float("inf"),
                ).amin()
                compact_confidence_tensor = torch.minimum(
                    compact_confidence_tensor,
                    stage_margin,
                )
            #logit_p.append(last_p)
            #torch.set_printoptions(profile='full')
            #print('last_p',torch.exp(last_p)[:,31744:])
            #torch.set_printoptions(profile='default')
            #print(torch.exp(last_p)[:,31936:].shape)
            #print('topk_10_p',torch.mean(torch.exp(topk_p)))
            #print('topk_1_p',torch.mean(torch.exp(topk_p),dim=0)[0])

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            input_embeds = self.embed_tokens(input_ids)
            # print(input_ids.equal(input_ids0))

            if draft_workspace is None:
                ss_token.append(topk_index)
                scores_list.append(cu_scores)
            else:
                flat_count = int(cu_scores.numel())
                draft_workspace.scores[score_cursor : score_cursor + flat_count].copy_(
                    cu_scores.reshape(-1)
                )
                score_cursor += flat_count
                draft_workspace.tokens[token_cursor : token_cursor + int(topk_index.numel())].copy_(
                    topk_index.reshape(-1)
                )
                token_cursor += int(topk_index.numel())
                torch.index_select(
                    tree_mask,
                    -1,
                    out_ids,
                    out=draft_workspace.tree_mask_gather,
                )
                draft_workspace.tree_mask[..., :top_k].copy_(
                    draft_workspace.tree_mask_gather
                )
                draft_workspace.tree_mask[..., top_k : 2 * top_k].copy_(
                    self.tree_mask_init
                )
                tree_mask = draft_workspace.tree_mask
            if draft_workspace is None:
                tree_mask = torch.cat((tree_mask[:, :, :, out_ids], self.tree_mask_init), dim=3)

            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        if compact_mode == "auto":
            # One scalar is required on the host to choose the already
            # captured verifier graph.  Do it once, after all draft stages,
            # rather than once per stage.
            confidence = float(compact_confidence_tensor.item())
            medium_threshold = float(
                getattr(self, "_flagos_compact_tree_min_confidence", 0.0)
            )
            tight_threshold = float(
                getattr(
                    self,
                    "_flagos_compact_tree_tight_min_confidence",
                    float("inf"),
                )
            )
            tight_available = int(
                getattr(self, "_flagos_compact_tree_tight_max_paths", 0)
            ) > 0
            if math.isfinite(confidence) and tight_available and confidence >= tight_threshold:
                compact_level = "tight"
            elif math.isfinite(confidence) and confidence >= medium_threshold:
                compact_level = "medium"
            else:
                compact_level = "full"
            self._flagos_compact_tree_confidence = confidence
            self._flagos_compact_tree_level = compact_level
            self._flagos_compact_tree_active = compact_level != "full"
            self._flagos_compact_tree_gate_evaluations = int(
                getattr(self, "_flagos_compact_tree_gate_evaluations", 0)
            ) + 1
            counter_name = f"_flagos_compact_tree_{compact_level}_selections"
            setattr(self, counter_name, int(getattr(self, counter_name, 0)) + 1)

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        if draft_workspace is not None:
            scores_list = draft_workspace.scores[:score_cursor]
            ss_token_list = draft_workspace.tokens[:token_cursor]
            parents_flat = draft_workspace.parents[:parent_cursor]
        else:
            scores_list = torch.cat(scores_list, dim=0).view(-1)
            ss_token_list = torch.cat(ss_token, dim=0).view(-1)
            parents_flat = torch.cat(parents_list, dim=0)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        if draft_workspace is not None:
            draft_workspace.draft_tokens[:, 0].copy_(sample_token.reshape(-1))
            draft_workspace.draft_tokens[:, 1:].copy_(
                ss_token_list[top_scores_index].reshape(1, -1)
            )
            draft_tokens = draft_workspace.draft_tokens
        else:
            draft_tokens = ss_token_list[top_scores_index]
            draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)[None]

        draft_parents = parents_flat[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        if draft_workspace is None:
            # The persistent path already owns a batch dimension.
            draft_tokens = draft_tokens
        operator_buffers = _flagos_build_draft_tree_buffers(
            self,
            mask_index,
            total_tokens,
            hidden_states,
            logits_processor,
        )
        if operator_buffers is not None:
            retrieve_indices, tree_mask, tree_position_ids = operator_buffers
        else:
            mask_index_list = mask_index.tolist()
            tree_mask = torch.eye(total_tokens + 1).bool()
            tree_mask[:, 0] = True
            for i in range(total_tokens):
                tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
            tree_position_ids = torch.sum(tree_mask, dim=1) - 1
            tree_mask = tree_mask.float()[None, None]

            max_depth = torch.max(tree_position_ids) + 1
            noleaf_index = torch.unique(mask_index).tolist()
            noleaf_num = len(noleaf_index) - 1
            leaf_num = total_tokens - noleaf_num
            retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
            retrieve_indices = retrieve_indices.tolist()
            rid = 0
            position_ids_list = tree_position_ids.tolist()
            for i in range(total_tokens + 1):
                if i not in noleaf_index:
                    cid = i
                    node_depth = position_ids_list[i]
                    for j in reversed(range(node_depth + 1)):
                        retrieve_indices[rid][j] = cid
                        cid = mask_index_list[cid - 1]
                    rid += 1
            if logits_processor is not None:
                maxitem = total_tokens + 5

                def custom_sort(lst):
                    return [value if value >= 0 else maxitem for value in lst]

                retrieve_indices = sorted(retrieve_indices, key=custom_sort)
            retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
            tree_position_ids = tree_position_ids.to(hidden_states.device).unsqueeze(0)

        del parents_list, scores_list, ss_token, ss_token_list, parents_flat, draft_parents, mask_index
        #print('draft tokens',draft_tokens)
        #print('retrieve indices',retrieve_indices)
        #print('tree mask',tree_mask)
        #print('tree posisition ids',tree_position_ids)
        #if return_logit:
        #    logit_p = torch.stack(logit_p)
        #    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids,logit_p
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):
    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)

class PMMModel(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=10, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["language_model.model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        #self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.motion_embeds = nn.Embedding(6,config.hidden_size, self.padding_idx)

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)
        #print('init tree',self.tree_mask_init.device)
    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][-2])
        #print(attention_mask)
        #exit()
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        #print(combined_attention_mask[0][0][-2:])
        #print('tree_mask_3',self.tree_mask)
        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree mask/ combined attention mask shape')
            #print(tree_mask.shape)
            #print(combined_attention_mask.shape)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        #print(combined_attention_mask[0][0][-2:])

        return combined_attention_mask
    def _prepare_parallel_decoder_attention_mask(self, attention_mask, input_shape ,inputs_embeds, past_key_values_length,forward_idx):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #print('attention mask')
        combined_attention_mask = None
        seq_length = torch.count_nonzero(attention_mask,dim=1) + 6-forward_idx
        #print(seq_length)
        if input_shape[-1] > 1:
            combined_attention_mask = _make_parallel_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        #dtype = torch.float32
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][0])
        #exit()
        #print('parallel mask:')
        #print(combined_attention_mask.shape)
        for i in range(len(seq_length)):
            #print(seq_length[i])
            #print(combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]])
            #print(seq_length[i])
            combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]]=0.00
            #print(combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]])
            #print(combined_attention_mask[i][0][0][-20:])
        #print(combined_attention_mask[0][0][-1])
        #print(combined_attention_mask[0][0][0])
        #exit()
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][-2])
        #print(attention_mask)
        #exit()
        #if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #    expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=seq_length).to(
        #        inputs_embeds.device
        #    )
        #    combined_attention_mask = (
        #        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        #    )
        #print(combined_attention_mask[0][0][-2:])
        #print('tree_mask_3',self.tree_mask)
        # [MODIFIED] add tree mask
        '''if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree mask/ combined attention mask shape')
            #print(tree_mask.shape)
            #print(combined_attention_mask.shape)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min'''
        #print(combined_attention_mask[0][0][-2:])
        combined_attention_mask = combined_attention_mask.to(torch.float32)
        return combined_attention_mask

    def forward(
            self,
            hidden_states=None,
            input_ids=None,
            input_embeddings=None,
            forward_current_idx=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        #print('forward_current_idx',forward_current_idx)
        #print('attention mask shape',attention_mask.shape)
        #print('forward attention mask',attention_mask[:,-20:])
        #print(use_cache)
        #print('tree mask 2',self.tree_mask)
        #print(attention_mask.shape)
        batch_size, overall_length, _ = input_embeddings.shape
        seq_length = torch.count_nonzero(attention_mask,dim=1)
        #print(batch_size)
        #print(seq_length)
        #print(overall_length)
        #print(attention_mask[:,-20:])
        #print(input_embeddings)
        #exit()
        #print('input shape:',batch_size,seq_length)
        #print((seq_length))
        #print(input_embeddings.shape)
        #print(hidden_states.shape)
        seq_length_with_past = seq_length
        #print(seq_length)
        #exit()
        past_key_values_length = 0
        #print(input_ids.shape)
        #with torch.no_grad():
        #    inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()
        #print(inputs_embeds.shape)
        if std is not None:
             noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
             inputs_embeds=inputs_embeds+noise

        '''if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            #print('past kv length',past_key_values_length)
            #print('seq length with past',seq_length_with_past)
            #print('input embeddings shape')
            #print(input_embeddings.shape)
            #print('tree mask',self.tree_mask)
            #print('past kv length',past_key_values_length)
            seq_length_with_past = seq_length_with_past + past_key_values_length'''
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, overall_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, overall_length)
        else:
            position_ids = position_ids.view(-1, overall_length).long()

        #position_ids=position_ids//4
        #if attention_mask is None:
        #    attention_mask = torch.ones(
        #        (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
        #    )
        #print(past_key_values_length)
        #print(attention_mask[0][0][-2])
        #exit()
        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        #print(inputs_embeds)
        #inputs_embeds = input_embeddings
        #print('input embeds dtype',input_embeddings.dtype)
        motion_embedding = self.motion_embeds(torch.tensor([i for i in range(6)]).to(input_embeddings.device)).to(input_embeddings.dtype)
       #print(motion_embedding.shape)
        prefix_length = torch.count_nonzero(attention_mask,dim=1)
        #print(prefix_length)
        #print(motion_embedding.repeat([batch_size,1,1]).shape)
        #for i in range(batch_size):
        #    input_embeddings[i][prefix_length[i]:prefix_length[i]+(6-forward_current_idx),:]=motion_embedding[forward_current_idx:,:]
        #input_embeddings[:,prefix_length:prefix_length+(6-forward_current_idx),:]=motion_embedding.repeat([batch_size,1,1])[:,forward_current_idx:,:]
        #print('input_embeddings')
        #print(input_embeddings.shape)
        #print('input hidden')
        #print(hidden_states.shape)
        #print(prefix_length+(6-forward_current_idx))
        #input_embeddings[]
        #print('motion embedding dtype',motion_embedding.shape)
        #print(motion_embedding.shape)
        #hidden_states = input_embeddings
        #print(hidden_states.shape)
        #for i in range(batch_size):
        #    input_embeddings[i][seq_length[i]:seq_length[i]+6]=motion_embedding
        attention_mask = self._prepare_parallel_decoder_attention_mask(
            attention_mask, (batch_size, overall_length), input_embeddings, past_key_values_length,forward_current_idx
        )
        #print(attention_mask.shape)
        #print(attention_mask[0][0][-10])
        #exit()
        #print('attention mask')
        #print(attention_mask.shape)
        #print(attention_mask[0][0])
        #print('forward input shape',hidden_states.shape)
        #print(hidden_states.dtype)
        #print(inputs_embeds.shape)
        #print(hidden_states.shape)
        #exit()
        #hidden_states.to(input_embeddings.device)
        #print(input_embeddings.device)
        #print(hidden_states.device)
        #hidden_states=input_embeddings
        hidden_states = self.fc(torch.cat((input_embeddings, hidden_states), dim=-1))
        #prev_hidden = hidden_states.clone()
        for i in range(batch_size):
            hidden_states[i][prefix_length[i]:prefix_length[i]+(6-forward_current_idx),:]=motion_embedding[forward_current_idx:,:]
            #print('fc overwrite',prefix_length[i],prefix_length[i]+(6-forward_current_idx))
            #print('after fc',hidden_states)
            #cmp_tensor = (prev_hidden == hidden_states)[:,:,0].all()
            #print('compare',cmp_tensor)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            #if past_key_values:
            #    print('past key value',past_key_value[0].shape)
            if self.gradient_checkpointing and self.training:
                #print('training')

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                #print(hidden_states.shape)
                #print(attention_mask.shape)
                #print(attention_mask[0][0][-3])
                #print(position_ids)
                #exit()
                #print(use_cache)
                #print(idx)
                #print(past_key_value)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            #print(hidden_states.shape)
            #exit()
            #print(layer_outputs)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        #print(hidden_states[0][0])
        #for i in range(batch_size):
        #    hidden_state_mask = torch.full((overall_length,1), 0, device=device)
        #    hidden_state_mask[prefix_length[i]:prefix_length[i]+(6-forward_current_idx)][0]=1
        #hidden_states = hidden_states * hidden_state_mask
        #print(hidden_states[0][0])
        #print(len(next_decoder_cache))
        #print(len(next_decoder_cache[0]))
        #print(hidden_states.shape)
       # exit()
        if use_cache:
            return hidden_states, next_decoder_cache
        #exit()

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None
        persistent_cache = getattr(
            self, "_flagos_persistent_draft_kv_cache_object", None
        )
        if persistent_cache is not None:
            persistent_cache.clear()
    #TODO:Modify this function to generate draft tokens in parallel.
    @torch.no_grad()
    def _eval_top_k(self,hidden_states,input_tokens,input_embeds,head,forward_id,logits_processor):
        #
        accept_threshold=2
        top_k = 10
        #exit()
        #print(input_embeds.shape)
        input_embeds = input_embeds[1:].unsqueeze(0)
        #print(input_embeds.shape)
        #print(hidden_states.shape)
        hidden_states = hidden_states[:].unsqueeze(0)
        #print(hidden_states.shape)
        #print(hidden_states.shape)
        #input_tokens = input_tokens[1:]
        print('new sample')
        #for idx in range(6):
        for idx in [0]:
            print('idx',idx)
            forward_id = idx
            attention_mask = (torch.tensor([1] * (hidden_states.shape[1]-(6-forward_id)-1)+[0]*(6-forward_id+1)).to(hidden_states.device)).unsqueeze(0)
            #print(('prefix len',hidden_states.shape[1]-(6-forward_id)))
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds,attention_mask = attention_mask,forward_current_idx=forward_id,
                                                    use_cache=True)
            #print(out_hidden.shape)
            last_hidden = out_hidden[:, hidden_states.shape[1]-(6-forward_id)-1:-1]
            #print(last_hidden.shape)
            last_headout = head(last_hidden)
            #print('last headout',last_headout)
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            #input_tokens = input_tokens[1:]
            #topk_index[0] = topk_index[0][idx:]
            #print(topk_index.shape)
            input_tokens = input_tokens[1:]
            for i in range(6-idx):
                topk_index[0][i] = torch.abs(topk_index[0][i] - input_tokens[i])
            print(topk_index)
            in_bool = torch.abs(topk_index)<=accept_threshold
            print('in_bool')
            print(in_bool)
            print('in_list')
            print(in_bool.any(dim=2))
        return

    def topK_genrate_0(self, hidden_states, input_tokens,input_embeds, head, logits_processor):
        #print('tree mask 1',self.tree_mask)
        #print('hidden state shape',hidden_states.shape)
        #print('input tokens',input_tokens.shape)
        #print('input embeds',input_embeds.shape)
        n_parallel_decode=6
        #output:
        self.training = False
        input_embeds = input_embeds.to(hidden_states.device)
        input_embeds = input_embeds[:,1:,:]
        total_tokens = self.total_tokens
        #depth = self.depth
        top_k = self.top_k

        sample_token = input_tokens[:, -1].to(hidden_states.device)

        scores_list = []
        parents_list = []
        ss_token = []

        len_posi = input_embeds.shape[1]
        self.reset()
        #print(self.tree_mask)
        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            #print('kv len',kv_len)
            #print(hidden_states.shape)
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states=hidden_states, input_embeddings=input_embeds,use_cache=True, output_attentions=False)
        self.stable_kv = past_key_values
        kv_len = past_key_values[0][0].shape[2]
        #print('initilized complete')
        #print('out_hidden shape',out_hidden.shape)
        last_hidden = out_hidden[:, -n_parallel_decode:]
        #print('last hidden',last_hidden)
        #print(last_hidden.shape)
        #print('past_key_values',past_key_values.shape)
        last_headout = head(last_hidden)
        #print('last headout',last_headout)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        #print('topk',topk_index)
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        #print('topk_idx',topk_index)
        input_embeds = self.embed_tokens(input_ids)
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        #print(input_hidden.shape)
        #print(input_embeds.shape)
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self.tree_mask_init=self.tree_mask_init.to(self.embed_tokens.weight.device)
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        #print(input_ids)
        #print(past_key_values.shape)
        # 4
        for i in range(depth):
            #print(i)
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_embeddings=input_embeds, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            input_embeds = self.embed_tokens(input_ids)
            #print(input_ids.shape)
            # print(input_ids.equal(input_ids0))
            #print(tree_mask.device)
            #print(out_ids.device)
            #print(self.tree_mask_init.device)
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1,dtype=torch.float32).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        tree_position_ids = tree_position_ids.unsqueeze(0)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_tokens,input_embeds, head, logits_processor):
        #print('tree mask 1',self.tree_mask)
        #generate top-k token at the corresponding positions in parallel.
        #print('hidden state shape',hidden_states.shape)
        #print('input tokens',input_tokens.shape)
        #print('input embeds',input_embeds.shape)
        self.training = False
        input_embeds = input_embeds.to(hidden_states.device)
        input_embeds = input_embeds[:,1:,:]
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_tokens[:, -1].to(hidden_states.device)

        scores_list = []
        parents_list = []
        ss_token = []

        len_posi = input_embeds.shape[1]
        self.reset()
        #print(self.tree_mask)
        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            #print(hidden_states.shape)
            #print(input_embeds.shape)
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds,
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states=hidden_states, input_embeddings=input_embeds,use_cache=True, output_attentions=False)
        self.stable_kv = past_key_values
        #print('initilized complete')
        #print('out_hidden shape',out_hidden.shape)
        last_hidden = out_hidden[:, -1]
        #print('last hidden',last_hidden)
        #print(last_hidden.shape)
        #print('past_key_values',past_key_values.shape)
        last_headout = head(last_hidden)
        #print('last headout',last_headout)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        #print('topk',topk_index)
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        #print('topk_idx',topk_index)
        input_embeds = self.embed_tokens(input_ids)
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        #print(input_hidden.shape)
        #print(input_embeds.shape)
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self.tree_mask_init=self.tree_mask_init.to(self.embed_tokens.weight.device)
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        #print(input_ids)
        #print(past_key_values.shape)
        # 4
        for i in range(depth):
            #print(i)
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_embeddings=input_embeds, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            input_embeds = self.embed_tokens(input_ids)
            #print(input_ids.shape)
            # print(input_ids.equal(input_ids0))
            #print(tree_mask.device)
            #print(out_ids.device)
            #print(self.tree_mask_init.device)
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        draft_tokens = draft_tokens[None]
        operator_buffers = _flagos_build_draft_tree_buffers(
            self,
            mask_index,
            total_tokens,
            hidden_states,
            logits_processor,
        )
        if operator_buffers is not None:
            retrieve_indices, tree_mask, tree_position_ids = operator_buffers
        else:
            mask_index_list = mask_index.tolist()
            tree_mask = torch.eye(total_tokens + 1).bool()
            tree_mask[:, 0] = True
            for i in range(total_tokens):
                tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
            tree_position_ids = torch.sum(tree_mask, dim=1) - 1
            tree_mask = tree_mask.float()[None, None]

            max_depth = torch.max(tree_position_ids) + 1
            noleaf_index = torch.unique(mask_index).tolist()
            noleaf_num = len(noleaf_index) - 1
            leaf_num = total_tokens - noleaf_num
            retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
            retrieve_indices = retrieve_indices.tolist()
            rid = 0
            position_ids_list = tree_position_ids.tolist()
            for i in range(total_tokens + 1):
                if i not in noleaf_index:
                    cid = i
                    node_depth = position_ids_list[i]
                    for j in reversed(range(node_depth + 1)):
                        retrieve_indices[rid][j] = cid
                        cid = mask_index_list[cid - 1]
                    rid += 1
            if logits_processor is not None:
                maxitem = total_tokens + 5

                def custom_sort(lst):
                    return [value if value >= 0 else maxitem for value in lst]

                retrieve_indices = sorted(retrieve_indices, key=custom_sort)
            retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
            tree_position_ids = tree_position_ids.to(hidden_states.device).unsqueeze(0)

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents, mask_index

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class PMMModel1(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=10, threshold=1.0):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["language_model.model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        #self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        #self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        self.motion_embeds = nn.Embedding(6,config.hidden_size, self.padding_idx)

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)
        #print('init tree',self.tree_mask_init.device)
    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][-2])
        #print(attention_mask)
        #exit()
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        #print(combined_attention_mask[0][0][-2:])
        #print('tree_mask_3',self.tree_mask)
        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree mask/ combined attention mask shape')
            #print(tree_mask.shape)
            #print(combined_attention_mask.shape)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min
        #print(combined_attention_mask[0][0][-2:])

        return combined_attention_mask
    def _prepare_parallel_decoder_attention_mask(self, attention_mask, input_shape ,inputs_embeds, past_key_values_length,forward_idx):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #print('attention mask')
        combined_attention_mask = None
        seq_length = torch.count_nonzero(attention_mask,dim=1) + 6-forward_idx
        #print(seq_length)
        if input_shape[-1] > 1:
            combined_attention_mask = _make_parallel_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        #dtype = torch.float32
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][0])
        #exit()
        #print('parallel mask:')
        #print(combined_attention_mask.shape)
        for i in range(len(seq_length)):
            #print(seq_length[i])
            #print(combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]])
            #print(seq_length[i])
            combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]]=0.00
            #print(combined_attention_mask[i][0][:seq_length[i]][:,:seq_length[i]])
            #print(combined_attention_mask[i][0][0][-20:])
        #print(combined_attention_mask[0][0][-1])
        #print(combined_attention_mask[0][0][0])
        #exit()
        #print(combined_attention_mask.shape)
        #print(combined_attention_mask[0][0][-2])
        #print(attention_mask)
        #exit()
        #if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #    expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=seq_length).to(
        #        inputs_embeds.device
        #    )
        #    combined_attention_mask = (
        #        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        #    )
        #print(combined_attention_mask[0][0][-2:])
        #print('tree_mask_3',self.tree_mask)
        # [MODIFIED] add tree mask
        '''if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree mask/ combined attention mask shape')
            #print(tree_mask.shape)
            #print(combined_attention_mask.shape)
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min'''
        #print(combined_attention_mask[0][0][-2:])
        combined_attention_mask = combined_attention_mask.to(torch.float32)
        return combined_attention_mask

    def forward(
            self,
            hidden_states=None,
            input_ids=None,
            input_embeddings=None,
            forward_current_idx=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        #print('forward_current_idx',forward_current_idx)
        #print('attention mask shape',attention_mask.shape)
        #print('forward attention mask',attention_mask[:,-20:])
        #print(use_cache)
        #print('tree mask 2',self.tree_mask)
        #print(attention_mask.shape)
        batch_size, overall_length, _ = input_embeddings.shape
        seq_length = torch.count_nonzero(attention_mask,dim=1)
        #print(batch_size)
        #print(seq_length)
        #print(overall_length)
        #print(attention_mask[:,-20:])
        #print(input_embeddings)
        #exit()
        #print('input shape:',batch_size,seq_length)
        #print((seq_length))
        #print(input_embeddings.shape)
        #print(hidden_states.shape)
        seq_length_with_past = seq_length
        #print(seq_length)
        #exit()
        past_key_values_length = 0
        #print(input_ids.shape)
        #with torch.no_grad():
        #    inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()
        #print(inputs_embeds.shape)
        if std is not None:
             noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
             inputs_embeds=inputs_embeds+noise

        '''if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            #print('past kv length',past_key_values_length)
            #print('seq length with past',seq_length_with_past)
            #print('input embeddings shape')
            #print(input_embeddings.shape)
            #print('tree mask',self.tree_mask)
            #print('past kv length',past_key_values_length)
            seq_length_with_past = seq_length_with_past + past_key_values_length'''
        if position_ids is None:
            device = input_embeddings.device if input_embeddings is not None else input_embeddings.device
            position_ids = torch.arange(
                past_key_values_length, overall_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, overall_length)
        else:
            position_ids = position_ids.view(-1, overall_length).long()

        #position_ids=position_ids//4
        #if attention_mask is None:
        #    attention_mask = torch.ones(
        #        (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
        #    )
        #print(past_key_values_length)
        #print(attention_mask[0][0][-2])
        #exit()
        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        #print(inputs_embeds)
        #inputs_embeds = input_embeddings
        #print('input embeds dtype',input_embeddings.dtype)
        motion_embedding = self.motion_embeds(torch.tensor([i for i in range(6)]).to(input_embeddings.device)).to(input_embeddings.dtype)
       #print(motion_embedding.shape)
        prefix_length = torch.count_nonzero(attention_mask,dim=1)
        #print(prefix_length)
        #print(motion_embedding.repeat([batch_size,1,1]).shape)
        #for i in range(batch_size):
        #    input_embeddings[i][prefix_length[i]:prefix_length[i]+(6-forward_current_idx),:]=motion_embedding[forward_current_idx:,:]
        #input_embeddings[:,prefix_length:prefix_length+(6-forward_current_idx),:]=motion_embedding.repeat([batch_size,1,1])[:,forward_current_idx:,:]
        #print('input_embeddings')
        #print(input_embeddings.shape)
        #print('input hidden')
        #print(hidden_states.shape)
        #print(prefix_length+(6-forward_current_idx))
        #input_embeddings[]
        #print('motion embedding dtype',motion_embedding.shape)
        #print(motion_embedding.shape)
        #hidden_states = input_embeddings
        #print(hidden_states.shape)
        #for i in range(batch_size):
        #    input_embeddings[i][seq_length[i]:seq_length[i]+6]=motion_embedding
        attention_mask = self._prepare_parallel_decoder_attention_mask(
            attention_mask, (batch_size, overall_length), input_embeddings, past_key_values_length,forward_current_idx
        )
        #print(attention_mask.shape)
        #print(attention_mask[0][0][-10])
        #exit()
        #print('attention mask')
        #print(attention_mask.shape)
        #print(attention_mask[0][0])
        #print('forward input shape',hidden_states.shape)
        #print(hidden_states.dtype)
        #print(inputs_embeds.shape)
        #print(hidden_states.shape)
        #exit()
        #hidden_states.to(input_embeddings.device)
        #print(input_embeddings.device)
        #print(hidden_states.device)
        hidden_states=input_embeddings
        #hidden_states = self.fc(torch.cat((input_embeddings, hidden_states), dim=-1))
        #prev_hidden = hidden_states.clone()
        for i in range(batch_size):
            hidden_states[i][prefix_length[i]:prefix_length[i]+(6-forward_current_idx),:]=motion_embedding[forward_current_idx:,:]
            #print('fc overwrite',prefix_length[i],prefix_length[i]+(6-forward_current_idx))
            #print('after fc',hidden_states)
            #cmp_tensor = (prev_hidden == hidden_states)[:,:,0].all()
            #print('compare',cmp_tensor)
        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            #if past_key_values:
            #    print('past key value',past_key_value[0].shape)
            if self.gradient_checkpointing and self.training:
                #print('training')

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                #print(hidden_states.shape)
                #print(attention_mask.shape)
                #print(attention_mask[0][0][-3])
                #print(position_ids)
                #exit()
                #print(use_cache)
                #print(idx)
                #print(past_key_value)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            #print(hidden_states.shape)
            #exit()
            #print(layer_outputs)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        #print(hidden_states[0][0])
        #for i in range(batch_size):
        #    hidden_state_mask = torch.full((overall_length,1), 0, device=device)
        #    hidden_state_mask[prefix_length[i]:prefix_length[i]+(6-forward_current_idx)][0]=1
        #hidden_states = hidden_states * hidden_state_mask
        #print(hidden_states[0][0])
        #print(len(next_decoder_cache))
        #print(len(next_decoder_cache[0]))
        #print(hidden_states.shape)
       # exit()
        if use_cache:
            return hidden_states, next_decoder_cache
        #exit()

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None
    #TODO:Modify this function to generate draft tokens in parallel.
    @torch.no_grad()
    def _eval_top_k(self,hidden_states,input_tokens,input_embeds,head,forward_id,logits_processor):
        #
        accept_threshold=2
        top_k = 10
        #exit()
        #print(input_embeds.shape)
        input_embeds = input_embeds[1:].unsqueeze(0)
        #print(input_embeds.shape)
        #print(hidden_states.shape)
        hidden_states = hidden_states[:].unsqueeze(0)
        #print(hidden_states.shape)
        #print(hidden_states.shape)
        #input_tokens = input_tokens[1:]
        print('new sample')
        #for idx in range(6):
        for idx in [0]:
            print('idx',idx)
            forward_id = idx
            attention_mask = (torch.tensor([1] * (hidden_states.shape[1]-(6-forward_id)-1)+[0]*(6-forward_id+1)).to(hidden_states.device)).unsqueeze(0)
            #print(('prefix len',hidden_states.shape[1]-(6-forward_id)))
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds,attention_mask = attention_mask,forward_current_idx=forward_id,
                                                    use_cache=True)
            #print(out_hidden.shape)
            last_hidden = out_hidden[:, hidden_states.shape[1]-(6-forward_id)-1:-1]
            #print(last_hidden.shape)
            last_headout = head(last_hidden)
            #print('last headout',last_headout)
            last_p = self.logsoftmax(last_headout)
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            #input_tokens = input_tokens[1:]
            #topk_index[0] = topk_index[0][idx:]
            #print(topk_index.shape)
            input_tokens = input_tokens[1:]
            for i in range(6-idx):
                topk_index[0][i] = torch.abs(topk_index[0][i] - input_tokens[i])
            print(topk_index)
            in_bool = torch.abs(topk_index)<=accept_threshold
            print('in_bool')
            print(in_bool)
            print('in_list')
            print(in_bool.any(dim=2))
        return

    def topK_genrate_0(self, hidden_states, input_tokens,input_embeds, head, logits_processor):
        #print('tree mask 1',self.tree_mask)
        #print('hidden state shape',hidden_states.shape)
        #print('input tokens',input_tokens.shape)
        #print('input embeds',input_embeds.shape)
        n_parallel_decode=6
        #output:
        self.training = False
        input_embeds = input_embeds.to(hidden_states.device)
        input_embeds = input_embeds[:,1:,:]
        total_tokens = self.total_tokens
        #depth = self.depth
        top_k = self.top_k

        sample_token = input_tokens[:, -1].to(hidden_states.device)

        scores_list = []
        parents_list = []
        ss_token = []

        len_posi = input_embeds.shape[1]
        self.reset()
        #print(self.tree_mask)
        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            #print('kv len',kv_len)
            #print(hidden_states.shape)
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states=hidden_states, input_embeddings=input_embeds,use_cache=True, output_attentions=False)
        self.stable_kv = past_key_values
        kv_len = past_key_values[0][0].shape[2]
        #print('initilized complete')
        #print('out_hidden shape',out_hidden.shape)
        last_hidden = out_hidden[:, -n_parallel_decode:]
        #print('last hidden',last_hidden)
        #print(last_hidden.shape)
        #print('past_key_values',past_key_values.shape)
        last_headout = head(last_hidden)
        #print('last headout',last_headout)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        #print('topk',topk_index)
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        #print('topk_idx',topk_index)
        input_embeds = self.embed_tokens(input_ids)
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        #print(input_hidden.shape)
        #print(input_embeds.shape)
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self.tree_mask_init=self.tree_mask_init.to(self.embed_tokens.weight.device)
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        #print(input_ids)
        #print(past_key_values.shape)
        # 4
        for i in range(depth):
            #print(i)
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_embeddings=input_embeds, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            input_embeds = self.embed_tokens(input_ids)
            #print(input_ids.shape)
            # print(input_ids.equal(input_ids0))
            #print(tree_mask.device)
            #print(out_ids.device)
            #print(self.tree_mask_init.device)
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1,dtype=torch.float32).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        tree_position_ids = tree_position_ids.unsqueeze(0)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_tokens,input_embeds, head, logits_processor):
        #print('tree mask 1',self.tree_mask)
        #generate top-k token at the corresponding positions in parallel.
        #print('hidden state shape',hidden_states.shape)
        #print('input tokens',input_tokens.shape)
        #print('input embeds',input_embeds.shape)
        self.training = False
        input_embeds = input_embeds.to(hidden_states.device)
        input_embeds = input_embeds[:,1:,:]
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_tokens[:, -1].to(hidden_states.device)

        scores_list = []
        parents_list = []
        ss_token = []

        len_posi = input_embeds.shape[1]
        self.reset()
        #print(self.tree_mask)
        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            #print(hidden_states.shape)
            #print(input_embeds.shape)
            out_hidden, past_key_values = self(hidden_states, input_embeddings=input_embeds,
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states=hidden_states, input_embeddings=input_embeds,use_cache=True, output_attentions=False)
        self.stable_kv = past_key_values
        #print('initilized complete')
        #print('out_hidden shape',out_hidden.shape)
        last_hidden = out_hidden[:, -1]
        #print('last hidden',last_hidden)
        #print(last_hidden.shape)
        #print('past_key_values',past_key_values.shape)
        last_headout = head(last_hidden)
        #print('last headout',last_headout)
        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        #print('topk',topk_index)
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        #print('topk_idx',topk_index)
        input_embeds = self.embed_tokens(input_ids)
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        #print(input_hidden.shape)
        #print(input_embeds.shape)
        tree_mask = self.tree_mask_init.to(self.embed_tokens.weight.device)
        self.tree_mask_init=self.tree_mask_init.to(self.embed_tokens.weight.device)
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
        #print(input_ids)
        #print(past_key_values.shape)
        # 4
        for i in range(depth):
            #print(i)
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_embeddings=input_embeds, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            input_embeds = self.embed_tokens(input_ids)
            #print(input_ids.shape)
            # print(input_ids.equal(input_ids0))
            #print(tree_mask.device)
            #print(out_ids.device)
            #print(self.tree_mask_init.device)
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        # with Timer("mask1"):
        #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
        #     tree_mask0[0][0] = True
        #     for i in range(total_tokens):
        #         #tree_mask0[i + 1][0]=True
        #         tree_mask0[i + 1][i + 1] = True
        #         p=mask_index_list[i]
        #         tree_mask0[i + 1][p] = True
        #         while p:
        #             p=mask_index_list[p-1]
        #             tree_mask0[i + 1][p] = True
        #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
        #
        # print(tree_mask0.equal(tree_mask))
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        tree_position_ids = tree_position_ids.unsqueeze(0)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc

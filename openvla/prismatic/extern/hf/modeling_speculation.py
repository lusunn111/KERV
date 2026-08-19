import copy
import json
import time

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from openvla.specdecoding.model.cnets import MMModel,PMMModel
from openvla.specdecoding.model.cnets import EConfig
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig
import safetensors

from .utils import *
from experiments.robot.tool_utils import kalman_predict_from_history
from .kv_cache import initialize_past_key_values

import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache

logger = logging.get_logger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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


def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

class LlamaSpecForCausalLM(LlamaForCausalLM):
    def __init__(self,config:LlamaConfig,attn_implementation):
        super().__init__(config=config)
        self.tree_mask = None
        return
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
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

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('customized tree mask')
        #print(attention_mask.shape)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        flagos_action_head = bool(getattr(self, "_flagos_action_head_enabled", False))
        if flagos_action_head:
            action_vocab_size = int(self._flagos_action_vocab_size)
            action_logit_offset = int(self._flagos_action_logit_offset)
            action_weight = self.lm_head.weight[
                action_logit_offset : action_logit_offset + action_vocab_size
            ]
            action_bias = getattr(self.lm_head, "bias", None)
            if action_bias is not None:
                action_bias = action_bias[
                    action_logit_offset : action_logit_offset + action_vocab_size
                ]
            logits = F.linear(hidden_states, action_weight, action_bias)
            self._flagos_action_head_hit_count = int(
                getattr(self, "_flagos_action_head_hit_count", 0)
            ) + 1
            should_record = bool(getattr(self, "_flagos_action_head_record", False))
            record_once = bool(getattr(self, "_flagos_action_head_record_once", True))
            if should_record and (
                not record_once or self._flagos_action_head_hit_count == 1
            ):
                record_path = getattr(self, "_flagos_action_head_log_path", None)
                if record_path:
                    record = {
                        "operator": "kerv_verifier_action_lm_head",
                        "implementation": "flagos_action_vocab_linear",
                        "hidden_shape": list(hidden_states.shape),
                        "input_dtype": str(hidden_states.dtype).replace("torch.", ""),
                    "full_vocab_size": int(self.lm_head.weight.shape[0]),
                    "logical_vocab_size": int(
                        getattr(self.config, "vocab_size", self.lm_head.weight.shape[0])
                    ),
                        "action_vocab_size": action_vocab_size,
                        "action_token_offset": action_logit_offset,
                    }
                    with Path(record_path).open("a", encoding="utf-8") as output_file:
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        #print('past kv type',type(outputs.past_key_values))
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        #labels: torch.LongTensor = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        #print('customized forward!!!!!')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if isinstance(past_key_values, Cache):
                past_seen_tokens = past_key_values.get_seq_length()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()
                #print('use old cache')
        #print('past seen tokens',past_seen_tokens)
        #print(past_seen_tokens)
        #if hasattr(self, "tree_mask") and self.tree_mask is not None:
        #    cache_position = position_ids
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        #print(cache_position)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        #else:
        #    cache_position = position_ids
        #print('cache position',cache_position)
        #print('past seen tokens',past_seen_tokens)
        #TODO:Update this function to fullfill the requirements.
        external_causal_mask = getattr(self, "_flagos_external_causal_mask", None)
        if external_causal_mask is not None:
            causal_mask = external_causal_mask
        else:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        #print('causal mack',causal_mask[0][0][-20:][:,-20:])
        #print(self.tree_mask)
        #print(position_ids)
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        #print(hidden_states.shape)
        #print(causal_mask.shape)
        #print('verify position ids',position_ids)
        #print()
        #print('model forward use cache')
        #print(use_cache)

        for decoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            if bool(getattr(next_decoder_cache, "_flagos_persistent_tree_cache", False)):
                next_cache = next_decoder_cache
            else:
                next_cache = (
                    next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
                )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            #loss = None,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        #disable this feature
        #to specify the attention mask.
        '''if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                #print('return None')
                return None'''

        dtype, device = input_tensor.dtype, input_tensor.device
        #print(attention_mask.shape)
        #print(input_tensor.shape)
        #print('cache_position',cache_position.shape)
        #print('past see tokens',past_seen_tokens)
        min_dtype = torch.finfo(dtype).min
        #max_dtype = torch.finfo(dtype).max
        sequence_length = input_tensor.shape[1]
        #print('sequence length',sequence_length)
        #print('attention mask',attention_mask)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
            #print('static cache')
        else:  # dynamic cache
            if hasattr(self, "tree_mask") and self.tree_mask is not None:
                target_length=past_seen_tokens + sequence_length
            elif isinstance(attention_mask, torch.Tensor):
                target_length = attention_mask.shape[-1]
            else:
                target_length = past_seen_tokens + sequence_length + 1
            '''target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )'''
        #print('target length',target_length)

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if hasattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value"):
            causal_mask = torch.triu(causal_mask, diagonal=1+past_seen_tokens)
        elif sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        #print('causal mask',causal_mask[:, -2])
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        #print('causal mask shape',causal_mask.shape)
        #print('causal mask num',causal_mask[0][0][-1])
        #print('attention mask shape',attention_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print('causal mask',causal_mask[:, :, -2])
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            #print('update based on attention mask')
            if attention_mask.dim() == 2:
                #print('dim = 2')
                mask_length = attention_mask.shape[-1]
                #print('mask length',mask_length)
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                #print(padding_mask)
                padding_mask = padding_mask == 0
                #print('padding mask',padding_mask)
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                #print('dim = 4')
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice
        #print('no tree mask')
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            #print('tree_mask',tree_mask)
            tree_len = tree_mask.size(-1)
            causal_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = min_dtype
        #else:
        #    print('no tree mask')
        #print('final tree mask')
        #print(causal_mask.shape)
        #print('causal mask',causal_mask[:, :, -2])
        #print(causal_mask.shape)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class SpecVLAforActionPrediction(nn.Module):
    '''def __init__(self,openvla=None,head=None):
        self.base_model = openvla
        self.ea_layer = head'''
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            parallel_draft=False,
            total_token=None,
            depth=None,
            top_k=None,
            threshold=None,
            accept_threshold=None
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.language_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.language_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        #if not parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
        config = EConfig.from_pretrained(ea_model_path)
        #else:
        #    config = EConfig.from_pretrained(ea_model_path)

        self.accept_threshold=accept_threshold
        #print('init accept threshold',accept_threshold)
        self.norm_stats = base_model.norm_stats

        # Compute action bins
        self.bins = base_model.bins
        self.bin_centers = base_model.bin_centers

        self.state = 0 # 1为进入KF,SD 0：0为纯SD
        self.cnt_kf = 0 # 进入状态1时，KF次数统计
        self.cnt_sig = 0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = base_model.vocab_size
        self.get_action_stats = base_model.get_action_stats
        if parallel_draft:
            #model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}
            with open(ea_model_path+'/model.safetensors', "rb") as f:
                safetensors_model = f.read()
                #pytorch_model = safetensors.torch.deserialize(safetensors_model)
                #print(state_dict.keys())
                self.ea_layer = PMMModel(config, path=base_model_name_or_path,load_emb=True)
                ea_layer_state_dict = safetensors.torch.load(safetensors_model)
        else:
            self.ea_layer = MMModel(config, path=base_model_name_or_path,load_emb=True)
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path)
        #self.ea_layer.init_tree()
        self.tree_mask = None
        low_memory = False

        device = base_model.language_model.model.layers[-1].self_attn.q_proj.weight.device
        #load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        # The draft embedding is replaced by the verifier embedding immediately below.
        # Allow smoke/deployment checkpoints to omit this redundant ~500 MB tensor,
        # but keep strict validation for every other draft parameter.
        ea_layer_state_dict.pop("embed_tokens.weight", None)
        incompatible = self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if key != "embed_tokens.weight"]
        if missing or unexpected:
            raise RuntimeError(
                "Invalid draft checkpoint: "
                f"missing keys={missing}, unexpected keys={unexpected}"
            )
        self.ea_layer.embed_tokens = self.base_model.language_model.model.embed_tokens
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.ea_layer.tree_mask = None
        self.ea_layer.tree_mode = None
        #print(self.ea_layer.fc.weight)
        #exit()
        #self.base_model.language_model = LlamaSpecForCausalLM(language_model)
        self._collect_errors = False
        self._error_stats = defaultdict(lambda: defaultdict(list))
        self._current_episode_key = None
        self._current_action_index = None
        self._error_output_path = None
        self._log_relaxed_mismatches = False
        self._rollout_max_steps: Optional[int] = None
        self._store_history_flag: bool = True
        self._current_history: Optional[Sequence[np.ndarray]] = None
        self._step_idx: int = -1
        self._kalman_config: Dict[str, Optional[Union[bool, float]]] = {
            "enabled": False,
            "process_var": None,
            "measurement_var": None,
        }
        self._kalman_history_window: Optional[int] = None
        self._kalman_tree_enabled: bool = True
        self._token_history: List[List[int]] = []
        self._last_action_token_ids_cpu: List[int] = []
        self._flagos_runtime_stats_path: Optional[Path] = None
        self._flagos_runtime_metadata: Dict[str, object] = {}
        self._flagos_runtime_writer = None
        self._flagos_timing_backend = "cpu"
        self._flagos_system_optimization_enabled = False
        self._flagos_decode_workspace = None
        self._flagos_process_warmup_done = False
    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
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

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )


        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = combined_attention_mask.min()

        return combined_attention_mask
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    def enable_error_collection(
        self,
        output_path: Union[str, Path],
        history_output_path: Optional[Union[str, Path]] = None,
        log_within_threshold: bool = False,
    ) -> None:
        """Enable speculative decoding error collection and set the output path.

        Args:
            output_path: path to save error stats (.npy)
            history_output_path: optional path to save accepted-actions history (.npy). If not set,
                a sibling file will be created next to `output_path`.
            log_within_threshold: when True, record mismatches even if they fall within the
                relaxed acceptance threshold. Useful when debugging relaxed speculative decoding.
        """
        self._collect_errors = True
        # error_stats: episode_key -> action_idx -> list of (correct, wrong, abs_diff, dof_index)
        self._error_stats = defaultdict(lambda: defaultdict(list))
        # accepted_actions: episode_key -> action_idx -> list of {"raw": [...], "discretized": [...]} per action
        self._accepted_actions = defaultdict(lambda: defaultdict(list))
        self._error_output_path = Path(output_path)
        self._history_output_path = Path(history_output_path) if history_output_path is not None else None
        self._log_relaxed_mismatches = log_within_threshold

    def enable_flagos_runtime_stats(
        self,
        output_path: Union[str, Path],
        *,
        flag_gems_enabled: bool = False,
        flag_gems_include: str = "",
        timing_backend: str = "cpu",
        async_logging: bool = False,
    ) -> None:
        """Record per-action Draft/Verify/Accept/Kalman statistics as JSONL.

        ``timing_backend=cuda_event`` records GPU phases without synchronizing
        at every boundary.  The action performs one final event synchronization
        when the JSONL record is assembled.  Logging can be moved to a bounded
        background writer so filesystem latency never enters the action path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._flagos_runtime_writer is not None:
            self._flagos_runtime_writer.close()
            self._flagos_runtime_writer = None
        path.write_text("", encoding="utf-8")
        self._flagos_runtime_stats_path = path
        timing_backend = str(timing_backend).lower()
        if timing_backend not in {"cpu", "cuda_event"}:
            raise ValueError("timing_backend must be cpu or cuda_event")
        self._flagos_timing_backend = timing_backend
        self._flagos_runtime_metadata = {
            "flag_gems_enabled": bool(flag_gems_enabled),
            "flag_gems_include": str(flag_gems_include),
            "timing_backend": timing_backend,
            "async_logging": bool(async_logging),
        }
        if async_logging:
            from .flagos_system_runtime import AsyncJsonlWriter

            self._flagos_runtime_writer = AsyncJsonlWriter(path)

    def close_flagos_runtime_stats(self) -> None:
        """Flush the optional asynchronous runtime writer."""
        writer = self._flagos_runtime_writer
        self._flagos_runtime_writer = None
        if writer is not None:
            writer.close()

    def _record_flagos_runtime_stats(self, record: Dict[str, object]) -> None:
        if self._flagos_runtime_stats_path is None:
            return
        payload = dict(self._flagos_runtime_metadata)
        payload.update(record)
        if self._flagos_runtime_writer is not None:
            self._flagos_runtime_writer.write(payload)
            return
        with self._flagos_runtime_stats_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def start_rollout(
        self,
        *,
        max_steps: Optional[int] = None,
        kalman_cfg: Optional[Dict[str, Optional[Union[bool, float]]]] = None,
        store_history: bool = True,
    ) -> None:
        """Reset per-episode speculative decoding context."""

        self._rollout_max_steps = int(max_steps) if max_steps is not None else None
        self._store_history_flag = store_history
        self._current_history = [] if store_history else None
        self._step_idx = -1
        self._token_history = []

        cfg = kalman_cfg or {}
        if "enabled" in cfg or "use_kalman" in cfg:
            self._kalman_config["enabled"] = bool(cfg.get("enabled") or cfg.get("use_kalman"))
        if "process_var" in cfg:
            self._kalman_config["process_var"] = cfg["process_var"]
        if "measurement_var" in cfg:
            self._kalman_config["measurement_var"] = cfg["measurement_var"]
        if "history_window" in cfg:
            value = cfg["history_window"]
            self._kalman_history_window = int(value) if value not in (None, False) else None
        if "tree_enabled" in cfg:
            self._kalman_tree_enabled = bool(cfg["tree_enabled"])

    def set_step_context(
        self,
        step_idx: int,
        accepted_actions: Optional[Sequence[np.ndarray]],
    ) -> None:
        """Track the current rollout step and accepted action history."""

        self._step_idx = int(step_idx)
        if not self._store_history_flag:
            self._current_history = None
            return

        if accepted_actions is None:
            self._current_history = []
        else:
            self._current_history = list(accepted_actions)

    def set_dynamic_threshold(self, value: Optional[float]) -> None:
        """Override the acceptance threshold for the next speculative step."""

        if value is None:
            return
        self.accept_threshold = float(value)

    def _run_prompt_stage_graph(self, model_inputs: Dict[str, object]):
        """Run the fixed-shape prompt forward through an optional CUDA Graph."""
        cache = getattr(self, "_flagos_prompt_stage_graph", None)
        if cache is None:
            return None
        from .flagos_stage_graph import StageGraphCache

        kwargs = dict(model_inputs)
        kwargs.update(
            {
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": True,
                "output_orig": True,
            }
        )
        result, mode = cache.run(self, kwargs)
        self._flagos_last_prompt_graph_mode = mode
        return result

    def _run_draft_stage_graph(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        past_key_values,
    ):
        """Run only the drafter transformer block through a fixed-shape graph.

        Top-k selection, tree construction, and Kalman augmentation remain
        outside the graph.  If a cache implementation mutates a layout that is
        not graph-safe, the runner returns ``None`` and the original forward is
        used for that request.
        """
        cache = getattr(self, "_flagos_draft_stage_graph", None)
        if cache is None:
            return None
        kwargs = {
            "hidden_states": hidden_states,
            "input_embeddings": input_embeddings,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

        def forward(**values):
            return self.ea_layer(
                values["hidden_states"],
                input_embeddings=values["input_embeddings"],
                past_key_values=values["past_key_values"],
                use_cache=True,
            )

        result, mode = cache.run(forward, kwargs)
        self._flagos_last_draft_graph_mode = mode
        if result is None:
            return None
        return result

    def set_kalman_enabled(self, enabled: bool) -> None:
        """Toggle the Kalman fallback flag for the current rollout."""

        self._kalman_config["enabled"] = bool(enabled)

    def set_kalman_tree_enabled(self, enabled: bool) -> None:
        """Enable or disable Kalman-based tree augmentation."""

        self._kalman_tree_enabled = bool(enabled)

    def set_rollout_metadata(
        self,
        *,
        max_steps: Optional[int] = None,
        process_var: Optional[float] = None,
        measurement_var: Optional[float] = None,
        history_window: Optional[int] = None,
        tree_enabled: Optional[bool] = None,
    ) -> None:
        """Update auxiliary rollout parameters without resetting the episode."""

        if max_steps is not None:
            self._rollout_max_steps = int(max_steps)
        if process_var is not None:
            self._kalman_config["process_var"] = process_var
        if measurement_var is not None:
            self._kalman_config["measurement_var"] = measurement_var
        if history_window is not None:
            self._kalman_history_window = int(history_window) if history_window >= 0 else None
        if tree_enabled is not None:
            self._kalman_tree_enabled = bool(tree_enabled)

    def predict_kf_chain_tokens(
        self,
        prefix_tokens: Sequence[int],
        remaining_dims: int,
    ) -> List[int]:
        """Predict a chain of fallback tokens using the configured Kalman module.

        The default implementation delegates to an optional ``kalman_predict_tokens`` hook on the
        model. When the hook is absent or returns an empty sequence, we fall back to repeating the
        last token in ``prefix_tokens`` so the tree still exposes a deterministic KF branch.
        """

        if remaining_dims <= 0:
            return []

        if not self._kalman_config.get("enabled", False):
            return []

        history = self._token_history if self._store_history_flag else None
        process_var = self._kalman_config.get("process_var")
        measurement_var = self._kalman_config.get("measurement_var")

        predictions: List[int] = []
        if hasattr(self, "kalman_predict_tokens"):
            try:
                raw = self.kalman_predict_tokens(
                    prefix_tokens=list(prefix_tokens),
                    remaining_dims=remaining_dims,
                    history=history,
                    process_var=process_var,
                    measurement_var=measurement_var,
                )
                predictions = list(raw) if raw is not None else []
            except Exception:
                predictions = []

        if not predictions:
            fallback_token = int(prefix_tokens[-1]) if prefix_tokens else int(getattr(self.tokenizer, "pad_token_id", 0))
            predictions = [fallback_token for _ in range(remaining_dims)]

        return [int(tok) for tok in predictions[:remaining_dims]]

    def predict_kf_chains_batched(
        self,
        prefix_batch: Sequence[Sequence[int]],
        remaining_dims_batch: Sequence[int],
    ) -> List[List[int]]:
        """Generate every Kalman tail after computing the history estimate once.

        With valid action history, ``kalman_predict_tokens`` is independent of
        the candidate prefix and only its tail slice changes. Without history,
        preserve the original per-prefix repeat-last-token fallback exactly.
        """
        if len(prefix_batch) != len(remaining_dims_batch):
            raise ValueError("prefix and remaining-dimension batches must align")
        if not prefix_batch:
            return []
        if not self._kalman_config.get("enabled", False):
            return [[] for _ in prefix_batch]
        try:
            action_dim = int(self.get_action_dim())
        except Exception:
            action_dim = 0
        history = self._token_history if self._store_history_flag else None
        process_var = self._kalman_config.get("process_var")
        measurement_var = self._kalman_config.get("measurement_var")
        full_prediction: List[int] = []
        if action_dim > 0 and hasattr(self, "kalman_predict_tokens"):
            try:
                raw = self.kalman_predict_tokens(
                    prefix_tokens=list(prefix_batch[0]),
                    remaining_dims=action_dim,
                    history=history,
                    process_var=process_var,
                    measurement_var=measurement_var,
                )
                full_prediction = list(raw) if raw is not None else []
            except Exception:
                full_prediction = []

        results: List[List[int]] = []
        for prefix, remaining in zip(prefix_batch, remaining_dims_batch):
            remaining = max(0, int(remaining))
            if remaining == 0:
                results.append([])
                continue
            if len(full_prediction) >= action_dim and action_dim > 0:
                start = max(0, action_dim - remaining)
                values = full_prediction[start:action_dim]
            else:
                fallback = int(prefix[-1]) if prefix else int(
                    getattr(self.tokenizer, "pad_token_id", 0)
                )
                values = [fallback] * remaining
            results.append([int(token) for token in values[:remaining]])
        self._flagos_kalman_batch_hits = int(
            getattr(self, "_flagos_kalman_batch_hits", 0)
        ) + 1
        self._flagos_kalman_batch_chains = int(
            getattr(self, "_flagos_kalman_batch_chains", 0)
        ) + len(results)
        return results

    def kalman_predict_tokens(
        self,
        *,
        prefix_tokens: Sequence[int],
        remaining_dims: int,
        history: Optional[Sequence[Sequence[int]]],
        process_var: Optional[float],
        measurement_var: Optional[float],
    ) -> List[int]:
        """Predict future action tokens via 1D Kalman filtering per DoF."""

        if history is None or remaining_dims <= 0:
            return []

        try:
            action_dim = int(self.get_action_dim())
        except Exception:
            return []

        if action_dim <= 0:
            return []

        valid_sequences: List[List[int]] = []
        for seq in history:
            if seq is None:
                continue
            seq_list = [int(tok) for tok in seq if tok is not None]
            if len(seq_list) >= action_dim:
                valid_sequences.append(seq_list[:action_dim])

        if not valid_sequences:
            return []

        window = self._kalman_history_window
        if window is not None and window > 0 and len(valid_sequences) > window:
            valid_sequences = valid_sequences[-window:]

        per_dof_history: List[List[float]] = [[] for _ in range(action_dim)]
        for seq in valid_sequences:
            for idx in range(action_dim):
                per_dof_history[idx].append(float(seq[idx]))

        default_process = float(process_var) if process_var is not None else 1.0
        default_measurement = float(measurement_var) if measurement_var is not None else 1e-3

        predictions_full: List[int] = []
        for idx in range(action_dim):
            series = per_dof_history[idx]
            if not series:
                if valid_sequences:
                    predictions_full.append(int(valid_sequences[-1][idx]))
                else:
                    fallback_tok = int(prefix_tokens[-1]) if prefix_tokens else int(getattr(self.tokenizer, "pad_token_id", 0))
                    predictions_full.append(fallback_tok)
                continue

            try:
                estimate = kalman_predict_from_history(
                    series,
                    process_variance=default_process,
                    measurement_variance=default_measurement,
                )
            except Exception:
                estimate = series[-1]

            predictions_full.append(int(round(estimate)))

        action_token_start = self.vocab_size - self.bin_centers.shape[0]
        token_min = action_token_start
        token_max = self.vocab_size - 1
        predictions_full = [max(token_min, min(token_max, tok)) for tok in predictions_full]

        start_idx = max(0, action_dim - remaining_dims)
        tail = predictions_full[start_idx:action_dim]
        if len(tail) < remaining_dims:
            fallback_token = int(prefix_tokens[-1]) if prefix_tokens else predictions_full[-1]
            tail.extend([fallback_token] * (remaining_dims - len(tail)))
        return tail[:remaining_dims]

    def set_logging_context(self, episode_key: Optional[str]) -> None:
        """Update the current logging context key (e.g., task_id_episode_id)."""
        self._current_episode_key = episode_key
        self._current_action_index = None

    def set_action_step(self, action_index: Optional[int]) -> None:
        """Track the current environment step for error attribution."""
        self._current_action_index = action_index

    def save_error_stats(self) -> None:
        """Persist collected speculative decoding errors to disk if enabled."""
        if not self._collect_errors or self._error_output_path is None:
            return

        output_dir = self._error_output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {episode: {action_idx: entries for action_idx, entries in action_map.items()} for episode, action_map in self._error_stats.items()}
        np.save(self._error_output_path, payload, allow_pickle=True)
        print(f"Saved speculative decoding error stats to {self._error_output_path}")

        # Save accepted-actions history to a sibling file or explicit history path
        if getattr(self, "_history_output_path", None) is None:
            actions_path = self._error_output_path.with_name(self._error_output_path.stem + "_accepted_actions.npy")
        else:
            actions_path = self._history_output_path
        actions_payload = {episode: {action_idx: entries for action_idx, entries in action_map.items()} for episode, action_map in self._accepted_actions.items()}
        np.save(actions_path, actions_payload, allow_pickle=True)
        print(f"Saved speculative decoding accepted actions to {actions_path}")

    @property
    def collect_errors(self) -> bool:
        return self._collect_errors

    def _record_candidate_mismatch(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        candidate_idx: int,
        accept_length: int,
        accept_threshold: Optional[int],
        accepted_tokens_before: int,
    ) -> None:
        """Record the first mismatch between draft and verifier tokens during strict decoding."""
        if not self._collect_errors or self._current_episode_key is None:
            return

        if accept_threshold not in (None, 0) and not self._log_relaxed_mismatches:
            return

        # candidates includes a start token; compare against verifier tokens for the best candidate.
        if candidates.size(1) <= 1:
            return

        # All tokens aligned; nothing to record.
        if accept_length >= candidates.size(1) - 1:
            return

        verifier_logits = logits[candidate_idx, :-1].detach().cpu()
        if accept_length >= verifier_logits.shape[0]:
            return

        verifier_tokens = torch.argmax(verifier_logits, dim=-1)
        verifier_tokens = verifier_tokens + int(
            getattr(self, "_flagos_action_logit_offset", 0)
        )
        wrong_token_tensor = candidates[candidate_idx, accept_length + 1]
        wrong_token = int(wrong_token_tensor.item())
        # Skip padding or invalid draft tokens
        if wrong_token < 0:
            return

        correct_token = int(verifier_tokens[accept_length].item())
        action_start = self.vocab_size - self.bin_centers.shape[0]
        # Filter out non-action tokens such as <eos>/<pad>
        if correct_token < action_start or wrong_token < action_start:
            return

        threshold_value = abs(correct_token - wrong_token)
        # Map accept_length -> DOF index (1-based) accounting for tokens already accepted
        dof_index = int(accepted_tokens_before + accept_length) + 1
        try:
            action_dim = int(self.get_action_dim())
        except Exception:
            action_dim = None
        if action_dim is not None and dof_index > action_dim:
            return
        action_idx = self._current_action_index if self._current_action_index is not None else -1
        # Store 4-tuple: (correct_token, wrong_token, abs_diff, dof_index)
        self._error_stats[self._current_episode_key][action_idx].append((correct_token, wrong_token, threshold_value, dof_index))
    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        return self.base_model.get_action_dim(unnorm_key)
    def forward(
        self,
        output_orig=False,
        input_embeds = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        #inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None
    ):
         #先这样，后面看结合需求怎么改，我的判断是得根据需要的数据模态把需要的内容放进去
         with torch.inference_mode():
            #reorganize the embeddings
            #print('forward not tested.')
            #print(output_hidden_states)
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values = pixel_values,
                labels = labels,
                inputs_embeds = input_embeds,
                past_key_values=past_key_values,
                use_cache = use_cache,
                output_attentions = output_attentions,
                output_hidden_states=True,
                output_projector_features=output_projector_features,
                return_dict=return_dict,
                position_ids=position_ids
            )
            #print(outputs.keys())
            if output_orig:
                orig = outputs.logits
            #print(len(outputs.hidden_states))
            hidden_states = outputs.hidden_states[-1]
            input_embeddings = outputs.hidden_states[0]
            #print(len(hidden_states))
            #print(torch.cat(hidden_states).shape)
            if output_orig:
                return outputs, orig, hidden_states,input_embeddings
            else:
                return outputs, hidden_states
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        return_hidden_states: bool = False,
        legacy_output_hidden: Optional[bool] = None,
        generate_mode = 'Speculative',
        #accept_threshold=None,
        **kwargs: str
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[torch.FloatTensor]]]:
        """Wrapper around .generate() that decodes predicted actions and can return hidden states.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalizing actions
            return_hidden_states: Whether to return the last hidden state
            legacy_output_hidden: Legacy parameter, equivalent to return_hidden_states
            **kwargs: Additional arguments for generate

        Returns:
            If return_hidden_states=False:
                unnormalized actions as numpy array
            Otherwise:
                Tuple of (unnormalized_actions, hidden_states)
        """
        # 处理参数，支持旧的参数命名方式
        if legacy_output_hidden is not None:
            return_hidden_states = legacy_output_hidden

        # 设置generate方法的参数
        if return_hidden_states:
            kwargs['output_hidden_states'] = True

        # 如果特殊的空标记不在提示末尾，则添加它
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            #print('add special token')
            #print(kwargs['attention_mask'])
            kwargs['attention_mask']=torch.cat(
                (kwargs['attention_mask'], torch.unsqueeze(torch.Tensor([1]), dim=0).to(input_ids.device)), dim=1
            ).to(int)
            #print(kwargs['attention_mask'])
        #print(kwargs)
        #exit()
        # 运行模型生成
        #print('base model generate')
        '''outputs = self.ea_forward(
            input_ids=input_ids,
            max_new_tokens=self.get_action_dim(unnorm_key),
            #return_dict=True,
            #return_dict_in_generate=True,
            **kwargs
        )'''
        if generate_mode == 'speculative':
            outputs = self.eagenerate(
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                #return_dict=True,
                #return_dict_in_generate=True,
                accept_threshold=self.accept_threshold,
                **kwargs
            )
            # with open('./log_eagenerate.txt', 'a') as f:
            #     f.write(f"[* outputs]{outputs}\n")
            #print(outputs)
        else:
            #print('ea_forward')
            outputs = self.ea_forward(
                input_ids=input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                #return_dict=True,
                #return_dict_in_generate=True,
                **kwargs
                )
        #print(outputs)
        #print(outputs.hidden_states[0][0].shape)
        #input()
        #print(outputs.shape)
        #print(-self.get_action_dim(unnorm_key))
       # exit()
        #print("LOCAL_TRANSFORMER.generate方法!!!!!!!")
        #print(outputs)
        # 获取生成的token IDs
        #print(generated_ids)
        if hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
            #eagle_generated_ids = eagle_outputs.sequences
        elif len(outputs)==2:
            generated_ids = outputs[0]
        else:
            generated_ids = outputs
            #eagle_generated_ids = eagle_outputs.sequences
        #print('generate ids',generated_ids)
        #print('output sequence',outputs)
        #print(-self.get_action_dim(unnorm_key))
        #exit()
        #print(generated_ids)
        # 从生成的tokens转换为动作值
        action_dim = self.get_action_dim(unnorm_key)
        if generate_mode == 'speculative' and len(self._last_action_token_ids_cpu) == action_dim:
            predicted_action_token_ids = np.asarray(self._last_action_token_ids_cpu, dtype=np.int64)
        else:
            predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
        #print('predicted_ids',predicted_action_token_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        #print('vocab_size',self.vocab_size)
        #print('final_action_token',discretized_actions)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        normalized_actions = self.bin_centers[discretized_actions]
       # print(normalized_actions)
        # 反归一化动作
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        # 如果需要返回隐藏状态
        if return_hidden_states:
            if (len(outputs)==2):
                tmp_hidden = outputs[1]
            else:
                tmp_hidden=outputs.hidden_states
            # 使用前向传播获取隐藏状态
            first_layer_hidden = []
            last_layer_hidden = []
            #print(len(outputs.hidden_states))
            for i in range(len(tmp_hidden)):
                last_layer_hidden.append(tmp_hidden[i][-1].cpu()[0])
                first_layer_hidden.append(tmp_hidden[i][0].cpu()[0])
            # 返回二元组: (动作, 隐藏状态)
            #print(last_layer_hidden[0].shape)
            return actions, predicted_action_token_ids,(first_layer_hidden,last_layer_hidden)

        # 否则只返回动作
        return actions
    @torch.no_grad()
    def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder
    ):
        #print(model_kwargs.keys())
        #print(outputs.keys())
        #exit()
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs
        )
        #print(model_kwargs["past_key_values"][0][0].shape)
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                #print('update attention mask')
                #print(attention_mask)
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            #else:
            #    print('no attention mask to update')
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs
    @torch.no_grad()
    def ea_forward(self,input_ids,max_new_tokens, logits_processor=None,output_hidden_states=False,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['attention_mask'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        if output_hidden_states:
            hidden_states = []
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states
            )
        if output_hidden_states:
            hidden_states.append(outputs.hidden_states)
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        #print(type(outputs['past_key_values']))
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        model_inputs['attention_mask']=outputs.attention_mask
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            #print(model_inputs)
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                #use_cache = True
            )
            if output_hidden_states:
                hidden_states.append(outputs.hidden_states)
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        #print('ea forward',output_hidden_states)
        if output_hidden_states:
            #print(outputs.hidden_states)
            return input_ids[:,input_len+1:],hidden_states[:-1]
        return input_ids[:,input_len+1:]
    def ea_forward_embed(self,input_ids,max_new_tokens, logits_processor=None,**kwargs):
        #prefill the past kv embeddings
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        # Initialize the past key and value states
        #use the openvla.forward to initilaize kv
        model = self
        '''if hasattr(model.base_model.language_model, "past_key_values"):
            past_key_values = model.base_model.language_model.past_key_values
            past_key_values_data = model.base_model.language_model.past_key_values_data
            current_length_data = model.base_model.language_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model)
            model.base_model.language_model.past_key_values = past_key_values
            model.base_model.language_model.past_key_values_data = past_key_values_data
            model.base_model.language_model.current_length_data = current_length_data
        #print(len(model.base_model.language_model.past_key_values))'''
        #print(model.base_model.language_model.past_key_values[0][1].data.shape)
        #exit()
        input_len = input_ids.shape[1]
        #reset_tree_mode(model.ea_layer)
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = model.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #print('model inputs:')
        #print(model_inputs['input_ids'])
        #print(model_inputs['pixel_values'])
        #exit()
        #print('start forwarding')
        #print(model_inputs)
        outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        #print('outputs')
        #print('loss',outputs.loss)
        #print('logits',outputs.logits)
        #print('past key values',outputs.past_key_values)
        #print('hidden states',outputs.hidden_states)
        #print('attentions',outputs.attentions)
        #print('projector',outputs.projector_features)
        #exit()
        input_len = input_ids.shape[1]-1
        input_embed_len = outputs['past_key_values'][0][0].shape[2]-1
        new_token = 0
        model_inputs["cache_position"] = torch.arange(input_embed_len, device=input_ids.device)
        model_inputs['use_cache']=True
        for idx in range(max_steps):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            #print(input_id)
            #exit()
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            model_inputs = self._update_model_kwargs_for_generation(
                outputs,
                model_inputs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_inputs['input_ids']=input_ids
            model_inputs = model.base_model.prepare_inputs_for_generation(**model_inputs)
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                #use_cache = True
            )
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > 1024:
                break
            if input_ids.shape[1] > 1960:
                break
        return input_ids[:,input_len:]
    def get_history(self):
        '''
        得到历史
        获取当前 episode 的运行时 action-token 历史。

        Kalman 执行不能依赖仅在 mismatch 诊断开启时才创建的
        ``_accepted_actions``，否则关闭日志会改变推理语义。
        以list形式返回，每个元素是一个ndarray，表示一个时间步的accepted action
        [[31869, 31884, 31872, 31891, 31902, 31926, 31744], [31869, 31884, 31872, 31891, 31896, 31931, 31744], [31826, 31884, 31872, 31893, 31824, 31929, 31744]]
        '''
        return [list(tokens) for tokens in self._token_history]
    def get_history_with_win_and_EOF_id(self,dof_id, window_size):
        '''
        获取自由度历史
        将[[31869, 31884, 31872, 31891, 31902, 31926, 31744], [31869, 31884, 31872, 31891, 31896, 31931, 31744], [31826, 31884, 31872, 31893, 31824, 31929, 31744]]
        按照DOF id取为一个list，并且要有窗口大小，从最后一个时间步开始往前取，如果不够就尽可能取就好
        '''
        # print("[*] in get_history_with_win_and_EOF_id")
        acts = self.get_history()
        # print(f"[*] full history: {acts}")
        if len(acts) == 0:
            return []
        action_dim = len(acts[0])
        history_with_win = []
        for i in range(window_size):
            idx = len(acts) - 1 - i
            if idx < 0:
                break
            history_with_win.append(acts[idx][dof_id])
        history_with_win.reverse()
        return history_with_win

    def kelman_predict_tokens(self):
        '''
        卡尔曼预测
        对当前每一个自由度，使用kalman filter预测下一个token

        '''
        action_dim = int(self.get_action_dim())
        history_window = self._kalman_history_window
        if history_window is None or history_window <= 0:
            try:
                history_window = len(self.get_history())
            except Exception:
                history_window = 0

        action_token_start = self.vocab_size - self.bin_centers.shape[0]
        token_min = action_token_start
        token_max = self.vocab_size - 1

        pad_token = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token is None:
            pad_token = token_min
        pad_token = int(max(token_min, min(token_max, pad_token)))

        ans: List[int] = []
        for dof_id in range(action_dim):
            try:
                history = self.get_history_with_win_and_EOF_id(dof_id, history_window)
            except Exception:
                history = []

            if not history:
                pred_token = pad_token
            else:
                try:
                    pred_token = int(kalman_predict_from_history(history))
                except Exception:
                    pred_token = int(history[-1])

            pred_token = max(token_min, min(token_max, int(pred_token)))
            ans.append(pred_token)
        return ans

    def _finalize_action_with_kf(
        self,
        input_ids: torch.LongTensor,
        input_len: int,
        candidates: torch.Tensor,
        best_candidate_idx: int,
        accept_length: int,
    ) -> Tuple[torch.LongTensor, int]:
        action_dim = int(self.get_action_dim())
        if action_dim <= 0:
            return input_ids, 0

        candidate_row = candidates[best_candidate_idx].detach()
        candidate_values = [int(token) for token in candidate_row.to("cpu").tolist()]
        start_token = candidate_values[0]
        accept_len = max(int(accept_length), 0)

        action_token_start = self.vocab_size - self.bin_centers.shape[0]
        token_min = action_token_start
        token_max = self.vocab_size - 1

        pad_token = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token is None:
            pad_token = token_min
        pad_token = int(max(token_min, min(token_max, pad_token)))

        def clamp_token(token_value: int) -> int:
            return int(max(token_min, min(token_max, token_value)))

        accepted_tokens: List[int] = []
        if accept_len > 0:
            accepted_tokens = [
                clamp_token(token)
                for token in candidate_values[1 : 1 + accept_len]
                if token >= 0
            ]

        candidate_tail = [
            clamp_token(token)
            for token in candidate_values[1 + accept_len :]
            if token >= 0
        ]

        fallback_token = clamp_token(accepted_tokens[-1] if accepted_tokens else (candidate_tail[0] if candidate_tail else pad_token))

        remaining_dims = max(action_dim - len(accepted_tokens), 0)
        kf_tail: List[int] = []
        if remaining_dims > 0:
            try:
                kf_full = self.kelman_predict_tokens()
            except Exception:
                kf_full = []
            if kf_full:
                offset = len(accepted_tokens)
                tail_slice = kf_full[offset : offset + remaining_dims]
                kf_tail = [clamp_token(int(tok)) for tok in tail_slice]

        combined_tokens = accepted_tokens + kf_tail
        if len(combined_tokens) < action_dim and candidate_tail:
            needed = action_dim - len(combined_tokens)
            combined_tokens.extend(candidate_tail[:needed])

        if len(combined_tokens) < action_dim:
            missing = action_dim - len(combined_tokens)
            combined_tokens.extend([fallback_token] * missing)

        final_dof_tokens = combined_tokens[:action_dim]

        append_tensor = torch.tensor(
            [start_token] + final_dof_tokens,
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(0)
        input_ids = torch.cat([input_ids, append_tensor], dim=-1)
        return input_ids, len(final_dof_tokens)

    @torch.no_grad()
    def eagenerate(
        self,
        input_ids,
        max_new_tokens,
        #return_dict=True,
        #return_dict_in_generate=True,
        log = False,
        accept_threshold=None,
        **kwargs
    ):
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        runtime_stats_enabled = bool(
            self._flagos_runtime_stats_path is not None
            and not getattr(self, "_flagos_suppress_runtime_stats", False)
        )
        system_optimization_enabled = bool(
            getattr(self, "_flagos_system_optimization_enabled", False)
        )
        workspace_optimization_enabled = bool(
            system_optimization_enabled
            and getattr(self, "_flagos_persistent_decode_workspace", False)
        )
        control_optimization_enabled = bool(
            system_optimization_enabled
            and getattr(self, "_flagos_persistent_control_buffer", False)
        )
        from .flagos_system_runtime import ControlBuffer, DecodeWorkspace, PhaseTimer

        phase_timer = PhaseTimer(
            input_ids.device,
            runtime_stats_enabled,
            getattr(self, "_flagos_timing_backend", "cpu"),
        )
        control_buffer = None
        if control_optimization_enabled and input_ids.device.type == "cuda":
            control_buffer = getattr(self, "_flagos_control_buffer", None)
            if control_buffer is None or control_buffer.device != input_ids.device:
                control_buffer = ControlBuffer(input_ids.device)
                self._flagos_control_buffer = control_buffer
        runtime_start = time.perf_counter()
        # Do not let the previous action's backend label leak into a new
        # record when an action terminates before Commit/Draft is reached.
        self._flagos_joint_graph_last_mode = "disabled"
        graph_capture_overhead_start = float(
            getattr(self, "_flagos_cuda_graph_capture_overhead_s", 0.0)
        )
        joint_graph_capture_overhead_start = float(
            getattr(self, "_flagos_joint_graph_capture_overhead_s", 0.0)
        )
        accept_lengths: List[int] = []
        best_candidate_indices: List[int] = []
        candidate_shapes: List[List[int]] = []
        kalman_tokens_added = 0
        next_token_control_transfers = 0
        fused_control_transfers = 0
        ended_with_stop_token = False
        self._last_action_token_ids_cpu = []
        phase_latency_s: Dict[str, object] = {
            "prompt_and_initial_draft": 0.0,
            "verify": [],
            "verify_capture_overhead": [],
            "verify_replay_net": [],
            "verify_graph_mode": [],
            "posterior_and_accept": [],
            "commit_and_next_draft": [],
            "kalman_completion": [],
        }

        def phase_start(name: str):
            if not runtime_stats_enabled:
                return None
            return phase_timer.start(name)

        def phase_end(start) -> float:
            if start is None:
                return 0.0
            return phase_timer.end(start)
        # Avoid modifying the input_ids in-place

        padding = getattr(self, "_flagos_padding_token_buffer", None)
        if (
            padding is None
            or padding.device != input_ids.device
            or padding.dtype != input_ids.dtype
        ):
            padding = torch.full(
                (1, 1), -1, device=input_ids.device, dtype=input_ids.dtype
            )
            self._flagos_padding_token_buffer = padding
        # The speculative path never mutates the caller's prompt in-place.
        # Phase 6 therefore avoids an action-sized clone before immediately
        # copying the prompt into DecodeWorkspace below.
        if not bool(getattr(self, "_flagos_inference_mode_enabled", False)):
            input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = max_new_tokens
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        # Prompt/vision prefill is not part of the fixed tree verifier graph.
        # Clear the previous action's resident tree mask before prefill.
        self.base_model.language_model._flagos_external_causal_mask = None
        if bool(
            getattr(self, "_flagos_persistent_runtime_enabled", False)
            or getattr(self, "_flagos_resident_kv_cache_enabled", False)
        ):
            from .flagos_persistent_cache import KervPersistentTreeCache

            persistent_cache = getattr(self, "_flagos_persistent_kv_cache", None)
            if persistent_cache is None:
                language_config = self.base_model.language_model.config
                device = input_ids.device
                dtype = self.base_model.language_model.model.layers[0].self_attn.q_proj.weight.dtype
                persistent_cache = KervPersistentTreeCache(
                    language_config,
                    prefix_capacity=int(self._flagos_persistent_prefix_capacity),
                    tree_capacity=int(self._flagos_persistent_tree_capacity),
                    device=device,
                    dtype=dtype,
                    fixed_workspace_layout=bool(
                        getattr(self, "_flagos_fixed_workspace_layout", False)
                    ),
                )
                self._flagos_persistent_kv_cache = persistent_cache
                self._flagos_persistent_allocations = int(
                    getattr(self, "_flagos_persistent_allocations", 0)
                ) + 1
            # Keep the phase-two operator selection on the resident cache as
            # well as on the model.  The commit helper is called from a
            # lower-level utility which otherwise cannot see the model's
            # FlagScale configuration.
            persistent_cache._flagos_embodied_ops_enabled = bool(
                getattr(self, "_flagos_embodied_ops_enabled", False)
            )
            persistent_cache._flagos_embodied_ops_include = set(
                getattr(self, "_flagos_embodied_ops_include", set())
            )
            persistent_cache._flagos_embodied_ops_backend = str(
                getattr(self, "_flagos_embodied_ops_backend", "auto")
            )
            persistent_cache.reset()
            model_inputs["past_key_values"] = persistent_cache
            model_inputs["use_cache"] = True
        reset_tree_mode(self.ea_layer)
        # CUDA Graph only captures the first Verify of each action (fixed
        # prompt-cache length). Bucket that tree, but keep later eager
        # fallbacks at their real node count to avoid padded GEMM regressions.
        self._flagos_tree_use_node_buckets = True
        time_0 = time.time()
        initialize_start = phase_start("prompt_and_initial_draft")
        with torch.profiler.record_function("KERV::PromptInitialDraft"):
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, prompt_hidden_states, sample_token, past_key_value_data,prompt_embeds,attention_mask = initialize_tree(model_inputs, self, logits_processor)
        if runtime_stats_enabled:
            phase_latency_s["prompt_and_initial_draft"] = phase_end(initialize_start)

        if workspace_optimization_enabled:
            # Allocate once per loaded model/action shape.  The workspace only
            # serves batch-one contiguous tensors; any incompatible input keeps
            # the existing allocation path below.
            workspace = getattr(self, "_flagos_decode_workspace", None)
            if workspace is None:
                workspace = DecodeWorkspace.create(
                    input_ids,
                    prompt_embeds,
                    max_new_tokens,
                    fixed_draft_input=bool(
                        getattr(self, "_flagos_fixed_draft_input_workspace", False)
                    ),
                )
                self._flagos_decode_workspace = workspace
            if workspace is not None and workspace.reset(input_ids, prompt_embeds):
                input_ids = workspace.input_ids[:, : input_ids.shape[1]]
                prompt_embeds = workspace.prompt_embeds[:, : prompt_embeds.shape[1]]
                self._flagos_decode_workspace_hits = int(
                    getattr(self, "_flagos_decode_workspace_hits", 0)
                ) + 1
            else:
                self._flagos_decode_workspace = None

        def append_candidate_padding(tokens: torch.Tensor) -> torch.Tensor:
            """Append the sentinel using a persistent buffer when possible."""
            if not workspace_optimization_enabled or tokens.ndim != 2 or tokens.shape[0] != 1:
                return torch.cat((tokens, padding), dim=1)
            buffers = getattr(self, "_flagos_candidate_padding_buffers", None)
            if buffers is None:
                buffers = {}
                self._flagos_candidate_padding_buffers = buffers
            key = (tokens.device, tokens.dtype, int(tokens.shape[1]))
            buffer = buffers.get(key)
            if buffer is None:
                buffer = torch.empty(
                    (1, int(tokens.shape[1]) + 1),
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
                buffers[key] = buffer
            buffer[:, :-1].copy_(tokens)
            buffer[:, -1].fill_(-1)
            self._flagos_candidate_padding_hits = int(
                getattr(self, "_flagos_candidate_padding_hits", 0)
            ) + 1
            return buffer

        input_len = input_ids.shape[1]-1
        max_length = max_length - self.ea_layer.total_tokens - 10
        # print(f'best_length:{max_length}')
        new_token = 0
            # 每4次使用 Speculative (SD) 后，进行1次 Kalman Fallback (KF)
            # 也就是说当 (step_idx+1) 对 5 取模为 0 时触发 KF：SD,SD,SD,SD, KF, ...
        force_direct_completion = self._step_idx >= 0 and ((self._step_idx + 1) % 4 == 0)
        # force_direct_completion = False #暂时关闭kf
        for idx in range(max_length):
            # with Timer("all"):
            cycle_begin_time = time.time()
            self.base_model.language_model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)
            verify_start = phase_start("verify")
            verify_capture_start = float(
                getattr(self, "_flagos_cuda_graph_capture_overhead_s", 0.0)
            )
            with torch.profiler.record_function("KERV::TreeVerify"):
                logits, hidden_state_new,hidden_embedding_new,past_kv_data_new,outputs= tree_decoding(
                    self,
                    prompt_embeds,
                    draft_tokens,
                    attention_mask,
                    past_key_value_data,
                    tree_position_ids,
                    #input_ids,
                    retrieve_indices,
                    #draft_logit=draft_logit
                )
            if runtime_stats_enabled:
                verify_elapsed = phase_end(verify_start)
                verify_capture_delta = max(
                    0.0,
                    float(getattr(self, "_flagos_cuda_graph_capture_overhead_s", 0.0))
                    - verify_capture_start,
                )
                phase_latency_s["verify"].append(verify_elapsed)
                phase_latency_s["verify_capture_overhead"].append(
                    verify_capture_delta
                )
                phase_latency_s["verify_replay_net"].append(
                    max(0.0, verify_elapsed - verify_capture_delta)
                )
                phase_latency_s["verify_graph_mode"].append(
                    str(getattr(self, "_flagos_last_cuda_graph_mode", "eager"))
                )
            draft_tokens = append_candidate_padding(draft_tokens)
            candidates = draft_tokens[0, retrieve_indices]
            # with open('./log_eagenerate.txt', 'a') as f:
            #     f.write(f"[* candidates]{candidates}\n")
            posterior_start = phase_start("posterior_and_accept")
            with torch.profiler.record_function("KERV::PosteriorAccept"):
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits,
                    candidates,
                    logits_processor,
                    accept_threshold=accept_threshold,
                    token_offset=int(getattr(self, "_flagos_action_logit_offset", 0)),
                    runtime_owner=self,
                )
            if runtime_stats_enabled:
                phase_latency_s["posterior_and_accept"].append(phase_end(posterior_start))
            # with open('./log_eagenerate.txt', 'a') as f:
            #     f.write(f"[* best_candidate]{best_candidate}\n")
            precomputed_token = None
            precomputed_token_is_eos = None
            fused_control_transfer = bool(
                getattr(self, "_flagos_fused_control_transfer_enabled", False)
                and logits_processor is None
                and not force_direct_completion
            )
            if isinstance(best_candidate, torch.Tensor) and isinstance(accept_length, torch.Tensor):
                if fused_control_transfer:
                    precomputed_token = torch.argmax(sample_p) + int(
                        getattr(self, "_flagos_action_logit_offset", 0)
                    )
                    if control_buffer is not None:
                        control_values = control_buffer.read_bundle(
                            best_candidate, accept_length, precomputed_token
                        )
                    else:
                        control_values = torch.stack(
                            (
                                best_candidate.to(torch.long),
                                accept_length.to(torch.long),
                                precomputed_token.to(torch.long),
                            )
                        ).detach().to("cpu").tolist()
                    best_idx, accept_len_int = int(control_values[0]), int(control_values[1])
                    precomputed_token_is_eos = (
                        int(control_values[2]) == self.tokenizer.eos_token_id
                    )
                    self._flagos_fused_control_transfer_hits = int(
                        getattr(self, "_flagos_fused_control_transfer_hits", 0)
                    ) + 1
                    fused_control_transfers += 1
                else:
                    if control_buffer is not None:
                        control_pair = control_buffer.read_pair(best_candidate, accept_length)
                    else:
                        control_pair = torch.stack(
                            (best_candidate.to(torch.long), accept_length.to(torch.long))
                        ).detach().to("cpu").tolist()
                    best_idx, accept_len_int = (int(control_pair[0]), int(control_pair[1]))
            else:
                best_idx, accept_len_int = int(best_candidate), int(accept_length)
            accept_lengths.append(accept_len_int)
            best_candidate_indices.append(best_idx)
            candidate_shapes.append([int(size) for size in candidates.shape])
            if self._collect_errors:
                generated_tokens_so_far = max(0, int(input_ids.shape[1]) - (int(input_len) + 1))
                self._record_candidate_mismatch(
                    logits,
                    candidates,
                    best_idx,
                    accept_len_int,
                    accept_threshold,
                    generated_tokens_so_far,
                )
            if force_direct_completion:
                kalman_start = phase_start("kalman_completion")
                with torch.profiler.record_function("KERV::KalmanCompletion"):
                    input_ids, tokens_added = self._finalize_action_with_kf(
                        input_ids,
                        input_len,
                        candidates,
                        best_idx,
                        accept_len_int,
                    )
                if runtime_stats_enabled:
                    phase_latency_s["kalman_completion"].append(phase_end(kalman_start))
                new_token += tokens_added
                kalman_tokens_added += int(tokens_added)
                break
            update_start = phase_start("commit_and_next_draft")
            if (
                not fused_control_transfer
                and (input_ids.shape[1] - input_len - 1 + accept_len_int) <= 6
            ):
                next_token_control_transfers += 1
            with torch.profiler.record_function("KERV::KVCommitNextDraft"):
                if bool(
                    getattr(self, "_flagos_bucket_first_verify_only", False)
                ):
                    self._flagos_tree_use_node_buckets = False
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token,prompt_embeds,past_key_value_data,attention_mask, should_stop = update_inference_inputs(
                    prompt_embeds,
                    #prompt_hidden_states,
                    input_ids,
                    input_len,
                    candidates,
                    best_idx,
                    accept_len_int,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_kv_data_new,
                    #current_length_data,
                    self,
                    hidden_state_new,
                    #hidden_embedding_new,
                    sample_p,
                    attention_mask,
                    precomputed_token=precomputed_token,
                    precomputed_token_is_eos=precomputed_token_is_eos,
                    return_stop=True,
                )
            if runtime_stats_enabled:
                phase_latency_s["commit_and_next_draft"].append(phase_end(update_start))
            if should_stop:
                ended_with_stop_token = True
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        #print('end loop')
        #print('check stop tokens')
        if ended_with_stop_token:
            input_ids = input_ids[:, :-1]

        action_dim = int(self.get_action_dim())
        generated_tokens_cpu: List[int] = []
        filtered_tokens: List[int] = []
        if self._collect_errors or self._store_history_flag or runtime_stats_enabled:
            generated_tokens_cpu = [
                int(token)
                for token in input_ids[0, input_len + 1 :].detach().to("cpu").tolist()
            ]
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            for token in generated_tokens_cpu:
                if token < 0:
                    continue
                if eos_id is not None and token == eos_id:
                    break
                if pad_id is not None and token == pad_id:
                    break
                filtered_tokens.append(token)
        # The direct Kalman completion appends ``start_token + action_dim``
        # tokens.  Historically ``predict_action`` decodes the *last*
        # ``action_dim`` tokens, while the Kalman history stores the first
        # ``action_dim`` generated tokens.  Keep those two views separate so
        # reusing this single device-to-host copy does not change the rollout.
        history_token_ids = filtered_tokens[:action_dim]
        action_token_ids = filtered_tokens[-action_dim:] if len(filtered_tokens) >= action_dim else []
        self._last_action_token_ids_cpu = action_token_ids

        # Record the accepted action tokens once per step (after trimming stop tokens)
        if self._collect_errors and hasattr(self, "_accepted_actions"):
            try:
                action_dim = int(self.get_action_dim())
            except Exception:
                action_dim = None
            if action_dim is not None:
                try:
                    valid_tokens = filtered_tokens
                    if len(valid_tokens) >= action_dim:
                        raw_tokens = valid_tokens[:action_dim]
                        discretized = self.vocab_size - np.array(raw_tokens, dtype=np.int64)
                        discretized = np.clip(
                            discretized - 1,
                            a_min=0,
                            a_max=self.bin_centers.shape[0] - 1,
                        ).tolist()
                        action_idx = (
                            self._current_action_index
                            if self._current_action_index is not None
                            else -1
                        )
                        self._accepted_actions[self._current_episode_key][action_idx].append(
                            {"raw": raw_tokens, "discretized": discretized}
                        )
                        # try:
                        #     pre_act = self.kelman_predict_tokens()
                        # except Exception:
                        #     pre_act = []
                        # try:
                        #     acts_win = self.get_history_with_win_and_EOF_id(1, self._kalman_history_window)
                        # except Exception:
                        #     acts_win = []
                        # try:
                        #     acts = self.get_history()
                        # except Exception:
                        #     acts = []
                        # with open('./log_action.txt', 'a') as f:
                        #     # f.write(f'[* pre_act]{pre_act}\n')
                        #     f.write(f'{self._current_episode_key}')
                        #     f.write(f'[* acts_win]{acts_win}\n')
                        #     f.write(f'[* acts]{acts}\n')
                        #     f.write(f'[* KF_pre]{pre_act}\n')


                except Exception:
                    pass

        if self._store_history_flag:
            try:
                action_dim_hist = int(self.get_action_dim())
            except Exception:
                action_dim_hist = None
            if action_dim_hist is not None:
                if len(filtered_tokens) >= action_dim_hist:
                    accepted_slice = history_token_ids[:action_dim_hist]
                    self._token_history.append(accepted_slice)
                    window = self._kalman_history_window
                    if window is not None and window > 0 and len(self._token_history) > window:
                        self._token_history = self._token_history[-window:]

        if runtime_stats_enabled:
            resolved_phase_latency = phase_timer.resolve()
            if resolved_phase_latency.get("prompt_and_initial_draft"):
                phase_latency_s["prompt_and_initial_draft"] = resolved_phase_latency[
                    "prompt_and_initial_draft"
                ][-1]
            for phase_name in (
                "verify",
                "posterior_and_accept",
                "commit_and_next_draft",
                "kalman_completion",
            ):
                values = resolved_phase_latency.get(phase_name)
                if values:
                    phase_latency_s[phase_name] = values
            runtime_latency = time.perf_counter() - runtime_start
            graph_capture_overhead = max(
                0.0,
                (
                    float(getattr(self, "_flagos_cuda_graph_capture_overhead_s", 0.0))
                    - graph_capture_overhead_start
                )
                + (
                    float(
                        getattr(self, "_flagos_joint_graph_capture_overhead_s", 0.0)
                    )
                    - joint_graph_capture_overhead_start
                ),
            )
            # ``auto`` treats static-tree attention as a measured candidate,
            # not an unconditional default.  Collect enough real Verify
            # replays to suppress one-off noise, then keep it only when the
            # configured 25.5 ms gate is met.  The explicit ``triton`` mode
            # remains available for cross-chip experiments.
            if (
                str(getattr(self, "_flagos_static_tree_attention_mode", "off"))
                == "auto"
                and not bool(
                    getattr(
                        self,
                        "_flagos_static_tree_attention_gate_evaluated",
                        False,
                    )
                )
            ):
                samples = getattr(
                    self, "_flagos_static_tree_attention_gate_samples_ms", []
                )
                samples.extend(
                    float(value) * 1000.0
                    for value, mode in zip(
                        phase_latency_s["verify_replay_net"],
                        phase_latency_s["verify_graph_mode"],
                    )
                    if mode == "replay"
                )
                self._flagos_static_tree_attention_gate_samples_ms = samples
                if len(samples) >= 24:
                    ordered = sorted(samples)
                    midpoint = len(ordered) // 2
                    median_ms = (
                        ordered[midpoint]
                        if len(ordered) % 2
                        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
                    )
                    threshold_ms = float(
                        getattr(
                            self,
                            "_flagos_static_tree_attention_gate_threshold_ms",
                            25.5,
                        )
                    )
                    enabled_after_gate = median_ms <= threshold_ms
                    self.base_model.language_model._flagos_static_tree_attention_runtime_enabled = enabled_after_gate
                    self._flagos_static_tree_attention_gate_evaluated = True
                    self._flagos_static_tree_attention_gate_median_ms = median_ms
                    if not enabled_after_gate:
                        # Drop the single candidate graph before capturing the
                        # native route.  Capture counters remain cumulative and
                        # therefore auditable.
                        self._flagos_verifier_cuda_graph_cache.clear()
                    print(
                        "[FlagOS/KERV] static-tree attention auto gate: "
                        f"median={median_ms:.3f}ms threshold={threshold_ms:.3f}ms "
                        f"selected={enabled_after_gate}",
                        flush=True,
                    )
            generated_token_count = max(0, int(input_ids.shape[1]) - (int(input_len) + 1))
            graph_warmup_counters = getattr(
                self, "_flagos_graph_warmup_counters", {}
            )

            def formal_graph_count(stage: str, counter: str) -> int:
                total = int(
                    getattr(self, f"_flagos_{stage}_graph_{counter}", 0)
                )
                warmup = int(
                    graph_warmup_counters.get(f"{stage}_{counter}", 0)
                )
                return max(0, total - warmup)

            runtime_record = {
                "episode": self._current_episode_key,
                "action_index": self._current_action_index,
                "step_index": self._step_idx,
                "accept_threshold": None if accept_threshold is None else float(accept_threshold),
                "cycle_count": len(accept_lengths),
                "accept_lengths": accept_lengths,
                "mean_accept_length": (
                    float(sum(accept_lengths)) / len(accept_lengths) if accept_lengths else 0.0
                ),
                "best_candidate_indices": best_candidate_indices,
                "candidate_shapes": candidate_shapes,
                "forced_kalman": bool(force_direct_completion),
                "kalman_tokens_added": kalman_tokens_added,
                "generated_token_count": generated_token_count,
                "action_token_ids": action_token_ids,
                "kalman_history_token_ids": history_token_ids,
                "draft_tree_tokens": int(self.ea_layer.total_tokens),
                "generation_latency_s": runtime_latency,
                "graph_capture_overhead_s": graph_capture_overhead,
                "steady_replay_latency_s": max(
                    0.0, runtime_latency - graph_capture_overhead
                ),
                "system_optimization": {
                    "enabled": bool(
                        getattr(self, "_flagos_system_optimization_enabled", False)
                    ),
                    "phase6_a100_fast_path": bool(
                        getattr(self, "_flagos_phase6_a100_fast_path", False)
                    ),
                    "timing_backend": str(
                        getattr(self, "_flagos_timing_backend", "cpu")
                    ),
                    "decode_workspace": bool(
                        getattr(self, "_flagos_decode_workspace", None) is not None
                    ),
                    "decode_workspace_hits": int(
                        getattr(self, "_flagos_decode_workspace_hits", 0)
                    ),
                    "workspace_append_hits": int(
                        getattr(self, "_flagos_decode_workspace_append_hits", 0)
                    ),
                    "workspace_token_hits": int(
                        getattr(self, "_flagos_decode_workspace_token_hits", 0)
                    ),
                    "candidate_padding_hits": int(
                        getattr(self, "_flagos_candidate_padding_hits", 0)
                    ),
                    "control_buffer_transfers": int(
                        getattr(
                            getattr(self, "_flagos_control_buffer", None),
                            "transfers",
                            0,
                        )
                    ),
                    "prompt_graph_mode": str(
                        getattr(self, "_flagos_last_prompt_graph_mode", "disabled")
                    ),
                    "draft_graph_mode": str(
                        getattr(self, "_flagos_last_draft_graph_mode", "disabled")
                    ),
                    "stage_graph_captures": int(
                        getattr(self, "_flagos_stage_graph_captures", 0)
                    ),
                    "stage_graph_fallbacks": int(
                        getattr(self, "_flagos_stage_graph_fallbacks", 0)
                    ),
                    "stage_graph_capture_overhead_s": float(
                        getattr(self, "_flagos_stage_graph_capture_overhead_s", 0.0)
                    ),
                    "prompt_graph_captures": int(
                        getattr(self, "_flagos_prompt_graph_captures", 0)
                    ),
                    "prompt_graph_replays": int(
                        getattr(self, "_flagos_prompt_graph_replays", 0)
                    ),
                    "prompt_graph_fallbacks": int(
                        getattr(self, "_flagos_prompt_graph_fallbacks", 0)
                    ),
                    "draft_graph_captures": int(
                        getattr(self, "_flagos_draft_graph_captures", 0)
                    ),
                    "draft_graph_replays": int(
                        getattr(self, "_flagos_draft_graph_replays", 0)
                    ),
                    "draft_graph_fallbacks": int(
                        getattr(self, "_flagos_draft_graph_fallbacks", 0)
                    ),
                    "prompt_graph_formal_captures": formal_graph_count(
                        "prompt", "captures"
                    ),
                    "prompt_graph_formal_replays": formal_graph_count(
                        "prompt", "replays"
                    ),
                    "prompt_graph_formal_fallbacks": formal_graph_count(
                        "prompt", "fallbacks"
                    ),
                    "draft_graph_formal_captures": formal_graph_count(
                        "draft", "captures"
                    ),
                    "draft_graph_formal_replays": formal_graph_count(
                        "draft", "replays"
                    ),
                    "draft_graph_formal_fallbacks": formal_graph_count(
                        "draft", "fallbacks"
                    ),
                    "stage_graph_disable_reason": getattr(
                        self, "_flagos_stage_graph_disable_reason", None
                    ),
                    "draft_workspace_enabled": bool(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_persistent_draft_workspace",
                            False,
                        )
                    ),
                    "draft_workspace_hits": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_draft_workspace_hits",
                            0,
                        )
                    ),
                    "draft_workspace_allocations": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_draft_workspace_allocations",
                            0,
                        )
                    ),
                    "prompt_token_cache_hits": int(
                        getattr(self, "_flagos_prompt_token_cache_hits", 0)
                    ),
                    "pinned_image_buffer_hits": int(
                        getattr(self, "_flagos_pinned_image_buffer_hits", 0)
                    ),
                    "persistent_draft_kv_cache": bool(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_persistent_draft_kv_cache",
                            False,
                        )
                    ),
                    "persistent_draft_kv_allocations": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_persistent_draft_kv_allocations",
                            0,
                        )
                    ),
                    "persistent_draft_kv_fallbacks": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_persistent_draft_kv_fallbacks",
                            0,
                        )
                    ),
                    "commit_draft_overlap_hits": int(
                        getattr(self, "_flagos_commit_draft_overlap_hits", 0)
                    ),
                    "commit_draft_stream_overlap_mode": str(
                        getattr(
                            self,
                            "_flagos_commit_draft_stream_overlap_mode",
                            "off",
                        )
                    ),
                    "commit_draft_cuda_graph_mode": str(
                        getattr(self, "_flagos_commit_draft_cuda_graph_mode", "off")
                    ),
                    "commit_draft_selected_backend": str(
                        getattr(
                            self,
                            "_flagos_commit_draft_selected_backend",
                            "serial",
                        )
                    ),
                    "joint_graph_mode": str(
                        getattr(self, "_flagos_joint_graph_last_mode", "disabled")
                    ),
                    "joint_graph_captures": int(
                        getattr(self, "_flagos_joint_graph_captures", 0)
                    ),
                    "joint_graph_replays": int(
                        getattr(self, "_flagos_joint_graph_replays", 0)
                    ),
                    "joint_graph_fallbacks": int(
                        getattr(self, "_flagos_joint_graph_fallbacks", 0)
                    ),
                    "joint_graph_capture_overhead_s": float(
                        getattr(self, "_flagos_joint_graph_capture_overhead_s", 0.0)
                    ),
                    "joint_graph_disable_reason": getattr(
                        self, "_flagos_joint_graph_disable_reason", None
                    ),
                    "episode_device_warmup_mode": str(
                        getattr(self, "_flagos_episode_device_warmup_mode", "off")
                    ),
                    "episode_device_warmup_scheduled": int(
                        getattr(self, "_flagos_episode_warmup_scheduled", 0)
                    ),
                    "episode_device_warmup_waits": int(
                        getattr(self, "_flagos_episode_warmup_waits", 0)
                    ),
                    "episode_device_warmup_duration_ms": float(
                        getattr(self, "_flagos_episode_warmup_duration_ms_value", 0.0)
                    ),
                    "episode_device_warmup_affinity": getattr(
                        self, "_flagos_numa_affinity", None
                    ),
                    "fixed_draft_input_workspace": bool(
                        getattr(self, "_flagos_fixed_draft_input_workspace", False)
                    ),
                    "allocator_backend": str(
                        getattr(self, "_flagos_allocator_backend", "auto")
                    ),
                    "task_graph_warmups": int(
                        getattr(self, "_flagos_task_graph_warmups", 0)
                    ),
                },
                "phase_latency_s": phase_latency_s,
                "host_transfer_points": {
                    "posterior_control_pair": max(
                        0, len(accept_lengths) - fused_control_transfers
                    ),
                    "posterior_control_bundle": fused_control_transfers,
                    "next_token_scalar": next_token_control_transfers,
                    "kalman_candidate_row": int(force_direct_completion),
                    "final_action_tokens": 1,
                    "mismatch_logits": len(accept_lengths) if self._collect_errors else 0,
                },
                "action_head": {
                    "enabled": bool(getattr(self, "_flagos_action_head_enabled", False)),
                    "full_vocab_size": int(
                        self.vocab_size
                    ),
                    "action_vocab_size": int(
                        getattr(self, "_flagos_action_vocab_size", self.vocab_size)
                    ),
                    "token_offset": int(
                        getattr(self, "_flagos_action_logit_offset", 0)
                    ),
                    "cumulative_hits": int(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_action_head_hit_count",
                            0,
                        )
                    ),
                },
                "w8a16": dict(
                    getattr(self, "_flagos_w8a16_manifest", {})
                ),
                "tree_builder": {
                    "enabled": bool(
                        getattr(self, "_flagos_tree_builder_enabled", False)
                    ),
                    "cumulative_hits": int(
                        getattr(self, "_flagos_tree_builder_hit_count", 0)
                    ),
                    "static_runtime": bool(
                        getattr(self, "_flagos_static_tree_runtime_enabled", False)
                    ),
                    "verify_max_paths": int(
                        getattr(self, "_flagos_tree_verify_max_paths", 0)
                    ),
                    "max_depth": int(
                        getattr(self, "_flagos_tree_max_depth", 0)
                    ),
                    "compact_tree_mode": str(
                        getattr(self, "_flagos_compact_tree_mode", "off")
                    ),
                    "compact_tree_active": bool(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_active",
                            getattr(self, "_flagos_compact_tree_active", False),
                        )
                    ),
                    "compact_tree_confidence": getattr(
                        getattr(self, "ea_layer", None),
                        "_flagos_compact_tree_confidence",
                        getattr(self, "_flagos_compact_tree_confidence", None),
                    ),
                    "compact_tree_level": str(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_level",
                            getattr(self, "_flagos_compact_tree_level", "full"),
                        )
                    ),
                    "compact_tree_gate_evaluations": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_gate_evaluations",
                            0,
                        )
                    ),
                    "compact_tree_full_selections": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_full_selections",
                            0,
                        )
                    ),
                    "compact_tree_medium_selections": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_medium_selections",
                            0,
                        )
                    ),
                    "compact_tree_tight_selections": int(
                        getattr(
                            getattr(self, "ea_layer", None),
                            "_flagos_compact_tree_tight_selections",
                            0,
                        )
                    ),
                    "static_hits": int(
                        getattr(self, "_flagos_static_tree_hit_count", 0)
                    ),
                    "cache_hits": int(
                        getattr(self, "_flagos_tree_cache_hits", 0)
                    ),
                    "cache_misses": int(
                        getattr(self, "_flagos_tree_cache_misses", 0)
                    ),
                    "operatorized": bool(
                        getattr(self, "_flagos_tree_operator_enabled", False)
                    ),
                    "operator_build_hits": int(
                        getattr(self, "_flagos_tree_operator_build_hits", 0)
                    ),
                    "operator_pack_hits": int(
                        getattr(self, "_flagos_tree_operator_pack_hits", 0)
                    ),
                    "draft_tree_operator_hits": int(
                        getattr(
                            self.ea_layer,
                            "_flagos_draft_tree_operator_hits",
                            0,
                        )
                    ),
                    "operator_build_time_ms": float(
                        getattr(self, "_flagos_tree_operator_build_time_s", 0.0)
                    ) * 1000.0,
                    "verify_accept_operatorized": bool(
                        getattr(self, "_flagos_verify_accept_operator_enabled", False)
                    ),
                    "verify_accept_operator_hits": int(
                        getattr(self, "_flagos_verify_accept_operator_hits", 0)
                    ),
                    "embodied_verify_accept_hits": int(
                        getattr(self, "_flagos_embodied_verify_accept_hits", 0)
                    ),
                    "embodied_tree_pack_hits": int(
                        getattr(self, "_flagos_embodied_tree_pack_hits", 0)
                    ),
                    "tree_embed_pack_hits": int(
                        getattr(self, "_flagos_tree_embed_pack_hits", 0)
                    ),
                    "tree_embed_resident_hits": int(
                        getattr(self, "_flagos_tree_embed_resident_hits", 0)
                    ),
                    "embodied_static_tree_attention_hits": int(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_static_tree_attention_hits",
                            0,
                        )
                    ),
                    "static_tree_attention_mode": str(
                        getattr(self, "_flagos_static_tree_attention_mode", "off")
                    ),
                    "static_tree_attention_gate_evaluated": bool(
                        getattr(
                            self,
                            "_flagos_static_tree_attention_gate_evaluated",
                            False,
                        )
                    ),
                    "static_tree_attention_gate_median_ms": getattr(
                        self,
                        "_flagos_static_tree_attention_gate_median_ms",
                        None,
                    ),
                    "static_tree_attention_selected": bool(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_static_tree_attention_runtime_enabled",
                            False,
                        )
                    ),
                    "embodied_ops_enabled": bool(
                        getattr(self, "_flagos_embodied_ops_enabled", False)
                    ),
                    "verify_accept_aten_hits": int(
                        getattr(self, "_flagos_verify_accept_aten_hits", 0)
                    ),
                    "verify_accept_selected_backend": getattr(
                        self, "_flagos_verify_accept_selected_backend", None
                    ),
                    "verify_accept_native_ms": getattr(
                        self, "_flagos_verify_accept_native_ms", None
                    ),
                    "verify_accept_triton_ms": getattr(
                        self, "_flagos_verify_accept_triton_ms", None
                    ),
                    "fused_control_transfer": bool(
                        getattr(self, "_flagos_fused_control_transfer_enabled", False)
                    ),
                    "fused_control_transfer_hits": int(
                        getattr(self, "_flagos_fused_control_transfer_hits", 0)
                    ),
                    "node_buckets": list(
                        getattr(self, "_flagos_tree_node_buckets", ())
                    ),
                    "compile_verifier": bool(
                        getattr(self, "_flagos_compile_verifier_enabled", False)
                    ),
                    "compile_verifier_mode": str(
                        getattr(self, "_flagos_compile_verifier_mode", "off")
                    ),
                    "compiled_verifier_hits": int(
                        getattr(self, "_flagos_compiled_verifier_hits", 0)
                    ),
                    "inductor_verifier_graph_captures": int(
                        getattr(self, "_flagos_inductor_verifier_graph_captures", 0)
                    ),
                    "inductor_verifier_fallbacks": int(
                        getattr(self, "_flagos_inductor_verifier_fallbacks", 0)
                    ),
                    "inductor_verifier_last_error": getattr(
                        self, "_flagos_inductor_verifier_last_error", None
                    ),
                    "cuda_graph_captures": int(
                        getattr(self, "_flagos_cuda_graph_captures", 0)
                    ),
                    "cuda_graph_replay_hits": int(
                        getattr(self, "_flagos_cuda_graph_replay_hits", 0)
                    ),
                    "cuda_graph_fallbacks": int(
                        getattr(self, "_flagos_cuda_graph_fallbacks", 0)
                    ),
                    "cuda_graph_audit_eager": int(
                        getattr(self, "_flagos_cuda_graph_audit_eager", 0)
                    ),
                    "cuda_graph_formal_captures": max(
                        0,
                        int(getattr(self, "_flagos_cuda_graph_captures", 0))
                        - int(
                            graph_warmup_counters.get("verifier_captures", 0)
                        ),
                    ),
                    "cuda_graph_formal_replays": max(
                        0,
                        int(getattr(self, "_flagos_cuda_graph_replay_hits", 0))
                        - int(graph_warmup_counters.get("verifier_replays", 0)),
                    ),
                    "cuda_graph_formal_fallbacks": max(
                        0,
                        int(getattr(self, "_flagos_cuda_graph_fallbacks", 0))
                        - int(
                            graph_warmup_counters.get("verifier_fallbacks", 0)
                        ),
                    ),
                    "cuda_graph_formal_audit_eager": max(
                        0,
                        int(getattr(self, "_flagos_cuda_graph_audit_eager", 0))
                        - int(
                            graph_warmup_counters.get(
                                "verifier_audit_eager", 0
                            )
                        ),
                    ),
                    "cuda_graph_input_alias_hits": int(
                        getattr(self, "_flagos_cuda_graph_input_alias_hits", 0)
                    ),
                    "cuda_graph_capture_overhead_s": float(
                        getattr(self, "_flagos_cuda_graph_capture_overhead_s", 0.0)
                    ),
                    "kalman_batch_enabled": bool(
                        getattr(self, "_flagos_kalman_batch_enabled", False)
                    ),
                    "kalman_batch_hits": int(
                        getattr(self, "_flagos_kalman_batch_hits", 0)
                    ),
                    "kalman_batch_chains": int(
                        getattr(self, "_flagos_kalman_batch_chains", 0)
                    ),
                    "kv_commit_batched": bool(
                        getattr(self, "_flagos_kv_commit_batched_enabled", False)
                    ),
                    "kv_commit_batched_hits": int(
                        getattr(self, "_flagos_kv_commit_batched_hits", 0)
                    ),
                    "persistent_runtime": bool(
                        getattr(self, "_flagos_persistent_runtime_enabled", False)
                    ),
                    "resident_kv_cache": bool(
                        getattr(self, "_flagos_resident_kv_cache_enabled", False)
                    ),
                    "persistent_allocations": int(
                        getattr(self, "_flagos_persistent_allocations", 0)
                    ),
                    "persistent_kv_commit_hits": int(
                        getattr(self, "_flagos_persistent_kv_commit_hits", 0)
                    ),
                    "phase2_kv_accept_commit_hits": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "phase2_commit_hits",
                            0,
                        )
                    ),
                    "persistent_prefix_capacity": int(
                        getattr(self, "_flagos_persistent_prefix_capacity", 0)
                    ),
                    "persistent_tree_capacity": int(
                        getattr(self, "_flagos_persistent_tree_capacity", 0)
                    ),
                    "exact_node_templates": bool(
                        getattr(self, "_flagos_exact_node_templates", False)
                    ),
                    "persistent_prefix_length": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "prefix_length",
                            0,
                        )
                    ),
                    "resident_mask_allocations": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "mask_buffer_allocations",
                            0,
                        )
                    ),
                    "resident_mask_reuses": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "mask_buffer_reuses",
                            0,
                        )
                    ),
                    "resident_mask_materializations": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "mask_materializations",
                            0,
                        )
                    ),
                    "resident_mask_materialization_skips": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "mask_materialization_skips",
                            0,
                        )
                    ),
                    "rotary_key_direct_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "rotary_key_direct_writes",
                            0,
                        )
                    ),
                    "key_copy_fallbacks": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "key_copy_fallbacks",
                            0,
                        )
                    ),
                    "value_copy_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "value_copy_writes",
                            0,
                        )
                    ),
                    "rotary_value_direct_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "rotary_value_direct_writes",
                            0,
                        )
                    ),
                    "tree_value_copy_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "tree_value_copy_writes",
                            0,
                        )
                    ),
                    "prompt_value_copy_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "prompt_value_copy_writes",
                            0,
                        )
                    ),
                    "tree_value_direct_writes": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "tree_value_direct_writes",
                            0,
                        )
                    ),
                    "fixed_workspace_layout": bool(
                        getattr(self, "_flagos_fixed_workspace_layout", False)
                    ),
                    "commit_scratch_hits": int(
                        getattr(
                            getattr(self, "_flagos_persistent_kv_cache", None),
                            "commit_scratch_hits",
                            0,
                        )
                    ),
                    "last_shape": dict(
                        getattr(self, "_flagos_static_tree_last_stats", {})
                    ),
                },
                "rope_cache": {
                    "enabled": bool(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_rope_cache_installed",
                            False,
                        )
                    ),
                    "cumulative_hits": int(
                        getattr(
                            getattr(
                                self.base_model.language_model,
                                "_flagos_rope_cache",
                                None,
                            ),
                            "hits",
                            0,
                        )
                    ),
                    "cumulative_misses": int(
                        getattr(
                            getattr(
                                self.base_model.language_model,
                                "_flagos_rope_cache",
                                None,
                            ),
                            "misses",
                            0,
                        )
                    ),
                },
                "tree_attention_mask": {
                    "enabled": bool(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_tree_attention_mask_installed",
                            False,
                        )
                    ),
                    "cumulative_hits": int(
                        getattr(
                            self.base_model.language_model,
                            "_flagos_tree_attention_mask_hits",
                            0,
                        )
                    ),
                },
            }
            self._record_flagos_runtime_stats(runtime_record)
            print(
                f"[FlagOS/KERV] step={self._step_idx} cycles={len(accept_lengths)} "
                f"accept={accept_lengths} kalman={force_direct_completion} "
                f"latency_s={runtime_latency:.6f}",
                flush=True,
            )

        if not log:
            return input_ids[:,input_len+1:]
        else:
            return input_ids, new_token, idx
    def eval_topk(self,input_ids, logits_processor=None,**kwargs):
        #token = torch.tensor(token).to(input_ids.device)
        temperature=0.0
        top_p=0.0
        top_k=0.0
        self.tree_mask=None
        self.base_model.language_model.tree_mask=None
        #input_len = input_ids.shape[1]-1
        max_length=2048
        logits_processor = None
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        #self.ea_layer.reset_kv()

        # Initialize the past key and value states
        tokenizer = self.get_tokenizer()
        max_steps = 6
        #model = self
        #print('base model')
        model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)
        #这里直接用ea_forward那最后一个位置的hidden state
        #print(kwargs)
        kwargs['return_hidden_states']=True
        #print(kwargs)
        #exit()
        action,tokens,hidden = self.predict_action(
               input_ids, logits_processor=None,**kwargs
            )
        token = torch.tensor(tokens).to(input_ids.device)
        input_embeds = hidden[0]
        hidden_states = hidden[1]
        hidden_states = torch.cat([item for item in hidden[1]],dim=0).to(input_ids.device)
        input_embeds = torch.cat([item for item in hidden[0]],dim=0).to(input_ids.device)
        #print(input_embeds.device)
        input_token_embeds = self.ea_layer.embed_tokens(torch.tensor([2]).to(input_ids.device))
        ea_layer_input_embeds = torch.cat((input_embeds,input_token_embeds),dim=0)
        self.ea_layer._eval_top_k(hidden_states,token, ea_layer_input_embeds,self.base_model.language_model.lm_head, 0,logits_processor)
        return action,None,None

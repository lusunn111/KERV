"""GPU-resident KV workspace for template-static KERV verification.

One allocation contains the committed prefix and a temporary tree workspace.
CUDA Graphs are keyed by exact prefix and bounded tree lengths, preserving the
original attention extent while keeping every template address stable.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import Cache


class KervPersistentTreeCache(Cache):
    """Persistent prefix KV plus a fixed-size temporary tree workspace."""

    _flagos_persistent_tree_cache = True

    def __init__(
        self,
        config,
        *,
        prefix_capacity: int,
        tree_capacity: int,
        device: torch.device,
        dtype: torch.dtype,
        fixed_workspace_layout: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(config.num_hidden_layers)
        self.num_key_value_heads = int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        )
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.prefix_capacity = int(prefix_capacity)
        self.tree_capacity = int(tree_capacity)
        self.fixed_workspace_layout = bool(fixed_workspace_layout)
        if self.prefix_capacity <= 0 or self.tree_capacity <= 0:
            raise ValueError("persistent cache capacities must be positive")

        # [layer, K/V, batch, kv_head, prefix+tree, head_dim]
        self.storage = torch.empty(
            (
                self.num_layers,
                2,
                1,
                self.num_key_value_heads,
                self.prefix_capacity + self.tree_capacity,
                self.head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        self.prefix_length = 0
        self.tree_length = 0
        self.tree_mode = False
        self.reset_count = 0
        self.tree_forward_count = 0
        self.commit_count = 0
        self._pending_write_start = 0
        self._pending_write_end = 0
        self._mask_buffers = {}
        self.mask_buffer_allocations = 0
        self.mask_buffer_reuses = 0
        self.mask_materializations = 0
        self.mask_materialization_skips = 0
        self.rotary_key_direct_writes = 0
        self.key_copy_fallbacks = 0
        self.value_copy_writes = 0
        self.rotary_value_direct_writes = 0
        self.tree_value_copy_writes = 0
        self.prompt_value_copy_writes = 0
        self.tree_value_direct_writes = 0
        self.prompt_value_direct_writes = 0
        self._direct_value_views: dict[int, tuple[int, int]] = {}
        self._ancestor_buffers: dict[int, torch.Tensor] = {}
        self._ancestor_score_buffers: dict[int, torch.Tensor] = {}
        self._ancestor_topk_index_buffers: dict[int, torch.Tensor] = {}
        self._ancestor_column_buffers: dict[int, torch.Tensor] = {}
        self.current_ancestor_indices: Optional[torch.Tensor] = None
        self.prefix_length_tensor = torch.zeros(
            (), device=device, dtype=torch.int32
        )
        self.empty_position_ids = torch.empty(
            (0,), device=device, dtype=torch.long
        )
        self._commit_scratch = torch.empty(
            (
                self.num_layers,
                2,
                1,
                self.num_key_value_heads,
                8,
                self.head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        self.commit_scratch_hits = 0

        # Compile every action-length specialization before timed generation.
        # The persistent cache is created on the first action, which is the
        # documented warmup action and excluded from steady measurements.
        if self.storage.is_cuda:
            from .flagos_persistent_ops import commit_tree_rows_

            for accepted in range(1, 8):
                warm_indices = torch.arange(
                    accepted, device=self.device, dtype=torch.long
                )
                commit_tree_rows_(self.storage, warm_indices, 0)
            torch.cuda.synchronize(self.device)

    @property
    def device(self) -> torch.device:
        return self.storage.device

    @property
    def dtype(self) -> torch.dtype:
        return self.storage.dtype

    def reset(self) -> None:
        """Start a new action query without reallocating device storage."""
        self.prefix_length = 0
        self.tree_length = 0
        self.tree_mode = False
        self._pending_write_start = 0
        self._pending_write_end = 0
        self._direct_value_views.clear()
        self.current_ancestor_indices = None
        self.prefix_length_tensor.zero_()
        self.reset_count += 1

    def capture_graph_state(self) -> dict[str, int | bool]:
        """Snapshot the Python-side cache metadata after graph capture.

        CUDA Graph replay updates the resident tensors but does not re-run the
        Python bookkeeping in ``update``.  Retaining this tiny state makes a
        prompt prefill graph indistinguishable from the eager cache transition
        to the following verifier stage.
        """

        return {
            "prefix_length": int(self.prefix_length),
            "tree_length": int(self.tree_length),
            "tree_mode": bool(self.tree_mode),
            "pending_write_start": int(self._pending_write_start),
            "pending_write_end": int(self._pending_write_end),
        }

    def restore_graph_state(self, state) -> None:
        if not state:
            return
        self.prefix_length = int(state["prefix_length"])
        self.tree_length = int(state["tree_length"])
        self.tree_mode = bool(state["tree_mode"])
        self._pending_write_start = int(state["pending_write_start"])
        self._pending_write_end = int(state["pending_write_end"])
        # The device-side cursor is read by graph-captured attention and KV
        # writes.  Keep it synchronized with the lightweight host metadata
        # whenever a graph entry is restored.
        self.prefix_length_tensor.fill_(self.prefix_length)

    def prepare_graph_replay(self, state) -> None:
        """Restore metadata before a Commit--Draft graph replay."""

        self.restore_graph_state(state)

    def finish_graph_replay(self, _input_state, output_state) -> None:
        """Restore the post-commit cursor after graph replay."""

        self.restore_graph_state(output_state)

    def begin_tree(self, tree_length: int) -> None:
        tree_length = int(tree_length)
        if tree_length <= 0 or tree_length > self.tree_capacity:
            raise ValueError(
                "Persistent verification tree exceeds the resident workspace: "
                f"capacity={self.tree_capacity}, actual={tree_length}"
            )
        self.tree_length = tree_length
        self.tree_mode = True
        self.tree_forward_count += 1

    def end_tree(self) -> None:
        self.tree_mode = False

    @property
    def tree_workspace_start(self) -> int:
        return self.prefix_capacity if self.fixed_workspace_layout else self.prefix_length

    def _write_span(self, layer_idx: int, token_count: int) -> tuple[int, int]:
        if self.tree_mode:
            start = self.tree_workspace_start
            if token_count != self.tree_length:
                raise ValueError("resident tree write length mismatch")
        elif layer_idx == 0:
            start = int(self.prefix_length)
        else:
            start = int(self._pending_write_start)
        return start, start + int(token_count)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        del layer_idx
        return int(self.prefix_length)

    def get_max_length(self) -> int:
        return int(self.prefix_capacity)

    def rotary_key_destination(
        self, layer_idx: int, token_count: int
    ) -> torch.Tensor:
        """Return the exact K slot used by the following Cache.update call."""

        layer_idx = int(layer_idx)
        token_count = int(token_count)
        start, end = self._write_span(layer_idx, token_count)
        capacity = self.prefix_capacity + self.tree_capacity
        if end > capacity:
            raise RuntimeError("resident rotary K write exceeds cache capacity")
        return self.storage[layer_idx, 0, ..., start:end, :]

    def rotary_value_destination(
        self, layer_idx: int, token_count: int
    ) -> torch.Tensor:
        """Return the fixed V slot paired with ``rotary_key_destination``."""

        layer_idx = int(layer_idx)
        start, end = self._write_span(layer_idx, int(token_count))
        if end > self.prefix_capacity + self.tree_capacity:
            raise RuntimeError("resident rotary V write exceeds cache capacity")
        return self.storage[layer_idx, 1, ..., start:end, :]

    def mark_rotary_value_written(
        self, layer_idx: int, token_count: int
    ) -> None:
        """Mark a V span written by the fused RoPE/KV operator."""

        start, end = self._write_span(int(layer_idx), int(token_count))
        self._direct_value_views[int(layer_idx)] = (start, end)

    @staticmethod
    def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
        return bool(
            left.data_ptr() == right.data_ptr()
            and left.shape == right.shape
            and left.stride() == right.stride()
            and left.dtype == right.dtype
            and left.device == right.device
        )

    def _store_layer_kv(
        self,
        layer_idx: int,
        start: int,
        end: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        destination_key = self.storage[layer_idx, 0, ..., start:end, :]
        destination_value = self.storage[layer_idx, 1, ..., start:end, :]
        if self._same_tensor_view(destination_key, key_states):
            self.rotary_key_direct_writes += 1
        else:
            destination_key.copy_(key_states)
            self.key_copy_fallbacks += 1
        direct_value_span = self._direct_value_views.pop(layer_idx, None)
        if (
            self._same_tensor_view(destination_value, value_states)
            or direct_value_span == (start, end)
        ):
            self.rotary_value_direct_writes += 1
            if self.tree_mode:
                self.tree_value_direct_writes += 1
            else:
                self.prompt_value_direct_writes += 1
        else:
            destination_value.copy_(value_states)
            self.value_copy_writes += 1
            if self.tree_mode:
                self.tree_value_copy_writes += 1
            else:
                self.prompt_value_copy_writes += 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"invalid KV layer index: {layer_idx}")
        token_count = int(key_states.shape[-2])

        if self.tree_mode:
            if token_count != self.tree_length:
                raise ValueError(
                    "Tree KV input must match the selected static template: "
                    f"expected={self.tree_length}, actual={token_count}"
                )
            # Within one graph template the prefix length is fixed, so the tree
            # follows it contiguously and preserves the original attention
            # reduction extent (important for boundary acceptance decisions).
            start = self.tree_workspace_start
            end = start + self.tree_length
            self._store_layer_kv(
                layer_idx, start, end, key_states, value_states
            )
            return (
                self.storage[layer_idx, 0, ..., :end, :],
                self.storage[layer_idx, 1, ..., :end, :],
            )

        # KERV prompt/incremental writes are contiguous. Record the span once
        # on layer 0 and reuse it for the remaining 31 layers. This avoids a
        # ``cache_position.max().item()`` device synchronization per layer.
        if layer_idx == 0:
            start = int(self.prefix_length)
            end = start + token_count
            if end > self.prefix_capacity:
                raise RuntimeError(
                    f"KERV prefix cache overflow: {end}>{self.prefix_capacity}"
                )
            self._pending_write_start = start
            self._pending_write_end = end
            self.prefix_length = end
            self.prefix_length_tensor.fill_(end)
        else:
            start = int(self._pending_write_start)
            end = int(self._pending_write_end)
            if end - start != token_count:
                raise RuntimeError("non-contiguous persistent KV layer update")
        self._store_layer_kv(layer_idx, start, end, key_states, value_states)
        return (
            self.storage[layer_idx, 0, ..., :end, :],
            self.storage[layer_idx, 1, ..., :end, :],
        )

    def build_tree_attention_mask(
        self,
        tree_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
        materialize: bool = True,
    ) -> torch.Tensor:
        """Create the additive mask for one fixed-address graph template."""
        if int(tree_mask.shape[-1]) != self.tree_length:
            raise ValueError("tree mask does not match the selected tree template")
        from kerv_flagos.tree_attention_mask import build_tree_causal_mask

        tree_start = self.tree_workspace_start
        key = (
            int(self.tree_length),
            int(tree_start if self.fixed_workspace_layout else self.prefix_length),
            dtype,
        )
        output = self._mask_buffers.get(key)
        if output is None:
            output = torch.empty(
                (1, 1, self.tree_length, tree_start + self.tree_length),
                device=self.device,
                dtype=dtype,
            )
            self._mask_buffers[key] = output
            self.mask_buffer_allocations += 1
        else:
            self.mask_buffer_reuses += 1
        if materialize:
            output = build_tree_causal_mask(
                tree_mask.contiguous(),
                self.prefix_length,
                dtype,
                output=output,
                tree_start=tree_start,
            )
            self.mask_materializations += 1
        else:
            # Static-tree Attention consumes the resident ancestor table and
            # never reads the dense additive mask.  Retain only a stable-shape
            # placeholder required by the HF layer interface.
            self.mask_materialization_skips += 1
        # Build the tree ancestry once per Verify, then keep the fixed-width
        # device buffer resident for all decoder layers and CUDA Graph replay.
        ancestor_buffer = self._ancestor_buffers.get(self.tree_length)
        if ancestor_buffer is None:
            ancestor_buffer = torch.empty(
                (self.tree_length, 8), device=self.device, dtype=torch.long
            )
            self._ancestor_buffers[self.tree_length] = ancestor_buffer
            self._ancestor_score_buffers[self.tree_length] = torch.empty(
                (self.tree_length, self.tree_length),
                device=self.device,
                dtype=torch.long,
            )
            self._ancestor_topk_index_buffers[self.tree_length] = torch.empty(
                (self.tree_length, 8), device=self.device, dtype=torch.long
            )
            self._ancestor_column_buffers[self.tree_length] = torch.arange(
                self.tree_length, device=self.device, dtype=torch.long
            ).expand(self.tree_length, -1).clone()
        local_mask = tree_mask.reshape(self.tree_length, self.tree_length).bool()
        scores = self._ancestor_score_buffers[self.tree_length]
        scores.copy_(self._ancestor_column_buffers[self.tree_length])
        scores.masked_fill_(~local_mask, -1)
        torch.topk(
            scores,
            8,
            dim=-1,
            out=(
                ancestor_buffer,
                self._ancestor_topk_index_buffers[self.tree_length],
            ),
        )
        ancestor_buffer.add_(tree_start)
        ancestor_buffer.masked_fill_(ancestor_buffer < tree_start, -1)
        self.current_ancestor_indices = ancestor_buffer
        return output

    def commit_tree(self, tree_indices: torch.Tensor, previous_length: int) -> "KervPersistentTreeCache":
        """Commit accepted tree nodes into the resident prefix in place."""
        previous_length = int(previous_length)
        if previous_length != self.prefix_length:
            raise RuntimeError(
                "Persistent KV prefix length mismatch: "
                f"cache={self.prefix_length}, caller={previous_length}"
            )
        indices = tree_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        accepted = int(indices.numel())
        end = previous_length + accepted
        if end > self.prefix_capacity:
            raise RuntimeError(f"KERV prefix cache overflow during commit: {end}")

        embodied_include = set(
            getattr(self, "_flagos_embodied_ops_include", set())
        )
        if bool(getattr(self, "_flagos_embodied_ops_enabled", False)) and (
            "kerv_kv_accept_commit" in embodied_include
        ):
            # The phase-two op consumes the same resident [L,2,B,H,T,D]
            # storage as the existing commit kernel.  It takes a clone for
            # overlapping paths in its native fallback, preserving the exact
            # gather-before-write semantics of the original implementation.
            from kerv_flagos.embodied_ops import kerv_kv_accept_commit

            if self.fixed_workspace_layout:
                indices = indices + (self.prefix_capacity - previous_length)
            scratch = self._commit_scratch[..., :accepted, :]
            kerv_kv_accept_commit(
                self.storage, indices, previous_length, scratch=scratch
            )
            self.commit_scratch_hits += 1
            self.prefix_length = end
            self.prefix_length_tensor.fill_(end)
            self.tree_mode = False
            self.commit_count += 1
            self.phase2_commit_hits = int(
                getattr(self, "phase2_commit_hits", 0)
            ) + 1
            return self
        from .flagos_persistent_ops import commit_tree_rows_

        if self.fixed_workspace_layout:
            absolute_indices = indices + self.prefix_capacity
            scratch = self._commit_scratch[..., :accepted, :]
            torch.index_select(
                self.storage,
                -2,
                absolute_indices,
                out=scratch,
            )
            self.storage[..., previous_length:end, :].copy_(scratch)
            self.commit_scratch_hits += 1
        else:
            commit_tree_rows_(self.storage, indices, previous_length)
        self.prefix_length = end
        self.prefix_length_tensor.fill_(end)
        self.tree_mode = False
        self.commit_count += 1
        return self

    def to_legacy_cache(self):
        # LlamaSpecForCausalLM recognizes this marker and keeps the object.
        return self

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, layer_idx: int):
        """Expose the committed prefix through the legacy layer interface.

        Prismatic derives the prompt embedding boundary from the first KV
        layer before entering tree verification.
        """
        layer_idx = int(layer_idx)
        return (
            self.storage[layer_idx, 0, ..., : self.prefix_length, :],
            self.storage[layer_idx, 1, ..., : self.prefix_length, :],
        )

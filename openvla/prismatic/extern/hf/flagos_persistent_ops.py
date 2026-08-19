"""Triton kernels for KERV's GPU-resident static verifier runtime."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _commit_tree_rows_kernel(
    storage,
    tree_indices,
    previous_length,
    row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    accepted: tl.constexpr,
    block: tl.constexpr,
):
    """Gather one accepted path and overwrite its committed prefix in place.

    A program owns one independent ``(layer, K/V, batch, head)`` row. All
    source values are loaded before stores are issued, so overlapping source
    and destination token ranges remain well-defined without a temporary
    full-layer tensor.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    valid = offsets < accepted * head_dim
    accepted_index = offsets // head_dim
    feature_index = offsets - accepted_index * head_dim
    source_node = tl.load(tree_indices + accepted_index, mask=valid, other=0)
    base = row * row_stride
    source = base + (previous_length + source_node) * head_dim + feature_index
    values = tl.load(storage + source, mask=valid)
    destination = base + (previous_length + accepted_index) * head_dim + feature_index
    tl.store(storage + destination, values, mask=valid)


def commit_tree_rows_(
    storage: torch.Tensor,
    tree_indices: torch.Tensor,
    previous_length: int,
) -> None:
    """Commit selected tree K/V rows directly into a contiguous resident cache."""
    if not storage.is_cuda or not tree_indices.is_cuda:
        raise ValueError("persistent tree commit requires CUDA tensors")
    if not storage.is_contiguous():
        raise ValueError("persistent tree storage must be contiguous")
    indices = tree_indices.to(dtype=torch.long).reshape(-1)
    accepted = int(indices.numel())
    if accepted == 0:
        return
    head_dim = int(storage.shape[-1])
    token_capacity = int(storage.shape[-2])
    row_count = int(storage.numel() // (token_capacity * head_dim))
    block = triton.next_power_of_2(accepted * head_dim)
    _commit_tree_rows_kernel[(row_count,)](
        storage,
        indices,
        int(previous_length),
        token_capacity * head_dim,
        head_dim,
        accepted,
        block,
        num_warps=4,
    )

"""Triton implementation of KERV greedy Verify-Accept reduction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_accept_length_pathwise_kernel(
    logits,
    candidates,
    accept_lengths,
    logits_stride_path: tl.constexpr,
    logits_stride_position: tl.constexpr,
    candidates_stride_path: tl.constexpr,
    n_positions: tl.constexpr,
    vocab_size: tl.constexpr,
    threshold: tl.constexpr,
    token_offset: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    """Compute one accepted prefix per path without materializing matches."""
    path = tl.program_id(0)
    vocab_offsets = tl.arange(0, BLOCK_VOCAB)
    alive = 1
    accepted = 0
    for position in tl.static_range(0, n_positions):
        row = logits + path * logits_stride_path + position * logits_stride_position
        values = tl.load(
            row + vocab_offsets,
            mask=vocab_offsets < vocab_size,
            other=-float("inf"),
        )
        predicted = tl.argmax(values, axis=0) + token_offset
        candidate = tl.load(candidates + path * candidates_stride_path + position + 1)
        alive = alive & (tl.abs(candidate - predicted) <= threshold)
        accepted += alive
    tl.store(accept_lengths + path, accepted)


@triton.jit
def _best_candidate_from_lengths_kernel(
    accept_lengths,
    best_candidate,
    best_length,
    n_paths: tl.constexpr,
    BLOCK_PATHS: tl.constexpr,
):
    paths = tl.arange(0, BLOCK_PATHS)
    valid = paths < n_paths
    accepted = tl.load(accept_lengths + paths, mask=valid, other=-1)
    tl.store(best_candidate, tl.argmax(accepted, axis=0))
    tl.store(best_length, tl.max(accepted, axis=0))


@torch.library.custom_op("flagos_kerv::verify_accept", mutates_args=())
def verify_accept(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    threshold: float,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first best path and its accepted prefix length on device."""
    if not logits.is_cuda or not candidates.is_cuda:
        raise ValueError("flagos_kerv::verify_accept requires CUDA tensors")
    if logits.ndim != 3 or candidates.ndim != 2:
        raise ValueError("expected logits[P,S,V] and candidates[P,S]")
    if logits.shape[0] != candidates.shape[0] or logits.shape[1] != candidates.shape[1]:
        raise ValueError("logits and candidates path/sequence dimensions must match")
    n_paths = int(logits.shape[0])
    n_positions = int(logits.shape[1] - 1)
    vocab_size = int(logits.shape[2])
    if n_paths <= 0 or n_positions <= 0 or vocab_size <= 0:
        raise ValueError("Verify-Accept dimensions must be positive")
    block_vocab = triton.next_power_of_2(vocab_size)
    block_paths = triton.next_power_of_2(n_paths)
    accept_lengths = torch.empty((n_paths,), dtype=torch.int32, device=logits.device)
    best_candidate = torch.empty((), dtype=torch.int64, device=logits.device)
    best_length = torch.empty((), dtype=torch.int64, device=logits.device)
    _candidate_accept_length_pathwise_kernel[(n_paths,)](
        logits,
        candidates,
        accept_lengths,
        logits.stride(0),
        logits.stride(1),
        candidates.stride(0),
        n_positions=n_positions,
        vocab_size=vocab_size,
        threshold=int(round(float(threshold))),
        token_offset=int(token_offset),
        BLOCK_VOCAB=block_vocab,
        num_warps=8,
    )
    _best_candidate_from_lengths_kernel[(1,)](
        accept_lengths,
        best_candidate,
        best_length,
        n_paths=n_paths,
        BLOCK_PATHS=block_paths,
        num_warps=1,
    )
    return best_candidate, best_length


@verify_accept.register_fake
def _verify_accept_fake(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    threshold: float,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del candidates, threshold, token_offset
    return (
        torch.empty((), dtype=torch.int64, device=logits.device),
        torch.empty((), dtype=torch.int64, device=logits.device),
    )


def load_flagos_kerv_verify_ops() -> None:
    """Force registration and fail if the operator schema is unavailable."""
    torch.ops.flagos_kerv.verify_accept

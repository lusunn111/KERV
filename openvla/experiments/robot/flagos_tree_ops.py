"""Loader and Python facade for the FlagOS KERV candidate-tree operators."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


_EXTENSION_LOADED = False


def load_flagos_kerv_tree_ops(verbose: bool = False) -> None:
    """Build/load the portable CPU operator and register it under ``flagos_kerv``."""
    global _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return
    try:
        torch.ops.flagos_kerv.build_verification_subtree
        _EXTENSION_LOADED = True
        return
    except (AttributeError, RuntimeError):
        pass

    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "kerv_tree_ops.cpp"
    if not source.exists():
        raise FileNotFoundError(f"KERV tree operator source is missing: {source}")
    load(
        name="flagos_kerv_tree_ops_v1",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG"],
        verbose=verbose,
        is_python_module=False,
    )
    # Fail immediately if registration did not happen; no silent Python fallback.
    torch.ops.flagos_kerv.build_verification_subtree
    torch.ops.flagos_kerv.pack_verification_tokens
    torch.ops.flagos_kerv.pack_verification_tokens_padded
    _EXTENSION_LOADED = True


def build_verification_subtree(
    parent_indices: torch.Tensor,
    max_paths: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    load_flagos_kerv_tree_ops()
    return torch.ops.flagos_kerv.build_verification_subtree(parent_indices, max_paths)


def pack_verification_tokens(
    draft_tokens: torch.Tensor,
    kept_indices: torch.Tensor,
) -> torch.Tensor:
    load_flagos_kerv_tree_ops()
    return torch.ops.flagos_kerv.pack_verification_tokens(draft_tokens, kept_indices)


def pack_verification_tokens_padded(
    draft_tokens: torch.Tensor,
    kept_indices: torch.Tensor,
    target_nodes: int,
) -> torch.Tensor:
    load_flagos_kerv_tree_ops()
    return torch.ops.flagos_kerv.pack_verification_tokens_padded(
        draft_tokens,
        kept_indices,
        target_nodes,
    )

import copy
import random
from collections import defaultdict

# typing
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def _build_parent_maps(retrieve_indices: torch.Tensor) -> Tuple[dict, dict]:
    parent = {0: 0}
    children = defaultdict(set)
    if retrieve_indices is None or retrieve_indices.numel() == 0:
        return parent, children

    for path in retrieve_indices.tolist():
        prev = None
        for raw_idx in path:
            idx = int(raw_idx)
            if idx < 0:
                break
            if prev is None:
                prev = idx
                continue
            if idx not in parent:
                parent[idx] = prev
            children[prev].add(idx)
            prev = idx
    return parent, children


def _collect_token_path(node_idx: int, parent: dict, draft_tokens: torch.Tensor) -> List[int]:
    path_tokens: List[int] = []
    current = node_idx
    while True:
        path_tokens.append(int(draft_tokens[0, current].item()))
        if current == 0:
            break
        current = parent.get(current, 0)
    return list(reversed(path_tokens))


def _rebuild_tree_buffers(
    parent: dict,
    children: dict,
    node_count: int,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = draft_tokens.device
    mask_dtype = tree_mask.dtype if tree_mask is not None else torch.float32
    adjacency = torch.zeros((node_count, node_count), dtype=mask_dtype, device=device)

    for node in range(node_count):
        current = node
        while True:
            adjacency[node, current] = 1.0
            if current == 0:
                break
            current = parent.get(current, 0)

    tree_mask_new = adjacency.unsqueeze(0).unsqueeze(0)
    tree_position_ids = (adjacency.sum(dim=-1).to(torch.long) - 1).unsqueeze(0)

    leaf_nodes = [idx for idx in range(1, node_count) if len(children.get(idx, ())) == 0]
    if not leaf_nodes:
        # Avoid empty retrieve indices by treating the deepest node as a leaf.
        leaf_nodes = [node_count - 1] if node_count > 1 else [0]

    paths: List[List[int]] = []
    max_path_len = 0
    for leaf in leaf_nodes:
        path: List[int] = []
        current = leaf
        while True:
            path.append(current)
            if current == 0:
                break
            current = parent.get(current, 0)
        path = list(reversed(path))
        paths.append(path)
        max_path_len = max(max_path_len, len(path))

    retrieve = torch.full(
        (len(paths), max_path_len),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    for row_idx, path in enumerate(paths):
        retrieve[row_idx, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)

    return retrieve, tree_mask_new, tree_position_ids


def _rebuild_tree_buffers_batched(
    parent: dict,
    children: dict,
    node_count: int,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the small dynamic tree on host and transfer each buffer once.

    The original helper allocates ``adjacency`` on CUDA and fills it through
    Python scalar assignments, producing one tiny GPU launch per ancestor.
    KERV trees contain only tens of nodes, so host construction plus one copy
    is both cheaper and exactly equivalent.
    """
    device = draft_tokens.device
    mask_dtype = tree_mask.dtype if tree_mask is not None else torch.float32
    adjacency = torch.zeros((node_count, node_count), dtype=mask_dtype)
    for node in range(node_count):
        current = node
        while True:
            adjacency[node, current] = 1.0
            if current == 0:
                break
            current = parent.get(current, 0)

    tree_position_ids = (adjacency.sum(dim=-1).to(torch.long) - 1).unsqueeze(0)
    leaf_nodes = [idx for idx in range(1, node_count) if len(children.get(idx, ())) == 0]
    if not leaf_nodes:
        leaf_nodes = [node_count - 1] if node_count > 1 else [0]

    paths: List[List[int]] = []
    for leaf in leaf_nodes:
        path: List[int] = []
        current = leaf
        while True:
            path.append(current)
            if current == 0:
                break
            current = parent.get(current, 0)
        paths.append(list(reversed(path)))
    max_path_len = max(len(path) for path in paths)
    retrieve = torch.full((len(paths), max_path_len), -1, dtype=torch.long)
    for row_idx, path in enumerate(paths):
        retrieve[row_idx, : len(path)] = torch.tensor(path, dtype=torch.long)

    return (
        retrieve.to(device),
        adjacency.unsqueeze(0).unsqueeze(0).to(device),
        tree_position_ids.to(device),
    )


def _build_static_verification_plan(
    parent: dict,
    children: dict,
    node_count: int,
    max_paths: int,
    mask_dtype: torch.dtype,
    max_depth: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Build a profile-selected verification subtree entirely on the host.

    KERV's candidate paths are ordered by the drafter.  A profile-selected
    ``max_paths`` therefore keeps the highest-priority prefix of leaves, then
    closes that set over all ancestors.  The leaf dimension is fixed while the
    number of unique nodes may still vary after KERV adds Kalman branches.  A
    topology cache is therefore reused only when the complete parent signature
    matches; callers must not assume that every iteration has the same node
    dimension.
    """
    leaf_nodes = [idx for idx in range(1, node_count) if len(children.get(idx, ())) == 0]
    if not leaf_nodes:
        leaf_nodes = [node_count - 1] if node_count > 1 else [0]

    full_paths: List[List[int]] = []
    for leaf in leaf_nodes:
        path: List[int] = []
        current = leaf
        while True:
            path.append(current)
            if current == 0:
                break
            current = parent.get(current, 0)
        full_paths.append(list(reversed(path)))

    selected_paths = full_paths
    if max_paths > 0:
        selected_paths = full_paths[: min(max_paths, len(full_paths))]

    # A compact template keeps the root plus ``max_depth`` action positions.
    # Prefixes can collapse to the same path after truncation, so deduplicate
    # them while preserving the drafter's priority order.  This branch is
    # intentionally host-side and is only used by the opt-in compact mode;
    # the exact/operatorized path remains unchanged.
    if max_depth > 0:
        compacted: List[List[int]] = []
        seen = set()
        for path in selected_paths:
            clipped = path[: 1 + int(max_depth)]
            key = tuple(clipped)
            if key not in seen:
                seen.add(key)
                compacted.append(clipped)
        selected_paths = compacted or [[0]]

    kept_old_nodes = sorted({node for path in selected_paths for node in path})
    old_to_new = {old: new for new, old in enumerate(kept_old_nodes)}
    static_node_count = len(kept_old_nodes)
    adjacency = torch.zeros((static_node_count, static_node_count), dtype=mask_dtype)
    for new_node, old_node in enumerate(kept_old_nodes):
        current = old_node
        while True:
            if current in old_to_new:
                adjacency[new_node, old_to_new[current]] = 1.0
            if current == 0:
                break
            current = parent.get(current, 0)

    max_path_len = max(len(path) for path in selected_paths)
    retrieve = torch.full((len(selected_paths), max_path_len), -1, dtype=torch.long)
    for row_idx, path in enumerate(selected_paths):
        mapped_path = [old_to_new[node] for node in path]
        retrieve[row_idx, : len(mapped_path)] = torch.tensor(mapped_path, dtype=torch.long)

    tree_position_ids = (adjacency.sum(dim=-1).to(torch.long) - 1).unsqueeze(0)
    stats = {
        "full_nodes": int(node_count),
        "verified_nodes": int(static_node_count),
        "full_paths": int(len(full_paths)),
        "verified_paths": int(len(selected_paths)),
        "max_depth": int(max_depth),
    }
    return (
        torch.tensor(kept_old_nodes, dtype=torch.long),
        retrieve,
        adjacency.unsqueeze(0).unsqueeze(0),
        tree_position_ids,
        stats,
    )


def _get_static_verification_plan(
    model,
    parent: dict,
    children: dict,
    node_count: int,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Materialize or reuse a fixed-width KERV verification topology."""
    configured_max_paths = int(getattr(model, "_flagos_tree_verify_max_paths", 0))
    configured_max_depth = int(getattr(model, "_flagos_tree_max_depth", 0))
    compact_owner = getattr(model, "ea_layer", model)
    compact_mode = str(
        getattr(model, "_flagos_compact_tree_mode", getattr(compact_owner, "_flagos_compact_tree_mode", "off"))
    ).lower()
    compact_level = str(
        getattr(
            compact_owner,
            "_flagos_compact_tree_level",
            getattr(model, "_flagos_compact_tree_level", "full"),
        )
    ).lower()
    compact_active = compact_mode == "on"
    if compact_mode == "auto":
        compact_active = bool(
            getattr(compact_owner, "_flagos_compact_tree_active", getattr(model, "_flagos_compact_tree_active", False))
            and getattr(compact_owner, "_flagos_compact_tree_confidence", getattr(model, "_flagos_compact_tree_confidence", None)) is not None
        )
    # ``auto`` falls back to the exact configured topology, never to a
    # partially compacted graph.  This makes the confidence gate lossless.
    if compact_mode == "auto" and compact_level == "full":
        # Accuracy-gated routing must return to the proven Phase-6 topology,
        # rather than the unbounded 49-path tree.  This is an explicit safe
        # fallback and is configured independently from the compact tiers.
        max_paths = int(
            getattr(
                compact_owner,
                "_flagos_compact_tree_full_max_paths",
                getattr(model, "_flagos_compact_tree_full_max_paths", 0),
            )
        )
        max_depth = int(
            getattr(
                compact_owner,
                "_flagos_compact_tree_full_max_depth",
                getattr(model, "_flagos_compact_tree_full_max_depth", 0),
            )
        )
    elif compact_active and compact_level == "tight":
        max_paths = int(
            getattr(
                compact_owner,
                "_flagos_compact_tree_tight_max_paths",
                getattr(model, "_flagos_compact_tree_tight_max_paths", 0),
            )
        )
        max_depth = int(
            getattr(
                compact_owner,
                "_flagos_compact_tree_tight_max_depth",
                getattr(model, "_flagos_compact_tree_tight_max_depth", 0),
            )
        )
    else:
        max_paths = configured_max_paths if compact_active or compact_mode == "off" else 0
        max_depth = configured_max_depth if compact_active else 0
    cache_enabled = bool(getattr(model, "_flagos_tree_cache_topology", True))
    parent_signature = tuple(int(parent.get(idx, 0)) for idx in range(node_count))
    configured_node_buckets = tuple(
        getattr(model, "_flagos_tree_node_buckets", ())
    )
    node_buckets = (
        configured_node_buckets
        if bool(getattr(model, "_flagos_tree_use_node_buckets", True))
        else ()
    )
    cache_key = (
        max_paths,
        max_depth,
        compact_active,
        compact_level,
        node_buckets,
        parent_signature,
    )
    cache = getattr(model, "_flagos_static_tree_cache", None)
    if cache is None:
        cache = {}
        model._flagos_static_tree_cache = cache

    cached = cache.get(cache_key) if cache_enabled else None
    if cached is None:
        mask_dtype = tree_mask.dtype if tree_mask is not None else torch.float32
        use_operator = bool(getattr(model, "_flagos_tree_operator_enabled", False)) and not compact_active and max_depth <= 0
        if use_operator:
            operator_start = time.perf_counter()
            parent_indices = torch.tensor(
                [int(parent.get(idx, 0)) for idx in range(node_count)],
                dtype=torch.long,
            )
            kept, retrieve, static_mask, position_ids, stats_tensor = (
                torch.ops.flagos_kerv.build_verification_subtree(
                    parent_indices,
                    max_paths,
                )
            )
            if static_mask.dtype != mask_dtype:
                static_mask = static_mask.to(mask_dtype)
            stat_values = stats_tensor.tolist()
            stats = {
                "full_nodes": int(stat_values[0]),
                "verified_nodes": int(stat_values[1]),
                "full_paths": int(stat_values[2]),
                "verified_paths": int(stat_values[3]),
            }
            model._flagos_tree_operator_build_hits = int(
                getattr(model, "_flagos_tree_operator_build_hits", 0)
            ) + 1
            model._flagos_tree_operator_build_time_s = float(
                getattr(model, "_flagos_tree_operator_build_time_s", 0.0)
            ) + (time.perf_counter() - operator_start)
        else:
            kept, retrieve, static_mask, position_ids, stats = (
                _build_static_verification_plan(
                    parent,
                    children,
                    node_count,
                    max_paths,
                    mask_dtype,
                    max_depth,
                )
            )
        real_nodes = int(stats["verified_nodes"])
        target_nodes = real_nodes
        if node_buckets and not bool(
            getattr(model, "_flagos_exact_node_templates", False)
        ):
            target_nodes = next(
                (bucket for bucket in node_buckets if bucket >= real_nodes),
                node_buckets[-1],
            )
            if target_nodes < real_nodes:
                raise ValueError(
                    f"Tree with {real_nodes} nodes exceeds the largest bucket "
                    f"{node_buckets[-1]}"
                )
        if target_nodes > real_nodes:
            padded_mask = torch.zeros(
                (1, 1, target_nodes, target_nodes),
                dtype=static_mask.dtype,
            )
            padded_mask[:, :, :real_nodes, :real_nodes] = static_mask
            dummy_indices = torch.arange(real_nodes, target_nodes)
            padded_mask[0, 0, dummy_indices, 0] = 1
            padded_mask[0, 0, dummy_indices, dummy_indices] = 1
            static_mask = padded_mask
            padded_positions = torch.ones((1, target_nodes), dtype=torch.long)
            padded_positions[:, :real_nodes] = position_ids
            position_ids = padded_positions
        stats["padded_nodes"] = int(target_nodes)
        stats["compact_active"] = bool(compact_active)
        stats["compact_level"] = compact_level
        stats["configured_max_paths"] = int(configured_max_paths)
        stats["configured_max_depth"] = int(configured_max_depth)
        device_plan = (
            kept.to(draft_tokens.device),
            retrieve.to(draft_tokens.device),
            static_mask.to(draft_tokens.device),
            position_ids.to(draft_tokens.device),
            stats,
        )
        if cache_enabled:
            # The topology space is tiny in KERV, but keep a hard bound to
            # avoid retaining device buffers for pathological dynamic inputs.
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[cache_key] = device_plan
        model._flagos_tree_cache_misses = int(
            getattr(model, "_flagos_tree_cache_misses", 0)
        ) + 1
        return device_plan

    model._flagos_tree_cache_hits = int(
        getattr(model, "_flagos_tree_cache_hits", 0)
    ) + 1
    return cached


def _augment_tree_with_kf_branches_batched(
    model,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact KERV Kalman-tree construction with batched host/device traffic."""
    action_dim = int(model.get_action_dim())
    parent_map, children_map = _build_parent_maps(retrieve_indices)
    total_nodes = draft_tokens.shape[1]
    for idx in range(1, total_nodes):
        parent_map.setdefault(idx, 0)
        children_map.setdefault(parent_map[idx], set()).add(idx)

    if tree_position_ids is not None and tree_position_ids.numel() > 0:
        depth_values = tree_position_ids[0, :total_nodes].detach().to("cpu").tolist()
        node_depth = {idx: int(value) for idx, value in enumerate(depth_values)}
    else:
        node_depth = {}
    node_depth[0] = 0
    draft_token_values = [
        int(value) for value in draft_tokens[0].detach().to("cpu").tolist()
    ]

    def depth_for(node: int) -> int:
        if node in node_depth:
            return node_depth[node]
        current = node
        depth_val = 0
        seen = set()
        while current not in seen:
            seen.add(current)
            if current == 0:
                break
            current = parent_map.get(current, 0)
            depth_val += 1
        node_depth[node] = depth_val
        return depth_val

    branch_nodes: List[int] = []
    branch_prefixes: List[List[int]] = []
    branch_remaining: List[int] = []
    for node_idx in range(1, total_nodes):
        current_depth = depth_for(node_idx)
        remaining = action_dim - current_depth
        if remaining <= 0:
            continue
        prefix_tokens: List[int] = []
        current = node_idx
        while True:
            prefix_tokens.append(draft_token_values[current])
            if current == 0:
                break
            current = parent_map.get(current, 0)
        prefix_tokens.reverse()
        branch_nodes.append(node_idx)
        branch_prefixes.append(prefix_tokens)
        branch_remaining.append(remaining)

    if bool(getattr(model, "_flagos_kalman_batch_enabled", False)) and hasattr(
        model, "predict_kf_chains_batched"
    ):
        branch_tokens = model.predict_kf_chains_batched(
            branch_prefixes,
            branch_remaining,
        )
    else:
        branch_tokens = [
            model.predict_kf_chain_tokens(prefix, remaining)
            for prefix, remaining in zip(branch_prefixes, branch_remaining)
        ]

    appended_tokens: List[int] = []
    next_node = total_nodes
    for node_idx, remaining, kf_tokens in zip(
        branch_nodes,
        branch_remaining,
        branch_tokens,
    ):
        if not kf_tokens:
            continue
        previous = node_idx
        for token_value in kf_tokens[:remaining]:
            appended_tokens.append(int(token_value))
            parent_map[next_node] = previous
            children_map.setdefault(previous, set()).add(next_node)
            node_depth[next_node] = depth_for(previous) + 1
            previous = next_node
            next_node += 1

    if appended_tokens:
        appended = torch.tensor(
            [appended_tokens], dtype=draft_tokens.dtype, device=draft_tokens.device
        )
        draft_tokens = torch.cat((draft_tokens, appended), dim=1)
    if bool(getattr(model, "_flagos_static_tree_runtime_enabled", False)):
        (
            kept_indices,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            static_stats,
        ) = _get_static_verification_plan(
            model,
            parent_map,
            children_map,
            draft_tokens.shape[1],
            draft_tokens,
            tree_mask,
        )
        embodied_pack = bool(getattr(model, "_flagos_embodied_ops_enabled", False)) and (
            "kerv_static_tree_pack"
            in getattr(model, "_flagos_embodied_ops_include", set())
        )
        if embodied_pack:
            target_nodes = int(static_stats.get("padded_nodes", kept_indices.numel()))
            draft_tokens = torch.ops.flagos_embodied.kerv_static_tree_pack(
                draft_tokens,
                kept_indices.to(draft_tokens.device),
                target_nodes,
            )
            model._flagos_embodied_tree_pack_hits = int(
                getattr(model, "_flagos_embodied_tree_pack_hits", 0)
            ) + 1
        elif bool(getattr(model, "_flagos_tree_operator_enabled", False)):
            target_nodes = int(static_stats.get("padded_nodes", kept_indices.numel()))
            if target_nodes > kept_indices.numel():
                draft_tokens = torch.ops.flagos_kerv.pack_verification_tokens_padded(
                    draft_tokens,
                    kept_indices,
                    target_nodes,
                )
            else:
                draft_tokens = torch.ops.flagos_kerv.pack_verification_tokens(
                    draft_tokens,
                    kept_indices,
                )
            model._flagos_tree_operator_pack_hits = int(
                getattr(model, "_flagos_tree_operator_pack_hits", 0)
            ) + 1
        else:
            draft_tokens = draft_tokens.index_select(1, kept_indices)
        model._flagos_static_tree_last_stats = static_stats
        model._flagos_static_tree_hit_count = int(
            getattr(model, "_flagos_static_tree_hit_count", 0)
        ) + 1
    else:
        retrieve_indices, tree_mask, tree_position_ids = _rebuild_tree_buffers_batched(
            parent_map,
            children_map,
            draft_tokens.shape[1],
            draft_tokens,
            tree_mask,
        )
    model._flagos_tree_builder_hit_count = int(
        getattr(model, "_flagos_tree_builder_hit_count", 0)
    ) + 1
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def augment_tree_with_kf_branches(
    model,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not getattr(model, "_kalman_tree_enabled", True):
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    kalman_enabled = getattr(model, "_kalman_config", {}).get("enabled", False)
    if not kalman_enabled:
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    if not hasattr(model, "get_action_dim"):
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    try:
        action_dim = int(model.get_action_dim())
    except Exception:
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    if action_dim <= 0:
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    if bool(getattr(model, "_flagos_tree_builder_enabled", False)):
        with torch.profiler.record_function("KERV::CandidateTreeBuilder"):
            return _augment_tree_with_kf_branches_batched(
                model,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
            )

    parent_map, children_map = _build_parent_maps(retrieve_indices)

    total_nodes = draft_tokens.shape[1]
    for idx in range(1, total_nodes):
        parent_map.setdefault(idx, 0)
        children_map.setdefault(parent_map[idx], set()).add(idx)

    device = draft_tokens.device
    node_depth = {}
    if tree_position_ids is not None and tree_position_ids.numel() > 0:
        for idx in range(min(tree_position_ids.shape[1], total_nodes)):
            node_depth[idx] = int(tree_position_ids[0, idx].item())
    node_depth[0] = 0

    def depth_for(node: int) -> int:
        if node in node_depth:
            return node_depth[node]
        current = node
        depth_val = 0
        seen = set()
        while current not in seen:
            seen.add(current)
            if current == 0:
                break
            current = parent_map.get(current, 0)
            depth_val += 1
        node_depth[node] = depth_val
        return depth_val

    original_nodes = list(range(1, total_nodes))
    for node_idx in original_nodes:
        current_depth = depth_for(node_idx)
        remaining = action_dim - current_depth
        if remaining <= 0:
            continue

        prefix_tokens = _collect_token_path(node_idx, parent_map, draft_tokens)
        kf_tokens = model.predict_kf_chain_tokens(prefix_tokens, remaining)
        if not kf_tokens:
            continue

        prev = node_idx
        for token_val in kf_tokens:
            if remaining <= 0:
                break
            token_tensor = torch.tensor([[token_val]], dtype=draft_tokens.dtype, device=device)
            draft_tokens = torch.cat([draft_tokens, token_tensor], dim=1)
            new_idx = draft_tokens.shape[1] - 1
            parent_map[new_idx] = prev
            children_map.setdefault(prev, set()).add(new_idx)
            node_depth[new_idx] = depth_for(prev) + 1
            prev = new_idx
            remaining -= 1

    # Rebuild the auxiliary buffers to reflect the new nodes
    new_total_nodes = draft_tokens.shape[1]
    retrieve_indices, tree_mask, tree_position_ids = _rebuild_tree_buffers(
        parent_map,
        children_map,
        new_total_nodes,
        draft_tokens,
        tree_mask,
    )

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(model_inputs, model, logits_processor):
   # model.ea_layer.reset_kv()
    #model_inputs['use_cache']=True
    #print(model.tree)
    prompt_graph = getattr(model, "_flagos_prompt_stage_graph", None)
    if prompt_graph is not None:
        graph_result = model._run_prompt_stage_graph(model_inputs)
    else:
        graph_result = None
    if graph_result is None:
        outputs, orig, hidden_states, model_embeds = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
            output_orig=True,
            #use_cache=True
        )
    else:
        outputs, orig, hidden_states, model_embeds = graph_result
    #outputs['use_cache']=True
    #这里是[p,e_0]
    #print(outputs.keys())
    #print('past kv shape')
    ##print(len(outputs.past_key_values[0]))
    #print(len(outputs.past_key_values[0][0]))
    #print(outputs.past_key_values[0][0][0].shape)
    #exit()
    input_embeds = model_embeds
    hidden_states = hidden_states[:,:,:]
    if logits_processor is not None:
        print("NULL logits processor")
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1]) + int(
            getattr(model, "_flagos_action_logit_offset", 0)
        )
        token = token[None, None]
    #print(input_id)
    #input_ids = torch.cat((input_ids,token))
   #print(input_ids)
    #print('token,',token)
    #model.ea_layer.reset_kv()
    input_ids = token
    input_token_embeds = model.ea_layer.embed_tokens(input_ids)
    #print(input_embeds[:,-1]==input_token_embeds)
    #print(input_embeds.shape)
    ea_layer_input_embeds = torch.cat((input_embeds,input_token_embeds),dim=1)
    #print(input_token_embeds.shape)
    #exit()
    #print(outputs.multimodal_labels)
    #print('hidden states shape',hidden_states.shape)
    #print('input ids shape',input_ids)
    #print('ea layer embeds',ea_layer_input_embeds)
    #print()

    # Clone the output hidden states
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids,ea_layer_input_embeds,model.base_model.language_model.lm_head, logits_processor)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = augment_tree_with_kf_branches(
        model,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
    )
    # 将draft_tokens
    # print(f"[* draft_tokens] {draft_tokens}")
    # 存到'./log.txt'
    # 创建或打开文件，覆盖
    # with open('./log_init_tree.txt', 'a') as f:
    #     f.write(f"[* draft_tokens]{draft_tokens}\n")
    #     f.write(f'[* retrieve_indices]{retrieve_indices}\n')
    #     f.write(f'[* tree_mask]{tree_mask}\n')
    #     f.write(f'[* tree_position_ids]{tree_position_ids}\n')

    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token, outputs.past_key_values, input_embeds, outputs.attention_mask


def reset_tree_mode(
        model,
):
    model.tree_mask = None
    model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates

def tree_decoding(
        model,
        prompt_embeds,
        tree_candidates,
        attention_mask,
        past_key_values,
        tree_position_ids,
        #input_ids,
        retrieve_indices,
        draft_logit = None
):
    persistent_cache = bool(
        getattr(past_key_values, "_flagos_persistent_tree_cache", False)
    )
    if persistent_cache and bool(
        getattr(model, "_flagos_persistent_input_buffers", False)
    ):
        buffers = getattr(model, "_flagos_position_id_buffers", None)
        if buffers is None:
            buffers = {}
            model._flagos_position_id_buffers = buffers
        position_key = (
            tree_position_ids.device.type,
            tree_position_ids.device.index,
            tuple(tree_position_ids.shape),
            tree_position_ids.dtype,
        )
        position_ids = buffers.get(position_key)
        if position_ids is None:
            position_ids = torch.empty_like(tree_position_ids)
            buffers[position_key] = position_ids
        torch.add(
            tree_position_ids,
            int(prompt_embeds.shape[1]),
            out=position_ids,
        )
    else:
        position_ids = tree_position_ids + prompt_embeds.shape[1]
    #position_ids = torch.cat((torch.tensor([i for i in range(prompt_embeds.shape[1])]).to(tree_position_ids.device).unsqueeze(0),position_ids),dim=1)
    #print(position_ids.shape)
    #print(tree_candidates)
    #print(output_orig)
    #print(len(past_key_values))
    #print((past_key_values[0][0].shape))
    #print('prompt embedding',prompt_embeds.shape)
    #print('tree decoding')
    #print('position ids',position_ids)
    #print('attention mask',attention_mask)
    #print('past key values',past_key_values)
    #print(tree_position_ids)
    #print(past_key_values[0][0].shape)
    #exit()
    #input_ids = draft_tokens
    #past kv?
    #position_ids = position_ids
    #print('tree candidate shape',tree_candidates.shape)
    #print('tree attn positional id')
    #print(position_ids)
    # with
    # with open('./log_tree_candidates.txt', 'a') as f:
    #     f.write(f"[* tree_candidates]{tree_candidates}\n")
    embodied_include = set(
        getattr(model, "_flagos_embodied_ops_include", set())
    )
    if persistent_cache and bool(
        getattr(model, "_flagos_persistent_input_buffers", False)
    ):
        # Bind one embedding buffer to every fixed tree bucket.  CUDA Graphs
        # capture these exact addresses, so subsequent verifier calls only
        # refresh the Token rows and do not copy another [T,H] tensor into a
        # graph-private input allocation.
        embedding_weight = model.base_model.language_model.model.embed_tokens.weight
        embedding_buffers = getattr(model, "_flagos_tree_embedding_buffers", None)
        if embedding_buffers is None:
            embedding_buffers = {}
            model._flagos_tree_embedding_buffers = embedding_buffers
        embedding_key = (
            tree_candidates.device.type,
            tree_candidates.device.index,
            int(tree_candidates.shape[1]),
            int(embedding_weight.shape[1]),
            embedding_weight.dtype,
        )
        text_embedding = embedding_buffers.get(embedding_key)
        if text_embedding is None:
            text_embedding = torch.empty(
                (1, int(tree_candidates.shape[1]), int(embedding_weight.shape[1])),
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
            embedding_buffers[embedding_key] = text_embedding
        torch.index_select(
            embedding_weight,
            0,
            tree_candidates.reshape(-1),
            out=text_embedding[0],
        )
        model._flagos_tree_embed_resident_hits = int(
            getattr(model, "_flagos_tree_embed_resident_hits", 0)
        ) + 1
    elif (
        bool(getattr(model, "_flagos_embodied_ops_enabled", False))
        and "kerv_tree_embed_pack" in embodied_include
        and tree_candidates.ndim == 2
        and tree_candidates.shape[0] == 1
    ):
        # The tree builder has already selected the candidate-node order.  A
        # contiguous index vector preserves that order while allowing the
        # phase-two op to combine the fixed-node padding and embedding write
        # into the resident verifier input buffer.  More general gather maps
        # continue through the native embedding path.
        from kerv_flagos.embodied_ops import kerv_tree_embed_pack

        # Packed tree sizes repeat across verification steps. Reuse the
        # device-side contiguous index vector instead of allocating and
        # populating an ``arange`` on every replay.
        cache = getattr(model, "_flagos_tree_embed_kept_cache", None)
        if cache is None:
            cache = {}
            model._flagos_tree_embed_kept_cache = cache
        cache_key = (
            tree_candidates.device.type,
            tree_candidates.device.index,
            int(tree_candidates.shape[1]),
        )
        kept = cache.get(cache_key)
        if kept is None:
            kept = torch.arange(
                tree_candidates.shape[1], device=tree_candidates.device, dtype=torch.long
            )
            cache[cache_key] = kept
        text_embedding = kerv_tree_embed_pack(
            tree_candidates,
            kept,
            model.base_model.language_model.model.embed_tokens.weight,
            int(tree_candidates.shape[1]),
        )
        model._flagos_tree_embed_pack_hits = int(
            getattr(model, "_flagos_tree_embed_pack_hits", 0)
        ) + 1
    else:
        text_embedding = model.base_model.language_model.model.embed_tokens(tree_candidates)
    #model.ea_layer.embed_tokens(tree_candidates[:,0,:])
    #print('assumption equal',prompt_embeds[:,-1,:]==model.ea_layer.embed_tokens(tree_candidates[:,0])[0])
    inputs_embeds = text_embedding
    #print('position ids')
    #print(position_ids)
    #inputs_embeds = torch.cat((prompt_embeds,text_embedding),dim=1)
    #print('input embed shape')
    #print(inputs_embeds.shape)
    #print('attention mask shape')
    #print(attention_mask)
    #print('past kv shape')
    #print(past_key_values[0][0][0].shape)
    #print(position_ids.shape)
    if persistent_cache:
        past_key_values.begin_tree(int(tree_candidates.shape[1]))
        external_mask = past_key_values.build_tree_attention_mask(
            model.base_model.language_model.tree_mask,
            dtype=inputs_embeds.dtype,
            materialize=not bool(
                getattr(
                    model.base_model.language_model,
                    "_flagos_static_tree_attention_runtime_enabled",
                    False,
                )
            ),
        )
        model.base_model.language_model._flagos_external_causal_mask = external_mask
    else:
        model.base_model.language_model._flagos_external_causal_mask = None
    verifier_forward = model.forward
    graph_logits_gathered = False
    if bool(getattr(model, "_flagos_compile_verifier_enabled", False)):
        compile_mode = str(
            getattr(model, "_flagos_compile_verifier_mode", "reduce-overhead")
        )
        if compile_mode in {"cuda-graph", "inductor-cuda-graph"}:
            from openvla.experiments.robot.flagos_verifier_graph import (
                run_verifier_cuda_graph,
            )

            graph_retrieve_indices = retrieve_indices
            if persistent_cache and bool(
                getattr(model, "_flagos_persistent_input_buffers", False)
            ):
                retrieve_buffers = getattr(
                    model, "_flagos_retrieve_index_buffers", None
                )
                if retrieve_buffers is None:
                    retrieve_buffers = {}
                    model._flagos_retrieve_index_buffers = retrieve_buffers
                retrieve_key = (
                    retrieve_indices.device.type,
                    retrieve_indices.device.index,
                    tuple(retrieve_indices.shape),
                    retrieve_indices.dtype,
                )
                graph_retrieve_indices = retrieve_buffers.get(retrieve_key)
                if graph_retrieve_indices is None:
                    graph_retrieve_indices = torch.empty_like(retrieve_indices)
                    retrieve_buffers[retrieve_key] = graph_retrieve_indices
                graph_retrieve_indices.copy_(retrieve_indices)
            model._flagos_last_cuda_graph_logits_gathered = False
            graph_outputs = run_verifier_cuda_graph(
                model,
                inputs_embeds,
                position_ids,
                graph_retrieve_indices,
                past_key_values,
            )
            if graph_outputs is not None:
                outputs, tree_logits, hidden_state, input_embeddings = graph_outputs
                graph_logits_gathered = bool(
                    getattr(model, "_flagos_last_cuda_graph_logits_gathered", False)
                )
                model._flagos_compiled_verifier_hits = int(
                    getattr(model, "_flagos_compiled_verifier_hits", 0)
                ) + 1
            else:
                outputs, tree_logits, hidden_state, input_embeddings = model.forward(
                    input_embeds=inputs_embeds,
                    output_orig=True,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    return_dict=True,
                    position_ids=position_ids,
                    use_cache=True,
                )
        else:
            verifier_forward = getattr(model, "_flagos_compiled_verifier_forward", None)
            if verifier_forward is None:
                verifier_forward = torch.compile(
                    model.forward,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=False,
                )
                model._flagos_compiled_verifier_forward = verifier_forward
            model._flagos_compiled_verifier_hits = int(
                getattr(model, "_flagos_compiled_verifier_hits", 0)
            ) + 1
            outputs, tree_logits, hidden_state, input_embeddings = verifier_forward(
                input_embeds=inputs_embeds,
                output_orig=True,
                attention_mask=None,
                past_key_values=past_key_values,
                return_dict=True,
                position_ids=position_ids,
                use_cache=True,
            )
    else:
        outputs, tree_logits, hidden_state, input_embeddings = verifier_forward(
            input_embeds=inputs_embeds,
            output_orig=True,
            attention_mask=None,
            past_key_values=past_key_values,
            return_dict=True,
            position_ids=position_ids,
            use_cache=True,
        )
    #)
    #print(outputs.keys())
    #print(len(outputs))
    #retrieve_indices = retrieve_indices + (past_key_values[0][0].shape[-2])
    #print('tree logits',tree_logits.shape)
    #print('retrieve indices',retrieve_indices)
    #retrieve_indices = retrieve_indices
    #print(tree_logits.shape)
    logits = tree_logits if graph_logits_gathered else tree_logits[0, retrieve_indices]
    # with open('./log_tree_logits.txt', 'a') as f:
    #     f.write(f"[* logits]{logits.shape}\n")
    #draft_logits = draft_logit[0, retrieve_indices]
    return logits, hidden_state,input_embeddings, outputs.past_key_values,outputs





def _native_greedy_verify_accept(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        accept_threshold,
        token_offset: int,
):
    candidates_device = candidates.to(logits.device)
    predicted = torch.argmax(logits[:, :-1], dim=-1) + int(token_offset)
    if accept_threshold is None:
        posterior_mask = (candidates_device[:, 1:] == predicted).int()
    else:
        posterior_mask = (
            torch.abs(candidates_device[:, 1:] - predicted) <= accept_threshold
        ).int()
    candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
    return (
        torch.argmax(candidates_accept_length).to(torch.long),
        candidates_accept_length.max(),
    )


def _select_verify_accept_backend(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        accept_threshold,
        token_offset: int,
        runtime_owner,
) -> str:
    """Select Triton only when it is both exact and measurably faster."""
    selected = getattr(runtime_owner, "_flagos_verify_accept_selected_backend", None)
    embodied_enabled = bool(getattr(runtime_owner, "_flagos_embodied_ops_enabled", False))
    embodied_include = getattr(runtime_owner, "_flagos_embodied_ops_include", set())
    if embodied_enabled and "kerv_verify_accept_control" in embodied_include:
        return "embodied"
    if selected in {"triton", "aten"}:
        return selected

    candidates_device = candidates.to(logits.device)
    threshold = 0.0 if accept_threshold is None else float(accept_threshold)

    def native_call():
        return _native_greedy_verify_accept(
            logits, candidates_device, accept_threshold, token_offset
        )

    def triton_call():
        return torch.ops.flagos_kerv.verify_accept(
            logits, candidates_device, threshold, int(token_offset)
        )

    selected = "aten"
    native_ms = float("inf")
    triton_ms = float("inf")
    try:
        native_result = native_call()
        triton_result = triton_call()
        exact = all(
            torch.equal(lhs.detach(), rhs.detach())
            for lhs, rhs in zip(native_result, triton_result)
        )
        if exact:
            # These kernels are sub-millisecond, so a short test is easily
            # biased by the A100's idle clock. Warm both paths long enough to
            # reach a stable clock, then use the median of alternating trials.
            for _ in range(100):
                native_call()
                triton_call()
            torch.cuda.synchronize(logits.device)

            def measure(call, iterations=200):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    call()
                end.record()
                end.synchronize()
                return float(start.elapsed_time(end)) / iterations

            native_trials = []
            triton_trials = []
            for trial in range(3):
                calls = (
                    ((native_call, native_trials), (triton_call, triton_trials))
                    if trial % 2 == 0
                    else ((triton_call, triton_trials), (native_call, native_trials))
                )
                for call, values in calls:
                    values.append(measure(call))
            native_ms = sorted(native_trials)[1]
            triton_ms = sorted(triton_trials)[1]
            # A small margin prevents timing noise from putting a regression
            # on the default closed-loop path.
            if triton_ms <= native_ms * 0.98:
                selected = "triton"
    except Exception as exc:
        runtime_owner._flagos_verify_accept_selection_error = repr(exc)

    runtime_owner._flagos_verify_accept_selected_backend = selected
    runtime_owner._flagos_verify_accept_native_ms = native_ms
    runtime_owner._flagos_verify_accept_triton_ms = triton_ms
    print(
        "[FlagOS/KERV] Verify-Accept backend="
        f"{selected}; native={native_ms:.6f}ms triton={triton_ms:.6f}ms"
    )
    return selected


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        accept_threshold=None,
        token_offset: int = 0,
        runtime_owner=None,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        if bool(
            getattr(runtime_owner, "_flagos_verify_accept_operator_enabled", False)
        ) or (
            bool(getattr(runtime_owner, "_flagos_embodied_ops_enabled", False))
            and "kerv_verify_accept_control"
            in getattr(runtime_owner, "_flagos_embodied_ops_include", set())
        ):
            backend = _select_verify_accept_backend(
                logits,
                candidates,
                accept_threshold,
                int(token_offset),
                runtime_owner,
            )
            if backend == "embodied":
                threshold = 0.0 if accept_threshold is None else float(accept_threshold)
                best_candidate, accept_length = torch.ops.flagos_embodied.kerv_verify_accept_control(
                    logits,
                    candidates.to(logits.device),
                    threshold,
                    int(token_offset),
                )
                runtime_owner._flagos_embodied_verify_accept_hits = int(
                    getattr(runtime_owner, "_flagos_embodied_verify_accept_hits", 0)
                ) + 1
                return best_candidate, accept_length, logits[best_candidate, accept_length]
            if backend == "triton":
                threshold = 0.0 if accept_threshold is None else float(accept_threshold)
                best_candidate, accept_length = torch.ops.flagos_kerv.verify_accept(
                    logits,
                    candidates.to(logits.device),
                    threshold,
                    int(token_offset),
                )
                runtime_owner._flagos_verify_accept_operator_hits = int(
                    getattr(runtime_owner, "_flagos_verify_accept_operator_hits", 0)
                ) + 1
                return best_candidate, accept_length, logits[best_candidate, accept_length]
            runtime_owner._flagos_verify_accept_aten_hits = int(
                getattr(runtime_owner, "_flagos_verify_accept_aten_hits", 0)
            ) + 1
        #print('evaluate posterior')
        #print('posterior mask')
        #print('candidates shape',candidates.shape)
        #print('logits shape',logits.shape)
        #print('candidats',candidates[:, 1:])
        #print('logits',torch.argmax(logits[:, :-1], dim=-1))
        # Find the tokens that match the maximum logits for each position in the sequence
        best_candidate, accept_length = _native_greedy_verify_accept(
            logits, candidates, accept_threshold, int(token_offset)
        )
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


def _commit_past_key_values(
    past_key_values,
    select_indices: torch.Tensor,
    previous_length: int,
    model,
):
    """Compact accepted verifier KV entries, optionally across all layers."""
    if bool(getattr(past_key_values, "_flagos_persistent_tree_cache", False)):
        tree_indices = select_indices.to(torch.long) - int(previous_length)
        past_key_values.commit_tree(tree_indices, int(previous_length))
        model._flagos_persistent_kv_commit_hits = int(
            getattr(model, "_flagos_persistent_kv_commit_hits", 0)
        ) + 1
        return past_key_values
    if not bool(getattr(model, "_flagos_kv_commit_batched_enabled", False)):
        committed = list(past_key_values)
        for layer_index in range(len(committed)):
            layer_values = committed[layer_index]
            layer_values = torch.cat(
                (layer_values[0].unsqueeze(0), layer_values[1].unsqueeze(0)),
                dim=0,
            )
            layer_indices = select_indices.to(layer_values.device)
            selected = layer_values.index_select(-2, layer_indices)
            end = previous_length + selected.shape[-2]
            layer_values[..., previous_length:end, :].copy_(selected)
            committed[layer_index] = layer_values[..., :end, :]
        return committed

    first = past_key_values[0]
    if isinstance(first, torch.Tensor):
        # Subsequent iterations already expose each layer as [K/V, B, H, S, D].
        packed = torch.stack(tuple(past_key_values), dim=0)
    else:
        # Hugging Face returns a tuple of (K, V) pairs on the first iteration.
        flat_values = tuple(value for layer in past_key_values for value in layer)
        packed = torch.stack(flat_values, dim=0).view(
            len(past_key_values), 2, *flat_values[0].shape
        )
    packed_indices = select_indices.to(packed.device)
    selected = packed.index_select(-2, packed_indices)
    end = previous_length + selected.shape[-2]
    packed[..., previous_length:end, :].copy_(selected)
    model._flagos_kv_commit_batched_hits = int(
        getattr(model, "_flagos_kv_commit_batched_hits", 0)
    ) + 1
    # Unbind returns per-layer views backed by one contiguous allocation. This
    # matches the verifier's existing [K/V, B, H, S, D] layer contract.
    return list(packed[..., :end, :].unbind(0))


@torch.no_grad()
def update_inference_inputs(
        prompt_embeds,
        #prompt_hidden_states,
        input_ids,
        input_len,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        #current_length_data,
        model,
        hidden_state_new,
        sample_p,
        attention_mask,
        precomputed_token=None,
        precomputed_token_is_eos=None,
        return_stop=False,
):
    workspace = getattr(model, "_flagos_decode_workspace", None)
    workspace_enabled = bool(
        getattr(model, "_flagos_system_optimization_enabled", False)
        and getattr(model, "_flagos_persistent_decode_workspace", False)
        and workspace is not None
    )
    prev_input_len = prompt_embeds.shape[1]
    end_loop = False
    if (input_ids.shape[1]-input_len-1+accept_length)>6:
        accept_length=max(6-(input_ids.shape[1]-input_len-1),0)
        end_loop = True
    #print('end loop',end_loop)

    select_indices = (retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len)
    accepted_tokens = candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)

    # The verifier cache and drafter cache are disjoint.  Commit the accepted
    # verifier rows on a dedicated stream while the default stream prepares
    # embeddings and executes the next Draft.  The event join below is placed
    # immediately before returning to the next Verify, so no host or device-
    # wide synchronization is introduced.
    commit_done_event = None
    joint_graph = getattr(model, "_flagos_joint_commit_draft_graph", None)
    joint_requested = bool(
        joint_graph is not None
        and torch.cuda.is_available()
        and accepted_tokens.is_cuda
        and bool(
            getattr(past_key_values_data_list, "_flagos_persistent_tree_cache", False)
        )
    )
    overlap_mode = str(
        getattr(model, "_flagos_commit_draft_stream_overlap_mode", "off")
    ).lower()
    overlap_enabled = bool(
        overlap_mode in {"auto", "on"}
        and torch.cuda.is_available()
        and accepted_tokens.is_cuda
        and bool(
            getattr(past_key_values_data_list, "_flagos_persistent_tree_cache", False)
        )
        and not joint_requested
    )
    if overlap_enabled:
        commit_stream = getattr(model, "_flagos_commit_stream", None)
        if commit_stream is None:
            # Torch 2.6 does not expose get_stream_priority_range on every
            # build. A dedicated default-priority stream preserves the
            # intended overlap and remains portable across supported builds.
            commit_stream = torch.cuda.Stream(device=accepted_tokens.device)
            model._flagos_commit_stream = commit_stream
            model._flagos_commit_inputs_ready_event = torch.cuda.Event()
            model._flagos_commit_done_event = torch.cuda.Event()
        producer_stream = torch.cuda.current_stream(accepted_tokens.device)
        inputs_ready = model._flagos_commit_inputs_ready_event
        inputs_ready.record(producer_stream)
        with torch.cuda.stream(commit_stream):
            commit_stream.wait_event(inputs_ready)
            past_key_values_data_list = _commit_past_key_values(
                past_key_values_data_list,
                select_indices,
                prev_input_len,
                model,
            )
            commit_done_event = model._flagos_commit_done_event
            commit_done_event.record(commit_stream)
        model._flagos_commit_draft_overlap_hits = int(
            getattr(model, "_flagos_commit_draft_overlap_hits", 0)
        ) + 1
    accepted_embeds = model.ea_layer.embed_tokens(accepted_tokens)
    if workspace_enabled:
        native_input_ids = input_ids
        native_prompt_embeds = prompt_embeds
        next_input_ids = workspace.append_tokens(accepted_tokens)
        next_prompt_embeds = workspace.append_embeddings(accepted_embeds)
        if next_input_ids is not None and next_prompt_embeds is not None:
            input_ids = next_input_ids
            prompt_embeds = next_prompt_embeds
            model._flagos_decode_workspace_append_hits = int(
                getattr(model, "_flagos_decode_workspace_append_hits", 0)
            ) + 1
        else:
            workspace_enabled = False
            input_ids = native_input_ids
            prompt_embeds = native_prompt_embeds
            # Restore the workspace cursor if only one of the two appends
            # succeeded before a capacity/layout fallback.
            workspace.reset(native_input_ids, native_prompt_embeds)
    if not workspace_enabled:
        input_ids = torch.cat([input_ids, accepted_tokens], dim=-1)
        prompt_embeds = torch.cat([prompt_embeds, accepted_embeds], dim=1)
    if not overlap_enabled and not joint_requested:
        past_key_values_data_list = _commit_past_key_values(
            past_key_values_data_list,
            select_indices,
            prev_input_len,
            model,
        )
    # Gather only the accepted path.  The old two-stage indexing materialized
    # the complete [batch, paths, accepted_length, hidden] tensor first.
    path_indices = retrieve_indices[best_candidate, : accept_length + 1]
    accept_hidden_state_new = hidden_state_new[:, path_indices]
    prob = sample_p
    if precomputed_token is not None:
        token = precomputed_token.reshape(1, 1).to(input_ids.device)
    elif logits_processor is not None:
        token = torch.multinomial(prob, 1) + int(
            getattr(model, "_flagos_action_logit_offset", 0)
        )
        token = token[None]
    else:
        token = torch.argmax(prob) + int(
            getattr(model, "_flagos_action_logit_offset", 0)
        )
        token = token[None, None]
    if end_loop:
        eos_token = getattr(model, "_flagos_eos_token_buffer", None)
        if (
            eos_token is None
            or eos_token.device != token.device
            or eos_token.dtype != token.dtype
        ):
            eos_token = torch.full(
                (1, 1),
                int(model.tokenizer.eos_token_id),
                device=token.device,
                dtype=token.dtype,
            )
            model._flagos_eos_token_buffer = eos_token
        token = eos_token
        token_is_eos = True
    elif precomputed_token_is_eos is not None:
        token_is_eos = bool(precomputed_token_is_eos)
    else:
        # One explicit scalar transfer replaces the implicit `if tensor == eos`
        # synchronization and is returned to the caller for loop control.
        token_is_eos = int(token.reshape(-1)[0].item()) == model.tokenizer.eos_token_id
    if workspace_enabled:
        # The sampled token is a temporary drafter input.  Do not append it to
        # the committed token cursor: the native ``torch.cat`` below returns a
        # separate tensor and the next Verify starts from ``input_ids`` without
        # this token.
        input_tokens = workspace.draft_token_input(input_ids, token.to(input_ids.device))
        if input_tokens is None:
            workspace_enabled = False
        else:
            # The drafter's fused linear path expects the same contiguous
            # layout as the historical ``torch.cat`` result.  Keep the large
            # persistent token buffer, but materialize only this tiny control
            # sequence when handing it to the drafter.
            input_tokens = input_tokens.contiguous()
            model._flagos_decode_workspace_token_hits = int(
                getattr(model, "_flagos_decode_workspace_token_hits", 0)
            ) + 1
    else:
        input_tokens = None
    if input_tokens is None:
        input_tokens = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    if token_is_eos:
        new_token += accept_length + 1
        result = (input_tokens, None, None, None, None, new_token, None, None, None)
        return result + (True,) if return_stop else result
    input_token_embeds = model.ea_layer.embed_tokens(token)
    native_input_token_embeds = input_token_embeds
    if workspace_enabled:
        input_token_embeds = workspace.append_token_embedding(input_token_embeds)
        ea_layer_input_embeds = (
            workspace.draft_input(prompt_embeds, input_token_embeds)
            if input_token_embeds is not None
            else None
        )
        if ea_layer_input_embeds is not None and not ea_layer_input_embeds.is_contiguous():
            # Preserve the exact contiguous input layout used by the native
            # cat path.  The workspace still reuses the persistent prefix and
            # token embedding storage before this small boundary copy.
            ea_layer_input_embeds = ea_layer_input_embeds.contiguous()
    else:
        ea_layer_input_embeds = None
    if ea_layer_input_embeds is None:
        ea_layer_input_embeds = torch.cat(
            (prompt_embeds, native_input_token_embeds), dim=1
        )
    ea_layer_input_hiddens = accept_hidden_state_new
    prefetched_draft_result = None
    joint_mode = "disabled"
    if joint_requested:
        import inspect

        supports_prefetch = "prefetched_result" in inspect.signature(
            model.ea_layer.topK_genrate
        ).parameters
        if not supports_prefetch:
            joint_requested = False
            _commit_past_key_values(
                past_key_values_data_list,
                select_indices,
                prev_input_len,
                model,
            )
            joint_mode = "unsupported_drafter_fallback"
        else:
            stable_draft_kv = getattr(model.ea_layer, "stable_kv", None)
        if joint_requested and stable_draft_kv is not None and hasattr(stable_draft_kv, "valid_length"):
            kv_len = int(stable_draft_kv.valid_length)
            draft_input_embeddings = ea_layer_input_embeds[:, 1:, :][:, kv_len:, :]

            def joint_forward(
                select_indices,
                hidden_states,
                draft_input_embeddings,
                prefix_length_marker,
            ):
                del prefix_length_marker
                _commit_past_key_values(
                    past_key_values_data_list,
                    select_indices,
                    prev_input_len,
                    model,
                )
                return model.ea_layer(
                    hidden_states,
                    input_embeddings=draft_input_embeddings,
                    past_key_values=stable_draft_kv,
                    use_cache=True,
                )

            joint_kwargs = {
                "select_indices": select_indices,
                "hidden_states": ea_layer_input_hiddens,
                "draft_input_embeddings": draft_input_embeddings,
                # The commit cache keeps a Python-side prefix cursor.  It is
                # part of the graph key so a graph captured at one decode
                # position is never replayed at another position.
                "prefix_length_marker": torch.tensor(
                    [int(prev_input_len)],
                    device=accepted_tokens.device,
                    dtype=torch.long,
                ),
            }
            prefetched_draft_result, joint_mode = joint_graph.run(
                joint_forward,
                joint_kwargs,
                stateful_caches=(past_key_values_data_list, stable_draft_kv),
            )
            if prefetched_draft_result is None:
                # Preserve the Phase-6 two-stream path if capture is not
                # graph-safe for this prefix/layout.
                _commit_past_key_values(
                    past_key_values_data_list,
                    select_indices,
                    prev_input_len,
                    model,
                )
            model._flagos_joint_graph_last_mode = joint_mode
        elif joint_requested:
            _commit_past_key_values(
                past_key_values_data_list,
                select_indices,
                prev_input_len,
                model,
            )
            model._flagos_joint_graph_last_mode = "eager_fallback"

    topk_kwargs = {}
    if prefetched_draft_result is not None:
        # MMModel exposes the optional prefetched-result boundary.  Other
        # drafter variants retain the original signature and use the safe
        # native path above.
        if supports_prefetch:
            topk_kwargs["prefetched_result"] = prefetched_draft_result
        else:
            prefetched_draft_result = None
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
        ea_layer_input_hiddens,
        input_tokens,
        ea_layer_input_embeds,
        model.base_model.language_model.lm_head,
        logits_processor,
        **topk_kwargs,
    )
    if overlap_enabled:
        draft_graph_mode = str(
            getattr(model, "_flagos_last_draft_graph_mode", "eager")
        )
        model._flagos_commit_draft_selected_backend = (
            "two_stream_commit_with_draft_graph"
            if draft_graph_mode in {"replay", "capture_replay"}
            else "two_stream_eager"
        )
    elif joint_requested:
        model._flagos_commit_draft_selected_backend = (
            "joint_commit_draft_graph"
            if prefetched_draft_result is not None
            else "two_stream_commit_with_draft_graph"
        )
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = augment_tree_with_kf_branches(
        model,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
    )
    if commit_done_event is not None:
        torch.cuda.current_stream(accepted_tokens.device).wait_event(commit_done_event)
    new_token += accept_length + 1
    result = (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        prompt_embeds,
        past_key_values_data_list,
        attention_mask,
    )
    return result + (False,) if return_stop else result


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)

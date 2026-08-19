"""Selective FlagGems registration used by the FlagScale KERV entrypoint."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


_FLAG_GEMS_LIBRARY = None
_FLAG_GEMS_REGISTRAR = None


def enable_embodied_ops(
    enabled: bool,
    include: str,
    backend: str,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
    strict: bool = True,
    model=None,
) -> Dict[str, Any]:
    """Install the unified FlagOS KERV operator namespace.

    The FlagScale source root is placed on ``PYTHONPATH`` by the launcher, so
    the KERV subprocess can use the same extension package as other FlagOS
    inference entries without copying kernels into the model repository.
    """

    from KERVRuntimeOptimization.embodied_ops import configure_kerv_ops

    manifest = configure_kerv_ops(
        enabled=enabled,
        include=include,
        backend=backend,
        record=record,
        record_once=record_once,
        log_path=log_path,
        manifest_path=manifest_path,
        strict=strict,
    )
    if model is not None:
        model._flagos_embodied_ops_enabled = bool(enabled)
        model._flagos_embodied_ops_include = set(manifest["requested_operators"])
        model._flagos_embodied_ops_backend = str(backend)
        model._flagos_tree_embed_pack_hits = 0
        if (
            enabled
            and str(backend).lower() == "triton"
            and "kerv_static_tree_attention" in model._flagos_embodied_ops_include
        ):
            from KERVRuntimeOptimization.embodied_ops.static_tree_attention import (
                install_static_tree_attention,
            )

            targets = install_static_tree_attention(model)
            if not targets and strict:
                raise RuntimeError(
                    "kerv_static_tree_attention requested but no compatible LlamaAttention layers were found"
                )
            manifest["static_tree_attention_targets"] = targets
            if manifest_path:
                Path(manifest_path).write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
    return manifest


def _as_jsonable(values: Iterable[Any]) -> list[str]:
    return sorted(str(value) for value in values)


def enable_flag_gems(
    enabled: bool,
    include: str,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
    rms_norm_backend: str = "flag_gems",
) -> Dict[str, Any]:
    """Register only the explicitly requested ATen operators and record the result."""
    requested_ops = [name.strip() for name in include.split(",") if name.strip()]
    manifest: Dict[str, Any] = {
        "enabled": enabled,
        "requested_ops": requested_ops,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())

    if enabled:
        try:
            global _FLAG_GEMS_LIBRARY, _FLAG_GEMS_REGISTRAR

            import flag_gems
            import triton

            if not requested_ops:
                raise ValueError("FlagGems is enabled but no operators were requested")
            if record and not log_path:
                raise ValueError("FlagGems recording requires flag_gems_log_path")

            if log_path:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            adaptive_rms_norm = rms_norm_backend == "adaptive" and "rms_norm" in requested_ops
            flag_gems_ops = [
                name for name in requested_ops if not (adaptive_rms_norm and name == "rms_norm")
            ]
            if rms_norm_backend not in ("flag_gems", "adaptive"):
                raise ValueError(f"Unsupported RMSNorm backend: {rms_norm_backend}")

            if flag_gems_ops:
                flag_gems.only_enable(
                    include=flag_gems_ops,
                    record=record,
                    once=record_once,
                    path=log_path,
                )

            registered_ops = list(flag_gems.all_registered_ops()) if flag_gems_ops else []
            registered_keys = list(flag_gems.all_registered_keys()) if flag_gems_ops else []
            if adaptive_rms_norm:
                from flag_gems.runtime.register import Register
                from KERVRuntimeOptimization.adaptive_rms_norm import (
                    configure_adaptive_recording,
                    install_llama_rms_norm_bridge,
                    rms_norm_aten_inference,
                )

                if _FLAG_GEMS_LIBRARY is not None:
                    raise RuntimeError("Adaptive operators have already been registered")
                _FLAG_GEMS_LIBRARY = torch.library.Library("aten", "IMPL")
                _FLAG_GEMS_REGISTRAR = Register(
                    (("rms_norm", rms_norm_aten_inference),),
                    lib=_FLAG_GEMS_LIBRARY,
                )
                registered_ops.append("rms_norm")
                registered_keys.append("rms_norm")
                configure_adaptive_recording(log_path, record, record_once)
                llama_rms_norm_bridge_count = install_llama_rms_norm_bridge(True)

            manifest.update(
                {
                    "flag_gems_version": getattr(flag_gems, "__version__", "unknown"),
                    "triton_version": triton.__version__,
                    "registered_ops": _as_jsonable(registered_ops),
                    "registered_keys": _as_jsonable(registered_keys),
                    "rms_norm_backend": rms_norm_backend,
                    "rms_norm_inference_only": adaptive_rms_norm,
                    "llama_rms_norm_bridge": bool(
                        adaptive_rms_norm and llama_rms_norm_bridge_count
                    ),
                    "llama_rms_norm_bridge_classes": int(
                        llama_rms_norm_bridge_count if adaptive_rms_norm else 0
                    ),
                }
            )
        except Exception as exc:
            raise RuntimeError(f"FlagGems registration failed: {exc}") from exc

    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    state = "enabled" if enabled else "disabled"
    print(f"[FlagOS/FlagGems] {state}; operators={requested_ops}", flush=True)
    return manifest


def enable_linear_fusion(
    model,
    enabled: bool,
    qkv_rows: str,
    gate_up_rows: str,
    qkv_input_sizes: str,
    gate_up_input_sizes: str,
    swiglu_rows: str,
    swiglu_input_sizes: str,
    swiglu_backend: str,
    add_rms_norm_rows: str,
    add_rms_norm_input_sizes: str,
    add_rms_norm_backend: str,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Bridge the loaded OpenVLA model to FlagOS grouped Linear fusion."""
    from KERVRuntimeOptimization.adaptive_linear_fusion import (
        enable_linear_fusion as enable_flagos_linear_fusion,
    )

    return enable_flagos_linear_fusion(
        model=model,
        enabled=enabled,
        qkv_rows=qkv_rows,
        gate_up_rows=gate_up_rows,
        qkv_input_sizes=qkv_input_sizes,
        gate_up_input_sizes=gate_up_input_sizes,
        swiglu_rows=swiglu_rows,
        swiglu_input_sizes=swiglu_input_sizes,
        swiglu_backend=swiglu_backend,
        add_rms_norm_rows=add_rms_norm_rows,
        add_rms_norm_input_sizes=add_rms_norm_input_sizes,
        add_rms_norm_backend=add_rms_norm_backend,
        record=record,
        record_once=record_once,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def enable_rotary_cache(
    model,
    enabled: bool,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Bridge the OpenVLA verifier to the FlagOS per-forward RoPE cache."""
    from KERVRuntimeOptimization.rotary_cache import enable_rotary_cache as enable_flagos_rope_cache

    return enable_flagos_rope_cache(
        model=model,
        enabled=enabled,
        record=record,
        record_once=record_once,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def enable_rotary_fusion(
    model,
    enabled: bool,
    resident_key_write: bool,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Bridge KERV verifier/drafter RoPE calls to the FlagOS fused kernel."""
    from KERVRuntimeOptimization.adaptive_rotary_fusion import (
        enable_rotary_fusion as enable_flagos_rotary_fusion,
    )

    return enable_flagos_rotary_fusion(
        model=model,
        enabled=enabled,
        resident_key_write=resident_key_write,
        record=record,
        record_once=record_once,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def enable_zero_copy_kv_return(
    enabled: bool,
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Avoid the per-layer full-cache K/V packing allocation."""
    from openvla.specdecoding.model.modeling_llama_kv import (
        enable_flagos_zero_copy_kv_return,
    )

    enable_flagos_zero_copy_kv_return(enabled)
    manifest: Dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": "flagos_kerv_zero_copy_kv_return",
        "native_path": "cat K and V into [2,B,H,S,D] per layer",
        "optimized_path": "return resident (K,V) views",
        "native_fallback": True,
        "torch_version": torch.__version__,
    }
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    state = "enabled" if enabled else "disabled"
    print(f"[FlagOS/KERV] zero-copy K/V return {state}", flush=True)
    return manifest


def enable_tree_attention_mask(
    model,
    enabled: bool,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Bridge the verifier tree mask to the FlagOS Triton construction kernel."""
    from KERVRuntimeOptimization.tree_attention_mask import (
        enable_tree_attention_mask as enable_flagos_tree_attention_mask,
    )

    return enable_flagos_tree_attention_mask(
        model=model,
        enabled=enabled,
        record=record,
        record_once=record_once,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def enable_action_head(
    model,
    enabled: bool,
    action_vocab_size: int,
    record: bool,
    record_once: bool,
    log_path: Optional[str],
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Restrict the KERV verifier LM head to the action-token vocabulary.

    The speculative drafter continues to use the original full-vocabulary
    ``lm_head``.  Only the verifier logits materialized by the OpenVLA language
    model are narrowed, so this adapter does not simultaneously change the
    drafter's cumulative full-vocabulary log-probability scores.
    """
    if not hasattr(model, "base_model") or not hasattr(model.base_model, "language_model"):
        raise TypeError("KERV action head requires a SpecVLA model with a language_model")
    language_model = model.base_model.language_model
    if not hasattr(language_model, "lm_head"):
        raise TypeError("KERV language model does not expose lm_head")

    physical_vocab_size = int(language_model.lm_head.weight.shape[0])
    # OpenVLA pads the checkpoint LM-head rows to 32064 while its logical
    # tokenizer/action vocabulary remains 32000.  Slice from the logical
    # vocabulary boundary so padded rows never enter action decoding.
    full_vocab_size = int(getattr(model, "vocab_size", physical_vocab_size))
    if full_vocab_size > physical_vocab_size:
        raise ValueError(
            f"logical vocab {full_vocab_size} exceeds LM-head rows {physical_vocab_size}"
        )
    action_vocab_size = int(action_vocab_size)
    if action_vocab_size <= 0 or action_vocab_size >= full_vocab_size:
        raise ValueError(
            f"action_vocab_size must be in (0, {full_vocab_size}), got {action_vocab_size}"
        )
    action_token_offset = full_vocab_size - action_vocab_size
    if hasattr(model, "bin_centers"):
        expected_action_vocab = int(model.bin_centers.shape[0])
        # OpenVLA uses one additional saturated boundary token next to the
        # 255 action-bin centers (the rollout reaches token 31744).  Accepting
        # bins+1 keeps that boundary in the restricted verifier vocabulary.
        if action_vocab_size not in (expected_action_vocab, expected_action_vocab + 1):
            raise ValueError(
                "Configured action vocabulary does not match OpenVLA action bins: "
                f"configured={action_vocab_size}, bins={expected_action_vocab}, "
                f"allowed=({expected_action_vocab}, {expected_action_vocab + 1})"
            )

    if record and enabled and not log_path:
        raise ValueError("Action-head recording requires log_path")
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    language_model._flagos_action_head_enabled = bool(enabled)
    language_model._flagos_action_vocab_size = action_vocab_size
    language_model._flagos_action_logit_offset = action_token_offset
    language_model._flagos_action_head_record = bool(record)
    language_model._flagos_action_head_record_once = bool(record_once)
    language_model._flagos_action_head_log_path = str(log_path) if log_path else None
    language_model._flagos_action_head_hit_count = 0
    model._flagos_action_head_enabled = bool(enabled)
    model._flagos_action_vocab_size = action_vocab_size
    model._flagos_action_logit_offset = action_token_offset if enabled else 0

    manifest: Dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": "flagos_kerv_verifier_action_lm_head",
        "scope": "verifier_only",
        "drafter_full_vocabulary_preserved": True,
        "full_vocab_size": full_vocab_size,
        "physical_lm_head_rows": physical_vocab_size,
        "action_vocab_size": action_vocab_size,
        "action_token_offset": action_token_offset,
        "materialized_logit_reduction": full_vocab_size / action_vocab_size,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    state = "enabled" if enabled else "disabled"
    print(
        f"[FlagOS/KERV] verifier action head {state}; "
        f"vocab={full_vocab_size}->{action_vocab_size} offset={action_token_offset}",
        flush=True,
    )
    return manifest


def enable_w8a16_quantization(
    model,
    enabled: bool,
    groups: str = "verify_mlp,verify_attention,vision_mlp,draft_linear",
    min_speedup: float = 1.05,
    strict: bool = False,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the optional accuracy-gated W8A16 path.

    Quantization is deliberately opt-in.  A model is never modified when the
    backend is unavailable (or its compiled extension does not match the
    running PyTorch), which keeps the BF16 path bit-for-bit unchanged.  The
    selected groups are recorded so the 30/200/1000 episode gate can promote
    only a measured configuration.
    """
    requested = tuple(
        sorted({item.strip() for item in str(groups).split(",") if item.strip()})
    )
    manifest: Dict[str, Any] = {
        "enabled": bool(enabled),
        "requested_groups": list(requested),
        "selected_groups": [],
        "implementation": "torchao_int8_weight_only_w8a16",
        "min_local_speedup": float(min_speedup),
        "precision_gate": "trajectory_and_p95",
        "fallback": True,
        "torch_version": torch.__version__,
    }
    if not enabled:
        manifest["reason"] = "disabled_by_default_bf16_safe_path"
    else:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for W8A16 inference")
            import torchao  # noqa: F401
            from torchao.quantization import Int8WeightOnlyConfig, quantize_

            # TorchAO wheels with an incompatible C++ extension import but
            # fail at the first quantization call.  Probe it before mutating
            # any module; this is important for strict accuracy semantics.
            try:
                import torchao._C as _torchao_c  # noqa: F401
            except Exception as exc:
                raise RuntimeError(f"TorchAO C++ backend unavailable: {exc}") from exc

            wanted = set(requested)

            def _group_for(name: str) -> Optional[str]:
                lowered = name.lower()
                if "ea_layer" in lowered or "drafter" in lowered:
                    return "draft_linear"
                if "vision" in lowered and ".mlp." in lowered:
                    return "vision_mlp"
                if ".mlp." in lowered:
                    return "verify_mlp"
                if "self_attn" in lowered or "attention" in lowered:
                    return "verify_attention"
                return None

            def _filter(module, name: str) -> bool:
                return isinstance(module, torch.nn.Linear) and _group_for(name) in wanted

            quantize_(model, Int8WeightOnlyConfig(), filter_fn=_filter)
            selected = sorted(
                {
                    group
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Module)
                    and _group_for(name) in wanted
                }
            )
            if not selected:
                raise RuntimeError("no compatible Linear modules matched W8A16 groups")
            manifest.update(
                {
                    "selected_groups": selected,
                    "fallback": False,
                    "reason": "backend_available; trajectory_gate_required",
                }
            )
            model._flagos_w8a16_selected_groups = tuple(selected)
            model._flagos_w8a16_min_speedup = float(min_speedup)
        except Exception as exc:
            manifest["reason"] = str(exc)
            if strict:
                raise RuntimeError(f"W8A16 requested but unavailable: {exc}") from exc

    model._flagos_w8a16_enabled = bool(enabled and not manifest["fallback"])
    model._flagos_w8a16_manifest = manifest
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[FlagOS/KERV] W8A16 {'selected' if model._flagos_w8a16_enabled else 'BF16 fallback'}; "
        f"groups={list(requested)} reason={manifest.get('reason', '')}",
        flush=True,
    )
    return manifest


def enable_draft_logsoftmax_topk(
    enabled: bool,
    manifest_path: Optional[str],
) -> Dict[str, Any]:
    """Fuse the Drafter's full-vocabulary LogSoftmax and TopK reduction."""
    from openvla.specdecoding.model.cnets import (
        enable_flagos_draft_logsoftmax_topk,
    )

    enable_flagos_draft_logsoftmax_topk(enabled)
    manifest: Dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": "flagos_kerv_two_stage_logsoftmax_topk",
        "scope": "drafter_full_vocabulary",
        "supported_k": 8,
        "supported_vocabulary": [16384, 65536],
        "native_fallback": True,
        "inference_only": True,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[FlagOS/KERV] Draft LogSoftmax-TopK fusion "
        f"{'enabled' if enabled else 'disabled'}",
        flush=True,
    )
    return manifest


def enable_tree_builder(
    model,
    enabled: bool,
    manifest_path: Optional[str],
    static_runtime: bool = False,
    verify_max_paths: int = 0,
    max_depth: int = 0,
    compact_tree_mode: str = "off",
    compact_tree_min_confidence: float = 0.0,
    compact_tree_full_max_paths: int = 0,
    compact_tree_full_max_depth: int = 0,
    compact_tree_tight_max_paths: int = 0,
    compact_tree_tight_max_depth: int = 0,
    compact_tree_tight_min_confidence: float = float("inf"),
    cache_topology: bool = True,
    operatorized: bool = False,
    draft_tree_operatorized: bool = False,
    verify_accept_operatorized: bool = False,
    node_buckets: str = "",
    compile_verifier: bool = False,
    compile_mode: str = "reduce-overhead",
    cuda_graph_max_entries: int = 1,
    cuda_graph_capture_past_length: int = -1,
    kalman_batch: bool = False,
    kv_commit_batched: bool = False,
    fused_control_transfer: bool = False,
    persistent_runtime: bool = False,
    resident_kv_cache: bool = False,
    persistent_prefix_capacity: int = 288,
    persistent_tree_capacity: int = 256,
    exact_node_templates: bool = False,
    fixed_workspace_layout: bool = False,
    prewarm_graph_buckets: bool = False,
    persistent_input_buffers: bool = False,
    static_tree_attention: str = "off",
) -> Dict[str, Any]:
    """Enable batched construction and profile-guided verification pruning."""
    if not hasattr(model, "get_action_dim"):
        raise TypeError("KERV tree builder requires a speculative action model")
    if static_runtime and not enabled:
        raise ValueError("Static tree runtime requires the batched tree builder")
    if verify_max_paths < 0:
        raise ValueError("verify_max_paths must be non-negative")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    compact_tree_mode = str(compact_tree_mode).lower()
    # Draccus/YAML 1.1 may deserialize the textual value ``off`` as a bool
    # before assigning it to this string field.
    if compact_tree_mode == "false":
        compact_tree_mode = "off"
    if compact_tree_mode not in {"off", "auto", "on"}:
        raise ValueError("compact_tree_mode must be one of: off, auto, on")
    if compact_tree_mode == "on" and verify_max_paths <= 0:
        raise ValueError("compact_tree_mode=on requires verify_max_paths > 0")
    if compact_tree_mode != "off" and not static_runtime:
        raise ValueError("Compact verification trees require static_runtime")
    if compact_tree_min_confidence < 0:
        raise ValueError("compact_tree_min_confidence must be non-negative")
    if compact_tree_full_max_paths < 0 or compact_tree_full_max_depth < 0:
        raise ValueError("Full fallback tree dimensions must be non-negative")
    if compact_tree_full_max_depth and not compact_tree_full_max_paths:
        raise ValueError("Full fallback depth requires a positive path count")
    if compact_tree_tight_max_paths < 0 or compact_tree_tight_max_depth < 0:
        raise ValueError("Tight compact-tree dimensions must be non-negative")
    if compact_tree_tight_min_confidence < compact_tree_min_confidence:
        raise ValueError(
            "Tight compact-tree confidence must be no lower than the medium threshold"
        )
    if compact_tree_tight_max_paths and not compact_tree_tight_max_depth:
        raise ValueError("Tight compact-tree paths require a positive depth")
    if compact_tree_tight_max_depth and not compact_tree_tight_max_paths:
        raise ValueError("Tight compact-tree depth requires a positive path count")
    if operatorized and not static_runtime:
        raise ValueError("Operatorized tree construction requires static_runtime")
    if draft_tree_operatorized and not operatorized:
        raise ValueError("Draft-tree operatorization requires the registered tree operator")
    node_buckets = "" if node_buckets is None else str(node_buckets)
    node_bucket_values = tuple(
        value.strip()
        for value in node_buckets.split(",")
        if value.strip() and value.strip().lower() not in {"none", "null"}
    )
    parsed_buckets = tuple(
        sorted(
            set(
                int(value.strip())
                for value in node_bucket_values
            )
        )
    )
    if any(value <= 0 for value in parsed_buckets):
        raise ValueError(f"node_buckets must be positive: {parsed_buckets}")
    if parsed_buckets and not static_runtime:
        raise ValueError("Node bucketing requires static_runtime")
    if compile_verifier and not parsed_buckets:
        raise ValueError("Verifier compilation requires fixed node buckets")
    if resident_kv_cache and not enabled:
        raise ValueError("Resident KV cache requires the KERV tree builder")
    if persistent_runtime or resident_kv_cache:
        if persistent_prefix_capacity <= 0 or persistent_tree_capacity <= 0:
            raise ValueError("Persistent capacities must be positive")
    if persistent_runtime:
        if (
            not static_runtime
            or not compile_verifier
            or compile_mode not in {"cuda-graph", "inductor-cuda-graph"}
        ):
            raise ValueError(
                "Persistent runtime requires static_runtime and a CUDA-graph verifier mode"
            )
    static_tree_attention = str(static_tree_attention).lower()
    if static_tree_attention not in {"off", "auto", "triton"}:
        raise ValueError(
            "static_tree_attention must be one of: off, auto, triton"
        )
    if fixed_workspace_layout:
        if not persistent_runtime or int(persistent_prefix_capacity) != 288:
            raise ValueError(
                "Fixed KERV workspace requires persistent_runtime and "
                "persistent_prefix_capacity=288"
            )
    if prewarm_graph_buckets and not compile_verifier:
        raise ValueError("Graph bucket prewarm requires verifier compilation")
    if persistent_input_buffers and not persistent_runtime:
        raise ValueError("Persistent verifier inputs require persistent_runtime")
    if static_tree_attention != "off":
        embodied_include = set(getattr(model, "_flagos_embodied_ops_include", set()))
        if "kerv_static_tree_attention" not in embodied_include:
            raise ValueError(
                "Static tree attention requires kerv_static_tree_attention in embodied_ops.include"
            )
        if not parsed_buckets or max(parsed_buckets) > int(persistent_tree_capacity):
            raise ValueError(
                "Persistent runtime requires static tree buckets no larger than "
                "persistent_tree_capacity"
            )
    if cuda_graph_max_entries <= 0:
        raise ValueError("cuda_graph_max_entries must be positive")
    if compile_mode not in (
        "default",
        "reduce-overhead",
        "max-autotune",
        "cuda-graph",
        "inductor-cuda-graph",
    ):
        raise ValueError(f"Unsupported compile_mode: {compile_mode}")
    if operatorized:
        from openvla.experiments.robot.flagos_tree_ops import (
            load_flagos_kerv_tree_ops,
        )

        # Registration is mandatory for this mode. A build/registration error
        # aborts model startup instead of silently returning to Python loops.
        load_flagos_kerv_tree_ops()
    if verify_accept_operatorized:
        from openvla.experiments.robot.flagos_verify_ops import (
            load_flagos_kerv_verify_ops,
        )

        load_flagos_kerv_verify_ops()
    model._flagos_tree_builder_enabled = bool(enabled)
    model._flagos_tree_builder_hit_count = 0
    model._flagos_static_tree_runtime_enabled = bool(static_runtime)
    model._flagos_tree_verify_max_paths = int(verify_max_paths)
    model._flagos_tree_max_depth = int(max_depth)
    model._flagos_compact_tree_mode = compact_tree_mode
    model._flagos_compact_tree_min_confidence = float(compact_tree_min_confidence)
    model._flagos_compact_tree_full_max_paths = int(compact_tree_full_max_paths)
    model._flagos_compact_tree_full_max_depth = int(compact_tree_full_max_depth)
    model._flagos_compact_tree_tight_max_paths = int(compact_tree_tight_max_paths)
    model._flagos_compact_tree_tight_max_depth = int(compact_tree_tight_max_depth)
    model._flagos_compact_tree_tight_min_confidence = float(
        compact_tree_tight_min_confidence
    )
    model._flagos_compact_tree_active = compact_tree_mode == "on"
    model._flagos_compact_tree_confidence = None
    model._flagos_compact_tree_level = "medium" if compact_tree_mode == "on" else "full"
    model._flagos_tree_cache_topology = bool(cache_topology)
    model._flagos_tree_operator_enabled = bool(operatorized)
    model._flagos_tree_operator_build_hits = 0
    model._flagos_tree_operator_pack_hits = 0
    model._flagos_tree_operator_build_time_s = 0.0
    # The Draft model owns the first 49-node candidate tree. Propagate the
    # operator mode so its base mask/retrieval construction uses the same
    # registered C++ implementation as the Kalman-augmented verification tree.
    model.ea_layer._flagos_draft_tree_operator_enabled = bool(draft_tree_operatorized)
    model.ea_layer._flagos_compact_tree_mode = compact_tree_mode
    model.ea_layer._flagos_compact_tree_min_confidence = float(
        compact_tree_min_confidence
    )
    model.ea_layer._flagos_compact_tree_full_max_paths = int(
        compact_tree_full_max_paths
    )
    model.ea_layer._flagos_compact_tree_full_max_depth = int(
        compact_tree_full_max_depth
    )
    model.ea_layer._flagos_compact_tree_tight_max_paths = int(
        compact_tree_tight_max_paths
    )
    model.ea_layer._flagos_compact_tree_tight_max_depth = int(
        compact_tree_tight_max_depth
    )
    model.ea_layer._flagos_compact_tree_tight_min_confidence = float(
        compact_tree_tight_min_confidence
    )
    model.ea_layer._flagos_compact_tree_active = compact_tree_mode == "on"
    model.ea_layer._flagos_compact_tree_confidence = None
    model.ea_layer._flagos_compact_tree_level = (
        "medium" if compact_tree_mode == "on" else "full"
    )
    model.ea_layer._flagos_compact_tree_gate_evaluations = 0
    model.ea_layer._flagos_compact_tree_full_selections = 0
    model.ea_layer._flagos_compact_tree_medium_selections = 0
    model.ea_layer._flagos_compact_tree_tight_selections = 0
    model.ea_layer._flagos_draft_tree_operator_hits = 0
    model._flagos_verify_accept_operator_enabled = bool(verify_accept_operatorized)
    model._flagos_verify_accept_operator_hits = 0
    model._flagos_verify_accept_aten_hits = 0
    model._flagos_verify_accept_selected_backend = None
    model._flagos_verify_accept_native_ms = None
    model._flagos_verify_accept_triton_ms = None
    model._flagos_tree_node_buckets = parsed_buckets
    model._flagos_bucket_first_verify_only = bool(
        compile_verifier and compile_mode == "cuda-graph" and not persistent_runtime
    )
    model._flagos_tree_use_node_buckets = True
    model._flagos_compile_verifier_enabled = bool(compile_verifier)
    model._flagos_compile_verifier_mode = str(compile_mode)
    model._flagos_compiled_verifier_forward = None
    model._flagos_inductor_verifier_forward = None
    if compile_verifier and compile_mode == "inductor-cuda-graph":
        # Inductor performs fixed-shape fusion and GEMM autotuning.  Its own
        # CUDA graphs are disabled because the resulting kernels are captured
        # by KERV's persistent verifier graph below.
        model._flagos_inductor_verifier_forward = torch.compile(
            model.forward,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=False,
        )
    model._flagos_inductor_verifier_graph_captures = 0
    model._flagos_inductor_verifier_fallbacks = 0
    model._flagos_inductor_verifier_last_error = None
    model._flagos_compiled_verifier_hits = 0
    model._flagos_verifier_cuda_graph_cache = {}
    effective_graph_entries = int(cuda_graph_max_entries)
    if fixed_workspace_layout and parsed_buckets:
        # Prefix length is deliberately absent from the fixed-workspace graph
        # signature, so one capture per tree bucket is sufficient.
        effective_graph_entries = min(
            effective_graph_entries, len(parsed_buckets)
        )
    model._flagos_cuda_graph_max_entries = effective_graph_entries
    model._flagos_cuda_graph_capture_past_length = int(
        cuda_graph_capture_past_length
    )
    model._flagos_cuda_graph_captures = 0
    model._flagos_cuda_graph_replay_hits = 0
    model._flagos_cuda_graph_fallbacks = 0
    model._flagos_cuda_graph_audit_eager = 0
    model._flagos_static_tree_attention_audit_signature = None
    model._flagos_cuda_graph_capture_overhead_s = 0.0
    model._flagos_last_cuda_graph_mode = "disabled"
    model._flagos_kalman_batch_enabled = bool(kalman_batch)
    model._flagos_kalman_batch_hits = 0
    model._flagos_kalman_batch_chains = 0
    model._flagos_kv_commit_batched_enabled = bool(kv_commit_batched)
    model._flagos_kv_commit_batched_hits = 0
    model._flagos_fused_control_transfer_enabled = bool(fused_control_transfer)
    model._flagos_fused_control_transfer_hits = 0
    model._flagos_persistent_runtime_enabled = bool(persistent_runtime)
    model._flagos_resident_kv_cache_enabled = bool(resident_kv_cache)
    model._flagos_persistent_prefix_capacity = int(persistent_prefix_capacity)
    model._flagos_persistent_tree_capacity = int(persistent_tree_capacity)
    model._flagos_exact_node_templates = bool(exact_node_templates)
    model._flagos_fixed_workspace_layout = bool(fixed_workspace_layout)
    model._flagos_prewarm_graph_buckets = bool(prewarm_graph_buckets)
    model._flagos_persistent_input_buffers = bool(persistent_input_buffers)
    model._flagos_static_tree_attention_mode = static_tree_attention
    model.base_model.language_model._flagos_static_tree_attention_runtime_enabled = (
        static_tree_attention != "off"
    )
    model._flagos_static_tree_attention_gate_evaluated = False
    model._flagos_static_tree_attention_gate_samples_ms = []
    model._flagos_static_tree_attention_gate_median_ms = None
    model._flagos_static_tree_attention_gate_threshold_ms = 25.5
    model._flagos_persistent_kv_cache = None
    model._flagos_persistent_allocations = 0
    model._flagos_persistent_kv_commit_hits = 0
    model._flagos_static_tree_hit_count = 0
    model._flagos_tree_cache_hits = 0
    model._flagos_tree_cache_misses = 0
    model._flagos_static_tree_cache = {}
    model._flagos_static_tree_last_stats = {}
    if static_tree_attention != "off":
        from KERVRuntimeOptimization.embodied_ops.static_tree_attention import (
            install_static_tree_attention,
        )

        targets = install_static_tree_attention(model)
        if not targets:
            raise RuntimeError(
                "Static tree attention requested but no compatible LlamaAttention layers were found"
            )
    manifest: Dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": (
            "flagos_kerv_profiled_fixed_width_tree"
            if static_runtime
            else "flagos_kerv_batched_tree_builder"
        ),
        "semantic_mode": (
            "confidence_gated_compact"
            if compact_tree_mode != "off"
            else ("profile_pruned" if verify_max_paths > 0 else "exact")
        ),
        "host_tree_construction": True,
        "single_token_append": False,
        "single_buffer_transfer": True,
        "static_runtime": bool(static_runtime),
        "verify_max_paths": int(verify_max_paths),
        "max_depth": int(max_depth),
        "compact_tree_mode": compact_tree_mode,
        "compact_tree_min_confidence": float(compact_tree_min_confidence),
        "compact_tree_full_max_paths": int(compact_tree_full_max_paths),
        "compact_tree_full_max_depth": int(compact_tree_full_max_depth),
        "compact_tree_tight_max_paths": int(compact_tree_tight_max_paths),
        "compact_tree_tight_max_depth": int(compact_tree_tight_max_depth),
        "compact_tree_tight_min_confidence": float(
            compact_tree_tight_min_confidence
        ),
        "cache_topology": bool(cache_topology),
        "fixed_verification_width": bool(static_runtime and verify_max_paths > 0),
        "ancestor_closed_subtree": bool(static_runtime),
        "operatorized": bool(operatorized),
        "operator_namespace": "flagos_kerv" if operatorized else None,
        "operators": (
            ["build_verification_subtree", "pack_verification_tokens"]
            if operatorized
            else []
        ),
        "tree_metadata_backend": "cpp_cpu" if operatorized else "python_cpu",
        "draft_tree_operatorized": bool(draft_tree_operatorized),
        "draft_tree_metadata_backend": (
            "cpp_cpu" if draft_tree_operatorized else "python_cpu"
        ),
        "verification_pack_backend": "aten_index_select" if operatorized else "aten_eager",
        "verify_accept_operatorized": bool(verify_accept_operatorized),
        "verify_accept_backend": "adaptive_triton_or_aten" if verify_accept_operatorized else "aten_eager",
        "node_buckets": list(parsed_buckets),
        "compile_verifier": bool(compile_verifier),
        "compile_mode": str(compile_mode),
        "cuda_graph_max_entries": effective_graph_entries,
        "configured_cuda_graph_max_entries": int(cuda_graph_max_entries),
        "cuda_graph_capture_past_length": int(cuda_graph_capture_past_length),
        "cuda_graph_kv_copy_backend": "aten_foreach_copy",
        "kalman_batch": bool(kalman_batch),
        "kv_commit_batched": bool(kv_commit_batched),
        "fused_control_transfer": bool(fused_control_transfer),
        "persistent_runtime": bool(persistent_runtime),
        "resident_kv_cache": bool(resident_kv_cache),
        "persistent_prefix_capacity": int(persistent_prefix_capacity),
        "persistent_tree_capacity": int(persistent_tree_capacity),
        "exact_node_templates": bool(exact_node_templates),
        "fixed_workspace_layout": bool(fixed_workspace_layout),
        "prewarm_graph_buckets": bool(prewarm_graph_buckets),
        "persistent_input_buffers": bool(persistent_input_buffers),
        "static_tree_attention": static_tree_attention,
        "persistent_kv_backend": "gpu_resident_fixed_address",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    state = "enabled" if enabled else "disabled"
    print(
        f"[FlagOS/KERV] tree builder {state}; static={bool(static_runtime)} "
        f"verify_max_paths={int(verify_max_paths)} cache={bool(cache_topology)} "
        f"operatorized={bool(operatorized)}",
        flush=True,
    )
    return manifest

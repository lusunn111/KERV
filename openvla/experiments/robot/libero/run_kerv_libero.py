"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import json
import wandb
import torch
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.flagos_ops import (
    enable_action_head,
    enable_draft_logsoftmax_topk,
    enable_embodied_ops,
    enable_flag_gems,
    enable_linear_fusion,
    enable_w8a16_quantization,
    enable_rotary_cache,
    enable_rotary_fusion,
    enable_zero_copy_kv_return,
    enable_tree_attention_mask,
    enable_tree_builder,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.tool_utils import compute_dynamic_threshold
from openvla.prismatic.extern.hf.flagos_system_runtime import (
    apply_numa_affinity,
    schedule_episode_device_warmup,
    wait_episode_device_warmup,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_spec: bool = True
    parallel_draft: bool = False
    accept_threshold: int = 9
    max_planning_steps: int = 300
    accept_threshold_start: int = 15
    accept_threshold_min: int = 3
    threshold_decay: str = "linear"
    use_kalman_fallback: bool = True
    use_kalman_tree: bool = True
    kalman_process_var: float = 1.0
    kalman_measurement_var: float = 1e-3
    kalman_history_window: Optional[int] = None
    store_action_history: bool = True
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Optional[Union[str, Path]] = None
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                    # Number of rollouts per task
    max_tasks: Optional[int] = None                  # Optional short-run cap for deployment smoke tests
    max_episode_steps: Optional[int] = None          # Optional policy-step cap for deployment smoke tests

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    collect_spec_errors: bool = False                # Whether to log speculative decoding mismatches
    spec_error_output: Optional[str] = None          # Optional explicit path to save mismatch stats
    spec_history_output: Optional[str] = None        # Optional explicit path to save accepted-actions history
    runtime_stats_path: Optional[str] = None         # Optional per-action FlagOS KERV profile JSONL
    flagos_timing_backend: str = "cpu"              # cpu or cuda_event
    flagos_async_runtime_logging: bool = False       # write JSONL off the action path

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # FlagOS/FlagGems selective operator integration
    use_flag_gems: bool = False
    flag_gems_include: str = "rms_norm"
    flag_gems_record: bool = True
    flag_gems_record_once: bool = True
    flag_gems_log_path: Optional[str] = None
    flag_gems_manifest_path: Optional[str] = None
    flag_gems_rms_norm_backend: str = "flag_gems"

    # Unified FlagOS embodied operator namespace.  Individual model bridges
    # remain opt-in below; this switch only registers the public operators.
    use_flagos_embodied_ops: bool = False
    flagos_embodied_ops_include: str = ""
    flagos_embodied_ops_backend: str = "auto"
    flagos_embodied_ops_record: bool = False
    flagos_embodied_ops_record_once: bool = True
    flagos_embodied_ops_log_path: Optional[str] = None
    flagos_embodied_ops_manifest_path: Optional[str] = None
    flagos_embodied_ops_strict: bool = True

    # FlagOS grouped Linear fusion. Only allowlisted flattened row counts use
    # the packed GEMM; all dynamic decoding shapes retain the native path.
    use_flagos_linear_fusion: bool = False
    flagos_qkv_fusion_rows: str = ""
    flagos_gate_up_fusion_rows: str = ""
    flagos_qkv_fusion_input_sizes: str = ""
    flagos_gate_up_fusion_input_sizes: str = ""
    flagos_swiglu_fusion_rows: str = ""
    flagos_swiglu_fusion_input_sizes: str = ""
    flagos_swiglu_fusion_backend: str = "native_inplace"
    flagos_add_rms_norm_fusion_rows: str = ""
    flagos_add_rms_norm_fusion_input_sizes: str = ""
    flagos_add_rms_norm_fusion_backend: str = "native_inplace"
    flagos_linear_fusion_record: bool = True
    flagos_linear_fusion_record_once: bool = True
    flagos_linear_fusion_log_path: Optional[str] = None
    flagos_linear_fusion_manifest_path: Optional[str] = None
    use_flagos_rope_cache: bool = False
    flagos_rope_cache_record: bool = True
    flagos_rope_cache_record_once: bool = True
    flagos_rope_cache_log_path: Optional[str] = None
    flagos_rope_cache_manifest_path: Optional[str] = None
    use_flagos_rope_fusion: bool = False
    flagos_rope_resident_key_write: bool = True
    flagos_rope_fusion_record: bool = True
    flagos_rope_fusion_record_once: bool = True
    flagos_rope_fusion_log_path: Optional[str] = None
    flagos_rope_fusion_manifest_path: Optional[str] = None
    use_flagos_zero_copy_kv_return: bool = False
    flagos_zero_copy_kv_manifest_path: Optional[str] = None
    use_flagos_tree_attention_mask: bool = False
    flagos_tree_attention_mask_record: bool = True
    flagos_tree_attention_mask_record_once: bool = True
    flagos_tree_attention_mask_log_path: Optional[str] = None
    flagos_tree_attention_mask_manifest_path: Optional[str] = None

    # KERV-specific FlagOS verifier head. The drafter deliberately retains its
    # original full-vocabulary scoring so this switch changes one path at a time.
    use_flagos_action_head: bool = False
    flagos_action_vocab_size: int = 256
    flagos_action_head_record: bool = True
    flagos_action_head_record_once: bool = True
    flagos_action_head_log_path: Optional[str] = None
    flagos_action_head_manifest_path: Optional[str] = None
    # Accuracy-gated W8A16 is experimental and remains disabled on the BF16
    # safe path.  The child writes a manifest even when it falls back.
    use_flagos_w8a16: bool = False
    flagos_w8a16_groups: str = "verify_mlp,verify_attention,vision_mlp,draft_linear"
    flagos_w8a16_min_speedup: float = 1.05
    flagos_w8a16_strict: bool = False
    flagos_w8a16_manifest_path: Optional[str] = None
    use_flagos_draft_logsoftmax_topk: bool = False
    flagos_draft_logsoftmax_topk_manifest_path: Optional[str] = None
    use_flagos_tree_builder: bool = False
    flagos_tree_builder_manifest_path: Optional[str] = None
    flagos_tree_static_runtime: bool = False
    flagos_tree_verify_max_paths: int = 0
    flagos_tree_max_depth: int = 0
    flagos_compact_tree_mode: str = "off"       # off | auto | on
    flagos_compact_tree_min_confidence: float = 0.0
    flagos_compact_tree_full_max_paths: int = 0
    flagos_compact_tree_full_max_depth: int = 0
    flagos_compact_tree_tight_max_paths: int = 0
    flagos_compact_tree_tight_max_depth: int = 0
    flagos_compact_tree_tight_min_confidence: float = float("inf")
    flagos_tree_cache_topology: bool = True
    flagos_tree_operatorized: bool = False
    flagos_draft_tree_operatorized: bool = False
    flagos_verify_accept_operatorized: bool = False
    flagos_tree_node_buckets: str = ""
    flagos_compile_verifier: bool = False
    flagos_compile_verifier_mode: str = "reduce-overhead"
    flagos_cuda_graph_max_entries: int = 1
    flagos_cuda_graph_capture_past_length: int = -1
    flagos_kalman_batch: bool = False
    flagos_kv_commit_batched: bool = False
    flagos_fused_control_transfer: bool = False
    flagos_persistent_runtime: bool = False
    flagos_resident_kv_cache: bool = False
    flagos_persistent_prefix_capacity: int = 288
    flagos_persistent_tree_capacity: int = 256
    flagos_exact_node_templates: bool = False
    flagos_fixed_workspace_layout: bool = False
    flagos_prewarm_graph_buckets: bool = False
    flagos_persistent_input_buffers: bool = False
    flagos_static_tree_attention: str = "off"
    # Optional benchmark hygiene: raise the GPU to its steady clock after the
    # long checkpoint/environment setup. This work is outside action timing.
    flagos_cuda_warmup_seconds: float = 0.0

    # Fourth-stage lossless runtime path.  It changes only allocation,
    # timing, and control-buffer plumbing; all model/operator routes remain
    # independently gated by their existing options.
    flagos_system_optimization: bool = False
    flagos_prompt_cuda_graph: str = "off"             # off | auto | on
    flagos_draft_cuda_graph: str = "off"              # off | auto | on
    flagos_shared_graph_pool: bool = True
    flagos_process_level_warmup: bool = False
    flagos_persistent_control_buffer: bool = False
    flagos_persistent_decode_workspace: bool = False
    flagos_async_logs: bool = False
    flagos_verifier_gate_during_warmup: bool = False
    flagos_prompt_token_cache: bool = False
    flagos_pinned_image_buffer: bool = False
    flagos_async_image_transfer: bool = False
    flagos_persistent_draft_kv_cache: bool = False
    flagos_draft_graph_accept_buckets: str = "1,2,3,4,5,6,7"
    flagos_commit_draft_cuda_graph: str = "off"      # off | auto | on
    flagos_commit_draft_stream_overlap: str = "off"  # off | auto | on
    flagos_task_level_graph_warmup: bool = False
    flagos_inference_mode: bool = False
    flagos_allocator_backend: str = "auto"
    # Phase-7 system-only paths.  They do not alter model/operator semantics.
    flagos_episode_device_warmup: str = "off"       # off | auto | on
    flagos_episode_warmup_duration_ms: str = "auto"
    flagos_joint_commit_draft_graph: str = "off"    # off | auto | on
    flagos_joint_graph_accept_buckets: str = "1,2,3,4,5,6,7"
    flagos_fixed_draft_input_workspace: bool = False
    flagos_numa_affinity: str = "off"               # off | auto | on

    # fmt: on


def warmup_cuda_for_measurement(seconds: float) -> None:
    """Warm CUDA clocks after model loading without changing model state."""
    seconds = float(seconds)
    if seconds <= 0.0 or not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    left = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    right = torch.randn_like(left)
    torch.cuda.synchronize(device)
    begin = time.perf_counter()
    iterations = 0
    while True:
        for _ in range(20):
            torch.mm(left, right)
            iterations += 1
        torch.cuda.synchronize(device)
        if time.perf_counter() - begin >= seconds:
            break
    print(
        f"[FlagOS/KERV] CUDA measurement warmup {seconds:.2f}s "
        f"({iterations} GEMMs)",
        flush=True,
    )


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Pin the launcher process before model construction and pinned-buffer
    # allocation.  Failure to read topology is intentionally non-fatal.
    applied_affinity = apply_numa_affinity(cfg.flagos_numa_affinity)
    if applied_affinity:
        print(f"[FlagOS/KERV] CPU affinity: {applied_affinity}", flush=True)

    # Register FlagOS operators before constructing the model or executing a forward pass.
    enable_flag_gems(
        enabled=cfg.use_flag_gems,
        include=cfg.flag_gems_include,
        record=cfg.flag_gems_record,
        record_once=cfg.flag_gems_record_once,
        log_path=cfg.flag_gems_log_path,
        manifest_path=cfg.flag_gems_manifest_path,
        rms_norm_backend=cfg.flag_gems_rms_norm_backend,
    )
    enable_embodied_ops(
        enabled=cfg.use_flagos_embodied_ops,
        include=cfg.flagos_embodied_ops_include,
        backend=cfg.flagos_embodied_ops_backend,
        record=cfg.flagos_embodied_ops_record,
        record_once=cfg.flagos_embodied_ops_record_once,
        log_path=cfg.flagos_embodied_ops_log_path,
        manifest_path=cfg.flagos_embodied_ops_manifest_path,
        strict=cfg.flagos_embodied_ops_strict,
    )

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # Attach the operator selection to the loaded model so verifier helpers
    # can route only the explicitly enabled KERV paths through flagos_embodied.
    enable_embodied_ops(
        enabled=cfg.use_flagos_embodied_ops,
        include=cfg.flagos_embodied_ops_include,
        backend=cfg.flagos_embodied_ops_backend,
        record=cfg.flagos_embodied_ops_record,
        record_once=cfg.flagos_embodied_ops_record_once,
        log_path=cfg.flagos_embodied_ops_log_path,
        manifest_path=cfg.flagos_embodied_ops_manifest_path,
        strict=cfg.flagos_embodied_ops_strict,
        model=model,
    )

    # The verifier head is configured after loading, when the exact OpenVLA
    # vocabulary and action-bin count are available.
    enable_action_head(
        model=model,
        enabled=cfg.use_flagos_action_head,
        action_vocab_size=cfg.flagos_action_vocab_size,
        record=cfg.flagos_action_head_record,
        record_once=cfg.flagos_action_head_record_once,
        log_path=cfg.flagos_action_head_log_path,
        manifest_path=cfg.flagos_action_head_manifest_path,
    )
    enable_w8a16_quantization(
        model=model,
        enabled=cfg.use_flagos_w8a16,
        groups=cfg.flagos_w8a16_groups,
        min_speedup=cfg.flagos_w8a16_min_speedup,
        strict=cfg.flagos_w8a16_strict,
        manifest_path=cfg.flagos_w8a16_manifest_path,
    )
    enable_draft_logsoftmax_topk(
        enabled=cfg.use_flagos_draft_logsoftmax_topk,
        manifest_path=cfg.flagos_draft_logsoftmax_topk_manifest_path,
    )
    enable_tree_builder(
        model=model,
        enabled=cfg.use_flagos_tree_builder,
        manifest_path=cfg.flagos_tree_builder_manifest_path,
        static_runtime=cfg.flagos_tree_static_runtime,
        verify_max_paths=cfg.flagos_tree_verify_max_paths,
        max_depth=cfg.flagos_tree_max_depth,
        compact_tree_mode=cfg.flagos_compact_tree_mode,
        compact_tree_min_confidence=cfg.flagos_compact_tree_min_confidence,
        compact_tree_full_max_paths=cfg.flagos_compact_tree_full_max_paths,
        compact_tree_full_max_depth=cfg.flagos_compact_tree_full_max_depth,
        compact_tree_tight_max_paths=cfg.flagos_compact_tree_tight_max_paths,
        compact_tree_tight_max_depth=cfg.flagos_compact_tree_tight_max_depth,
        compact_tree_tight_min_confidence=cfg.flagos_compact_tree_tight_min_confidence,
        cache_topology=cfg.flagos_tree_cache_topology,
        operatorized=cfg.flagos_tree_operatorized,
        draft_tree_operatorized=cfg.flagos_draft_tree_operatorized,
        verify_accept_operatorized=cfg.flagos_verify_accept_operatorized,
        node_buckets=cfg.flagos_tree_node_buckets,
        compile_verifier=cfg.flagos_compile_verifier,
        compile_mode=cfg.flagos_compile_verifier_mode,
        cuda_graph_max_entries=cfg.flagos_cuda_graph_max_entries,
        cuda_graph_capture_past_length=cfg.flagos_cuda_graph_capture_past_length,
        kalman_batch=cfg.flagos_kalman_batch,
        kv_commit_batched=cfg.flagos_kv_commit_batched,
        fused_control_transfer=cfg.flagos_fused_control_transfer,
        persistent_runtime=cfg.flagos_persistent_runtime,
        resident_kv_cache=cfg.flagos_resident_kv_cache,
        persistent_prefix_capacity=cfg.flagos_persistent_prefix_capacity,
        persistent_tree_capacity=cfg.flagos_persistent_tree_capacity,
        exact_node_templates=cfg.flagos_exact_node_templates,
        fixed_workspace_layout=cfg.flagos_fixed_workspace_layout,
        prewarm_graph_buckets=cfg.flagos_prewarm_graph_buckets,
        persistent_input_buffers=cfg.flagos_persistent_input_buffers,
        static_tree_attention=cfg.flagos_static_tree_attention,
    )
    model._flagos_verifier_gate_during_warmup = bool(
        cfg.flagos_verifier_gate_during_warmup
    )

    # Packed weights must be constructed after the model reaches its final
    # CUDA device and dtype.
    # Packed BF16 Linear fusion and weight-only replacement own the same
    # module objects.  Never stack them implicitly: an accepted W8A16
    # candidate is evaluated with its own manifest and the untouched BF16
    # fusion path remains the fallback.
    linear_fusion_enabled = bool(
        cfg.use_flagos_linear_fusion
        and not getattr(model, "_flagos_w8a16_enabled", False)
    )
    enable_linear_fusion(
        model=model,
        enabled=linear_fusion_enabled,
        qkv_rows=cfg.flagos_qkv_fusion_rows,
        gate_up_rows=cfg.flagos_gate_up_fusion_rows,
        qkv_input_sizes=cfg.flagos_qkv_fusion_input_sizes,
        gate_up_input_sizes=cfg.flagos_gate_up_fusion_input_sizes,
        swiglu_rows=cfg.flagos_swiglu_fusion_rows,
        swiglu_input_sizes=cfg.flagos_swiglu_fusion_input_sizes,
        swiglu_backend=cfg.flagos_swiglu_fusion_backend,
        add_rms_norm_rows=cfg.flagos_add_rms_norm_fusion_rows,
        add_rms_norm_input_sizes=cfg.flagos_add_rms_norm_fusion_input_sizes,
        add_rms_norm_backend=cfg.flagos_add_rms_norm_fusion_backend,
        record=cfg.flagos_linear_fusion_record,
        record_once=cfg.flagos_linear_fusion_record_once,
        log_path=cfg.flagos_linear_fusion_log_path,
        manifest_path=cfg.flagos_linear_fusion_manifest_path,
    )
    enable_rotary_cache(
        model=model,
        enabled=cfg.use_flagos_rope_cache,
        record=cfg.flagos_rope_cache_record,
        record_once=cfg.flagos_rope_cache_record_once,
        log_path=cfg.flagos_rope_cache_log_path,
        manifest_path=cfg.flagos_rope_cache_manifest_path,
    )
    enable_rotary_fusion(
        model=model,
        enabled=cfg.use_flagos_rope_fusion,
        resident_key_write=cfg.flagos_rope_resident_key_write,
        record=cfg.flagos_rope_fusion_record,
        record_once=cfg.flagos_rope_fusion_record_once,
        log_path=cfg.flagos_rope_fusion_log_path,
        manifest_path=cfg.flagos_rope_fusion_manifest_path,
    )
    enable_zero_copy_kv_return(
        enabled=cfg.use_flagos_zero_copy_kv_return,
        manifest_path=cfg.flagos_zero_copy_kv_manifest_path,
    )
    enable_tree_attention_mask(
        model=model,
        enabled=cfg.use_flagos_tree_attention_mask,
        record=cfg.flagos_tree_attention_mask_record,
        record_once=cfg.flagos_tree_attention_mask_record_once,
        log_path=cfg.flagos_tree_attention_mask_log_path,
        manifest_path=cfg.flagos_tree_attention_mask_manifest_path,
    )

    if cfg.runtime_stats_path:
        if not hasattr(model, "enable_flagos_runtime_stats"):
            raise RuntimeError("The selected KERV model does not expose FlagOS runtime statistics")
        model.enable_flagos_runtime_stats(
            cfg.runtime_stats_path,
            flag_gems_enabled=cfg.use_flag_gems,
            flag_gems_include=cfg.flag_gems_include,
            timing_backend=cfg.flagos_timing_backend,
            async_logging=cfg.flagos_async_runtime_logging or cfg.flagos_async_logs,
        )

    # Runtime-only optimizations are deliberately configured after all model
    # and operator bridges.  They do not alter thresholds, tree topology,
    # Kalman settings, or numerical kernels.
    model._flagos_system_optimization_enabled = bool(
        cfg.flagos_system_optimization
    )
    phase6_a100_fast_path = bool(
        cfg.flagos_system_optimization
        and torch.cuda.is_available()
        and "A100" in torch.cuda.get_device_name(torch.cuda.current_device())
    )
    model._flagos_phase6_a100_fast_path = phase6_a100_fast_path
    model._flagos_prompt_cuda_graph_mode = (
        str(cfg.flagos_prompt_cuda_graph).lower()
        if phase6_a100_fast_path
        else "off"
    )
    model._flagos_draft_cuda_graph_mode = (
        str(cfg.flagos_draft_cuda_graph).lower()
        if phase6_a100_fast_path
        else "off"
    )
    model._flagos_shared_graph_pool = bool(cfg.flagos_shared_graph_pool)
    model._flagos_process_level_warmup = bool(cfg.flagos_process_level_warmup)
    model._flagos_persistent_control_buffer = bool(
        cfg.flagos_persistent_control_buffer
    )
    model._flagos_persistent_decode_workspace = bool(
        cfg.flagos_persistent_decode_workspace
    )
    model._flagos_inference_mode_enabled = bool(cfg.flagos_inference_mode)
    model._flagos_task_level_graph_warmup = bool(
        cfg.flagos_task_level_graph_warmup and phase6_a100_fast_path
    )
    model._flagos_commit_draft_cuda_graph_mode = (
        str(cfg.flagos_commit_draft_cuda_graph).lower()
        if phase6_a100_fast_path
        else "off"
    )
    model._flagos_commit_draft_stream_overlap_mode = (
        str(cfg.flagos_commit_draft_stream_overlap).lower()
        if phase6_a100_fast_path
        else "off"
    )
    model._flagos_allocator_backend = str(cfg.flagos_allocator_backend).lower()
    model._flagos_episode_device_warmup_mode = str(
        cfg.flagos_episode_device_warmup
    ).lower()
    model._flagos_episode_warmup_duration_ms = str(
        cfg.flagos_episode_warmup_duration_ms
    )
    model._flagos_joint_commit_draft_graph_mode = str(
        cfg.flagos_joint_commit_draft_graph
    ).lower()
    model._flagos_joint_graph_accept_buckets = tuple(
        sorted(
            set(
                int(value.strip())
                for value in str(cfg.flagos_joint_graph_accept_buckets).split(",")
                if value.strip()
            )
        )
    )
    if any(value < 1 or value > 7 for value in model._flagos_joint_graph_accept_buckets):
        raise ValueError("flagos_joint_graph_accept_buckets must be within 1..7")
    model._flagos_fixed_draft_input_workspace = bool(
        cfg.flagos_fixed_draft_input_workspace
    )
    model._flagos_numa_affinity = applied_affinity
    model.ea_layer._flagos_persistent_draft_workspace = bool(
        cfg.flagos_system_optimization and cfg.flagos_persistent_decode_workspace
    )
    model.ea_layer._flagos_persistent_draft_kv_cache = bool(
        phase6_a100_fast_path and cfg.flagos_persistent_draft_kv_cache
    )
    model.ea_layer._flagos_draft_workspace_hits = 0
    model.ea_layer._flagos_draft_workspace_allocations = 0
    model._flagos_stage_graph_captures = 0
    model._flagos_stage_graph_fallbacks = 0
    model._flagos_stage_graph_capture_overhead_s = 0.0
    model._flagos_cuda_graph_pool = None
    model._flagos_joint_graph_captures = 0
    model._flagos_joint_graph_replays = 0
    model._flagos_joint_graph_fallbacks = 0
    model._flagos_joint_graph_capture_overhead_s = 0.0
    if cfg.flagos_system_optimization:
        from experiments.robot.flagos_stage_graph import (
            JointStageGraphCache,
            StageGraphCache,
        )

        def _graph_mode(value):
            # YAML 1.1 parses unquoted ``off``/``on`` as booleans.
            if value is False or str(value).lower() in {"false", "0", "none"}:
                return "off"
            if value is True or str(value).lower() in {"true", "1"}:
                return "on"
            return str(value).lower()

        prompt_mode = _graph_mode(model._flagos_prompt_cuda_graph_mode)
        draft_mode = _graph_mode(model._flagos_draft_cuda_graph_mode)
        commit_draft_mode = _graph_mode(
            model._flagos_commit_draft_cuda_graph_mode
        )
        commit_overlap_mode = _graph_mode(
            model._flagos_commit_draft_stream_overlap_mode
        )
        if prompt_mode not in {"off", "auto", "on"}:
            raise ValueError("flagos_prompt_cuda_graph must be off, auto, or on")
        if draft_mode not in {"off", "auto", "on"}:
            raise ValueError("flagos_draft_cuda_graph must be off, auto, or on")
        if commit_draft_mode not in {"off", "auto", "on"}:
            raise ValueError(
                "flagos_commit_draft_cuda_graph must be off, auto, or on"
            )
        if commit_overlap_mode not in {"off", "auto", "on"}:
            raise ValueError(
                "flagos_commit_draft_stream_overlap must be off, auto, or on"
            )
        # StageGraphCache accepts the existing off/auto policy; ``on`` is an
        # explicit opt-in with the same bounded entry budget and safe fallback.
        # Prompt prefill writes the resident KERV cache and remains eager until
        # its cache transition is explicitly made graph-safe.  The Draft
        # module, however, returns its own immutable tuple of KV tensors and
        # can be captured independently; its state transition is the returned
        # graph output rather than a write into the resident verifier cache.
        stateful_kv = bool(
            getattr(model, "_flagos_persistent_runtime_enabled", False)
            or getattr(model, "_flagos_resident_kv_cache_enabled", False)
        )
        # KervPersistentTreeCache exposes graph-state snapshot/restore hooks,
        # so prompt prefill can replay against the same resident addresses.
        # Non-persistent cache implementations keep the historical eager path.
        prompt_stage_graph_allowed = bool(
            not stateful_kv
            or getattr(model, "_flagos_persistent_runtime_enabled", False)
        )
        draft_stage_graph_allowed = True
        model._flagos_stage_graph_disable_reason = (
            "prompt_graph_requires_persistent_kv_cache"
            if not prompt_stage_graph_allowed
            else None
        )
        model._flagos_prompt_stage_graph = (
            StageGraphCache(model, prompt_mode, "prompt", max_entries=1)
            if prompt_mode != "off" and prompt_stage_graph_allowed
            else None
        )
        draft_graph_buckets = tuple(
            sorted(
                set(
                    int(value.strip())
                    for value in str(cfg.flagos_draft_graph_accept_buckets).split(",")
                    if value.strip()
                )
            )
        )
        if any(value < 1 or value > 7 for value in draft_graph_buckets):
            raise ValueError(
                "flagos_draft_graph_accept_buckets must be within 1..7"
            )
        model._flagos_draft_graph_accept_buckets = draft_graph_buckets
        model._flagos_draft_stage_graph = (
            StageGraphCache(
                model,
                draft_mode,
                "draft",
                # One initial-prompt signature plus one signature for each
                # accepted-token length.  Structural KV signatures allow all
                # later tuple instances to reuse these fixed graph buffers.
                max_entries=max(2, len(draft_graph_buckets) + 1),
            )
            if draft_mode != "off" and draft_stage_graph_allowed
            else None
        )
        model.ea_layer._flagos_draft_stage_graph_runner = (
            model._run_draft_stage_graph
            if model._flagos_draft_stage_graph is not None
            else None
        )
        joint_mode = _graph_mode(model._flagos_joint_commit_draft_graph_mode)
        if joint_mode not in {"off", "auto", "on"}:
            raise ValueError(
                "flagos_joint_commit_draft_graph must be off, auto, or on"
            )
        # The joint path is only legal with both persistent caches.  A failed
        # capture remains a normal Phase-6 fallback and never disables Draft
        # or Verifier Graph replay.
        joint_allowed = bool(
            phase6_a100_fast_path
            and joint_mode != "off"
            and getattr(model, "_flagos_persistent_runtime_enabled", False)
            and getattr(model, "_flagos_resident_kv_cache_enabled", False)
            and getattr(model.ea_layer, "_flagos_persistent_draft_kv_cache", False)
        )
        model._flagos_joint_commit_draft_graph = (
            JointStageGraphCache(
                model,
                joint_mode,
                max_entries=max(1, len(model._flagos_joint_graph_accept_buckets)),
            )
            if joint_allowed
            else None
        )
        if joint_allowed:
            model._flagos_joint_graph_disable_reason = None
        elif joint_mode == "off":
            model._flagos_joint_graph_disable_reason = "disabled_by_config"
        else:
            model._flagos_joint_graph_disable_reason = (
                "requires_a100_and_two_persistent_caches"
            )

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # These caches are input-side only.  They are enabled by the Phase 5
        # system profile and can be disabled independently without touching
        # model execution semantics.
        processor._flagos_prompt_token_cache_enabled = bool(
            cfg.flagos_prompt_token_cache
        )
        processor._flagos_pinned_image_buffer_enabled = bool(
            cfg.flagos_pinned_image_buffer and cfg.flagos_async_image_transfer
        )
        processor._flagos_prompt_cache_model = str(cfg.pretrained_checkpoint)

    # Initialize local logging
    target_dir = os.path.join(cfg.local_log_dir, "libero_goal_spec_relaxed")
    os.makedirs(target_dir,exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    error_output_path = None
    if cfg.use_spec and cfg.collect_spec_errors and hasattr(model, "enable_error_collection"):
        error_output_path = cfg.spec_error_output or os.path.join(target_dir, run_id + "_spec_errors.npy")
        history_output = cfg.spec_history_output or None
        model.enable_error_collection(
            error_output_path,
            history_output,
            log_within_threshold=True,
        )
        print(f"[Spec] Collecting verifier/draft mismatch stats at {error_output_path}")

    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "_relaxed_stats.json")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if cfg.max_tasks is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, cfg.max_tasks)
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Clock warmup is process-scoped.  Repeating a 1s GEMM warmup for every
    # episode makes multi-episode comparisons measure the harness rather than
    # KERV.  The legacy per-episode behavior remains available when the new
    # switch is disabled.
    if cfg.flagos_process_level_warmup and cfg.flagos_cuda_warmup_seconds > 0:
        warmup_cuda_for_measurement(cfg.flagos_cuda_warmup_seconds)
        model._flagos_process_warmup_done = True

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_task_times: Dict[str, Dict[str, List[List[float]]]] = {}
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        task_graph_warmed = False
        if cfg.flagos_task_level_graph_warmup:
            prompt_graph_cache = getattr(model, "_flagos_prompt_stage_graph", None)
            if prompt_graph_cache is not None:
                # Task prompts can have different token lengths. Keep only the
                # active task graph and capture the replacement outside the
                # measured episode path.
                prompt_graph_cache.entries.clear()

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_episode_times: List[List[List[float]]] = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            total_time = []
            done = False
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Overlap GPU clock warmup with LIBERO's mandatory settle steps;
            # the event is waited on immediately before the first measured
            # model action.  This avoids hiding warmup in action latency.
            if (
                bool(cfg.flagos_system_optimization)
                and str(cfg.flagos_episode_device_warmup).lower()
                in {"auto", "on"}
            ):
                schedule_episode_device_warmup(
                    model,
                    mode=cfg.flagos_episode_device_warmup,
                    duration_ms=cfg.flagos_episode_warmup_duration_ms,
                )

            # Setup
            t = 0
            replay_images = []
            episode_action_history: List[np.ndarray] = [] if cfg.store_action_history else None
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 170  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 240  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 220  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 470  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            if cfg.max_episode_steps is not None:
                max_steps = min(max_steps, cfg.max_episode_steps)

            planning_horizon = cfg.max_planning_steps or max_steps
            if cfg.use_spec and hasattr(model, "start_rollout"):
                kalman_cfg = {
                    "enabled": cfg.use_kalman_fallback,
                    "process_var": cfg.kalman_process_var,
                    "measurement_var": cfg.kalman_measurement_var,
                    "history_window": cfg.kalman_history_window,
                    "tree_enabled": cfg.use_kalman_tree,
                }
                try:
                    model.start_rollout(
                        max_steps=planning_horizon,
                        kalman_cfg=kalman_cfg,
                        store_history=cfg.store_action_history,
                    )
                except TypeError:
                    # Backwards compatibility for models without the extended signature.
                    if kalman_cfg is not None:
                        model.start_rollout(planning_horizon, kalman_cfg)
                    else:
                        model.start_rollout(planning_horizon)

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            if cfg.use_spec and (cfg.collect_spec_errors or cfg.runtime_stats_path) and hasattr(model, "set_logging_context"):
                episode_key = f"{task_id}_{episode_idx + 1}"
                model.set_logging_context(episode_key)
            if not cfg.flagos_process_level_warmup:
                warmup_cuda_for_measurement(cfg.flagos_cuda_warmup_seconds)
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # The wait is placed after simulator settling, so the
                    # warmup overlaps environment work and does not become a
                    # per-action synchronization cost.
                    if t == cfg.num_steps_wait:
                        wait_episode_device_warmup(model)

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    if cfg.use_spec and (cfg.collect_spec_errors or cfg.runtime_stats_path) and hasattr(model, "set_action_step"):
                        model.set_action_step(t - cfg.num_steps_wait + 1)

                    # Query model to get action
                    step_idx = t - cfg.num_steps_wait
                    dynamic_threshold = None
                    if cfg.use_spec:
                        dynamic_threshold = compute_dynamic_threshold(
                            step_idx=max(step_idx, 0),
                            total_steps=planning_horizon,
                            start=cfg.accept_threshold_start,
                            lower=cfg.accept_threshold_min,
                            schedule=cfg.threshold_decay,
                        )

                    def run_action_query():
                        return get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            return_time=True,
                            generate_mode='speculative',
                            history=episode_action_history,
                            step_idx=step_idx,
                            max_steps=planning_horizon,
                            dynamic_threshold=dynamic_threshold,
                            use_kalman=cfg.use_kalman_fallback,
                            kalman_process_var=cfg.kalman_process_var,
                            kalman_measurement_var=cfg.kalman_measurement_var,
                            kalman_history_window=cfg.kalman_history_window,
                            kalman_tree_enabled=cfg.use_kalman_tree,
                        )

                    if (
                        bool(
                            getattr(
                                model, "_flagos_task_level_graph_warmup", False
                            )
                        )
                        and not task_graph_warmed
                    ):
                        previous_suppress = bool(
                            getattr(model, "_flagos_suppress_runtime_stats", False)
                        )
                        previous_collect_errors = bool(
                            getattr(model, "_collect_errors", False)
                        )
                        model._flagos_suppress_runtime_stats = True
                        model._collect_errors = False
                        try:
                            run_action_query()
                        finally:
                            model._flagos_suppress_runtime_stats = previous_suppress
                            model._collect_errors = previous_collect_errors
                        # The first deterministic warmup action leaves the
                        # prompt-length Draft KV tuple available.  Capture all
                        # accepted-token input lengths against that same KV
                        # layout now, so formal episodes never pay a new Draft
                        # graph capture or capacity fallback.
                        draft_graph_cache = getattr(
                            model, "_flagos_draft_stage_graph", None
                        )
                        stable_draft_kv = getattr(
                            getattr(model, "ea_layer", None),
                            "stable_kv",
                            None,
                        )
                        if draft_graph_cache is not None and stable_draft_kv is not None:
                            draft_hidden_size = int(model.ea_layer.fc.out_features)
                            draft_embed_size = int(
                                model.ea_layer.embed_tokens.embedding_dim
                            )
                            draft_device = (
                                stable_draft_kv.valid_length_tensor.device
                                if hasattr(stable_draft_kv, "valid_length_tensor")
                                else stable_draft_kv[0][0].device
                            )
                            draft_dtype = model.ea_layer.fc.weight.dtype
                            # Match the formal action path: graph capture and
                            # all fixed-address KV mutations happen under
                            # inference mode.  PyTorch deliberately rejects an
                            # out-of-context mutation of inference tensors.
                            with torch.inference_mode():
                                for accepted_tokens in getattr(
                                    model,
                                    "_flagos_draft_graph_accept_buckets",
                                    (),
                                ):
                                    dummy_hidden = torch.zeros(
                                        (
                                            1,
                                            int(accepted_tokens),
                                            draft_hidden_size,
                                        ),
                                        device=draft_device,
                                        dtype=draft_dtype,
                                    )
                                    dummy_embeddings = torch.zeros(
                                        (
                                            1,
                                            int(accepted_tokens),
                                            draft_embed_size,
                                        ),
                                        device=draft_device,
                                        dtype=draft_dtype,
                                    )
                                    model._run_draft_stage_graph(
                                        dummy_hidden,
                                        dummy_embeddings,
                                        stable_draft_kv,
                                    )
                            torch.cuda.synchronize(draft_device)
                            if getattr(
                                model, "_flagos_stage_graph_last_traceback", None
                            ):
                                print(
                                    model._flagos_stage_graph_last_traceback,
                                    flush=True,
                                )
                            model.ea_layer.reset_kv()
                        # Capture every fixed verifier node bucket before the
                        # formal episode.  The resident input/KV addresses are
                        # the same buffers used by tree_decoding; formal
                        # actions only overwrite their contents and replay.
                        verifier_cache = getattr(
                            model, "_flagos_persistent_kv_cache", None
                        )
                        if (
                            verifier_cache is not None
                            and bool(
                                getattr(
                                    model,
                                    "_flagos_compile_verifier_enabled",
                                    False,
                                )
                            )
                            and str(
                                getattr(
                                    model,
                                    "_flagos_compile_verifier_mode",
                                    "",
                                )
                            )
                            in {"cuda-graph", "inductor-cuda-graph"}
                        ):
                            from openvla.experiments.robot.flagos_verifier_graph import (
                                run_verifier_cuda_graph,
                            )

                            language_model = model.base_model.language_model
                            embedding_weight = (
                                language_model.model.embed_tokens.weight
                            )
                            embedding_buffers = getattr(
                                model, "_flagos_tree_embedding_buffers", {}
                            )
                            model._flagos_tree_embedding_buffers = embedding_buffers
                            position_buffers = getattr(
                                model, "_flagos_position_id_buffers", {}
                            )
                            model._flagos_position_id_buffers = position_buffers
                            retrieve_buffers = getattr(
                                model, "_flagos_retrieve_index_buffers", {}
                            )
                            model._flagos_retrieve_index_buffers = retrieve_buffers
                            with torch.inference_mode():
                                for bucket in getattr(
                                    model, "_flagos_tree_node_buckets", ()
                                ):
                                    bucket = int(bucket)
                                    embedding_key = (
                                        embedding_weight.device.type,
                                        embedding_weight.device.index,
                                        bucket,
                                        int(embedding_weight.shape[1]),
                                        embedding_weight.dtype,
                                    )
                                    verifier_embeddings = embedding_buffers.get(
                                        embedding_key
                                    )
                                    if verifier_embeddings is None:
                                        verifier_embeddings = torch.zeros(
                                            (
                                                1,
                                                bucket,
                                                int(embedding_weight.shape[1]),
                                            ),
                                            device=embedding_weight.device,
                                            dtype=embedding_weight.dtype,
                                        )
                                        embedding_buffers[embedding_key] = (
                                            verifier_embeddings
                                        )
                                    verifier_embeddings.zero_()
                                    position_key = (
                                        embedding_weight.device.type,
                                        embedding_weight.device.index,
                                        (1, bucket),
                                        torch.long,
                                    )
                                    position_ids = position_buffers.get(position_key)
                                    if position_ids is None:
                                        position_ids = torch.zeros(
                                            (1, bucket),
                                            device=embedding_weight.device,
                                            dtype=torch.long,
                                        )
                                        position_buffers[position_key] = position_ids
                                    position_ids.zero_()
                                    retrieve_key = (
                                        embedding_weight.device.type,
                                        embedding_weight.device.index,
                                        (48, 8),
                                        torch.long,
                                    )
                                    retrieve_indices = retrieve_buffers.get(
                                        retrieve_key
                                    )
                                    if retrieve_indices is None:
                                        retrieve_indices = torch.zeros(
                                            (48, 8),
                                            device=embedding_weight.device,
                                            dtype=torch.long,
                                        )
                                        retrieve_buffers[retrieve_key] = (
                                            retrieve_indices
                                        )
                                    retrieve_indices.zero_()
                                    tree_mask = torch.eye(
                                        bucket,
                                        device=embedding_weight.device,
                                        dtype=torch.bool,
                                    )
                                    tree_mask[:, 0] = True
                                    tree_mask = tree_mask[None, None]
                                    language_model.tree_mask = tree_mask
                                    verifier_cache.begin_tree(bucket)
                                    language_model._flagos_external_causal_mask = (
                                        verifier_cache.build_tree_attention_mask(
                                            tree_mask,
                                            dtype=embedding_weight.dtype,
                                            materialize=False,
                                        )
                                    )
                                    run_verifier_cuda_graph(
                                        model,
                                        verifier_embeddings,
                                        position_ids,
                                        retrieve_indices,
                                        verifier_cache,
                                    )
                                verifier_cache.end_tree()
                        model._flagos_graph_warmup_counters = {
                            "prompt_captures": int(
                                getattr(model, "_flagos_prompt_graph_captures", 0)
                            ),
                            "prompt_replays": int(
                                getattr(model, "_flagos_prompt_graph_replays", 0)
                            ),
                            "prompt_fallbacks": int(
                                getattr(model, "_flagos_prompt_graph_fallbacks", 0)
                            ),
                            "draft_captures": int(
                                getattr(model, "_flagos_draft_graph_captures", 0)
                            ),
                            "draft_replays": int(
                                getattr(model, "_flagos_draft_graph_replays", 0)
                            ),
                            "draft_fallbacks": int(
                                getattr(model, "_flagos_draft_graph_fallbacks", 0)
                            ),
                            "verifier_captures": int(
                                getattr(model, "_flagos_cuda_graph_captures", 0)
                            ),
                            "verifier_replays": int(
                                getattr(model, "_flagos_cuda_graph_replay_hits", 0)
                            ),
                            "verifier_fallbacks": int(
                                getattr(model, "_flagos_cuda_graph_fallbacks", 0)
                            ),
                            "verifier_audit_eager": int(
                                getattr(model, "_flagos_cuda_graph_audit_eager", 0)
                            ),
                        }
                        if cfg.use_spec and hasattr(model, "start_rollout"):
                            model.start_rollout(
                                max_steps=planning_horizon,
                                kalman_cfg=kalman_cfg,
                                store_history=cfg.store_action_history,
                            )
                        if episode_action_history is not None:
                            episode_action_history.clear()
                        task_graph_warmed = True
                        model._flagos_task_graph_warmups = int(
                            getattr(model, "_flagos_task_graph_warmups", 0)
                        ) + 1

                    action, time = run_action_query()
                    raw_action = np.array(action, copy=True)
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if cfg.store_action_history and episode_action_history is not None:
                        episode_action_history.append(raw_action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    total_time.append(time)

                except Exception as e:
                    print(f"Caught exception: {e}")
                    traceback.print_exc()
                    log_file.write(f"Caught exception: {e}\n")
                    break
            #exit()
            task_episodes += 1
            total_episodes += 1
            task_episode_times.append([[float(ts[1]), float(ts[0])] for ts in total_time])

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
        #exit()
        all_task_times[str(task_id)] = {
            str(ep_idx): episode_times for ep_idx, episode_times in enumerate(task_episode_times)
        }
    with open(local_log_timefilepath,mode='w') as f:
        json.dump(all_task_times,f)
    # Compute per-task average inference time (step latency)
    for task_key, episodes in all_task_times.items():
        total_duration = 0.0
        total_segments = 0
        for episode_times in episodes.values():
            for start_ts, end_ts in episode_times:
                total_duration += max(0.0, end_ts - start_ts)
                total_segments += 1
        mean_duration = total_duration / total_segments if total_segments else 0.0
        timing_msg = (
            f"[Timing] task {task_key}: mean step latency {mean_duration:.6f}s "
            f"over {total_segments} steps ({len(episodes)} episodes)"
        )
        print(timing_msg)
        log_file.write(timing_msg + "\n")
        log_file.flush()
    # Save local log file
    if cfg.use_spec and cfg.collect_spec_errors and hasattr(model, "save_error_stats"):
        model.save_error_stats()
    if cfg.use_spec and (cfg.collect_spec_errors or cfg.runtime_stats_path):
        if hasattr(model, "set_logging_context"):
            model.set_logging_context(None)
        if hasattr(model, "set_action_step"):
            model.set_action_step(None)
    if hasattr(model, "close_flagos_runtime_stats"):
        model.close_flagos_runtime_stats()
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()

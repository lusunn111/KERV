# FlagOS operator and runtime integration

## Operator surface

KERV contributes 18 interfaces under `torch.ops.flagos_embodied`.

| Interface | Role | Release status |
|---|---|---|
| `kerv_silu_mul` | SiLU and elementwise multiply fusion | mainline |
| `kerv_add_rms_norm` | residual addition and RMSNorm fusion | mainline |
| `kerv_verify_accept_control` | token match, accepted length and path selection | mainline |
| `kerv_static_tree_pack` | static-tree template and candidate packing | mainline |
| `kerv_static_tree_attention` | prefix/ancestor-only tree attention | mainline |
| `kerv_action_projection_select` | action-vocabulary projection and selection | mainline |
| `kerv_value_cache_store` | resident V-cache write | available, disabled by default |
| `kerv_kv_commit` | accepted-path K/V commit | mainline |
| `kerv_tree_embed_pack` | token gather, embedding and tree-buffer packing | mainline |
| `kerv_rope_kv_store` | RoPE and resident K/V write | mainline |
| `kerv_action_verify_accept` | action projection and Verify-Accept fusion | mainline |
| `kerv_draft_action_topk` | drafter action projection and Top-K | mainline |
| `kerv_kv_accept_commit` | accepted-path gather and commit | mainline |
| `kerv_vision_add_layer_norm` | visual residual and LayerNorm fusion | mainline |
| `kerv_vision_bias_gelu` | visual bias and GELU fusion | mainline |
| `kerv_o_proj_residual_rms_norm` | attention epilogue fusion | experimental |
| `kerv_down_proj_residual_rms_norm` | MLP epilogue fusion | experimental |
| `kerv_logical_kv_commit` | logical-address-only K/V commit | experimental |

`backend=auto` keeps exact native fallbacks and only selects specialized
implementations for supported batch-one BF16/FP16 layouts.

## Transformer fusion

Two model-level fusion routes are enabled by measured row-count allowlists:

- **QKV fusion:** Q, K and V share the same input, so their weights are packed
  and the three projections are executed as one GEMM.
- **Gate-Up-SwiGLU fusion:** Gate and Up share the same input; one packed GEMM
  produces both branches, followed by an in-place SiLU/multiply epilogue.

The allowlists in `configs/kerv_libero_goal.yaml` are part of the A100
reference profile. Other dimensions fall back to the native CUDA path instead
of assuming that a Triton kernel is always faster.

## System execution

The final runtime also contains non-kernel optimizations:

- fixed-width, operatorized verification-tree templates;
- persistent prefix/tree workspaces and resident K/V cache;
- verifier, prompt and drafter CUDA Graph replay;
- shared graph memory pool and process-level warm-up;
- persistent decode/control buffers and fused CPU-GPU control transfer;
- pinned image input and asynchronous H2D copy;
- two-stream commit/next-draft overlap;
- cached prompt tokenization and asynchronous runtime logging.

Quantized W8A16, compact trees and the three experimental epilogues are not
enabled by the BF16 safe configuration.

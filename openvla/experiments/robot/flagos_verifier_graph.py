"""Shape-keyed CUDA Graph execution for KERV tree verification."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Tuple

import torch


def _clone_past_key_values(past_key_values):
    return tuple(
        tuple(value.clone() for value in layer_values)
        for layer_values in past_key_values
    )


def _copy_past_key_values(destination, source) -> None:
    destination_values = [
        value for layer_values in destination for value in layer_values
    ]
    source_values = [value for layer_values in source for value in layer_values]
    if len(destination_values) != len(source_values):
        raise ValueError("source and destination KV layer counts must match")
    # A verifier replay updates 32 K/V layer pairs. The foreach path submits
    # these copies as a batched multi-tensor operation instead of issuing one
    # Python/CUDA call per tensor.
    torch._foreach_copy_(destination_values, source_values)


def verifier_graph_signature(
    model, input_embeds, position_ids, retrieve_indices, past_key_values
):
    tree_mask = model.base_model.language_model.tree_mask
    if bool(getattr(past_key_values, "_flagos_persistent_tree_cache", False)):
        fixed_workspace = bool(
            getattr(past_key_values, "fixed_workspace_layout", False)
        )
        return (
            "persistent",
            tuple(input_embeds.shape),
            tuple(position_ids.shape),
            tuple(tree_mask.shape),
            None if fixed_workspace else int(past_key_values.prefix_length),
            int(past_key_values.tree_length),
            str(input_embeds.dtype),
        )
    return (
        tuple(input_embeds.shape),
        tuple(position_ids.shape),
        tuple(tree_mask.shape),
        tuple(
            tuple(tuple(value.shape) for value in layer_values)
            for layer_values in past_key_values
        ),
        str(input_embeds.dtype),
    )


@dataclass
class VerifierCudaGraphEntry:
    graph: torch.cuda.CUDAGraph
    input_embeds: torch.Tensor
    position_ids: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    past_key_values: Any
    external_causal_mask: Any
    persistent: bool
    persistent_inputs: bool
    static_tree_attention: bool
    outputs: Any

    @classmethod
    def capture(
        cls, model, input_embeds, position_ids, retrieve_indices, past_key_values
    ):
        language_model = model.base_model.language_model
        persistent = bool(
            getattr(past_key_values, "_flagos_persistent_tree_cache", False)
        )
        persistent_inputs = bool(
            persistent
            and getattr(model, "_flagos_persistent_input_buffers", False)
        )
        # Fixed-bucket buffers are populated before this call and retain the
        # same addresses for the lifetime of the model.  Capturing them in
        # place removes two replay-time device copies (the large [T,H]
        # embedding copy in particular) without changing model semantics.
        static_input_embeds = input_embeds if persistent_inputs else input_embeds.clone()
        static_position_ids = position_ids if persistent_inputs else position_ids.clone()
        # Retrieve layouts vary with the accepted candidate topology even
        # when the verifier node bucket is identical.  Keep path gathering
        # outside the captured graph so one graph can serve every topology in
        # a bucket.  The gather is a small device-only operation and avoids
        # both redundant captures and eager verifier fallbacks.
        static_retrieve_indices = retrieve_indices
        static_tree_mask = language_model.tree_mask.clone()
        if persistent:
            static_past_key_values = past_key_values
            external_mask = getattr(language_model, "_flagos_external_causal_mask", None)
            if external_mask is None:
                raise RuntimeError("Persistent verifier requires an external causal mask")
            static_external_causal_mask = external_mask.clone()
        else:
            static_past_key_values = _clone_past_key_values(past_key_values)
            static_external_causal_mask = None
        capture_stream = torch.cuda.Stream(device=input_embeds.device)
        capture_stream.wait_stream(torch.cuda.current_stream(input_embeds.device))

        compiled_forward = getattr(
            model, "_flagos_inductor_verifier_forward", None
        )
        using_inductor = compiled_forward is not None
        forward_fn = compiled_forward if compiled_forward is not None else model.forward

        def invoke_forward():
            return forward_fn(
                input_embeds=static_input_embeds,
                output_orig=True,
                attention_mask=None,
                past_key_values=static_past_key_values,
                return_dict=True,
                position_ids=static_position_ids,
                use_cache=True,
            )

        # Warm the exact model/shape on a side stream so libraries may create
        # handles and select kernels before stream capture begins.
        with torch.cuda.stream(capture_stream):
            try:
                language_model.tree_mask = static_tree_mask
                language_model._flagos_external_causal_mask = static_external_causal_mask
                for _ in range(2):
                    warm_outputs = invoke_forward()
                del warm_outputs
            except Exception as exc:
                if compiled_forward is None:
                    raise
                # A model/backend unsupported by Inductor keeps the already
                # validated native CUDA-graph path.  Record the reason so the
                # experimental mode never silently claims a compiler hit.
                model._flagos_inductor_verifier_last_error = repr(exc)
                model._flagos_inductor_verifier_fallbacks = int(
                    getattr(model, "_flagos_inductor_verifier_fallbacks", 0)
                ) + 1
                model._flagos_inductor_verifier_forward = None
                using_inductor = False
                forward_fn = model.forward
                language_model.tree_mask = static_tree_mask
                language_model._flagos_external_causal_mask = static_external_causal_mask
                for _ in range(2):
                    warm_outputs = invoke_forward()
                del warm_outputs
        capture_stream.synchronize()

        # Some cache implementations update the supplied KV tensors in place.
        # Restore every capture input after warmup so capture observes exactly
        # the same values as the caller's eager invocation.
        if not persistent_inputs:
            static_input_embeds.copy_(input_embeds)
            static_position_ids.copy_(position_ids)
        static_tree_mask.copy_(language_model.tree_mask)
        if persistent:
            static_external_causal_mask.copy_(
                getattr(language_model, "_flagos_external_causal_mask")
            )
        else:
            _copy_past_key_values(static_past_key_values, past_key_values)
        torch.cuda.synchronize(input_embeds.device)

        graph = torch.cuda.CUDAGraph()
        language_model.tree_mask = static_tree_mask
        language_model._flagos_external_causal_mask = static_external_causal_mask
        graph_pool = getattr(model, "_flagos_cuda_graph_pool", None)
        if graph_pool is None and bool(
            getattr(model, "_flagos_shared_graph_pool", False)
        ):
            graph_pool = torch.cuda.graph_pool_handle()
            model._flagos_cuda_graph_pool = graph_pool
        graph_kwargs = {"stream": capture_stream}
        if graph_pool is not None:
            graph_kwargs["pool"] = graph_pool
        with torch.cuda.graph(graph, **graph_kwargs):
            outputs = invoke_forward()
            outputs = (
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3],
            )
        torch.cuda.current_stream(input_embeds.device).wait_stream(capture_stream)
        if using_inductor:
            model._flagos_inductor_verifier_graph_captures = int(
                getattr(model, "_flagos_inductor_verifier_graph_captures", 0)
            ) + 1
        return cls(
            graph=graph,
            input_embeds=static_input_embeds,
            position_ids=static_position_ids,
            retrieve_indices=static_retrieve_indices,
            tree_mask=static_tree_mask,
            past_key_values=static_past_key_values,
            external_causal_mask=static_external_causal_mask,
            persistent=persistent,
            persistent_inputs=persistent_inputs,
            static_tree_attention=bool(
                getattr(
                    language_model,
                    "_flagos_static_tree_attention_runtime_enabled",
                    False,
                )
            ),
            outputs=outputs,
        )

    def replay(
        self, model, input_embeds, position_ids, retrieve_indices, past_key_values
    ):
        input_alias = bool(
            self.persistent_inputs
            and self.input_embeds.data_ptr() == input_embeds.data_ptr()
            and self.position_ids.data_ptr() == position_ids.data_ptr()
        )
        if input_alias:
            model._flagos_cuda_graph_input_alias_hits = int(
                getattr(model, "_flagos_cuda_graph_input_alias_hits", 0)
            ) + 1
        else:
            self.input_embeds.copy_(input_embeds)
            self.position_ids.copy_(position_ids)
        if not self.static_tree_attention:
            self.tree_mask.copy_(model.base_model.language_model.tree_mask)
        if self.persistent:
            if past_key_values is not self.past_key_values:
                raise RuntimeError("Persistent CUDA Graph KV workspace address changed")
            current_mask = getattr(
                model.base_model.language_model,
                "_flagos_external_causal_mask",
                None,
            )
            if current_mask is None:
                raise RuntimeError("Persistent verifier causal mask is missing")
            if not self.static_tree_attention:
                self.external_causal_mask.copy_(current_mask)
            model.base_model.language_model._flagos_external_causal_mask = (
                self.external_causal_mask
            )
        else:
            _copy_past_key_values(self.past_key_values, past_key_values)
        model.base_model.language_model.tree_mask = self.tree_mask
        self.graph.replay()
        return self.outputs


def run_verifier_cuda_graph(
    model,
    input_embeds,
    position_ids,
    retrieve_indices,
    past_key_values,
):
    """Capture/replay a bounded number of exact verifier Shape signatures."""
    signature = verifier_graph_signature(
        model, input_embeds, position_ids, retrieve_indices, past_key_values
    )
    # When bucket prewarming is enabled, every fixed verifier signature is
    # allowed to capture/replay during the untimed warmup window.  The old
    # policy admitted only one audit signature and forced all other buckets
    # through eager execution until 24 replay samples accumulated.  That
    # serialized otherwise independent fixed-shape graphs and made the gate
    # part of the measured rollout.  Correctness is still guarded by the
    # existing static-attention reference tests; the performance gate below
    # decides whether the selected static path remains enabled.
    prewarm_buckets = bool(
        getattr(model, "_flagos_prewarm_graph_buckets", False)
        and getattr(model, "_flagos_verifier_gate_during_warmup", True)
    )
    if (
        str(getattr(model, "_flagos_static_tree_attention_mode", "off"))
        == "auto"
        and not bool(
            getattr(
                model,
                "_flagos_static_tree_attention_gate_evaluated",
                False,
            )
        )
        and not prewarm_buckets
    ):
        # Capture exactly one representative tree bucket for the auto audit;
        # other buckets stay eager until the gate is resolved.  If the gate
        # rejects the candidate, at most one audit graph plus the native
        # buckets are captured, preserving the global six-graph budget.
        audit_signature = getattr(
            model, "_flagos_static_tree_attention_audit_signature", None
        )
        if audit_signature is None:
            audit_signature = signature
            model._flagos_static_tree_attention_audit_signature = signature
        if signature != audit_signature:
            model._flagos_cuda_graph_audit_eager = int(
                getattr(model, "_flagos_cuda_graph_audit_eager", 0)
            ) + 1
            model._flagos_last_cuda_graph_mode = "static_attention_audit_eager"
            return None
    cache: Dict[object, VerifierCudaGraphEntry] = getattr(
        model, "_flagos_verifier_cuda_graph_cache", None
    )
    if cache is None:
        cache = {}
        model._flagos_verifier_cuda_graph_cache = cache
    entry = cache.get(signature)
    if entry is not None:
        model._flagos_cuda_graph_replay_hits = int(
            getattr(model, "_flagos_cuda_graph_replay_hits", 0)
        ) + 1
        model._flagos_last_cuda_graph_mode = "replay"
        model._flagos_last_cuda_graph_logits_gathered = False
        return entry.replay(
            model, input_embeds, position_ids, retrieve_indices, past_key_values
        )

    capture_past_length = int(
        getattr(model, "_flagos_cuda_graph_capture_past_length", -1)
    )
    actual_past_length = int(past_key_values[0][0].shape[-2])
    if capture_past_length >= 0 and actual_past_length != capture_past_length:
        model._flagos_cuda_graph_fallbacks = int(
            getattr(model, "_flagos_cuda_graph_fallbacks", 0)
        ) + 1
        model._flagos_last_cuda_graph_mode = "eager_fallback"
        return None

    max_entries = int(getattr(model, "_flagos_cuda_graph_max_entries", 1))
    if len(cache) >= max_entries:
        model._flagos_cuda_graph_fallbacks = int(
            getattr(model, "_flagos_cuda_graph_fallbacks", 0)
        ) + 1
        model._flagos_last_cuda_graph_mode = "eager_fallback"
        return None

    torch.cuda.synchronize(input_embeds.device)
    capture_started = time.perf_counter()
    entry = VerifierCudaGraphEntry.capture(
        model, input_embeds, position_ids, retrieve_indices, past_key_values
    )
    torch.cuda.synchronize(input_embeds.device)
    capture_overhead_s = time.perf_counter() - capture_started
    cache[signature] = entry
    model._flagos_cuda_graph_captures = int(
        getattr(model, "_flagos_cuda_graph_captures", 0)
    ) + 1
    model._flagos_cuda_graph_capture_overhead_s = float(
        getattr(model, "_flagos_cuda_graph_capture_overhead_s", 0.0)
    ) + float(capture_overhead_s)
    # Capture cost is accounted separately. The actual request is executed by
    # the newly captured graph, so subtracting ``capture_overhead_s`` yields a
    # measured replay latency rather than an estimated eager replacement.
    model._flagos_last_cuda_graph_mode = "capture_replay"
    model._flagos_last_cuda_graph_logits_gathered = False
    return entry.replay(
        model, input_embeds, position_ids, retrieve_indices, past_key_values
    )

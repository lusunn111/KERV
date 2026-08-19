"""Optional CUDA Graph wrappers for fixed-shape KERV prompt/draft stages."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def _value_signature(value: Any) -> Any:
    """Build a structural graph key without depending on container identity.

    Draft KV state is returned as a tuple of tensors.  Keying that tuple by
    ``id`` limited the cache to one concrete Python object even when every
    tensor had the same graph-safe layout.  A structural signature lets one
    fixed graph entry own the persistent input buffers for that layout.
    """
    if torch.is_tensor(value):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            value.device.type,
            value.device.index,
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, _value_signature(value[key]))
                for key in sorted(value)
            ),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    # Stateful cache objects deliberately retain identity.  Their addresses
    # are part of the prompt graph contract and remain stable for the loaded
    # model lifetime.
    return (type(value).__qualname__, id(value))


def _signature(kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
    result = []
    for key in sorted(kwargs):
        value = kwargs[key]
        signature = _value_signature(value)
        # Shape-only signatures are correct for model tensors, but not for
        # tiny control tensors that select a stateful KV cursor.  Include the
        # value for these explicitly named markers so a Commit--Draft graph
        # captured at prefix 283 is never replayed at prefix 284.
        if (
            torch.is_tensor(value)
            and value.numel() <= 16
            and ("prefix" in key.lower() or "length" in key.lower())
        ):
            signature = (signature, tuple(value.detach().cpu().reshape(-1).tolist()))
        result.append((key, signature))
    return tuple(result)


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _clone_inputs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _clone_value(value) for key, value in kwargs.items()}


def _copy_value_(destination: Any, source: Any, name: str) -> None:
    if torch.is_tensor(source) and torch.is_tensor(destination):
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError(f"stage graph input changed for {name}")
        destination.copy_(source)
        return
    if isinstance(source, (tuple, list)) and isinstance(destination, (tuple, list)):
        if len(destination) != len(source):
            raise ValueError(f"stage graph input changed for {name}")
        for index, (dest_item, source_item) in enumerate(zip(destination, source)):
            _copy_value_(dest_item, source_item, f"{name}[{index}]")
        return
    if isinstance(source, dict) and isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise ValueError(f"stage graph input changed for {name}")
        for key in source:
            _copy_value_(destination[key], source[key], f"{name}.{key}")
        return
    if destination is not source and destination != source:
        raise ValueError(f"stage graph input changed for {name}")


@dataclass
class StageGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_inputs: Dict[str, Any]
    outputs: Any
    stateful_cache: Any = None
    stateful_cache_state: Any = None

    @classmethod
    def capture(
        cls,
        fn,
        kwargs: Dict[str, Any],
        *,
        pool=None,
    ):
        device = next(
            value.device
            for value in kwargs.values()
            if torch.is_tensor(value) and value.is_cuda
        )
        static_inputs = _clone_inputs(kwargs)
        stateful_cache = kwargs.get("past_key_values")
        input_cache_state = None
        if hasattr(stateful_cache, "capture_graph_state"):
            input_cache_state = stateful_cache.capture_graph_state()

        def restore_capture_input():
            if input_cache_state is not None and hasattr(
                stateful_cache, "restore_graph_state"
            ):
                stateful_cache.restore_graph_state(input_cache_state)
            elif hasattr(stateful_cache, "reset"):
                stateful_cache.reset()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                # Stateful KV workspaces must begin every warm call from the
                # exact same logical cursor while retaining fixed addresses.
                restore_capture_input()
                warm_outputs = fn(**static_inputs)
                del warm_outputs
        stream.synchronize()
        restore_capture_input()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, pool=pool):
            outputs = fn(**static_inputs)
        torch.cuda.current_stream(device).wait_stream(stream)
        output_cache_state = None
        if hasattr(stateful_cache, "capture_graph_state"):
            output_cache_state = stateful_cache.capture_graph_state()
        # Capturing executes the graph once.  Restore the caller's input state;
        # StageGraphCache performs one explicit replay before returning.
        restore_capture_input()
        return cls(
            graph=graph,
            static_inputs=static_inputs,
            outputs=outputs,
            stateful_cache=stateful_cache,
            stateful_cache_state=(input_cache_state, output_cache_state),
        )

    def replay(self, kwargs: Dict[str, Any]):
        for key, value in kwargs.items():
            static = self.static_inputs.get(key)
            _copy_value_(static, value, key)
        input_cache_state, output_cache_state = self.stateful_cache_state or (
            None,
            None,
        )
        if hasattr(self.stateful_cache, "prepare_graph_replay"):
            self.stateful_cache.prepare_graph_replay(input_cache_state)
        elif hasattr(self.stateful_cache, "reset"):
            self.stateful_cache.reset()
        self.graph.replay()
        if hasattr(self.stateful_cache, "finish_graph_replay"):
            self.stateful_cache.finish_graph_replay(
                input_cache_state, output_cache_state
            )
        elif output_cache_state is not None and hasattr(
            self.stateful_cache, "restore_graph_state"
        ):
            self.stateful_cache.restore_graph_state(output_cache_state)
        return self.outputs


class StageGraphCache:
    """Small bounded cache with capture accounting and safe eager fallback."""

    def __init__(self, owner, mode: str, name: str, max_entries: int = 2):
        self.owner = owner
        self.mode = str(mode).lower()
        self.name = name
        self.max_entries = int(max_entries)
        self.entries: Dict[Tuple[Any, ...], StageGraphEntry] = {}

    def run(self, fn, kwargs: Dict[str, Any]):
        if (
            self.mode == "off"
            or not torch.cuda.is_available()
            or not any(torch.is_tensor(value) and value.is_cuda for value in kwargs.values())
        ):
            return fn(**kwargs), "eager"
        key = _signature(kwargs)
        entry = self.entries.get(key)
        if entry is not None:
            try:
                outputs = entry.replay(kwargs)
                counter = f"_flagos_{self.name}_graph_replays"
                setattr(
                    self.owner,
                    counter,
                    int(getattr(self.owner, counter, 0)) + 1,
                )
                return outputs, "replay"
            except Exception as exc:
                setattr(
                    self.owner,
                    f"_flagos_{self.name}_graph_last_error",
                    repr(exc),
                )
                self.entries.pop(key, None)
        if self.mode == "auto" and len(self.entries) >= self.max_entries:
            self.owner._flagos_stage_graph_fallbacks = int(
                getattr(self.owner, "_flagos_stage_graph_fallbacks", 0)
            ) + 1
            counter = f"_flagos_{self.name}_graph_fallbacks"
            setattr(self.owner, counter, int(getattr(self.owner, counter, 0)) + 1)
            return fn(**kwargs), "eager_fallback"
        begin = time.perf_counter()
        try:
            pool = getattr(self.owner, "_flagos_cuda_graph_pool", None)
            if pool is None and bool(getattr(self.owner, "_flagos_shared_graph_pool", True)):
                pool = torch.cuda.graph_pool_handle()
                self.owner._flagos_cuda_graph_pool = pool
            entry = StageGraphEntry.capture(fn, kwargs, pool=pool)
        except Exception as exc:
            setattr(
                self.owner,
                f"_flagos_{self.name}_graph_last_error",
                repr(exc),
            )
            print(
                f"[FlagOS/KERV] {self.name} graph capture fallback: {exc}",
                flush=True,
            )
            self.owner._flagos_stage_graph_last_traceback = traceback.format_exc()
            self.owner._flagos_stage_graph_fallbacks = int(
                getattr(self.owner, "_flagos_stage_graph_fallbacks", 0)
            ) + 1
            counter = f"_flagos_{self.name}_graph_fallbacks"
            setattr(self.owner, counter, int(getattr(self.owner, counter, 0)) + 1)
            return fn(**kwargs), "eager_fallback"
        self.entries[key] = entry
        self.owner._flagos_stage_graph_capture_overhead_s = float(
            getattr(self.owner, "_flagos_stage_graph_capture_overhead_s", 0.0)
        ) + max(0.0, time.perf_counter() - begin)
        self.owner._flagos_stage_graph_captures = int(
            getattr(self.owner, "_flagos_stage_graph_captures", 0)
        ) + 1
        counter = f"_flagos_{self.name}_graph_captures"
        setattr(self.owner, counter, int(getattr(self.owner, counter, 0)) + 1)
        return entry.replay(kwargs), "capture_replay"


@dataclass
class JointStageGraphEntry:
    """A bounded graph entry for the Commit--Draft fused control path.

    The verifier and drafter use independent persistent cache objects.  The
    normal ``StageGraphEntry`` can track one stateful cache, while this entry
    restores both cache cursors around capture/replay.  Tensor contents remain
    resident at their original addresses; only the small Python metadata is
    restored between replays.
    """

    graph: torch.cuda.CUDAGraph
    static_inputs: Dict[str, Any]
    outputs: Any
    stateful_caches: Tuple[Any, ...]
    input_states: Tuple[Any, ...]
    output_states: Tuple[Any, ...]

    @staticmethod
    def _snapshot(cache: Any) -> Any:
        if hasattr(cache, "capture_graph_state"):
            return cache.capture_graph_state()
        return None

    @staticmethod
    def _restore(cache: Any, state: Any) -> None:
        if state is not None and hasattr(cache, "restore_graph_state"):
            cache.restore_graph_state(state)

    @classmethod
    def capture(cls, fn, kwargs, stateful_caches, *, pool=None):
        device = next(
            value.device
            for value in kwargs.values()
            if torch.is_tensor(value) and value.is_cuda
        )
        static_inputs = _clone_inputs(kwargs)
        caches = tuple(cache for cache in stateful_caches if cache is not None)
        input_states = tuple(cls._snapshot(cache) for cache in caches)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))

        def restore_inputs() -> None:
            for cache, state in zip(caches, input_states):
                cls._restore(cache, state)

        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    restore_inputs()
                    warm_outputs = fn(**static_inputs)
                    del warm_outputs
            stream.synchronize()
            restore_inputs()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=pool):
                outputs = fn(**static_inputs)
            stream.synchronize()
            output_states = tuple(cls._snapshot(cache) for cache in caches)
            restore_inputs()
        except Exception:
            restore_inputs()
            raise
        return cls(
            graph=graph,
            static_inputs=static_inputs,
            outputs=outputs,
            stateful_caches=caches,
            input_states=input_states,
            output_states=output_states,
        )

    def replay(self, kwargs):
        for key, value in kwargs.items():
            static = self.static_inputs.get(key)
            _copy_value_(static, value, key)
        for cache, state in zip(self.stateful_caches, self.input_states):
            self._restore(cache, state)
        self.graph.replay()
        for cache, state in zip(self.stateful_caches, self.output_states):
            self._restore(cache, state)
        return self.outputs


class JointStageGraphCache:
    """CUDA Graph cache for one Commit plus one Draft forward.

    The cache is deliberately conservative: a changed tensor shape, cache
    layout, or failed capture returns ``None`` to the caller.  The caller then
    uses the existing two-stream/eager implementation without changing any
    output semantics.
    """

    def __init__(self, owner, mode: str, max_entries: int = 7):
        self.owner = owner
        self.mode = str(mode).lower()
        self.max_entries = int(max_entries)
        self.entries: Dict[Tuple[Any, ...], JointStageGraphEntry] = {}
        # A persistent verifier cache is reset at every action query, but its
        # storage addresses and layout remain stable.  Resetting the cache is
        # therefore not a reason to discard graph entries: the graph key also
        # contains the prefix marker value, and replay restores the captured
        # cursor before execution.  Only replacement of the actual workspace
        # invalidates entries.
        self._cache_generation: Optional[Tuple[Any, ...]] = None

    def run(self, fn, kwargs, *, stateful_caches=()):
        if self.mode == "off" or not torch.cuda.is_available() or not any(
            torch.is_tensor(value) and value.is_cuda for value in kwargs.values()
        ):
            return None, "disabled"
        generation = tuple(
            (
                id(cache),
                getattr(getattr(cache, "storage", None), "data_ptr", lambda: None)(),
                getattr(cache, "prefix_capacity", None),
                getattr(cache, "tree_capacity", None),
            )
            for cache in stateful_caches
            if cache is not None
        )
        if self._cache_generation != generation:
            self.entries.clear()
            self._cache_generation = generation
        key = _signature(kwargs)
        entry = self.entries.get(key)
        if entry is not None:
            try:
                outputs = entry.replay(kwargs)
                self.owner._flagos_joint_graph_replays = int(
                    getattr(self.owner, "_flagos_joint_graph_replays", 0)
                ) + 1
                return outputs, "replay"
            except Exception as exc:
                self.owner._flagos_joint_graph_last_error = repr(exc)
                self.entries.pop(key, None)
        if self.mode == "auto" and len(self.entries) >= self.max_entries:
            self.owner._flagos_joint_graph_fallbacks = int(
                getattr(self.owner, "_flagos_joint_graph_fallbacks", 0)
            ) + 1
            return None, "eager_fallback"
        begin = time.perf_counter()
        try:
            pool = getattr(self.owner, "_flagos_cuda_graph_pool", None)
            if pool is None and bool(getattr(self.owner, "_flagos_shared_graph_pool", True)):
                pool = torch.cuda.graph_pool_handle()
                self.owner._flagos_cuda_graph_pool = pool
            entry = JointStageGraphEntry.capture(
                fn, kwargs, tuple(stateful_caches), pool=pool
            )
        except Exception as exc:
            self.owner._flagos_joint_graph_last_error = repr(exc)
            self.owner._flagos_joint_graph_fallbacks = int(
                getattr(self.owner, "_flagos_joint_graph_fallbacks", 0)
            ) + 1
            return None, "eager_fallback"
        self.entries[key] = entry
        self.owner._flagos_joint_graph_captures = int(
            getattr(self.owner, "_flagos_joint_graph_captures", 0)
        ) + 1
        self.owner._flagos_joint_graph_capture_overhead_s = float(
            getattr(self.owner, "_flagos_joint_graph_capture_overhead_s", 0.0)
        ) + max(0.0, time.perf_counter() - begin)
        outputs = entry.replay(kwargs)
        self.owner._flagos_joint_graph_replays = int(
            getattr(self.owner, "_flagos_joint_graph_replays", 0)
        ) + 1
        return outputs, "capture_replay"

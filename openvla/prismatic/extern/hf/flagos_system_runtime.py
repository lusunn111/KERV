"""Lossless runtime helpers for the FlagOS KERV integration.

The module deliberately contains no model or operator logic.  It only removes
bookkeeping overhead from the hot loop: CUDA event timing, asynchronous JSONL
logging, and reusable batch-one decode buffers.  Every helper has a conservative
native fallback so an unsupported layout never changes KERV semantics.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


def _parse_cpu_set(value: str) -> set[int]:
    """Parse Linux CPU-list syntax (for example ``0-27,56-83``)."""

    result: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            begin, end = part.split("-", 1)
            result.update(range(int(begin), int(end) + 1))
        else:
            result.add(int(part))
    return result


def resolve_gpu_cpu_affinity() -> Optional[set[int]]:
    """Return the CPU affinity reported by ``nvidia-smi topo -m``.

    The helper is intentionally best effort.  Containers without
    ``nvidia-smi`` or without topology permission simply keep the inherited
    affinity; no inference path depends on pinning.
    """

    explicit = os.environ.get("KERV_NUMA_CPUSET", "").strip()
    if explicit:
        try:
            cpus = _parse_cpu_set(explicit)
            return cpus or None
        except (TypeError, ValueError):
            return None
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = next((line.split() for line in lines if "CPU" in line and "Affinity" in line), None)
    if not header:
        return None
    try:
        # ``split`` separates the two-word column label into CPU/Affinity.
        cpu_index = header.index("CPU") + 1
    except ValueError:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        gpu_row = next(
            line.split() for line in lines if line.split() and line.split()[0] == f"GPU{int(visible)}"
        )
        return _parse_cpu_set(gpu_row[cpu_index])
    except (StopIteration, ValueError, IndexError):
        return None


def apply_numa_affinity(mode: str = "auto") -> Optional[str]:
    """Pin the process to the GPU-local CPU set when explicitly enabled."""

    mode = str(mode).lower()
    if mode not in {"auto", "on", "off"} or mode == "off":
        return None
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = resolve_gpu_cpu_affinity()
    if not cpus:
        return None
    try:
        os.sched_setaffinity(0, cpus)
    except (OSError, ValueError):
        return None
    return ",".join(str(cpu) for cpu in sorted(cpus))


def schedule_episode_device_warmup(
    model,
    *,
    mode: str = "off",
    duration_ms: str | float = "auto",
) -> bool:
    """Warm GPU clocks while LIBERO performs its mandatory settle steps.

    This only uses a private CUDA stream and two reusable BF16 matrices.  It
    never touches model parameters, KV caches, RNG state, or action history.
    The caller must invoke :func:`wait_episode_device_warmup` before the first
    measured action.
    """

    mode = str(mode).lower()
    if mode not in {"auto", "on"} or not torch.cuda.is_available():
        return False
    device = next(
        (parameter.device for parameter in model.parameters() if parameter.is_cuda),
        torch.device("cuda"),
    )
    try:
        duration = 50.0 if str(duration_ms).lower() == "auto" else float(duration_ms)
    except (TypeError, ValueError):
        duration = 50.0
    duration = max(10.0, min(duration, 250.0))
    buffers = getattr(model, "_flagos_episode_warmup_buffers", None)
    if buffers is None or buffers[0].device != device:
        left = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
        right = torch.randn_like(left)
        buffers = (left, right)
        model._flagos_episode_warmup_buffers = buffers
    stream = getattr(model, "_flagos_episode_warmup_stream", None)
    if stream is None or stream.device != device:
        # The warmup is deliberately lower priority than model work.  Keep a
        # portable fallback for older PyTorch builds that do not accept the
        # ``priority`` keyword.
        try:
            stream = torch.cuda.Stream(device=device, priority=2)
        except TypeError:
            stream = torch.cuda.Stream(device=device, priority=0)
        model._flagos_episode_warmup_stream = stream
    event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        # A bounded fixed count avoids a host-side polling loop.  The GEMM
        # stream overlaps with LIBERO's dummy environment steps.
        iterations = max(8, min(256, int(duration / 1.5)))
        for _ in range(iterations):
            torch.mm(buffers[0], buffers[1])
        event.record(stream)
    model._flagos_episode_warmup_event = event
    model._flagos_episode_warmup_pending = True
    model._flagos_episode_warmup_duration_ms = float(duration)
    model._flagos_episode_warmup_duration_ms_value = float(duration)
    model._flagos_episode_warmup_scheduled = int(
        getattr(model, "_flagos_episode_warmup_scheduled", 0)
    ) + 1
    return True


def wait_episode_device_warmup(model) -> float:
    """Make the default stream wait for the overlapped device warmup."""

    if not bool(getattr(model, "_flagos_episode_warmup_pending", False)):
        return 0.0
    event = getattr(model, "_flagos_episode_warmup_event", None)
    if event is None or not torch.cuda.is_available():
        model._flagos_episode_warmup_pending = False
        return 0.0
    device = event.device if hasattr(event, "device") else torch.device("cuda")
    torch.cuda.current_stream(device).wait_event(event)
    model._flagos_episode_warmup_pending = False
    model._flagos_episode_warmup_waits = int(
        getattr(model, "_flagos_episode_warmup_waits", 0)
    ) + 1
    return float(getattr(model, "_flagos_episode_warmup_duration_ms", 0.0))


class AsyncJsonlWriter:
    """Bounded background writer used for optional runtime diagnostics."""

    _STOP = object()

    def __init__(self, path: Path, max_queue: int = 256) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max_queue)
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="flagos-kerv-stats-writer",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            pending = 0
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        output.flush()
                        return
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    # Runtime diagnostics are not on the action critical
                    # path.  Flush in small batches rather than forcing a
                    # filesystem sync for every action record.
                    pending += 1
                    if pending >= 32:
                        output.flush()
                        pending = 0
                finally:
                    self._queue.task_done()

    def write(self, value: Dict[str, object]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(dict(value))
        except queue.Full:
            # Diagnostics must never block the inference stream.  The final
            # close() call waits for all records that did fit in the queue.
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.join()
        self._queue.put(self._STOP)
        self._thread.join(timeout=5.0)


@dataclass
class PhaseHandle:
    name: str
    start_event: Optional[torch.cuda.Event]
    cpu_start: float
    sample_index: int = -1


class PhaseTimer:
    """Record phases asynchronously and synchronize once per action.

    ``end`` returns a CPU wall-clock estimate for code that needs a scalar
    immediately.  ``resolve`` replaces it with CUDA event elapsed time after
    one final synchronization, which is the value emitted in runtime stats.
    """

    def __init__(self, device: torch.device, enabled: bool, backend: str) -> None:
        self.device = device
        self.enabled = bool(enabled)
        self.use_events = bool(
            self.enabled
            and backend == "cuda_event"
            and torch.cuda.is_available()
            and device.type == "cuda"
        )
        self._samples: List[Tuple[PhaseHandle, Optional[torch.cuda.Event]]] = []
        self._values: Dict[str, List[float]] = {}

    def start(self, name: str) -> PhaseHandle:
        event = None
        if self.use_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        return PhaseHandle(name=name, start_event=event, cpu_start=time.perf_counter())

    def end(self, handle: PhaseHandle) -> float:
        cpu_elapsed = time.perf_counter() - handle.cpu_start
        end_event = None
        if self.use_events and handle.start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(self.device))
        values = self._values.setdefault(handle.name, [])
        handle.sample_index = len(values)
        self._samples.append((handle, end_event))
        values.append(cpu_elapsed)
        return cpu_elapsed

    def resolve(self) -> Dict[str, List[float]]:
        if self.use_events and self._samples:
            # This is the only synchronization introduced by runtime stats for
            # the action.  Data-dependent .item()/to(cpu) calls remain intact.
            torch.cuda.synchronize(self.device)
            for handle, end_event in self._samples:
                if handle.start_event is not None and end_event is not None:
                    value = handle.start_event.elapsed_time(end_event) / 1000.0
                    values = self._values.get(handle.name, [])
                    if 0 <= handle.sample_index < len(values):
                        values[handle.sample_index] = max(0.0, value)
        return {key: list(values) for key, values in self._values.items()}


class DecodeWorkspace:
    """Reusable batch-one token/embedding buffers for KERV's update path."""

    def __init__(
        self,
        *,
        device: torch.device,
        token_dtype: torch.dtype,
        embed_dtype: torch.dtype,
        hidden_size: int,
        token_capacity: int,
        embed_capacity: int,
        fixed_draft_input: bool = False,
    ) -> None:
        self.device = device
        self.token_capacity = int(token_capacity)
        self.embed_capacity = int(embed_capacity)
        self.hidden_size = int(hidden_size)
        self.input_ids = torch.empty(
            (1, self.token_capacity), device=device, dtype=token_dtype
        )
        # ``input_ids`` is the committed history.  The token sampled after a
        # Verify pass is only a drafter input and must not advance that
        # history cursor (the native path builds it with a temporary cat).
        # In the fixed-input mode, the temporary sampled token occupies the
        # first free slot in the committed buffer.  The committed view keeps
        # its shorter length, and the next accepted append overwrites that
        # slot.  This removes a growing prefix D2D copy on every Draft cycle.
        self.fixed_draft_input = bool(fixed_draft_input)
        self.draft_input_ids = (
            self.input_ids
            if self.fixed_draft_input
            else torch.empty_like(self.input_ids)
        )
        self.prompt_embeds = torch.empty(
            (1, self.embed_capacity, self.hidden_size),
            device=device,
            dtype=embed_dtype,
        )
        self.draft_embeds = (
            self.prompt_embeds
            if self.fixed_draft_input
            else torch.empty_like(self.prompt_embeds)
        )
        self.token_embed = torch.empty(
            (1, 1, self.hidden_size), device=device, dtype=embed_dtype
        )
        self._token_length = 0
        self._embed_length = 0

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        max_new_tokens: int,
        fixed_draft_input: bool = False,
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return None
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            return None
        # Keep capacity bounded for long prompts while covering a full action
        # rollout and one verifier append without reallocating.
        token_capacity = max(int(input_ids.shape[1]) + int(max_new_tokens) + 16, 512)
        embed_capacity = max(int(prompt_embeds.shape[1]) + int(max_new_tokens) + 16, 512)
        return cls(
            device=input_ids.device,
            token_dtype=input_ids.dtype,
            embed_dtype=prompt_embeds.dtype,
            hidden_size=int(prompt_embeds.shape[-1]),
            token_capacity=token_capacity,
            embed_capacity=embed_capacity,
            fixed_draft_input=fixed_draft_input,
        )

    def reset(self, input_ids: torch.Tensor, prompt_embeds: torch.Tensor) -> bool:
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or prompt_embeds.ndim != 3
            or prompt_embeds.shape[0] != 1
            or input_ids.device != self.device
            or prompt_embeds.device != self.device
            or prompt_embeds.shape[-1] != self.hidden_size
            or input_ids.shape[1] > self.token_capacity
            or prompt_embeds.shape[1] > self.embed_capacity
        ):
            return False
        self.input_ids[:, : input_ids.shape[1]].copy_(input_ids)
        self.prompt_embeds[:, : prompt_embeds.shape[1]].copy_(prompt_embeds)
        self._token_length = int(input_ids.shape[1])
        self._embed_length = int(prompt_embeds.shape[1])
        return True

    def append_tokens(self, values: torch.Tensor) -> Optional[torch.Tensor]:
        values = values.to(device=self.device, dtype=self.input_ids.dtype)
        if values.ndim == 1:
            values = values.view(1, -1)
        if values.ndim != 2 or values.shape[0] != 1:
            return None
        end = self._token_length + int(values.shape[1])
        if end > self.token_capacity:
            return None
        self.input_ids[:, self._token_length : end].copy_(values)
        self._token_length = end
        return self.input_ids[:, :end]

    def draft_token_input(
        self, input_ids: torch.Tensor, token: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Build ``input_ids + token`` without changing committed history."""
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.device != self.device
            or input_ids.shape[1] + 1 > self.token_capacity
        ):
            return None
        token = token.to(device=self.device, dtype=self.draft_input_ids.dtype)
        if token.ndim == 1:
            token = token.view(1, -1)
        if token.ndim != 2 or token.shape[0] != 1 or token.shape[1] != 1:
            return None
        length = int(input_ids.shape[1])
        if not self.fixed_draft_input:
            self.draft_input_ids[:, :length].copy_(input_ids)
        self.draft_input_ids[:, length : length + 1].copy_(token)
        return self.draft_input_ids[:, : length + 1]

    def append_embeddings(self, values: torch.Tensor) -> Optional[torch.Tensor]:
        values = values.to(device=self.device, dtype=self.prompt_embeds.dtype)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[-1] != self.hidden_size:
            return None
        end = self._embed_length + int(values.shape[1])
        if end > self.embed_capacity:
            return None
        self.prompt_embeds[:, self._embed_length : end].copy_(values)
        self._embed_length = end
        return self.prompt_embeds[:, :end]

    def append_token_embedding(self, values: torch.Tensor) -> Optional[torch.Tensor]:
        values = values.to(device=self.device, dtype=self.prompt_embeds.dtype)
        if values.ndim != 3 or values.shape != self.token_embed.shape:
            return None
        self.token_embed.copy_(values)
        return self.token_embed

    def draft_input(self, prompt_embeds: torch.Tensor, token_embed: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            prompt_embeds.ndim != 3
            or token_embed.ndim != 3
            or prompt_embeds.shape[0] != 1
            or token_embed.shape != (1, 1, self.hidden_size)
            or prompt_embeds.shape[1] + 1 > self.embed_capacity
        ):
            return None
        length = int(prompt_embeds.shape[1])
        if not self.fixed_draft_input:
            self.draft_embeds[:, :length].copy_(prompt_embeds)
        self.draft_embeds[:, length : length + 1].copy_(token_embed)
        return self.draft_embeds[:, : length + 1]


class ControlBuffer:
    """Reusable device/host control exchange for Verify-Accept decisions."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.device_values = torch.empty((3,), device=device, dtype=torch.long)
        if device.type == "cuda":
            self.host_values = torch.empty((3,), dtype=torch.long, pin_memory=True)
        else:
            self.host_values = torch.empty((3,), dtype=torch.long)
        self.transfers = 0

    def _read(self, count: int) -> List[int]:
        self.host_values[:count].copy_(self.device_values[:count], non_blocking=True)
        if self.device.type == "cuda":
            # This is the one real data dependency for the control decision;
            # it replaces temporary stack/CPU copies, not model computation.
            torch.cuda.current_stream(self.device).synchronize()
        self.transfers += 1
        return [int(value) for value in self.host_values[:count].tolist()]

    def read_pair(self, first: torch.Tensor, second: torch.Tensor) -> List[int]:
        self.device_values[0].copy_(first.to(device=self.device, dtype=torch.long))
        self.device_values[1].copy_(second.to(device=self.device, dtype=torch.long))
        return self._read(2)

    def read_bundle(
        self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor
    ) -> List[int]:
        self.device_values[0].copy_(first.to(device=self.device, dtype=torch.long))
        self.device_values[1].copy_(second.to(device=self.device, dtype=torch.long))
        self.device_values[2].copy_(third.to(device=self.device, dtype=torch.long))
        return self._read(3)

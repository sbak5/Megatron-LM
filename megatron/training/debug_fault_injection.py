# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional fault injection for resiliency testing via nvidia-resiliency-ext.

Configure via CLI (see ``arguments.py``) or environment variables. CLI wins when set.

Environment variables::

    MEGATRON_FAULT_TYPE   - Fault enum name (GPU_SLEEP, GPU_ERROR, SEGFAULT, …)
    MEGATRON_FAULT_AT_ITER - 0-based iteration at which to inject
    MEGATRON_FAULT_RANK   - global rank to inject (omit for all ranks)
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch

from megatron.training.utils import print_rank_0


def _getenv_int(name: str) -> Optional[int]:
    s = os.environ.get(name)
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        print_rank_0(f"[MEGATRON_FAULT] Invalid {name}={s!r}, ignoring.")
        return None


class MegatronDebugFaultInjection:
    """Iteration- and rank-gated fault injection via nvidia-resiliency-ext."""

    def maybe_run(self, iteration: int, args: Any) -> None:
        """Inject the configured fault when ``iteration`` and rank match."""
        from nvidia_resiliency_ext.shared_utils.inject_fault import Fault, dispatch_fault_injection

        fault_type_str = getattr(args, 'megatron_fault_type', None) or os.environ.get(
            'MEGATRON_FAULT_TYPE'
        )
        at_iter = getattr(args, 'megatron_fault_at_iter', None)
        if at_iter is None:
            at_iter = _getenv_int('MEGATRON_FAULT_AT_ITER')
        fault_rank = getattr(args, 'megatron_fault_rank', None)
        if fault_rank is None:
            fault_rank = _getenv_int('MEGATRON_FAULT_RANK')

        if fault_type_str is None or at_iter is None:
            return
        if not torch.distributed.is_initialized():
            return
        if iteration != at_iter:
            return

        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()

        if fault_rank is not None and rank != int(fault_rank):
            return

        try:
            fault = Fault[fault_type_str]
        except KeyError:
            print_rank_0(
                f"[MEGATRON_FAULT] Unknown fault type {fault_type_str!r}. "
                f"Valid types: {[f.name for f in Fault]}"
            )
            return

        print(
            f"[MEGATRON_FAULT] global_rank={rank}/{world - 1}: "
            f"injecting {fault.name} at iteration {iteration}",
            flush=True,
        )

        dispatch_fault_injection(fault, 0, None)

        if fault in (Fault.GPU_SLEEP, Fault.GPU_ERROR):
            # Give the background thread a moment to submit the GPU work, then
            # synchronize on the main thread so training actually blocks/crashes.
            time.sleep(0.1)
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"[MEGATRON_FAULT] CUDA error after {fault.name}: {exc}", flush=True)
        else:
            # For signal/abort/exception faults give the dispatcher time to act.
            time.sleep(1)


_default_injector: Optional[MegatronDebugFaultInjection] = None


def get_megatron_debug_fault_injection() -> MegatronDebugFaultInjection:
    global _default_injector
    if _default_injector is None:
        _default_injector = MegatronDebugFaultInjection()
    return _default_injector


def maybe_megatron_debug_fault_injection(iteration: int, args: Any = None) -> None:
    """Entry point used from the training loop; ``args`` defaults to ``get_args()``."""
    if args is None:
        from megatron.training.global_vars import get_args

        args = get_args()
    get_megatron_debug_fault_injection().maybe_run(iteration, args)

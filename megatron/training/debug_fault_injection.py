# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional GPU hang / illegal-access fault injection for resiliency testing.

Configure via CLI (see ``arguments.py``) or environment variables. CLI wins when set.

Environment (backward compatible)::

    MEGATRON_FAULT_HANG_AT_ITER, MEGATRON_FAULT_HANG_RANK
    MEGATRON_FAULT_CRASH_AT_ITER, MEGATRON_FAULT_CRASH_RANK
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional, Tuple

import torch

from megatron.training.utils import print_rank_0

_MEGATRON_FAULT_GPU_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void megatron_fault_illegal_access_kernel() {
  int* p = nullptr;
  *p = 42;
}

void megatron_fault_illegal_access_launch() {
  megatron_fault_illegal_access_kernel<<<1, 1>>>();
}

// Spins until *flag != 0 (host keeps it zero → GPU busy forever).
__global__ void megatron_fault_spin_kernel(volatile int* flag) {
  while (*flag == 0) {
  }
}

void megatron_fault_hang_launch(int64_t flag_dev_ptr) {
  volatile int* p = reinterpret_cast<volatile int*>(static_cast<uintptr_t>(flag_dev_ptr));
  megatron_fault_spin_kernel<<<1, 1>>>(p);
}
"""

_MEGATRON_FAULT_GPU_CPP_DECLS = """
void megatron_fault_illegal_access_launch();
void megatron_fault_hang_launch(int64_t flag_dev_ptr);
"""


def _parse_optional_rank(
    args: Any, attr: str, env_name: str
) -> Tuple[Optional[int], Optional[str]]:
    """Return (rank or None if unset, invalid_env_string or None)."""
    v = getattr(args, attr, None)
    if v is not None:
        return int(v), None
    s = os.environ.get(env_name)
    if s is None:
        return None, None
    try:
        return int(s), None
    except ValueError:
        return None, s


def _parse_optional_iter(
    args: Any, attr: str, env_name: str
) -> Tuple[Optional[int], Optional[str]]:
    """Return (iteration or None if unset, invalid_env_string or None)."""
    v = getattr(args, attr, None)
    if v is not None:
        return int(v), None
    s = os.environ.get(env_name)
    if s is None:
        return None, None
    try:
        return int(s), None
    except ValueError:
        return None, s


class MegatronDebugFaultInjection:
    """JIT-compiled CUDA kernels for intentional GPU hang or crash; rank/iter from args + env."""

    _gpu_ext = None

    @classmethod
    def _get_gpu_ext(cls):
        if cls._gpu_ext is None:
            from torch.utils.cpp_extension import load_inline

            cls._gpu_ext = load_inline(
                name="megatron_fault_gpu",
                cpp_sources=_MEGATRON_FAULT_GPU_CPP_DECLS,
                cuda_sources=_MEGATRON_FAULT_GPU_CUDA,
                functions=[
                    "megatron_fault_illegal_access_launch",
                    "megatron_fault_hang_launch",
                ],
                with_cuda=True,
                verbose=False,
            )
        return cls._gpu_ext

    def launch_gpu_illegal_access_kernel(self) -> None:
        """Enqueue illegal access; caller should ``torch.cuda.synchronize()``."""
        if not torch.cuda.is_available():
            print(
                "[MEGATRON_FAULT] CUDA not available; cannot trigger GPU illegal access. "
                "Falling back to os._exit(139).",
                flush=True,
            )
            os._exit(139)
        self._get_gpu_ext().megatron_fault_illegal_access_launch()

    def launch_gpu_hang_kernel(self):
        """Enqueue spin on a device flag (stays 0); caller must ``synchronize()``."""
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._get_gpu_ext().megatron_fault_hang_launch(flag.data_ptr())
        return flag

    def maybe_run(self, iteration: int, args: Any) -> None:
        """If configured, hang or crash this rank at ``iteration`` (0-based, main-loop convention)."""
        hang_target, hang_bad = _parse_optional_iter(
            args, "megatron_fault_hang_at_iter", "MEGATRON_FAULT_HANG_AT_ITER"
        )
        crash_target, crash_bad = _parse_optional_iter(
            args, "megatron_fault_crash_at_iter", "MEGATRON_FAULT_CRASH_AT_ITER"
        )
        if hang_bad is not None:
            print_rank_0(
                f"[MEGATRON_FAULT] Invalid MEGATRON_FAULT_HANG_AT_ITER={hang_bad!r}"
            )
            hang_target = None
        if crash_bad is not None:
            print_rank_0(
                f"[MEGATRON_FAULT] Invalid MEGATRON_FAULT_CRASH_AT_ITER={crash_bad!r}"
            )
            crash_target = None

        if hang_target is None and crash_target is None:
            return
        if not torch.distributed.is_initialized():
            return

        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()

        if hang_target is not None and iteration == hang_target:
            hang_ok = True
            hang_rank, hr_bad = _parse_optional_rank(
                args, "megatron_fault_hang_rank", "MEGATRON_FAULT_HANG_RANK"
            )
            if hr_bad is not None:
                print_rank_0(
                    f"[MEGATRON_FAULT] Invalid MEGATRON_FAULT_HANG_RANK={hr_bad!r}"
                )
                hang_ok = False
            elif hang_rank is not None:
                hang_ok = rank == hang_rank
            if hang_ok:
                print(
                    f"[MEGATRON_FAULT] global_rank={rank}/{world - 1}: GPU spin + sync hang at "
                    f"iteration {iteration}. Kill the job to recover.",
                    flush=True,
                )
                if torch.cuda.is_available():
                    try:
                        self.launch_gpu_hang_kernel()
                        torch.cuda.synchronize()
                    except Exception as exc:
                        print(
                            f"[MEGATRON_FAULT] GPU hang failed ({type(exc).__name__}: {exc}); "
                            "falling back to CPU sleep.",
                            flush=True,
                        )
                        while True:
                            time.sleep(86400)
                else:
                    while True:
                        time.sleep(86400)

        if crash_target is None or iteration != crash_target:
            return

        crash_rank, cr_bad = _parse_optional_rank(
            args, "megatron_fault_crash_rank", "MEGATRON_FAULT_CRASH_RANK"
        )
        if cr_bad is not None:
            print_rank_0(
                f"[MEGATRON_FAULT] Invalid MEGATRON_FAULT_CRASH_RANK={cr_bad!r}"
            )
            return
        if crash_rank is not None and rank != crash_rank:
            return

        print(
            f"[MEGATRON_FAULT] global_rank={rank}: GPU illegal access at iteration "
            f"{iteration}",
            flush=True,
        )
        try:
            self.launch_gpu_illegal_access_kernel()
        except Exception as exc:
            print(
                f"[MEGATRON_FAULT] GPU crash JIT/launch failed ({type(exc).__name__}: {exc}); "
                "falling back to os._exit(139).",
                flush=True,
            )
            os._exit(139)
        torch.cuda.synchronize()


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

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Optional

from ..device import get_torch_device
from ..memory_utils import enable_memory_visualize
from .config import ProfilerConfig, TorchMemoryToolConfig


class TorchMemoryProfiler:
    """Profiler that dumps CUDA memory snapshots at step boundaries.

    Behavior:
    - On construction, do not record allocations (unselected steps stay unprofiled).
    - On start(step=X), enable allocator history and remember this step's subdirectory.
    - On stop(), dump a snapshot, then disable history even if dumping fails.
    - Propagate recording/dump failures instead of silently producing no artifact.
    """

    _memory_history_enabled: bool = False

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[TorchMemoryToolConfig] = None
    ):
        # Always respond to explicit start/stop calls for torch_memory tool,
        # regardless of per-role enable flag, to align with global step control.
        self.enable = True
        if not config:
            config = ProfilerConfig(ranks=[])
        self.config = config
        self.rank = rank
        self.this_step = False
        self.sub_dir = None

        # Get parameters from tool_config, with fallback to defaults
        if tool_config:
            self.trace_alloc_max_entries = tool_config.trace_alloc_max_entries
            self.stack_depth = tool_config.stack_depth
        else:
            self.trace_alloc_max_entries = 100_000
            self.stack_depth = 32

    def start(self, **kwargs):
        if not self.enable or not self._should_profile_this_rank() or self.this_step:
            return
        device = get_torch_device()
        if not device.is_available():
            raise RuntimeError("Memory-only profiling requires an available accelerator")
        enable_memory_visualize(trace_alloc_max_entries=self.trace_alloc_max_entries, stack_depth=self.stack_depth)
        TorchMemoryProfiler._memory_history_enabled = True
        profile_step = kwargs.get("profile_step", None)
        # Keep ranks aligned under same folder name
        self.sub_dir = f"step{profile_step}" if profile_step is not None else None
        self.this_step = True
        print(f"[MemoryProfiler] Recording step {profile_step}, rank {self.rank}", flush=True)

    def stop(self):
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        device = get_torch_device()
        try:
            out_dir = Path(self.config.save_path or "outputs/profile")
            if self.sub_dir is not None:
                out_dir = out_dir / self.sub_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"torch_memory_rank{self.rank}_pid{os.getpid()}.pickle"
            device.synchronize()
            device.memory._dump_snapshot(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Memory profiler did not produce a non-empty snapshot: {path}")
            print(f"[MemoryProfiler] Saved snapshot: {path}", flush=True)
        finally:
            # Resetting via clear_memory_history would re-enable recording and
            # continue profiling later, unselected steps. Disable it instead.
            if TorchMemoryProfiler._memory_history_enabled:
                device.memory._record_memory_history(enabled=None)
                TorchMemoryProfiler._memory_history_enabled = False

    def _should_profile_this_rank(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        # default rank 0
        return self.rank == 0

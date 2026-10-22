#!/usr/bin/env python3
"""Occupy Ascend NPU memory with a persistent bf16 tensor and keep the
device busy with matmul loops.

Usage (per card):
    ASCEND_RT_VISIBLE_DEVICES=<npu_id> python3 occupy_npu.py --rank 0 --gib 60
"""

import argparse
import os
import time

import torch
import torch_npu  # noqa: F401  (registers the npu backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="local rank; maps to npu:{rank}")
    parser.add_argument("--gib", type=int, default=60, help="GiB of bf16 tensor to keep resident")
    args = parser.parse_args()

    device = torch.device(f"npu:{args.rank}")
    torch.npu.set_device(device)

    # Reserve in chunks so the allocator can fit exactly ~gib GiB on a 64 GiB card.
    bytes_target = args.gib * 1024**3
    chunks = []
    chunk_bytes = 8 * 1024**3  # 8 GiB per chunk
    while sum(t.numel() * t.element_size() for t in chunks) < bytes_target:
        try:
            n = (chunk_bytes // 2)  # bf16: 2 bytes/elem
            chunks.append(torch.empty(n, dtype=torch.bfloat16, device=device))
        except torch_npu.NPUOutOfMemoryError:  # noqa: PERF203
            chunk_bytes //= 2
            if chunk_bytes < 1024**3:
                break
    held = sum(t.numel() * t.element_size() for t in chunks) / 1024**3
    print(f"[rank {args.rank}] held {held:.1f} GiB on {device}", flush=True)

    # Keep the device busy: matmul on a resident block, forever.
    block = chunks[0][:8192] if chunks else None
    a = block.unsqueeze(0) if block is not None else torch.empty(0, dtype=torch.bfloat16, device=device)
    tick = time.time()
    while True:
        if a.numel() > 0:
            _ = torch.matmul(a, a.transpose(1, 2))
        if time.time() - tick > 300:
            print(f"[rank {args.rank}] still holding {held:.1f} GiB", flush=True)
            tick = time.time()


if __name__ == "__main__":
    main()

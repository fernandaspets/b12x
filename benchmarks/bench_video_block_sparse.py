# SPDX-License-Identifier: Apache-2.0
"""Dense vs block-list sparse attention at MiniMax-H3 geometry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_video_block_sparse.py

The sparse list is a per-tile +/- radius window of 64-token blocks, so the density is about
(2*radius+1)/num_k_blocks -- the FastH3 VSA operating point is ~90% sparsity (density 0.10).
"""
from __future__ import annotations

import argparse
import subprocess

import torch

from b12x.attention import varlen
from b12x.preparation import PreparationSession, PreparedCall, require_prepared

TILE_M, BLOCK_K = 128, 64


def _window(S, radius, device):
    num_blocks = (S + BLOCK_K - 1) // BLOCK_K
    num_tiles = (S + TILE_M - 1) // TILE_M
    idx, off = [], [0]
    for m in range(num_tiles):
        c = (m * TILE_M) // BLOCK_K
        idx.extend(range(max(0, c - radius), min(num_blocks, c + radius + 1)))
        off.append(len(idx))
    return (torch.tensor(idx, device=device, dtype=torch.int32),
            torch.tensor(off, device=device, dtype=torch.int32))


def _time(S, H, D, radius, device, iters, warmup):
    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, S], device=device, dtype=torch.int32)
    sparse = radius is not None
    bi, bo = _window(S, radius, device) if sparse else (None, None)
    declaration = varlen.plan(
        q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False,
        block_sparse=sparse,
        num_q_tiles=((S + TILE_M - 1) // TILE_M if sparse else 0),
        total_blocks_cap=(int(bo[-1].item()) if sparse else 0),
    )
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        def prepare(state):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch,))

        session.prepare((declaration.request(name=f"vb-{sparse}", prepare_call=prepare),))
        state = require_prepared(declaration, "attention.varlen", device)
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S,
                              block_indices=bi, block_offsets=bo)
        for _ in range(warmup):
            state.run(binding)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            state.run(binding)
        end.record()
        torch.cuda.synchronize()
    density = (bo[-1].item() / (((S + TILE_M - 1) // TILE_M) * ((S + BLOCK_K - 1) // BLOCK_K))
               if sparse else 1.0)
    return start.elapsed_time(end) / iters, density


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=18748, help="H3 tokens per rank at USP=2")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--radius", type=int, default=14, help="blocks; ~90% sparsity at 293 blocks")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    limit = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits",
                            "-i", str(device.index)], capture_output=True, text=True).stdout.strip()
    print(f"# {props.name} sm={props.multi_processor_count} power_limit={limit}W commit={commit} torch={torch.__version__}")
    print(f"# S={a.seq} H={a.heads} D={a.head_dim} tile_m={TILE_M} block_k={BLOCK_K}")

    dense_ms, _ = _time(a.seq, a.heads, a.head_dim, None, device, a.iters, a.warmup)
    sparse_ms, density = _time(a.seq, a.heads, a.head_dim, a.radius, device, a.iters, a.warmup)
    print(f"  dense  {dense_ms:8.3f} ms/call")
    print(f"  sparse {sparse_ms:8.3f} ms/call  density={density:.4f}")
    print(f"  speedup {dense_ms / sparse_ms:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

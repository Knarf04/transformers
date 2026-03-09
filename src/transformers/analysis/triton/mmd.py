import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 256}, num_warps=4),
        triton.Config({"BLOCK_S": 512}, num_warps=8),
        triton.Config({"BLOCK_S": 1024}, num_warps=8),
        triton.Config({"BLOCK_S": 2048}, num_warps=8),
    ],
    key=["S"],
)
@triton.jit
def _mmd_ssd_last_kernel(
    # Pointers
    A_dt_ptr,   # (B, H, S) — already A * dt after softplus/clamp, permuted
    Q_ptr,      # (B, S)    — precomputed dot(C_last, B) per (b, s)
    dt_hks_ptr, # (B, H, S) — dt permuted to (B, H, S)
    out_ptr,    # (B, H)    — output mmd
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    # Strides for A_dt (B, H, S)
    stride_a_b, stride_a_h, stride_a_s,
    # Strides for Q (B, S)
    stride_q_b, stride_q_s,
    # Strides for dt_hks (B, H, S)
    stride_d_b, stride_d_h, stride_d_s,
    # Strides for out (B, H)
    stride_o_b, stride_o_h,
    # Block size
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # ---- Pass 1: compute suffix sum of A_dt from right to left ----
    # tail_sum[s] = A_dt[s+1] + A_dt[s+2] + ... + A_dt[S-1]
    # We accumulate from right, then L_last[s] = exp(tail_sum[s])
    #
    # We also need Q[b, s] * L_last[s] * dt_hks[b, h, s] for each s.
    # Then: aw[s] = |raw[s]|, normalized, weighted by distance (S-1-s).
    #
    # Two-pass approach:
    #   Pass 1: right-to-left to compute suffix sums and raw values, store aw_abs and sum
    #   Pass 2: left-to-right to compute normalized weighted sum

    # We process in blocks. First, right-to-left to get suffix sums.
    # For suffix sum: we need to process from right to left.
    # running_sum starts at 0 (for s = S-1, tail_sum = 0)

    a_base = A_dt_ptr + b * stride_a_b + h * stride_a_h
    q_base = Q_ptr + b * stride_q_b
    d_base = dt_hks_ptr + b * stride_d_b + h * stride_d_h

    # Number of full blocks
    num_blocks = tl.cdiv(S, BLOCK_S)

    # We need to store |raw| values for the normalization pass.
    # Use a two-pass approach: first compute sum of |raw|, then compute weighted sum.

    # ---- Right-to-left pass: compute sum_abs and sum_weighted ----
    running_suffix = 0.0
    sum_abs = 0.0
    sum_weighted = 0.0

    for block_idx in range(num_blocks - 1, -1, -1):
        offs = block_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < S

        # Load A_dt values for this block
        a_vals = tl.load(a_base + offs * stride_a_s, mask=mask, other=0.0)

        # Compute local suffix sum within block (right to left)
        # For position offs[i], tail_sum = local_suffix_from_right + running_suffix
        # We need exclusive suffix sum: tail_sum[s] = sum of A_dt[s+1..S-1]
        #
        # Within a block [start, start+BLOCK_S), process right to left:
        # For the rightmost element in the block: its suffix = running_suffix
        # Moving left: suffix accumulates A_dt values to its right

        # Reverse the block for cumsum, then reverse back
        # a_vals is indexed left to right: a_vals[0] = A_dt[block_start], ..., a_vals[BLOCK_S-1] = A_dt[block_start+BLOCK_S-1]
        # We want exclusive suffix sum: tail[i] = a_vals[i+1] + a_vals[i+2] + ... + a_vals[BLOCK_S-1] + running_suffix

        # Inclusive suffix sum from right (reversed cumsum)
        a_rev = tl.flip(a_vals, dim=0)
        a_rev_cumsum = tl.cumsum(a_rev, axis=0)
        a_inc_suffix = tl.flip(a_rev_cumsum, dim=0)  # inclusive suffix: a_inc_suffix[i] = a_vals[i] + ... + a_vals[BLOCK_S-1]

        # Exclusive suffix = inclusive suffix - a_vals[i] + running_suffix
        tail_sum = a_inc_suffix - a_vals + running_suffix

        # L_last = exp(tail_sum)
        L_last = tl.exp(tail_sum)

        # Load Q and dt
        q_vals = tl.load(q_base + offs * stride_q_s, mask=mask, other=0.0)
        d_vals = tl.load(d_base + offs * stride_d_s, mask=mask, other=0.0)

        # raw = Q * L_last * dt
        raw = q_vals * L_last * d_vals
        raw_abs = tl.abs(raw)

        # Zero out masked positions
        raw_abs = tl.where(mask, raw_abs, 0.0)

        # distance weight: (S - 1 - offs)
        dist_weight = (S - 1 - offs).to(tl.float32)
        dist_weight = tl.where(mask, dist_weight, 0.0)

        sum_abs += tl.sum(raw_abs, axis=0)
        sum_weighted += tl.sum(raw_abs * dist_weight, axis=0)

        # Update running suffix: add the entire block's sum for the next block to the left
        running_suffix += tl.sum(a_vals, axis=0)

    # mmd = sum_weighted / sum_abs
    mmd = tl.where(sum_abs > 0, sum_weighted / sum_abs, 0.0)

    out_offset = b * stride_o_b + h * stride_o_h
    tl.store(out_ptr + out_offset, mmd)


def mmd_ssd_last_triton(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    with torch.no_grad():
        # Preprocessing (same as PyTorch version)
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        batch, seqlen, nheads = dt.shape
        # A is broadcastable: (nheads,) — expand to match dt
        A_dt = (A * dt).permute(0, 2, 1).contiguous()  # (B, H, S)

        # Precompute Q = einsum("bgn, bsgn -> bs", C_last, B)
        C_last = C[:, -1, :, :]  # (B, G, N)
        Q = torch.einsum("bgn,bsgn->bs", C_last, B).contiguous()  # (B, S)

        dt_hks = dt.permute(0, 2, 1).contiguous()  # (B, H, S)

        # Output
        out = torch.empty(batch, nheads, device=dt.device, dtype=torch.float32)

        # Launch kernel: one program per (b, h) pair
        grid = (batch * nheads,)
        _mmd_ssd_last_kernel[grid](
            A_dt, Q, dt_hks, out,
            batch, nheads, seqlen,
            A_dt.stride(0), A_dt.stride(1), A_dt.stride(2),
            Q.stride(0), Q.stride(1),
            dt_hks.stride(0), dt_hks.stride(1), dt_hks.stride(2),
            out.stride(0), out.stride(1),
        )

        return out


# ---- Reference PyTorch implementation ----

def mmd_ssd_last_pytorch(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        A_dt = (A * dt).permute(0, 2, 1)    # (B, H, S)

        rev = A_dt.flip(dims=(-1,))
        rev_cumsum = torch.cumsum(rev, dim=-1)
        zeros = torch.zeros_like(rev_cumsum[..., :1])
        rev_cumsum_excl = torch.cat([zeros, rev_cumsum[..., :-1]], dim=-1)
        tail_sum = rev_cumsum_excl.flip(dims=(-1,))
        L_last = torch.exp(tail_sum)

        C_last = C[:, -1, :, :]
        Q = torch.einsum("bgn,bsgn->bs", C_last, B)
        dt_hks = dt.permute(0, 2, 1)

        raw = Q[:, None, :] * L_last * dt_hks
        aw = raw.abs()
        aw = aw / aw.sum(dim=-1, keepdim=True)
        S = aw.size(-1)
        dist = torch.arange(S - 1, -1, -1, dtype=aw.dtype, device=aw.device)

        mmd = torch.einsum("s,bhs->bh", dist, aw)

    return mmd


# ---- Parity and speed comparison ----

def test_parity_and_speed():
    import time

    torch.manual_seed(42)
    device = "cuda"

    # Typical shapes for Mamba2/Bamba
    batch, seqlen, nheads, ngroups, state_size = 4, 8192, 64, 8, 128

    dt = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.float32).abs() * 0.1
    A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
    B = torch.randn(batch, seqlen, ngroups, state_size, device=device, dtype=torch.float32) * 0.01
    C = torch.randn(batch, seqlen, ngroups, state_size, device=device, dtype=torch.float32) * 0.01
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.1

    # ---- Parity check ----
    out_pt = mmd_ssd_last_pytorch(dt, A, B, C, dt_bias=dt_bias)
    out_tr = mmd_ssd_last_triton(dt, A, B, C, dt_bias=dt_bias)

    max_diff = (out_pt - out_tr).abs().max().item()
    mean_diff = (out_pt - out_tr).abs().mean().item()
    print(f"[Parity] max |diff| = {max_diff:.6e}, mean |diff| = {mean_diff:.6e}")
    assert max_diff < 1e-2, f"Parity check failed: max diff = {max_diff}"
    print("[Parity] PASSED")

    # ---- Speed comparison ----
    n_warmup = 10
    n_iters = 100

    # Warmup
    for _ in range(n_warmup):
        _ = mmd_ssd_last_pytorch(dt, A, B, C, dt_bias=dt_bias)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = mmd_ssd_last_pytorch(dt, A, B, C, dt_bias=dt_bias)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - start) / n_iters * 1000

    # Warmup triton
    for _ in range(n_warmup):
        _ = mmd_ssd_last_triton(dt, A, B, C, dt_bias=dt_bias)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = mmd_ssd_last_triton(dt, A, B, C, dt_bias=dt_bias)
    torch.cuda.synchronize()
    tr_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"[Speed] PyTorch: {pt_time:.3f} ms/iter")
    print(f"[Speed] Triton:  {tr_time:.3f} ms/iter")
    print(f"[Speed] Speedup: {pt_time / tr_time:.2f}x")


if __name__ == "__main__":
    test_parity_and_speed()

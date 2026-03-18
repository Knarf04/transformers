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


# ---- State magnitude: chunkwise Triton version of state_mag_ssd_full_chunk ----
#
# Computes hidden state magnitude per token per head using the SSM recurrence:
#   mag_t = exp(A_h * dt_t) * mag_{t-1} + dt_t * ||B_t|| * ||x_{t,h}||
#
# This is the recurrence form of state_mag_ssd_full_chunk (O(S) vs O(S^2)),
# equivalent to summing L[t,s] * dt[s] * ||B[s]|| * ||x[s,h]|| over s <= t.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 256}, num_warps=4),
    ],
    key=["S"],
)
@triton.jit
def _state_mag_kernel(
    # All pointers are (BH, S) contiguous — flattened batch*heads rows
    decay_ptr,  # exp(A[h] * dt[b,s,h])
    inp_ptr,    # dt[b,s,h] * ||B[b,s]|| * ||x[b,s,h]||
    out_ptr,    # output: hidden state magnitude per token
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # indexes a (b, h) pair
    base = pid * S

    mag = 0.0
    num_blocks = tl.cdiv(S, BLOCK_S)
    for block_idx in range(num_blocks):
        for i in tl.static_range(BLOCK_S):
            s = block_idx * BLOCK_S + i
            if s < S:
                d = tl.load(decay_ptr + base + s)
                inp_val = tl.load(inp_ptr + base + s)
                mag = d * mag + inp_val
                tl.store(out_ptr + base + s, mag)


def state_mag_triton(dt, A, B, x, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    """Chunkwise Triton version of state_mag_ssd_full_chunk.

    Computes hidden state magnitude per token per head via the SSM recurrence.

    Args:
        dt: (B, S, H)
        A:  (H,)
        B:  (B, S, G, N)
        x:  (B, S, H, D)

    Returns:
        hidden_states: (B, nheads, S) — same layout as state_mag_ssd_full_chunk
    """
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt.float())
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        batch, seqlen, nheads = dt.shape

        decay = torch.exp((A * dt).float())  # (B, S, H)

        x_norm = torch.linalg.vector_norm(x.float(), dim=-1)           # (B, S, H)
        B_norm = torch.linalg.vector_norm(B.float(), dim=(-2, -1))     # (B, S)

        inp = dt.float() * B_norm.unsqueeze(-1) * x_norm               # (B, S, H)

        # Reshape to (BH, S) contiguous for kernel
        decay_flat = decay.permute(0, 2, 1).contiguous().reshape(-1, seqlen)
        inp_flat = inp.permute(0, 2, 1).contiguous().reshape(-1, seqlen)
        out_flat = torch.empty_like(inp_flat)

        BH = batch * nheads
        _state_mag_kernel[(BH,)](decay_flat, inp_flat, out_flat, seqlen)

        # Return (B, nheads, S) to match state_mag_ssd_full_chunk
        return out_flat.reshape(batch, nheads, seqlen)


# ---- Reference PyTorch implementations ----


def state_mag_pytorch(dt, A, B, x, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    """PyTorch reference for state_mag_triton. Returns (B, nheads, S)."""
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt.float())
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        batch, seqlen, nheads = dt.shape

        decay = torch.exp((A * dt).float())
        x_norm = torch.linalg.vector_norm(x.float(), dim=-1)
        B_norm = torch.linalg.vector_norm(B.float(), dim=(-2, -1))
        inp = dt.float() * B_norm.unsqueeze(-1) * x_norm

        mag = torch.zeros(batch, seqlen, nheads, device=dt.device, dtype=torch.float32)
        running = torch.zeros(batch, nheads, device=dt.device, dtype=torch.float32)
        for s in range(seqlen):
            running = decay[:, s, :] * running + inp[:, s, :]
            mag[:, s, :] = running

        return mag.permute(0, 2, 1).contiguous()


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

    # ---- State magnitude parity check ----
    print("\n---- State Magnitude ----")
    head_dim = 64
    x = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=torch.float32) * 0.1

    out_pt_mag = state_mag_pytorch(dt, A, B, x, dt_bias=dt_bias)
    out_tr_mag = state_mag_triton(dt, A, B, x, dt_bias=dt_bias)

    max_diff_mag = (out_pt_mag - out_tr_mag).abs().max().item()
    mean_diff_mag = (out_pt_mag - out_tr_mag).abs().mean().item()
    print(f"[Parity] max |diff| = {max_diff_mag:.6e}, mean |diff| = {mean_diff_mag:.6e}")
    assert max_diff_mag < 1e-2, f"State mag parity check failed: max diff = {max_diff_mag}"
    print("[Parity] PASSED")

    # Speed
    for _ in range(n_warmup):
        _ = state_mag_pytorch(dt, A, B, x, dt_bias=dt_bias)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = state_mag_pytorch(dt, A, B, x, dt_bias=dt_bias)
    torch.cuda.synchronize()
    pt_mag_time = (time.perf_counter() - start) / n_iters * 1000

    for _ in range(n_warmup):
        _ = state_mag_triton(dt, A, B, x, dt_bias=dt_bias)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = state_mag_triton(dt, A, B, x, dt_bias=dt_bias)
    torch.cuda.synchronize()
    tr_mag_time = (time.perf_counter() - start) / n_iters * 1000

    print(f"[Speed] PyTorch: {pt_mag_time:.3f} ms/iter")
    print(f"[Speed] Triton:  {tr_mag_time:.3f} ms/iter")
    print(f"[Speed] Speedup: {pt_mag_time / tr_mag_time:.2f}x")


# ---- State cosine similarity: Triton version of state_cosine_sim_full_chunk ----
#
# Uses the SSM recurrence h[t] = exp(A*dt[t]) * h[t-1] + dt[t] * (B[t] ⊗ x[t])
# instead of the quadratic O(S²) chunked computation. This is O(S * N * D) per (b, h).
#
# Grid: one program per (b, h, n_tile, d_tile).
# Each program maintains a (N_BLOCK, D_BLOCK) slice of the (N, D) state matrix in
# registers, scans through the full sequence, and writes the state to HBM at each
# sampled position p_i = (i+1)*interval - 1.


@triton.autotune(
    configs=[
        triton.Config({"N_BLOCK": 16, "D_BLOCK": 16}, num_warps=4),
        triton.Config({"N_BLOCK": 32, "D_BLOCK": 16}, num_warps=4),
        triton.Config({"N_BLOCK": 16, "D_BLOCK": 32}, num_warps=4),
        triton.Config({"N_BLOCK": 32, "D_BLOCK": 32}, num_warps=8),
        triton.Config({"N_BLOCK": 64, "D_BLOCK": 16}, num_warps=8),
        triton.Config({"N_BLOCK": 16, "D_BLOCK": 64}, num_warps=8),
    ],
    key=["S", "N", "D"],
)
@triton.jit
def _state_sample_kernel(
    dt_ptr,    # (B, S, H) — preprocessed (softplus + clamp applied)
    A_ptr,     # (H,)
    B_ptr,     # (B, S, G, N)
    x_ptr,     # (B, S, H, D)
    out_ptr,   # (B, H, npos, N, D) — output sampled states
    S, H, N, D,
    interval, npos, group_size,
    stride_dt_b, stride_dt_s, stride_dt_h,
    stride_A,
    stride_B_b, stride_B_s, stride_B_g, stride_B_n,
    stride_x_b, stride_x_s, stride_x_h, stride_x_d,
    stride_out_b, stride_out_h, stride_out_pos, stride_out_n, stride_out_d,
    N_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, N_BLOCK)
    d_tiles = tl.cdiv(D, D_BLOCK)
    nd_tiles = n_tiles * d_tiles

    bh_id     = pid // nd_tiles
    nd_id     = pid % nd_tiles
    n_tile_id = nd_id // d_tiles
    d_tile_id = nd_id % d_tiles

    b = bh_id // H
    h = bh_id % H
    g = h // group_size

    n_offs = n_tile_id * N_BLOCK + tl.arange(0, N_BLOCK)  # (N_BLOCK,)
    d_offs = d_tile_id * D_BLOCK + tl.arange(0, D_BLOCK)  # (D_BLOCK,)
    n_mask = n_offs < N
    d_mask = d_offs < D

    # Load A[h] once — scalar decay parameter for this head
    A_val = tl.load(A_ptr + h * stride_A)

    # Base pointers: incorporate all fixed indices so the s-loop only adds stride_*_s
    dt_base = dt_ptr + b * stride_dt_b + h * stride_dt_h                          # scalar
    B_base  = B_ptr  + b * stride_B_b  + g * stride_B_g  + n_offs * stride_B_n   # (N_BLOCK,)
    x_base  = x_ptr  + b * stride_x_b  + h * stride_x_h  + d_offs * stride_x_d  # (D_BLOCK,)

    # SSM state for this (b, h, n_tile, d_tile) block — lives in registers
    state = tl.zeros((N_BLOCK, D_BLOCK), dtype=tl.float32)

    for s in range(S):
        dt_val = tl.load(dt_base + s * stride_dt_s)
        decay  = tl.exp(A_val * dt_val)

        B_vals = tl.load(B_base + s * stride_B_s, mask=n_mask, other=0.0)  # (N_BLOCK,)
        x_vals = tl.load(x_base + s * stride_x_s, mask=d_mask, other=0.0)  # (D_BLOCK,)

        # h[s] = decay * h[s-1] + dt[s] * (B[s] ⊗ x[s])
        state = decay * state + dt_val * (B_vals[:, None] * x_vals[None, :])

        # Store at sampled position p_i = (i+1)*interval - 1, i.e. when (s+1) % interval == 0
        if (s + 1) % interval == 0:
            pos = s // interval
            out_ptrs = (out_ptr
                        + b * stride_out_b
                        + h * stride_out_h
                        + pos * stride_out_pos
                        + n_offs[:, None] * stride_out_n
                        + d_offs[None, :] * stride_out_d)
            tl.store(out_ptrs, state, mask=n_mask[:, None] & d_mask[None, :])


def state_cosine_sim_triton(dt, A, B, x, interval, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    """Triton implementation of state_cosine_sim_full_chunk.

    Computes pairwise cosine similarity between SSM hidden states at positions
    interval-1, 2*interval-1, ..., seqlen-1 via the SSM recurrence.
    This is O(S*N*D) per (b,h) vs O(S²) for the chunked PyTorch version.

    Args:
        dt:       (B, S, H)
        A:        (H,)
        B:        (B, S, G, N)
        x:        (B, S, H, D)
        interval: int — spacing between sampled positions; seqlen % interval == 0

    Returns:
        cosine_sim: (B, H, npos, npos) — strictly lower-triangular cosine similarity
    """
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt.float())
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        batch, seqlen, nheads = dt.shape
        ngroups    = B.shape[2]
        state_dim  = B.shape[3]  # N
        head_dim   = x.shape[3]  # D
        group_size = nheads // ngroups

        assert seqlen % interval == 0, f"seqlen ({seqlen}) must be divisible by interval ({interval})"
        npos = seqlen // interval

        dt = dt.contiguous().float()
        A  = A.contiguous().float()
        B  = B.contiguous().float()
        x  = x.contiguous().float()

        # Output: sampled states (B, H, npos, N, D)
        out = torch.zeros(batch, nheads, npos, state_dim, head_dim,
                          device=dt.device, dtype=torch.float32)

        grid = lambda meta: (
            batch * nheads
            * triton.cdiv(state_dim, meta["N_BLOCK"])
            * triton.cdiv(head_dim,  meta["D_BLOCK"]),
        )

        _state_sample_kernel[grid](
            dt, A, B, x, out,
            seqlen, nheads, state_dim, head_dim,
            interval, npos, group_size,
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        )

        # Flatten state dim and compute pairwise cosine similarity
        states_flat = out.reshape(batch, nheads, npos, state_dim * head_dim)
        norms       = torch.linalg.vector_norm(states_flat, dim=-1, keepdim=True)
        states_norm = states_flat / (norms + 1e-8)
        cosine_sim  = torch.matmul(states_norm, states_norm.transpose(-2, -1))

        # Retain only strictly lower triangle (j < i)
        mask = torch.tril(torch.ones(npos, npos, dtype=torch.bool, device=cosine_sim.device), diagonal=-1)
        cosine_sim = cosine_sim * mask

    return cosine_sim


# ---- Reference PyTorch implementation (recurrence-based) ----


def state_cosine_sim_pytorch(dt, A, B, x, interval, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    """Recurrence-based PyTorch reference for state_cosine_sim_triton.

    Mathematically equivalent to state_cosine_sim_full_chunk but uses the O(S*N*D)
    SSM recurrence rather than the quadratic chunked formulation.

    Args / Returns: same as state_cosine_sim_triton.
    """
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt.float())
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        batch, seqlen, nheads = dt.shape
        ngroups    = B.shape[2]
        state_dim  = B.shape[3]
        head_dim   = x.shape[3]
        group_size = nheads // ngroups
        npos       = seqlen // interval

        # Expand B from G groups to H heads: (B, S, H, N)
        B_exp = B.float().repeat_interleave(group_size, dim=2)

        state = torch.zeros(batch, nheads, state_dim, head_dim,
                            device=dt.device, dtype=torch.float32)
        sampled = []

        for s in range(seqlen):
            d    = torch.exp(A * dt[:, s, :])          # (B, H)
            dt_s = dt[:, s, :]                          # (B, H)
            B_s  = B_exp[:, s, :, :]                   # (B, H, N)
            x_s  = x[:, s, :, :].float()               # (B, H, D)

            outer = B_s[:, :, :, None] * x_s[:, :, None, :]              # (B, H, N, D)
            state = d[:, :, None, None] * state + dt_s[:, :, None, None] * outer

            if (s + 1) % interval == 0:
                sampled.append(state.cpu().clone())

        # (npos, B, H, N, D) -> (B, H, npos, N, D)
        states      = torch.stack(sampled, dim=0).permute(1, 2, 0, 3, 4)
        states_flat = states.reshape(batch, nheads, npos, state_dim * head_dim)

        norms       = torch.linalg.vector_norm(states_flat, dim=-1, keepdim=True)
        states_norm = states_flat / (norms + 1e-8)
        cosine_sim  = torch.matmul(states_norm, states_norm.transpose(-2, -1))

        mask = torch.tril(torch.ones(npos, npos, dtype=torch.bool), diagonal=-1)
        cosine_sim = cosine_sim * mask.to(cosine_sim.device)

    return cosine_sim


# ---- Parity and speed test for state cosine similarity ----


def test_state_cosine_sim_parity():
    import time
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mmd import state_cosine_sim_full_chunk

    torch.manual_seed(0)
    device = "cuda"

    # Small shapes for parity (the quadratic reference is expensive at large S)
    batch, seqlen, nheads, ngroups, state_dim, head_dim = 2, 512, 8, 2, 32, 32
    interval   = 64   # npos = 8
    chunk_size = 64   # for the quadratic reference

    dt     = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.float32).abs() * 0.1
    A      = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
    B      = torch.randn(batch, seqlen, ngroups, state_dim, device=device, dtype=torch.float32) * 0.01
    x      = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.1

    print("---- State Cosine Similarity ----")

    # Triton vs. recurrence PyTorch reference (both O(S*N*D))
    out_tr = state_cosine_sim_triton(dt, A, B, x, interval, dt_bias=dt_bias)
    out_pt = state_cosine_sim_pytorch(dt, A, B, x, interval, dt_bias=dt_bias)

    # Compare only the lower-triangle entries (upper is zero by construction)
    npos = seqlen // interval
    lt_mask = torch.tril(torch.ones(npos, npos, dtype=torch.bool), diagonal=-1)
    out_tr_lt = out_tr[:, :, lt_mask]
    out_pt_lt = out_pt[:, :, lt_mask].to(device)

    max_diff  = (out_tr_lt - out_pt_lt).abs().max().item()
    mean_diff = (out_tr_lt - out_pt_lt).abs().mean().item()
    print(f"[Parity vs recurrence ref] max |diff| = {max_diff:.6e}, mean |diff| = {mean_diff:.6e}")
    assert max_diff < 1e-3, f"Parity check failed (Triton vs recurrence ref): max diff = {max_diff}"
    print("[Parity vs recurrence ref] PASSED")

    # Triton vs. quadratic chunked reference (state_cosine_sim_full_chunk)
    out_quad = state_cosine_sim_full_chunk(
        dt, A, B, x, interval=interval, chunk_size=chunk_size,
        dt_bias=dt_bias, device=device,
    )
    out_tr_cpu  = out_tr.cpu()
    out_tr_lt2  = out_tr_cpu[:, :, lt_mask]
    out_quad_lt = out_quad[:, :, lt_mask]

    max_diff2  = (out_tr_lt2 - out_quad_lt).abs().max().item()
    mean_diff2 = (out_tr_lt2 - out_quad_lt).abs().mean().item()
    print(f"[Parity vs quadratic ref]  max |diff| = {max_diff2:.6e}, mean |diff| = {mean_diff2:.6e}")
    assert max_diff2 < 1e-3, f"Parity check failed (Triton vs quadratic ref): max diff = {max_diff2}"
    print("[Parity vs quadratic ref]  PASSED")

    # Speed comparison (larger shapes, no quadratic reference)
    batch2, seqlen2, nheads2, ngroups2, state_dim2, head_dim2 = 2, 4096, 32, 4, 64, 64
    interval2 = 128  # npos = 32
    dt2    = torch.randn(batch2, seqlen2, nheads2, device=device, dtype=torch.float32).abs() * 0.1
    A2     = -torch.rand(nheads2, device=device, dtype=torch.float32) * 0.5
    B2     = torch.randn(batch2, seqlen2, ngroups2, state_dim2, device=device, dtype=torch.float32) * 0.01
    x2     = torch.randn(batch2, seqlen2, nheads2, head_dim2, device=device, dtype=torch.float32) * 0.1
    dt_bias2 = torch.randn(nheads2, device=device, dtype=torch.float32) * 0.1

    n_warmup, n_iters = 5, 20

    for _ in range(n_warmup):
        _ = state_cosine_sim_pytorch(dt2, A2, B2, x2, interval2, dt_bias=dt_bias2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = state_cosine_sim_pytorch(dt2, A2, B2, x2, interval2, dt_bias=dt_bias2)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) / n_iters * 1000

    for _ in range(n_warmup):
        _ = state_cosine_sim_triton(dt2, A2, B2, x2, interval2, dt_bias=dt_bias2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = state_cosine_sim_triton(dt2, A2, B2, x2, interval2, dt_bias=dt_bias2)
    torch.cuda.synchronize()
    tr_time = (time.perf_counter() - t0) / n_iters * 1000

    print(f"\n[Speed] B={batch2} S={seqlen2} H={nheads2} N={state_dim2} D={head_dim2} interval={interval2}")
    print(f"[Speed] PyTorch (recurrence): {pt_time:.3f} ms/iter")
    print(f"[Speed] Triton:               {tr_time:.3f} ms/iter")
    print(f"[Speed] Speedup:              {pt_time / tr_time:.2f}x")


if __name__ == "__main__":
    test_parity_and_speed()
    print()
    test_state_cosine_sim_parity()

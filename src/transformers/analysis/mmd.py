import torch

def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

def mmd_gqa_last(query_states, key_states, attention_mask=None, scaling=1.0):
    with torch.no_grad():
        # Use fp32 for precise attention computation
        q_states = query_states.clone().detach().to(torch.float32)
        k_states = key_states.clone().detach().to(torch.float32)
        mask = None
        if attention_mask is not None:
            mask = attention_mask.clone().detach().to(torch.float32)

        B, H_q, L, d = q_states.shape
        H_k = k_states.shape[1]
        group_size = H_q // H_k

        q_last = q_states.view(B, H_k, group_size, L, d)[..., -1, :]

        k = k_states.transpose(-2, -1).unsqueeze(2)       # (B,H_k,1,d,L)
        k = k.expand(-1, -1, group_size, -1, -1)          # (B,H_k,group_size,d,L)

        scores = torch.einsum("bghd,bghdj->bghj", q_last, k) * scaling

        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            scores = scores.masked_fill(m == 0, float("-inf"))

        attn = torch.nn.functional.softmax(scores, dim=-1)                       # (B,H_k,group_size,L)
        attn_avg = attn.view(B, H_q, L).mean(dim=0)            # (L,) after mean over heads
        dist = torch.arange(L - 1, -1, -1, dtype=torch.float32, device=attn_avg.device)          # (L,)
        aw = torch.abs(attn_avg)
        aw = aw / aw.sum(dim=-1, keepdim=True)
        mmd = (aw * dist).sum(dim=-1)

    return mmd

def mmd_ssd_last(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit is not None:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        A_dt = (A * dt).permute(0, 2, 1)    # (B, H, S)

        rev = A_dt.flip(dims=(-1,))        # (B, H, S)
        rev_cumsum = torch.cumsum(rev, dim=-1)  # (B, H, S),
        zeros = torch.zeros_like(rev_cumsum[..., :1])
        rev_cumsum_excl = torch.cat([zeros, rev_cumsum[..., :-1]], dim=-1)
        tail_sum = rev_cumsum_excl.flip(dims=(-1,))  # (B, H, S)
        L_last = torch.exp(tail_sum)       # (B, H, S)

        C_last = C[:, -1, :, :]                        # (B, H, N)
        Q = torch.einsum("bhn,bshn->bs", C_last, B)
        dt_hks = dt.permute(0, 2, 1)                   # (B, H, S)

        raw = Q[:, None, :] * L_last * dt_hks                      # (B, H, S)
        aw  = raw.abs()
        aw  = aw / aw.sum(dim=-1, keepdim=True)        # normalize over S
        S = aw.size(-1)
        dist = torch.arange(S-1, -1, -1, dtype=aw.dtype, device=aw.device)

        mmd = torch.einsum("s,bhs->bh", dist, aw)

    return mmd

def mmd_ssd_full(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        seqlen = dt.shape[1]
        A = (A*dt).permute([0, 2, 1])
        L = torch.exp(segment_sum(A))
        M = torch.einsum("blgn, bsgn, bhls, bsh -> blhs", C, B, L, dt)    # Full Attention Map of Mamba2
        
        M = M.permute(0, 2, 1, 3).abs()
        
        idx = torch.arange(seqlen)
        dist = torch.tril(idx.unsqueeze(1) - idx.unsqueeze(0))
    return torch.sum(M * dist, dim=-1) / torch.sum(M, dim=-1)

def mmd_ssd_full_chunk(dt, A, B, C, chunk_size, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf")), device="cpu"):
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        A = (A*dt).permute([0, 2, 1]) # (batch, nheads, seqlen)
        
        batch, nheads, seqlen = A.shape
        num_chunks = -(-seqlen // chunk_size)

        mmd = torch.zeros(batch, nheads, seqlen, device=torch.device("cpu"))
        denom = torch.zeros(batch, nheads, seqlen, device=torch.device("cpu"))
        
        for i in range(num_chunks):
            i_start = i * chunk_size
            i_end = min((i+1) * chunk_size, seqlen)
            C_chunk = C[:, i_start:i_end, :, :].detach().to(device) # (batch, chunk_size, ngroups, nheads)
            A_i_chunk = A[:, :, i_start:i_end].detach().to(device)

            for j in range(num_chunks):
                j_start = j * chunk_size
                j_end = min((j+1) * chunk_size, seqlen)
                B_chunk = B[:, j_start:j_end, :, :].detach().to(device) # (batch, chunk_size, ngroups, nheads)
                dt_chunk = dt[:, j_start:j_end, :].detach().to(device) # (batch, chunk_size, nheads)

                if i_start == j_start: # diagonal
                    L_chunk = torch.exp(segment_sum(A_i_chunk))
                elif i_start > j_start: # off-diagonal
                    A_j_chunk = A[:, :, j_start:j_end].detach().to(device)
                    A_mid = A[:, :, j_end:i_start].detach().to(device)

                    A_C = torch.exp(torch.cumsum(A_i_chunk, dim=-1))
                    A_B = torch.exp(torch.flip(torch.cumsum(torch.flip(A_j_chunk, [-1]), dim=-1), [-1]) - A_j_chunk)
                    A_sum = torch.exp(torch.sum(A_mid, dim=-1)) # (batch, nheads)
                    
                    L_chunk = torch.einsum("bnl, bns, bn -> bnls", A_C, A_B, A_sum) # (batch, nheads, chunk_size, chunk_size)
                    
                    del A_C, A_B, A_sum, A_j_chunk, A_mid
                else:
                    del B_chunk, dt_chunk
                    continue
                
                # Materialize chunked attention matrix
                M = torch.einsum("blgn, bsgn, bhls, bsh -> blhs", C_chunk, B_chunk, L_chunk, dt_chunk).detach().cpu()
                M = M.permute(0, 2, 1, 3).abs()
        
                i_idx = torch.arange(i_start, i_end, device=device)  # shape (L_i,)
                j_idx = torch.arange(j_start, j_end, device=device)  # shape (L_j,)

                dist = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)   # shape (L_i, L_j)
                dist = dist.clamp(min=0)
                
                mmd[:, :, i_start:i_end] += torch.sum(M * dist, dim=-1)
                denom[:, :, i_start:i_end] += torch.sum(M, dim=-1)
                
                del B_chunk, L_chunk, dt_chunk, M, i_idx, j_idx, dist
                torch.cuda.empty_cache()
            del C_chunk, A_i_chunk
            torch.cuda.empty_cache()
        del A, dt
        torch.cuda.empty_cache()
    return mmd / denom

def state_cosine_sim_full_chunk(dt, A, B, x, interval, chunk_size, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf")), device="cpu"):
    """
    Compute pairwise cosine similarity between SSM hidden states at sampled positions.

    Samples the actual hidden state h_t at positions interval-1, 2*interval-1, ..., seqlen-1
    (npos = seqlen // interval positions). The state per head is an (N, D) matrix — the
    accumulated outer product of B (shape N) and x (shape D) weighted by L·dt — and is
    flattened to (N*D,) for cosine similarity. H is never merged, so the output is per-head.

    Returns a lower-triangular cosine similarity matrix: entry [b, h, i, j] holds
    cos_sim(h_{p_i}, h_{p_j}) for j < i, and 0 everywhere else (diagonal included in zeros).

    Follows the same chunked L[t, s] computation as state_mag_ssd_full_chunk.
    Three key/query relationships per inner (j) chunk:
      • j_end == i_end   — "diagonal": last key chunk in interval; uses segment_sum.
      • j_end <= i_start — "off-diagonal": key chunk entirely before interval; mirrors
                           A_C × A_B × A_sum from state_mag_ssd_full_chunk, with A_C
                           collapsed to its last element exp(Σ A_i_chunk).
      • otherwise        — key chunk inside the interval but not last; like off-diagonal
                           but the "middle" runs from j_end to i_end.

    Requires: seqlen % interval == 0  and  interval % chunk_size == 0.

    Args:
        dt:        (B, S, H)
        A:         (H,)
        B:         (B, S, G, N)  — G=ngroups, N=d_state
        x:         (B, S, H, D)  — H=nheads, D=d_head
        interval:  int — spacing between sampled positions
        chunk_size: int — inner key-chunk size for memory-efficient computation
        dt_bias, dt_softplus, dt_limit: same preprocessing as state_mag_ssd_full_chunk
        device:    computation device string

    Returns:
        cosine_sim: (B, H, npos, npos) — strictly lower-triangular cosine similarity
    """
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        A = (A * dt).permute([0, 2, 1])  # (batch, nheads, seqlen)

        batch, nheads, seqlen = A.shape
        ngroups   = B.shape[2]
        state_dim = B.shape[3]   # N = d_state
        head_dim  = x.shape[3]   # D = d_head
        group_size = nheads // ngroups

        assert seqlen % interval == 0 and interval % chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by interval ({interval}), "
            f"which must be divisible by chunk_size ({chunk_size})"
        )
        npos           = seqlen // interval
        num_key_chunks = seqlen // chunk_size

        sampled_states = []  # list of (B, H, N, D) CPU tensors, one per sampled position

        for i in range(npos):
            i_start = i * interval
            i_end   = i_start + interval
            A_i_chunk = A[:, :, i_start:i_end].detach().to(device)  # (B, H, interval)

            state_i = torch.zeros(batch, nheads, state_dim, head_dim,
                                  device=device, dtype=torch.float32)

            for j in range(num_key_chunks):
                j_start = j * chunk_size
                j_end   = j_start + chunk_size

                if j_start >= i_end:
                    break

                B_chunk  = B[:, j_start:j_end, :, :].detach().to(device)   # (B, s_len, G, N)
                dt_chunk = dt[:, j_start:j_end, :].detach().to(device)      # (B, s_len, H)
                x_chunk  = x[:, j_start:j_end, :, :].detach().to(device)   # (B, s_len, H, D)

                if j_end == i_end:
                    # Diagonal: last key chunk in interval.
                    # L[p_i, s] = last row of segment_sum within j chunk = suffix decay to i_end.
                    A_j_chunk = A[:, :, j_start:j_end].detach().to(device)
                    L_last = torch.exp(segment_sum(A_j_chunk))[:, :, -1, :]  # (B, H, s_len)
                    del A_j_chunk

                elif j_end <= i_start:
                    # Off-diagonal: key chunk is entirely before the query interval.
                    # L[p_i, s] = A_B[s] * A_sum * A_C  (same formula as state_mag_ssd_full_chunk,
                    # but A_C collapses to scalar exp(Σ A_i_chunk) since we only need p_i).
                    A_j_chunk = A[:, :, j_start:j_end].detach().to(device)
                    A_mid     = A[:, :, j_end:i_start].detach().to(device)
                    A_B   = torch.exp(torch.flip(torch.cumsum(torch.flip(A_j_chunk, [-1]), dim=-1), [-1]) - A_j_chunk)
                    A_sum = torch.exp(torch.sum(A_mid,     dim=-1))  # (B, H)
                    A_C   = torch.exp(torch.sum(A_i_chunk, dim=-1))  # (B, H)
                    L_last = A_B * (A_sum * A_C)[:, :, None]         # (B, H, s_len)
                    del A_j_chunk, A_mid, A_B, A_sum, A_C

                else:
                    # Within-interval (not last): j_start >= i_start, j_end < i_end.
                    # L[p_i, s] = A_B[s] * exp(Σ A[j_end:i_end])
                    A_j_chunk  = A[:, :, j_start:j_end].detach().to(device)
                    A_tail     = A[:, :, j_end:i_end].detach().to(device)
                    A_B        = torch.exp(torch.flip(torch.cumsum(torch.flip(A_j_chunk, [-1]), dim=-1), [-1]) - A_j_chunk)
                    A_sum_tail = torch.exp(torch.sum(A_tail, dim=-1))  # (B, H)
                    L_last = A_B * A_sum_tail[:, :, None]              # (B, H, s_len)
                    del A_j_chunk, A_tail, A_B, A_sum_tail

                # weight[b,h,s] = L_last[b,h,s] * dt[b,s,h]
                weight = L_last * dt_chunk.permute(0, 2, 1)  # (B, H, s_len)

                # Expand B from G groups to H heads: (B, s_len, H, N)
                B_exp = B_chunk.repeat_interleave(group_size, dim=2)

                # state_i[b,h,n,d] += Σ_s weight[b,h,s] * B_exp[b,s,h,n] * x_chunk[b,s,h,d]
                state_i += torch.einsum("bhs,bshn,bshd->bhnd", weight, B_exp, x_chunk)

                del B_chunk, dt_chunk, x_chunk, B_exp, L_last, weight
                torch.cuda.empty_cache()

            del A_i_chunk
            torch.cuda.empty_cache()
            sampled_states.append(state_i.cpu())

        del A, dt
        torch.cuda.empty_cache()

        # Stack and reshape: (npos, B, H, N, D) → (B, H, npos, N*D)
        states = torch.stack(sampled_states, dim=0)      # (npos, B, H, N, D)
        states_flat = states.permute(1, 2, 0, 3, 4)      # (B, H, npos, N, D)
        states_flat = states_flat.reshape(batch, nheads, npos, state_dim * head_dim)

        # L2-normalise over the N*D dimension, then pairwise cosine similarity
        norms = torch.linalg.vector_norm(states_flat, dim=-1, keepdim=True)  # (B, H, npos, 1)
        states_norm = states_flat / (norms + 1e-8)
        cosine_sim = torch.matmul(states_norm, states_norm.transpose(-2, -1))  # (B, H, npos, npos)

        # Retain only strictly lower triangle (j < i); zero diagonal and above
        mask = torch.tril(torch.ones(npos, npos, dtype=torch.bool), diagonal=-1)
        cosine_sim = cosine_sim * mask.to(cosine_sim.device)

    return cosine_sim


def state_mag_ssd_full_chunk(dt, A, B, x, chunk_size, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf")), device="cpu"):
    with torch.no_grad():
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = torch.nn.functional.softplus(dt)
        if dt_limit:
            dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

        A = (A*dt).permute([0, 2, 1]) # (batch, nheads, seqlen)
        
        batch, nheads, seqlen = A.shape
        num_chunks = -(-seqlen // chunk_size)

        hidden_states = torch.zeros(batch, nheads, seqlen, device=torch.device("cpu"))
        # denom = torch.zeros(batch, nheads, seqlen, device=torch.device("cpu"))
        
        for i in range(num_chunks):
            i_start = i * chunk_size
            i_end = min((i+1) * chunk_size, seqlen)
            A_i_chunk = A[:, :, i_start:i_end].detach().to(device)

            for j in range(num_chunks):
                j_start = j * chunk_size
                j_end = min((j+1) * chunk_size, seqlen)
                B_chunk = B[:, j_start:j_end, :, :].detach().to(device) # (batch, chunk_size, ngroups, nheads)
                dt_chunk = dt[:, j_start:j_end, :].detach().to(device) # (batch, chunk_size, nheads)
                x_chunk = x[:, j_start:j_end, :, :].detach().to(device) # (batch, chunk_size, nheads, head_dim)

                if i_start == j_start: # diagonal
                    L_chunk = torch.exp(segment_sum(A_i_chunk))
                elif i_start > j_start: # off-diagonal
                    A_j_chunk = A[:, :, j_start:j_end].detach().to(device)
                    A_mid = A[:, :, j_end:i_start].detach().to(device)

                    A_C = torch.exp(torch.cumsum(A_i_chunk, dim=-1))
                    A_B = torch.exp(torch.flip(torch.cumsum(torch.flip(A_j_chunk, [-1]), dim=-1), [-1]) - A_j_chunk)
                    A_sum = torch.exp(torch.sum(A_mid, dim=-1)) # (batch, nheads)
                    
                    L_chunk = torch.einsum("bnl, bns, bn -> bnls", A_C, A_B, A_sum) # (batch, nheads, chunk_size, chunk_size)
                    
                    del A_C, A_B, A_sum, A_j_chunk, A_mid
                else:
                    del B_chunk, dt_chunk, x_chunk
                    continue
                
                # Materialize chunked attention matrix
                # H = torch.einsum("bsgn, bhls, bsh, bshd -> bshdgn", B_chunk, L_chunk, dt_chunk, x_chunk)
                # H = torch.linalg.vector_norm(H, dim=(3, 4, 5)).permute(0, 2, 1).contiguous().detach().cpu()

                Lsum = L_chunk.sum(dim=2)                 # [b,h,s_len]
                F    = (dt_chunk * Lsum.permute(0,2,1))   # [b,s_len,h]
                x_norm = torch.linalg.vector_norm(x_chunk, dim=-1)        # [b,s_len,h]
                B_norm = torch.linalg.vector_norm(B_chunk, dim=(-2,-1))   # [b,s_len]
                H = (F.abs() * x_norm * B_norm.unsqueeze(-1))             # [b,s_len,h]
                H = H.permute(0,2,1).cpu()

                hidden_states[:, :, i_start:i_end] += H
                
                del B_chunk, L_chunk, dt_chunk, x_chunk, H
                torch.cuda.empty_cache()
            del A_i_chunk
            torch.cuda.empty_cache()
        del A, dt
        torch.cuda.empty_cache()
    return hidden_states
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
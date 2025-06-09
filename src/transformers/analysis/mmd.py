import torch

def mmd_from_gqa_inputs(query_states, key_states, attention_mask=None, scaling=1.0):
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

def mmd_from_ssd_inputs(dt, A, B, C, dt_bias=None, dt_softplus=True, dt_limit=(0.0, float("inf"))):
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
        Q = torch.einsum("bhn,bshn->bhs", C_last, B)
        dt_hks = dt.permute(0, 2, 1)                    # (B, H, S)

        raw = Q * L_last * dt_hks                       # (B, H, S)
        aw  = raw.abs()
        aw  = aw / aw.sum(dim=-1, keepdim=True)        # normalize over S
        S = aw.size(-1)
        dist = torch.arange(S-1, -1, -1, dtype=aw.dtype, device=aw.device)

        mmd = torch.einsum("s,bhs->bh", dist, aw)

    return mmd


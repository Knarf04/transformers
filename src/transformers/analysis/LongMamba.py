import torch
import numpy as np

merge_config = {
    "model_arch": "ours",
    "base_model": "state-spaces/mamba-1.4b",
    "save_para4debug":True,
    "seed": 123,
    "b": 1,
    "c": 1,
    "longbench_dataset": "none",
    "leval_dataset": "none",
    "deci_dataset": "none",
    "our_method": "channelwise_alpha"
}

def get_topk_mask_channelwise(delta_t, k=2000, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    k = int(np.min([L_for_dec, k]))
    delta_t_select = delta_t[:,:,:L_for_dec]
    # using torch.quantile to get the topk threshold
    topk_threshold = torch.quantile(delta_t_select, 1 - k/L_for_dec, dim=2, keepdim=True)
    mask = delta_t_select > topk_threshold
    mask = torch.cat([mask, torch.ones_like(delta_t[:,:,L_for_dec:], dtype=torch.bool)], dim=2)
    return mask

def get_top_k_token_indices(delta_t, k=2000, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    delta_t_norm = torch.norm(delta_t, p=2, dim=1)
    delta_t_norm = delta_t_norm[:,:L_for_dec]
    k = int(np.min([L_for_dec, k])) # k should be less than the sequence length
    _, not_decimated = torch.topk(delta_t_norm, k, dim=1, largest=True, sorted=False)
    not_decimated, _ = torch.sort(not_decimated.squeeze())
    
    not_decimated = torch.cat([not_decimated, torch.arange(L_for_dec,L).to(not_decimated.device)])
    return not_decimated

def get_channelwise_topAlpha(delta_t, alpha=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if alpha is None:
        print("Alpha is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)
    k_values = L_for_dec * alpha
    k_values = torch.clamp(k_values, max=L_for_dec)

    mask = torch.zeros_like(delta_t, dtype=torch.bool)

    _, sorted_indices = torch.sort(delta_t[:, :, :L_for_dec], descending=True, dim=-1)
    range_tensor = torch.arange(L_for_dec).view(1, 1, -1).expand(delta_t.size(0), delta_t.size(1), L_for_dec).to(k_values.device)
    topk_mask = range_tensor < k_values.view(delta_t.size(0), delta_t.size(1), 1)
    mask[:, :, :L_for_dec].scatter_(2, sorted_indices, topk_mask)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True
    
    return mask

def get_channelwise_topBound(delta_t, decay=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if decay is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)
    
    sorted_delta_t, sorted_indices = torch.sort(delta_t, descending=True, dim=-1)
    cumsum_delta = torch.cumsum(sorted_delta_t, dim=-1)
    cumsum_mask = cumsum_delta <= decay.view(1, -1, 1)
    topk_positions = cumsum_mask.sum(dim=-1)
    range_tensor = torch.arange(delta_t.size(-1)).view(1, 1, -1).to(topk_positions.device)

    mask = range_tensor < topk_positions.unsqueeze(-1)
    final_mask = torch.zeros_like(mask)
    final_mask.scatter_(2, sorted_indices, mask)
    
    if response_length > 0:
        final_mask[:, :, L_for_dec:] = True
    
    return final_mask

def get_channelwise_offline(delta_t, alpha=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length

    k_values = L_for_dec * alpha
    k_values = torch.clamp(k_values, max=L_for_dec).to(torch.int64)

    delta_t_ranked, _ = torch.sort(delta_t, descending=True, dim=-1)
    dt_thre = torch.gather(delta_t_ranked.squeeze(0), 1, k_values.unsqueeze(-1)).view(-1)

    mask = delta_t >= dt_thre.view(1, -1, 1)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True
    
    return mask, dt_thre

def get_channelwise_normalize(delta_t, decay=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if decay is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)

    delta_t_sum = torch.sum(delta_t, dim=-1, keepdim=False)
    norm = (decay.unsqueeze(0) / delta_t_sum).unsqueeze(-1)
    norm = norm.repeat(1, 1, L)
    # print(norm.shape)
    if response_length > 0:
        norm[:, :, L_for_dec:] = 1
    delta_t = delta_t*norm
    return delta_t

def get_channelwise_dt_threshold(delta_t, dt_thre=None, response_length=0):
    L = delta_t.shape[2]
    L_for_dec = L-response_length
    if dt_thre is None:
        print("Decay bound is none, switch to topk method")
        return get_topk_mask_channelwise(delta_t=delta_t, k=2000, response_length=response_length)

    mask = delta_t > dt_thre.unsqueeze(0).unsqueeze(-1)

    if response_length > 0:
        mask[:, :, L_for_dec:] = True

    return mask
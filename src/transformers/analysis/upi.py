import torch
import torch.nn.functional as F

# dt = scale_dt(self.upi_mask, dt, self.dt_bias)
def scale_dt(scale_mask, dt, dt_bias):
    assert (scale_mask.dim() == 0) or (
        scale_mask.dim() == 1 and scale_mask.size(0) == dt_bias.size(0)
    ), f"scale_mask must be scalar or of shape {(dt_bias.size(0),)}, got {tuple(scale_mask.shape)}"
    t = F.softplus(dt + dt_bias) / scale_mask
    y = torch.expm1(t).clamp_min(1e-6)
    return torch.log(y) - dt_bias

def dynamic_scale_mask(scale_mask, seq_len, seq_len_scaled=32768, seq_len_trained=4096):
    scale = max(0, (seq_len-seq_len_trained)/(seq_len_scaled-seq_len_trained))
    return scale * (scale_mask - 1) + 1

def scale_proper(scale, A, X, dt, dt_bias):
    X = rearrange(X, "b l (h p) -> b l h p", h=A.shape[0])
    dt_sp = F.softplus((dt + dt_bias).to(dtype=torch.float32)).to(dtype=dt.dtype)

    if abs(scale - 1) < 1e-6:
        return A, X, dt_sp
    
    sA = A * scale
    x_scale = torch.expm1(sA * dt_sp) / torch.expm1(A * dt_sp)
    sX = X * x_scale.unsqueeze(-1)
    return sA, sX, dt_sp
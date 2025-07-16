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

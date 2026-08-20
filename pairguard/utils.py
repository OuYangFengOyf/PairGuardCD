import random
import numpy as np
import torch

def seed_everything(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx.square() + gy.square() + 1e-8)

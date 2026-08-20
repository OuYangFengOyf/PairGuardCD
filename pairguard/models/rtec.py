import torch
from torch import nn
import torch.nn.functional as F
from pairguard.utils import tensor_gradient_magnitude

class RTEC(nn.Module):
    def __init__(
        self,
        c_b2=24,
        c_d2=32,
        c_z=24,
        hidden=48,
        kappa=2.0,
    ):
        super().__init__()
        self.kappa = float(kappa)
        in_channels = c_b2 + c_d2 + 1 + 1 + c_z
        self.aggregate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.fn_gate = nn.Conv2d(hidden, 1, 1)
        self.fn_direction = nn.Conv2d(hidden, 1, 1)
        self.fp_gate = nn.Conv2d(hidden, 1, 1)
        self.fp_direction = nn.Conv2d(hidden, 1, 1)

    def forward(self, b2, d2, base_probability, relation_code, base_logit):
        size = d2.shape[-2:]
        probability = F.interpolate(
            base_probability.detach(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        boundary = tensor_gradient_magnitude(probability)
        relation = F.interpolate(
            relation_code.detach(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        features = torch.cat(
            [
                b2.detach(),
                d2.detach(),
                probability,
                boundary,
                relation,
            ],
            dim=1,
        )
        hidden = self.aggregate(features)

        fn_response = torch.sigmoid(self.fn_gate(hidden))
        fp_response = torch.sigmoid(self.fp_gate(hidden))
        positive_direction = F.relu(self.fn_direction(hidden))
        negative_direction = F.relu(self.fp_direction(hidden))
        delta = (
            fn_response * positive_direction
            - fp_response * negative_direction
        ).clamp(-self.kappa, self.kappa)
        delta_up = F.interpolate(
            delta,
            size=base_logit.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        final_logit = base_logit + delta_up
        return {
            "RFN": fn_response,
            "RFP": fp_response,
            "delta_logit": delta,
            "final_logit": final_logit,
            "final_prob": torch.sigmoid(final_logit),
        }

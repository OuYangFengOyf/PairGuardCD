import torch
from torch import nn
from .backbone import SiameseMobileNetV3Small
from .ei_fusion import MultiScaleEIFusion
from .relation import DirectionalRelationHead
from .decoder import RelationGuidedDecoder
from .rtec import RTEC

class PairGuardCD(nn.Module):
    def __init__(self, pretrained_backbone=False, rtec_clip=2.0):
        super().__init__()
        self.backbone = SiameseMobileNetV3Small(pretrained_backbone)
        self.ei = MultiScaleEIFusion()
        self.relation = DirectionalRelationHead()
        self.decoder = RelationGuidedDecoder()
        self.rtec = RTEC(kappa=rtec_clip)

    def forward(self, img1, img2, use_rtec=True, use_relation=True):
        features1, features2 = self.backbone(img1, img2)
        fused = self.ei(features1, features2)

        if use_relation:
            relation = self.relation(features1, features2)
            relation_code = relation["ZR"]
        else:
            batch = img1.shape[0]
            relation_code = torch.zeros(
                batch,
                24,
                features1[8].shape[-2],
                features1[8].shape[-1],
                device=img1.device,
                dtype=features1[8].dtype,
            )
            relation = {"ZR": relation_code}

        decoded = self.decoder(
            fused,
            relation_code,
            img1.shape[-2:],
        )
        output = {**relation, **decoded, "B": fused}

        if use_rtec:
            correction = self.rtec(
                fused[2],
                decoded["D2"],
                decoded["base_prob"],
                relation_code,
                decoded["base_logit"],
            )
            output.update(correction)
        else:
            output["final_logit"] = decoded["base_logit"]
            output["final_prob"] = decoded["base_prob"]

        return output

    def freeze_except_rtec(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("rtec.")

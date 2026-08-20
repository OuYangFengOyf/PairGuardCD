from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .c2f_scpt import C2FSCPT
from .narp_us import RelationState, NARPUS, build_pseudo_label, null_reliability
from .relation_teacher import OffsetNullRelationTeacher
from .geometry import transform_points
from .correspondence import local_correspondence_evidence
from .proposal_builder import ProposalHypothesisBuilder

class OfflineTeacher:
    def __init__(self, feature_extractor, sam_adapter, selector, relation_teacher=None, c2f_scpt=None):
        if feature_extractor is None:
            raise ValueError("A DINOv2 feature extractor is required.")
        if sam_adapter is None:
            raise ValueError("A SAM2 proposal adapter is required.")
        if selector is None:
            raise ValueError("A trained NARP-US utility selector is required.")
        self.feature_extractor = feature_extractor
        self.scpt = c2f_scpt or C2FSCPT()
        self.proposal_builder = ProposalHypothesisBuilder(sam_adapter)
        self.narp = NARPUS(selector)
        self.rel_teacher = relation_teacher or OffsetNullRelationTeacher()

    @staticmethod
    def _u8(image):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.max() <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @staticmethod
    def _resize_feature(feature, size):
        feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        return F.normalize(feature, dim=1)

    def _relation_maps(self, accepted, trans12, trans21, output_hw, coarse_hw, fine_hw):
        height, width = output_hw
        valid12 = np.zeros((height, width), dtype=np.float32)
        valid21 = np.zeros_like(valid12)
        null12 = np.zeros_like(valid12)
        null21 = np.zeros_like(valid12)
        utility12 = np.zeros_like(valid12)
        utility21 = np.zeros_like(valid12)
        null_r12 = np.zeros_like(valid12)
        null_r21 = np.zeros_like(valid12)

        for hypothesis, _, usability in accepted:
            if hypothesis.state == RelationState.OBJECT_OBJECT:
                if hypothesis.mask1 is not None:
                    valid12[hypothesis.mask1 > 0] = 1.0
                    utility12[hypothesis.mask1 > 0] = np.maximum(
                        utility12[hypothesis.mask1 > 0],
                        usability,
                    )
                if hypothesis.mask2 is not None:
                    valid21[hypothesis.mask2 > 0] = 1.0
                    utility21[hypothesis.mask2 > 0] = np.maximum(
                        utility21[hypothesis.mask2 > 0],
                        usability,
                    )
            elif hypothesis.state == RelationState.OBJECT_NULL and hypothesis.mask1 is not None:
                mask = hypothesis.mask1 > 0
                null12[mask] = 1.0
                utility12[mask] = usability
                null_r12[mask] = null_reliability(
                    hypothesis.q_src,
                    hypothesis.search_reliability,
                    hypothesis.counterpart_evidence,
                )
            elif hypothesis.state == RelationState.NULL_OBJECT and hypothesis.mask2 is not None:
                mask = hypothesis.mask2 > 0
                null21[mask] = 1.0
                utility21[mask] = usability
                null_r21[mask] = null_reliability(
                    hypothesis.q_src,
                    hypothesis.search_reliability,
                    hypothesis.counterpart_evidence,
                )

        dx12 = np.zeros((height, width), dtype=np.float32)
        dy12 = np.zeros_like(dx12)
        q12 = np.zeros_like(dx12)
        dx21 = np.zeros_like(dx12)
        dy21 = np.zeros_like(dx12)
        q21 = np.zeros_like(dx12)

        for transports, dx_map, dy_map, q_map in (
            (trans12, dx12, dy12, q12),
            (trans21, dx21, dy21, q21),
        ):
            for transport in transports:
                ys, xs = np.where(transport.component_mask > 0)
                if len(xs) == 0:
                    continue
                points = np.stack([xs, ys], axis=1).astype(np.float32)
                mapped = transform_points(points, transport.matrix)
                displacement = mapped - points
                dx_map[ys, xs] = displacement[:, 0]
                dy_map[ys, xs] = displacement[:, 1]
                q_map[ys, xs] = transport.reliability

        def resize(array, size, mode="nearest"):
            tensor = torch.from_numpy(array).view(1, 1, height, width)
            return F.interpolate(tensor, size=size, mode=mode)

        coarse = {
            "valid12": resize(valid12, coarse_hw),
            "valid21": resize(valid21, coarse_hw),
            "null12": resize(null12, coarse_hw),
            "null21": resize(null21, coarse_hw),
            "utility12": resize(utility12, coarse_hw),
            "utility21": resize(utility21, coarse_hw),
        }
        fine = {
            "valid12": resize(valid12, fine_hw),
            "valid21": resize(valid21, fine_hw),
            "null12": resize(null12, fine_hw),
            "null21": resize(null21, fine_hw),
            "utility12": resize(utility12, fine_hw),
            "utility21": resize(utility21, fine_hw),
        }
        geometry = {
            "dx12": resize(dx12, coarse_hw) / 16.0,
            "dy12": resize(dy12, coarse_hw) / 16.0,
            "q12": resize(q12, coarse_hw),
            "dx21": resize(dx21, coarse_hw) / 16.0,
            "dy21": resize(dy21, coarse_hw) / 16.0,
            "q21": resize(q21, coarse_hw),
            "null_r12": resize(null_r12, coarse_hw),
            "null_r21": resize(null_r21, coarse_hw),
        }
        return coarse, fine, geometry

    def build(self, img1, img2, seed12, seed21):
        img1 = self._u8(img1)
        img2 = self._u8(img2)
        seed12 = np.asarray(seed12, dtype=np.float32)
        seed21 = np.asarray(seed21, dtype=np.float32)
        if seed12.shape != seed21.shape:
            raise ValueError("Bidirectional seed maps must have identical spatial dimensions.")

        height, width = seed12.shape
        fine_hw = (max(1, height // 8), max(1, width // 8))
        coarse_hw = (max(1, height // 16), max(1, width // 16))

        raw_z1 = self.feature_extractor(img1)
        raw_z2 = self.feature_extractor(img2)
        z1_fine = self._resize_feature(raw_z1, fine_hw)
        z2_fine = self._resize_feature(raw_z2, fine_hw)
        z1_coarse = self._resize_feature(raw_z1, coarse_hw)
        z2_coarse = self._resize_feature(raw_z2, coarse_hw)

        corr12, corr_valid12 = local_correspondence_evidence(
            z1_fine,
            z2_fine,
            seed12.shape,
        )
        corr21, corr_valid21 = local_correspondence_evidence(
            z2_fine,
            z1_fine,
            seed21.shape,
        )

        score12, trans12 = self.scpt(
            img1,
            img2,
            seed12,
            corr12,
            corr_valid12,
        )
        score21, trans21 = self.scpt(
            img2,
            img1,
            seed21,
            corr21,
            corr_valid21,
        )

        hypotheses = self.proposal_builder.build_direction(
            img1,
            img2,
            seed12,
            score12,
            trans12,
            "12",
        )
        hypotheses += self.proposal_builder.build_direction(
            img2,
            img1,
            seed21,
            score21,
            trans21,
            "21",
        )
        accepted, unknown, rejected = self.narp.select(hypotheses)

        strict_background = (
            (seed12 < 0.10)
            & (seed21 < 0.10)
            & (score12 < 0.20)
            & (score21 < 0.20)
            & corr_valid12.astype(bool)
            & corr_valid21.astype(bool)
        )
        ye, vpl = build_pseudo_label(
            seed12.shape,
            accepted,
            seed12,
            strict_background,
        )
        coarse_maps, fine_maps, geometry = self._relation_maps(
            accepted,
            trans12,
            trans21,
            seed12.shape,
            coarse_hw,
            fine_hw,
        )

        relation = self.rel_teacher.build(
            z1_coarse,
            z2_coarse,
            z1_fine,
            z2_fine,
            coarse_maps["valid12"],
            coarse_maps["valid21"],
            fine_maps["valid12"],
            fine_maps["valid21"],
            coarse_maps["null12"],
            coarse_maps["null21"],
            fine_maps["null12"],
            fine_maps["null21"],
            coarse_maps["utility12"],
            coarse_maps["utility21"],
            fine_maps["utility12"],
            fine_maps["utility21"],
            geometry,
        )

        cache = {
            "Ye": ye[None].astype(np.float32),
            "Vpl": vpl[None].astype(np.float32),
            "seed12": seed12[None].astype(np.float32),
            "seed21": seed21[None].astype(np.float32),
        }
        for key, value in relation.items():
            cache[key] = value.squeeze(0).detach().cpu().numpy().astype(np.float32)

        metadata = {
            "num_transports_12": len(trans12),
            "num_transports_21": len(trans21),
            "num_hypotheses": len(hypotheses),
            "num_accepted": len(accepted),
            "num_unknown": len(unknown),
            "num_rejected": len(rejected),
            "valid_coverage": float(vpl.mean()),
        }
        return cache, metadata

    def save(self, out_path, cache):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **cache)

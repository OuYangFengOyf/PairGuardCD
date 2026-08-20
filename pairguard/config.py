from dataclasses import dataclass

@dataclass
class PairGuardConfig:
    image_size: int = 256
    num_coarse_offsets: int = 169
    num_coarse_classes: int = 170
    num_fine_offsets: int = 25
    relation_code_channels: int = 24
    coarse_radius: int = 6
    fine_radius: int = 2
    rtec_clip: float = 2.0
    pseudo_weight: float = 1.0
    relation_weight: float = 0.5

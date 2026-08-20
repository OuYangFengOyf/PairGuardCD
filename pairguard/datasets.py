from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PIXEL_CACHE_KEYS = {"Ye", "Vpl", "seed12", "seed21"}
COARSE_CACHE_KEYS = {
    "coarse_12",
    "coarse_21",
    "rT_coarse_12",
    "rT_coarse_21",
    "teacher_dx12",
    "teacher_dy12",
    "teacher_dx21",
    "teacher_dy21",
}
FINE_CACHE_KEYS = {"fine_12", "fine_21", "rT_fine_12", "rT_fine_21"}

def _load_rgb(path):
    return TF.to_tensor(Image.open(path).convert("RGB"))

def _load_mask(path):
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)

def _photometric(tensor):
    image = TF.to_pil_image(tensor)
    if torch.rand(()) < 0.3:
        brightness = 1.0 + float(torch.empty(1).uniform_(-0.15, 0.15))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.15, 0.15))
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return TF.to_tensor(image)

def _crop_tensor(tensor, top, left, height, width):
    return tensor[..., top : top + height, left : left + width]

class PairListDataset(Dataset):
    def __init__(
        self,
        manifest,
        size=256,
        augment="none",
        normalize=True,
        crop_training=True,
        changed_crop_probability=0.6,
    ):
        self.size = int(size)
        self.augment = augment
        self.normalize = normalize
        self.crop_training = bool(crop_training)
        self.changed_crop_probability = float(changed_crop_probability)
        self.records = []

        for raw in Path(manifest).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [item.strip() for item in raw.split(",")]
            while len(parts) < 5:
                parts.append("")
            self.records.append(
                (
                    parts[0],
                    parts[1],
                    parts[2] or None,
                    parts[3] or None,
                    parts[4] or str(len(self.records)),
                )
            )

    @property
    def group_ids(self):
        return [record[4] for record in self.records]

    def __len__(self):
        return len(self.records)

    def _sample_crop(self, height, width, label, aligned):
        crop_h = min(self.size, height)
        crop_w = min(self.size, width)
        if crop_h == height and crop_w == width:
            return 0, 0, crop_h, crop_w

        step = 16 if aligned else 1
        max_top = height - crop_h
        max_left = width - crop_w

        if (
            label is not None
            and torch.rand(()) < self.changed_crop_probability
            and (label > 0.5).any()
        ):
            ys, xs = torch.where(label[0] > 0.5)
            index = int(torch.randint(0, len(xs), ()).item())
            center_y = int(ys[index].item())
            center_x = int(xs[index].item())
            top_min = max(0, center_y - crop_h + 1)
            top_max = min(center_y, max_top)
            left_min = max(0, center_x - crop_w + 1)
            left_max = min(center_x, max_left)
            top = int(torch.randint(top_min, top_max + 1, ()).item()) if top_max >= top_min else 0
            left = int(torch.randint(left_min, left_max + 1, ()).item()) if left_max >= left_min else 0
        else:
            top = int(torch.randint(0, max_top + 1, ()).item())
            left = int(torch.randint(0, max_left + 1, ()).item())

        if aligned:
            top = min(max_top, (top // step) * step)
            left = min(max_left, (left // step) * step)

        return top, left, crop_h, crop_w

    def _crop_cache(self, cache, top, left, crop_h, crop_w):
        output = {}
        for key in cache.files:
            array = cache[key]
            tensor = torch.from_numpy(array).float()
            if key in PIXEL_CACHE_KEYS:
                output[key] = _crop_tensor(tensor, top, left, crop_h, crop_w)
            elif key in COARSE_CACHE_KEYS:
                y = top // 16
                x = left // 16
                h = crop_h // 16
                w = crop_w // 16
                output[key] = _crop_tensor(tensor, y, x, h, w)
            elif key in FINE_CACHE_KEYS:
                y = top // 8
                x = left // 8
                h = crop_h // 8
                w = crop_w // 8
                output[key] = _crop_tensor(tensor, y, x, h, w)
            else:
                output[key] = tensor
        return output

    def _resize_if_needed(self, img1, img2, label):
        if img1.shape[-2:] == (self.size, self.size):
            return img1, img2, label
        img1 = TF.resize(img1, [self.size, self.size], antialias=True)
        img2 = TF.resize(img2, [self.size, self.size], antialias=True)
        if label is not None:
            label = TF.resize(
                label,
                [self.size, self.size],
                interpolation=TF.InterpolationMode.NEAREST,
            )
        return img1, img2, label

    def _geometric(self, img1, img2, label):
        if torch.rand(()) < 0.5:
            img1, img2 = TF.hflip(img1), TF.hflip(img2)
            if label is not None:
                label = TF.hflip(label)
        if torch.rand(()) < 0.5:
            img1, img2 = TF.vflip(img1), TF.vflip(img2)
            if label is not None:
                label = TF.vflip(label)
        rotation = int(torch.randint(0, 4, ()).item())
        if rotation:
            img1 = torch.rot90(img1, rotation, [-2, -1])
            img2 = torch.rot90(img2, rotation, [-2, -1])
            if label is not None:
                label = torch.rot90(label, rotation, [-2, -1])
        return img1, img2, label

    def __getitem__(self, idx):
        path1, path2, label_path, cache_path, group_id = self.records[idx]
        img1 = _load_rgb(path1)
        img2 = _load_rgb(path2)
        label = _load_mask(label_path) if label_path else None
        cache = np.load(cache_path, allow_pickle=False) if cache_path else None

        if img1.shape[-2:] != img2.shape[-2:]:
            raise ValueError(f"Image size mismatch: {path1} vs {path2}")

        height, width = img1.shape[-2:]
        use_crop = self.crop_training and self.augment in {"full", "photometric"}
        cached_sample = {}

        if use_crop:
            top, left, crop_h, crop_w = self._sample_crop(
                height,
                width,
                label,
                aligned=cache is not None,
            )
            img1 = _crop_tensor(img1, top, left, crop_h, crop_w)
            img2 = _crop_tensor(img2, top, left, crop_h, crop_w)
            if label is not None:
                label = _crop_tensor(label, top, left, crop_h, crop_w)
            if cache is not None:
                cached_sample = self._crop_cache(cache, top, left, crop_h, crop_w)
        elif cache is not None:
            for key in cache.files:
                cached_sample[key] = torch.from_numpy(cache[key]).float()

        img1, img2, label = self._resize_if_needed(img1, img2, label)

        if self.augment == "full":
            img1, img2, label = self._geometric(img1, img2, label)
            img1 = _photometric(img1)
            img2 = _photometric(img2)
        elif self.augment == "photometric":
            img1 = _photometric(img1)
            img2 = _photometric(img2)

        if self.normalize:
            img1 = TF.normalize(img1, IMAGENET_MEAN, IMAGENET_STD)
            img2 = TF.normalize(img2, IMAGENET_MEAN, IMAGENET_STD)

        sample = {
            "img1": img1,
            "img2": img2,
            "id": str(idx),
            "group_id": group_id,
        }
        if label is not None:
            sample["label"] = label
        sample.update(cached_sample)
        return sample

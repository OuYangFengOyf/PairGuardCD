from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def sliding_window_predict(model, img1, img2, crop_size=256, stride=128, use_rtec=True):
    if img1.shape[0] != 1 or img2.shape[0] != 1:
        raise ValueError("Sliding-window inference requires batch size 1.")
    height, width = img1.shape[-2:]
    if height <= crop_size and width <= crop_size:
        return model(img1, img2, use_rtec=use_rtec)["final_prob"]

    prediction = torch.zeros((1, 1, height, width), device=img1.device)
    count = torch.zeros_like(prediction)
    ys = list(range(0, max(1, height - crop_size + 1), stride))
    xs = list(range(0, max(1, width - crop_size + 1), stride))
    if not ys or ys[-1] != height - crop_size:
        ys.append(max(0, height - crop_size))
    if not xs or xs[-1] != width - crop_size:
        xs.append(max(0, width - crop_size))

    for y in ys:
        for x in xs:
            patch1 = img1[..., y : y + crop_size, x : x + crop_size]
            patch2 = img2[..., y : y + crop_size, x : x + crop_size]
            patch_height, patch_width = patch1.shape[-2:]
            if patch_height < crop_size or patch_width < crop_size:
                pad = (0, crop_size - patch_width, 0, crop_size - patch_height)
                patch1 = F.pad(patch1, pad)
                patch2 = F.pad(patch2, pad)
            patch_prediction = model(patch1, patch2, use_rtec=use_rtec)["final_prob"]
            patch_prediction = patch_prediction[..., :patch_height, :patch_width]
            prediction[..., y : y + patch_height, x : x + patch_width] += patch_prediction
            count[..., y : y + patch_height, x : x + patch_width] += 1

    return prediction / count.clamp_min(1)

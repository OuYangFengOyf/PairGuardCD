from pathlib import Path
import torch
from torch.utils.data import DataLoader

def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

def make_loader(dataset, batch_size=4, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

def save_checkpoint(path, model, optimizer=None, extra=None):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, output)

def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

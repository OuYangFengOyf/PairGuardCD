from pairguard.losses import bce_dice_from_logit
from .common import move_batch

def train_s0_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch["img1"],
            batch["img2"],
            use_rtec=False,
            use_relation=False,
        )
        loss = bce_dice_from_logit(output["base_logit"], batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))

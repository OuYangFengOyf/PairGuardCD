from pairguard.losses import rtec_typed_loss
from .common import move_batch

def train_s2_epoch(model, loader, optimizer, device):
    model.eval()
    model.rtec.train()
    total = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch["img1"],
            batch["img2"],
            use_rtec=True,
            use_relation=True,
        )
        loss = rtec_typed_loss(output, batch["label"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())

    return total / max(1, len(loader))

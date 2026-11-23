"""NLL training and self-describing checkpoints for the flat baseline."""
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from stateFormer.flat_transformer import FlatTransGenerator


def flat_nll(outputs, sequences, lengths, weights):
    batch, packets, fields = sequences.shape
    if len(outputs["logits"]) != fields:
        raise ValueError("Output head order does not match target fields")
    mask = torch.arange(packets, device=sequences.device)[None] < lengths[:, None]
    nll = sequences.new_zeros((), dtype=torch.float32)
    for field, logits in enumerate(outputs["logits"]):
        if logits.shape[:2] != (batch, packets):
            raise ValueError("Training requires one prediction per packet and field")
        target = sequences[:, :, field].masked_fill(~mask, 1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
        nll = nll + (loss.reshape(batch, packets) * mask * weights[:, None]).sum()
    # Preserve the packet model's field average, masks and label weights.
    return nll / fields


def save_flat_checkpoint(path, model, metadata):
    path = Path(path)
    state = dict(model_type=model.model_type, model_config=model.model_config(),
                 metadata=metadata, model_state_dict=model.state_dict())
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_flat_checkpoint(path, device, expected_metadata=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_type") != FlatTransGenerator.model_type:
        raise ValueError("Checkpoint is not a flat-transformer checkpoint")
    if expected_metadata is not None and state["metadata"] != expected_metadata:
        raise ValueError("Checkpoint fields, labels, bins or experiment identity differ")
    model = FlatTransGenerator(**state["model_config"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device), state["metadata"]


def train_flat(model, dataloader, epochs, device, checkpoint_dir=None, checkpoint=100, metadata=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    history = []
    for epoch in range(epochs):
        model.train()
        start = time.perf_counter()
        batches = 0
        for sequences, labels, lengths, weights in dataloader:
            sequences, labels, lengths, weights = [v.to(device) for v in (sequences, labels, lengths, weights)]
            inputs = model.shift_targets(sequences)
            loss = flat_nll(model(labels, lengths, inputs), sequences, lengths, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite flat-model NLL; stopping")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches += 1
        if batches == 0:
            raise ValueError("Empty dataloader")
        elapsed = time.perf_counter() - start
        record = dict(epoch=epoch + 1, nll=float(loss.detach()), seconds=elapsed, batches=batches)
        history.append(record)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {record['nll']}, NLL: {record['nll']}, time cost: {elapsed}.", flush=True)
        if checkpoint_dir is not None:
            save_flat_checkpoint(checkpoint_dir / "generator.pth", model, metadata)
            if checkpoint > 0 and (epoch + 1) % checkpoint == 0:
                save_flat_checkpoint(checkpoint_dir / f"generator_{epoch+1}.pth", model, metadata)
    return history

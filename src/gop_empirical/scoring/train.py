"""Shared Adam + early-stopping trainer for E1 (phones) and E2 (sequences)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def masked_mse(pred: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over real phones only (``pad_mask`` True = pad)."""
    valid = ~pad_mask
    if int(valid.sum()) == 0:
        raise RuntimeError("sequence batch has no real phones")
    err = (pred - target) ** 2
    return err[valid].mean()


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class _SeqDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pad_mask: np.ndarray,
        phone_ids: np.ndarray | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.pad_mask = torch.as_tensor(pad_mask, dtype=torch.bool)
        self.phone_ids = (
            None if phone_ids is None else torch.as_tensor(phone_ids, dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if self.phone_ids is None:
            return self.x[index], self.y[index], self.pad_mask[index]
        return self.x[index], self.y[index], self.pad_mask[index], self.phone_ids[index]


def phone_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    phone_ids: np.ndarray | None = None,
) -> DataLoader:
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32)
    if phone_ids is None:
        dataset: TensorDataset | Dataset = TensorDataset(tensor_x, tensor_y)
    else:
        dataset = TensorDataset(
            tensor_x, tensor_y, torch.as_tensor(np.array(phone_ids, copy=True), dtype=torch.long)
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
    )


def sequence_loader(
    x: np.ndarray,
    y: np.ndarray,
    pad_mask: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    phone_ids: np.ndarray | None = None,
) -> DataLoader:
    dataset = _SeqDataset(x, y, pad_mask, phone_ids=phone_ids)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size < 2 or pred.size != target.size:
        return float("nan")
    if float(pred.std()) < 1e-12 or float(target.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sequence: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    pred_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for batch in loader:
        if sequence:
            if len(batch) == 4:
                x, y, pad_mask, phone_ids = batch
                phone_ids = phone_ids.to(device)
            else:
                x, y, pad_mask = batch
                phone_ids = None
            x = x.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)
            pred = model(x, pad_mask) if phone_ids is None else model(x, pad_mask, phone_ids)
            loss = masked_mse(pred, y, pad_mask)
            valid = ~pad_mask
            n_valid = int(valid.sum().item())
            pred_flat = pred.detach()[valid]
            y_flat = y.detach()[valid]
        else:
            if len(batch) == 3:
                x, y, phone_ids = batch
                phone_ids = phone_ids.to(device)
            else:
                x, y = batch
                phone_ids = None
            x = x.to(device)
            y = y.to(device)
            pred = model(x) if phone_ids is None else model(x, phone_ids)
            loss = torch.mean((pred - y) ** 2)
            n_valid = int(y.numel())
            pred_flat = pred.detach().reshape(-1)
            y_flat = y.detach().reshape(-1)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * n_valid
        count += n_valid
        pred_chunks.append(pred_flat.cpu().numpy())
        y_chunks.append(y_flat.cpu().numpy())
    if count == 0:
        raise RuntimeError("empty epoch")
    pcc = _pearson(np.concatenate(pred_chunks), np.concatenate(y_chunks))
    return total / count, pcc


def train_regressor(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    sequence: bool,
) -> dict[str, Any]:
    set_torch_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    epochs_ran = 0
    kind = "transformer" if sequence else "mlp"
    for epoch in range(1, int(max_epochs) + 1):
        train_mse, train_pcc = _run_epoch(
            model, train_loader, optimizer=optimizer, device=device, sequence=sequence
        )
        val_mse, val_pcc = _run_epoch(
            model, val_loader, optimizer=None, device=device, sequence=sequence
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_pcc": train_pcc,
                "val_pcc": val_pcc,
            }
        )
        epochs_ran = epoch
        improved = val_mse + 1e-12 < best_val
        if improved:
            best_val = val_mse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        stopped = (not improved) and wait >= int(patience)
        mark = "*" if improved else " "
        print(
            f"  {kind} epoch {epoch}/{int(max_epochs)}{mark}  "
            f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
            f"train_pcc={train_pcc:.4f}  val_pcc={val_pcc:.4f}  "
            f"best_mse={best_val:.6f}@{best_epoch}",
            flush=True,
        )
        if stopped:
            print(
                f"  {kind} early stop at epoch {epoch} (patience={int(patience)})",
                flush=True,
            )
            break
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return {
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "epochs_ran": int(epochs_ran),
        "history": history,
    }


@torch.no_grad()
def predict_mlp(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    phone_ids: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    loader = phone_loader(
        x,
        np.zeros(x.shape[0], dtype=np.float64),
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        phone_ids=phone_ids,
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 3:
            xb, _, ids = batch
            pred = model(xb.to(device), ids.to(device))
        else:
            xb, _ = batch
            pred = model(xb.to(device))
        chunks.append(pred.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


@torch.no_grad()
def predict_transformer(
    model: torch.nn.Module,
    packed: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    n_rows: int,
) -> np.ndarray:
    model.eval()
    phone_ids = packed.get("phone_ids")
    loader = sequence_loader(
        packed["x"],
        packed["y"],
        packed["pad_mask"],
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        phone_ids=phone_ids,
    )
    out = np.full(int(n_rows), np.nan, dtype=np.float64)
    offset = 0
    row_indices: list[np.ndarray] = packed["row_indices"]
    for batch in loader:
        if len(batch) == 4:
            xb, _yb, mask, ids = batch
            pred = model(xb.to(device), mask.to(device), ids.to(device)).detach().cpu().numpy()
        else:
            xb, _yb, mask = batch
            pred = model(xb.to(device), mask.to(device)).detach().cpu().numpy()
        bsz = pred.shape[0]
        for j in range(bsz):
            idx = row_indices[offset + j]
            n = int(idx.size)
            out[idx] = pred[j, :n]
        offset += bsz
    if np.isnan(out).any():
        raise RuntimeError("transformer prediction missed some phone rows")
    return out

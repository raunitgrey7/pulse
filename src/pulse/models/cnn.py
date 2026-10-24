"""A small 1D CNN that classifies a raw vibration window.

Design choices, each tied to the physics or the deployment target.
Standardisation is part of the graph so every runtime (PyTorch, ONNX fp32,
ONNX int8) applies exactly the same preprocessing, and the removed scale is
fed to the classifier because absolute level is a cue in its own right. The
first convolution is a wide, strided learned filter bank; rectification and
pooling after it act as an envelope detector, so the later layers see the
impact modulation at a low frame rate. Bearing classes differ in how impacts
are modulated over tens of milliseconds, which is why the receptive field
has to cover most of the window: an earlier variant with a 9 ms receptive
field could not tell inner-race from ball defects (about 50 percent on those
two classes). The head flattens the remaining 16 frames instead of averaging
them, so modulation that spans the window is still visible. Training is
deterministic for a fixed seed: fixed thread count, seeded generator and
deterministic algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pulse.signals.dataset import Dataset
from pulse.signals.generator import CLASS_NAMES

N_CLASSES = len(CLASS_NAMES)


class Standardize(nn.Module):
    """Per-window zero mean, unit variance, plus the log of the scale that was removed.

    Shape cues (impulsiveness, harmonic ratios) live in the standardised
    waveform; the absolute level in g is a cue of its own (imbalance force
    grows with speed squared), so it is returned separately for the head.
    Written with ops every exporter supports.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        centred = x - mean
        std = torch.sqrt((centred * centred).mean(dim=-1, keepdim=True) + self.eps)
        return centred / std, torch.log(std).flatten(1)


def _block(c_in: int, c_out: int, kernel: int, stride: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool),
    )


class PulseCNN(nn.Module):
    """Filter-bank front end, four envelope-rate conv blocks, flattened linear head."""

    def __init__(self, n_classes: int = N_CLASSES, width: int = 16, n_samples: int = 2048) -> None:
        super().__init__()
        self.standardize = Standardize()
        self.features = nn.Sequential(
            _block(1, width, 32, 2, 4),
            _block(width, 2 * width, 7, 1, 2),
            _block(2 * width, 4 * width, 7, 1, 2),
            _block(4 * width, 4 * width, 7, 1, 2),
            _block(4 * width, 4 * width, 5, 1, 2),
        )
        n_frames = n_samples // 128
        self.head = nn.Linear(4 * width * n_frames + 1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z, log_scale = self.standardize(x)
        frames = self.features(z).flatten(1)
        return self.head(torch.cat([frames, log_scale], dim=1))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters; defaults train in about a minute on a laptop CPU."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    threads: int = 4
    seed: int = 0
    max_shift: int = 256


@dataclass
class TrainHistory:
    """Per-epoch loss and validation accuracy, plus the epoch that was kept."""

    train_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = -1


def set_deterministic(seed: int, threads: int) -> torch.Generator:
    """Pin everything that makes CPU training reproducible and return a seeded generator."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch.Generator().manual_seed(seed)


def _split(n: int, val_fraction: int | float, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(n, generator=gen)
    n_val = round(n * val_fraction)
    return perm[n_val:], perm[:n_val]


def _augment(xb: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Random circular shift: the phase of a window carries no label information."""
    if max_shift <= 0:
        return xb
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=gen))
    return torch.roll(xb, shifts=shift, dims=-1)


@torch.no_grad()
def predict_logits(model: PulseCNN, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Logits for windows of shape (batch, n_samples)."""
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch_size])).numpy() for i in range(0, x.shape[0], batch_size)]
    return np.concatenate(out, axis=0)


def _run_epoch(
    model: PulseCNN,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    x: torch.Tensor,
    y: torch.Tensor,
    config: TrainConfig,
    gen: torch.Generator,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    order = torch.randperm(x.shape[0], generator=gen)
    total = 0.0
    for start in range(0, x.shape[0], config.batch_size):
        idx = order[start : start + config.batch_size]
        xb = _augment(x[idx], config.max_shift, gen)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), y[idx])
        loss.backward()
        opt.step()
        sched.step()
        total += float(loss.detach()) * idx.numel()
    return total / x.shape[0]


def _validation_accuracy(model: PulseCNN, dataset: Dataset, val_idx: np.ndarray) -> float:
    """Accuracy on the validation indices, or NaN when there is no validation split."""
    if val_idx.size == 0:
        return float("nan")
    preds = predict_logits(model, dataset.x[val_idx]).argmax(axis=1)
    return float((preds == dataset.y[val_idx]).mean())


def train_cnn(dataset: Dataset, config: TrainConfig = TrainConfig()) -> tuple[PulseCNN, TrainHistory]:
    """Train on a held-in split, keep the weights of the best validation epoch."""
    gen = set_deterministic(config.seed, config.threads)
    x = torch.from_numpy(dataset.x)
    y = torch.from_numpy(dataset.y)
    train_idx, val_idx = _split(len(dataset), config.val_fraction, gen)
    model = PulseCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = config.epochs * int(np.ceil(train_idx.numel() / config.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config.learning_rate, total_steps=steps)
    history = TrainHistory()
    best_state: dict[str, torch.Tensor] = {}
    best_acc = -1.0
    for epoch in range(config.epochs):
        loss = _run_epoch(model, opt, sched, x[train_idx], y[train_idx], config, gen)
        acc = _validation_accuracy(model, dataset, val_idx.numpy())
        history.train_loss.append(loss)
        history.val_accuracy.append(acc)
        if val_idx.numel() == 0 or acc >= best_acc:
            best_acc, history.best_epoch = acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_cnn(model: PulseCNN, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size


def load_cnn(path: Path) -> PulseCNN:
    model = PulseCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

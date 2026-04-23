"""
Self-Pruning CNN — Tredence AI Engineering Internship Case Study
================================================================
Author  : <Your Name>
Dataset : CIFAR-10
Device  : CUDA (RTX 4070 optimised)

Run
---
    python main.py                        # default settings
    python main.py --epochs 15 --batch 256
    python main.py --infer --checkpoint best_model.pt
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                   # headless / non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 0.  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN ops (small perf hit, worth it for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PRUNABLE LINEAR LAYER
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that associates every weight scalar
    with a learnable gate_score.

    Forward pass
    ────────────
        gates         = sigmoid(gate_scores)      ∈ (0, 1)  per-weight
        pruned_weight = weight ⊙ gates            element-wise product
        output        = x @ pruned_weight.T + bias

    Gradients flow through *both* weight and gate_scores because all
    operations (sigmoid, element-wise multiply, linear) are differentiable.

    Sparsity penalty contribution
    ──────────────────────────────
        L1 of gates = Σ |sigmoid(gate_scores)|   (always positive, so = Σ gates)

    Why L1 drives sparsity: the L1 subgradient at gate=g is +λ regardless
    of magnitude, creating constant pressure toward zero — unlike L2 whose
    gradient vanishes near zero.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard learnable weight & bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight; initialised near 0 so
        # sigmoid(0) = 0.5, giving all connections a "neutral" start
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming init for weight (best for ReLU activations)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)          # (out, in)
        pruned_weight = self.weight * gates                      # element-wise
        return nn.functional.linear(x, pruned_weight, self.bias)

    # ── sparsity helpers ──────────────────────────────────────────────────────
    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of the gate values for this layer (scalar)."""
        return torch.sigmoid(self.gate_scores).sum()

    @torch.no_grad()
    def layer_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below *threshold* (i.e. effectively pruned)."""
        gates  = torch.sigmoid(self.gate_scores)
        pruned = (gates < threshold).sum().item()
        return pruned / gates.numel()

    @torch.no_grad()
    def gate_values(self) -> np.ndarray:
        """Flat numpy array of all gate values — used for plotting."""
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().ravel()

    # ── hard-prune ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def hard_prune(self, threshold: float = 1e-2) -> None:
        """
        Post-training: zero out weights whose gate < threshold.
        Converts soft probabilistic gates into a binary sparse mask.
        """
        mask = (torch.sigmoid(self.gate_scores) >= threshold).float()
        self.weight.mul_(mask)

    def __repr__(self) -> str:
        return (f"PrunableLinear(in={self.in_features}, "
                f"out={self.out_features}, "
                f"gate_shape={list(self.gate_scores.shape)})")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SELF-PRUNING CNN
# ══════════════════════════════════════════════════════════════════════════════

class SelfPruningCNN(nn.Module):
    """
    Architecture
    ────────────
    Feature extractor : 3× (Conv → BN → ReLU → Pool)   — standard nn layers
    Classifier        : 2× PrunableLinear                — gates applied here

    Design choices
    ──────────────
    • BatchNorm after every Conv → training stability & faster convergence
    • AdaptiveAvgPool2d(4,4) → fixed spatial size regardless of input dims
    • Only the classifier uses PrunableLinear; the conv backbone is kept dense
      (pruning conv filters requires structured pruning, out of scope here)
    • Classifier hidden dim 512 gives enough capacity to classify CIFAR-10
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            # Block 1 — 3→32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 32×32 → 16×16

            # Block 2 — 32→64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 16×16 → 8×8

            # Block 3 — 64→128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),        # → 4×4 spatial
        )

        flat_dim = 128 * 4 * 4  # = 2048

        # ── Prunable classifier ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            PrunableLinear(flat_dim, 512),
            nn.ReLU(inplace=True),
            PrunableLinear(512, num_classes),
        )

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

    # ── aggregate sparsity loss ───────────────────────────────────────────────
    def sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate penalties across all PrunableLinear layers."""
        return sum(
            m.sparsity_loss()
            for m in self.modules()
            if isinstance(m, PrunableLinear)
        )

    # ── overall sparsity metric ───────────────────────────────────────────────
    @torch.no_grad()
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Weighted average sparsity across all PrunableLinear layers."""
        total, pruned = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates  = torch.sigmoid(m.gate_scores)
                pruned += (gates < threshold).sum().item()
                total  += gates.numel()
        return pruned / total if total > 0 else 0.0

    # ── layer-wise breakdown ──────────────────────────────────────────────────
    @torch.no_grad()
    def layer_sparsity_report(self, threshold: float = 1e-2) -> dict:
        report = {}
        for name, m in self.named_modules():
            if isinstance(m, PrunableLinear):
                report[name] = round(m.layer_sparsity(threshold) * 100, 2)
        return report

    # ── all gate values for visualisation ────────────────────────────────────
    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        parts = [m.gate_values() for m in self.modules() if isinstance(m, PrunableLinear)]
        return np.concatenate(parts) if parts else np.array([])

    # ── apply hard pruning to the whole model ────────────────────────────────
    @torch.no_grad()
    def hard_prune(self, threshold: float = 1e-2) -> None:
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                m.hard_prune(threshold)

    # ── parameter count helper ───────────────────────────────────────────────
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.

    Augmentation (train only)
    ─────────────────────────
    • RandomHorizontalFlip  — mirrors the image left/right
    • RandomCrop(32,pad=4)  — shifts image by up to 4 pixels each axis
    Together they're the standard CIFAR-10 augmentation baseline.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)   # per-channel CIFAR-10 stats

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        data_root, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_loader  = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP  (one epoch)
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model      : SelfPruningCNN,
    loader     : DataLoader,
    optimizer  : optim.Optimizer,
    criterion  : nn.Module,
    lam        : float,
    device     : torch.device,
    scaler     : torch.cuda.amp.GradScaler,
) -> tuple[float, float]:
    """
    Returns (avg_total_loss, train_accuracy).

    Loss
    ────
        Total = CrossEntropy(logits, labels)  +  λ × L1(gates)

    Mixed Precision (AMP)
    ─────────────────────
    autocast() runs forward pass in fp16 where safe → ~2× throughput on
    Ampere GPUs.  GradScaler prevents fp16 underflow during backprop.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits     = model(x)
            cls_loss   = criterion(logits, y)
            spar_loss  = model.sparsity_loss()
            loss       = cls_loss + lam * spar_loss

        scaler.scale(loss).backward()
        # Gradient clipping (unscaled) to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs          = x.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == y).sum().item()
        n_samples  += bs

    return total_loss / n_samples, correct / n_samples


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model  : SelfPruningCNN,
    loader : DataLoader,
    device : torch.device,
) -> float:
    """Returns top-1 accuracy on *loader*."""
    model.eval()
    correct   = 0
    n_samples = 0

    for x, y in loader:
        x, y       = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        correct   += (logits.argmax(1) == y).sum().item()
        n_samples += x.size(0)

    return correct / n_samples


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when val accuracy stops improving for *patience* epochs.
    Also saves the best model checkpoint.
    """
    def __init__(self, patience: int = 5, delta: float = 1e-4, path: str = "best_model.pt"):
        self.patience   = patience
        self.delta      = delta
        self.path       = path
        self.best_score = -np.inf
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_acc: float, model: nn.Module) -> None:
        if val_acc > self.best_score + self.delta:
            self.best_score = val_acc
            self.counter    = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_gate_distribution(
    model  : SelfPruningCNN,
    lam    : float,
    prefix : str = ".",
) -> str:
    """
    Saves a high-quality histogram of all gate values.
    A successful result shows:
      • Large spike near 0   → most gates pruned
      • Small cluster > 0    → surviving important connections
    """
    gates = model.all_gate_values()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(gates, bins=200, color="#2563EB", alpha=0.85, edgecolor="none")
    ax.axvline(0.01, color="#DC2626", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")
    ax.set_title(f"Gate Value Distribution  |  λ = {lam:.0e}", fontsize=14)
    ax.set_xlabel("Gate value  [sigmoid(gate_score)]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(prefix, f"gate_dist_lam{lam:.0e}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_results_summary(results: list[dict], prefix: str = ".") -> str:
    """Accuracy & sparsity bar chart across λ values."""
    lambdas   = [str(r["lambda"]) for r in results]
    accs      = [r["test_accuracy"] * 100 for r in results]
    sparsities = [r["sparsity_pct"] for r in results]

    x   = np.arange(len(lambdas))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, accs,       w, label="Test Accuracy (%)", color="#2563EB", alpha=0.85)
    bars2 = ax2.bar(x + w/2, sparsities, w, label="Sparsity (%)",      color="#16A34A", alpha=0.85)

    ax1.set_ylabel("Test Accuracy (%)", color="#2563EB")
    ax2.set_ylabel("Sparsity (%)",      color="#16A34A")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ={l}" for l in lambdas])
    ax1.set_title("Accuracy vs Sparsity Trade-off", fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    path = os.path.join(prefix, "results_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PRINTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width

def print_epoch(epoch: int, total: int, loss: float, train_acc: float,
                test_acc: float, sparsity: float) -> None:
    print(
        f"  Ep {epoch:>3}/{total}  "
        f"loss={loss:.4f}  "
        f"train={train_acc*100:5.2f}%  "
        f"test={test_acc*100:5.2f}%  "
        f"sparse={sparsity*100:5.2f}%"
    )

def print_results_table(results: list[dict]) -> None:
    print("\n" + _hr("═"))
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print(_hr())
    for r in results:
        print(f"  {r['lambda']:<12.0e} {r['test_accuracy']*100:>14.2f} {r['sparsity_pct']:>14.2f}")
    print(_hr("═"))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  INFERENCE SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]

def run_inference(checkpoint: str, device: torch.device, test_loader: DataLoader) -> None:
    """Load a saved checkpoint and print per-class accuracy."""
    model = SelfPruningCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    class_correct = [0] * 10
    class_total   = [0] * 10

    with torch.no_grad():
        for x, y in test_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            preds  = logits.argmax(1)
            for pred, label in zip(preds, y):
                class_correct[label] += (pred == label).item()
                class_total[label]   += 1

    print(f"\n{_hr()}")
    print(f"  Inference from checkpoint: {checkpoint}")
    print(_hr())
    for i, cls in enumerate(CIFAR10_CLASSES):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {cls:<12}: {acc:5.1f}%")
    overall = 100 * sum(class_correct) / sum(class_total)
    print(_hr())
    print(f"  Overall accuracy: {overall:.2f}%\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Self-Pruning CNN — Tredence Case Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",      type=int,   default=10,
                   help="Training epochs per lambda")
    p.add_argument("--batch",       type=int,   default=128,
                   help="Batch size")
    p.add_argument("--workers",     type=int,   default=4,
                   help="DataLoader worker processes")
    p.add_argument("--lambdas",     type=float, nargs="+",
                   default=[1e-5, 1e-4, 1e-3],
                   help="Sparsity penalty coefficients to sweep")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--threshold",   type=float, default=1e-2,
                   help="Gate threshold for hard pruning & sparsity measurement")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--outdir",      type=str,   default="outputs",
                   help="Directory for checkpoints, plots, and JSON results")
    p.add_argument("--infer",       action="store_true",
                   help="Run inference only (requires --checkpoint)")
    p.add_argument("--checkpoint",  type=str,   default="outputs/best_model.pt",
                   help="Checkpoint path for inference mode")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = build_parser().parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*72}")
    print(f"  Self-Pruning CNN  |  device={device}  |  seed={args.seed}")
    print(f"{'═'*72}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch, num_workers=args.workers)

    # ── Inference-only mode ───────────────────────────────────────────────────
    if args.infer:
        run_inference(args.checkpoint, device, test_loader)
        return

    # ── Training sweep over λ values ──────────────────────────────────────────
    results  : list[dict] = []
    best_acc : float      = 0.0

    for lam in args.lambdas:
        print(f"\n{'─'*72}")
        print(f"  λ = {lam:.0e}   ({args.epochs} epochs)")
        print(f"{'─'*72}")

        # Fresh model + optimiser per λ (fair comparison)
        model     = SelfPruningCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        # Cosine LR schedule: smoothly anneals LR to near-zero by end
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        ckpt_path = str(outdir / f"model_lam{lam:.0e}.pt")
        stopper   = EarlyStopping(
            patience=args.patience if args.patience > 0 else 9999,
            path=ckpt_path,
        )

        epoch_log: list[dict] = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, lam, device, scaler)
            test_acc  = evaluate(model, test_loader, device)
            sparsity  = model.overall_sparsity(args.threshold)
            scheduler.step()

            print_epoch(epoch, args.epochs, train_loss,
                        train_acc, test_acc, sparsity)

            epoch_log.append({
                "epoch": epoch, "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4),
                "sparsity": round(sparsity, 4),
                "elapsed": round(time.time() - t0, 2),
            })

            if args.patience > 0:
                stopper(test_acc, model)
                if stopper.early_stop:
                    print(f"  ↳ Early stopping triggered at epoch {epoch}")
                    break

        # ── Hard pruning (post-training) ──────────────────────────────────────
        # Load best checkpoint before pruning
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        pre_prune_acc      = evaluate(model, test_loader, device)
        pre_prune_sparsity = model.overall_sparsity(args.threshold)

        model.hard_prune(args.threshold)

        post_prune_acc      = evaluate(model, test_loader, device)
        post_prune_sparsity = model.overall_sparsity(args.threshold)

        print(f"\n  Hard Pruning (threshold={args.threshold}):")
        print(f"  {'':4} Pre-prune  → acc={pre_prune_acc*100:.2f}%  sparsity={pre_prune_sparsity*100:.2f}%")
        print(f"  {'':4} Post-prune → acc={post_prune_acc*100:.2f}%  sparsity={post_prune_sparsity*100:.2f}%")

        layer_report = model.layer_sparsity_report(args.threshold)
        print(f"  Layer-wise sparsity: {layer_report}")

        # ── Save gate histogram ───────────────────────────────────────────────
        plot_path = plot_gate_distribution(model, lam, prefix=str(outdir))
        print(f"  Gate histogram saved → {plot_path}")

        # ── Track best overall model ──────────────────────────────────────────
        if post_prune_acc > best_acc:
            best_acc = post_prune_acc
            torch.save(model.state_dict(), str(outdir / "best_model.pt"))
            print(f"  ★  New best model saved (acc={best_acc*100:.2f}%)")

        results.append({
            "lambda":           lam,
            "test_accuracy":    round(post_prune_acc, 4),
            "sparsity_pct":     round(post_prune_sparsity * 100, 2),
            "layer_sparsity":   layer_report,
            "epoch_log":        epoch_log,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print_results_table(results)

    summary_path = str(outdir / "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {summary_path}")

    summary_plot = plot_results_summary(results, prefix=str(outdir))
    print(f"  Summary chart  → {summary_plot}\n")


if __name__ == "__main__":
    main()
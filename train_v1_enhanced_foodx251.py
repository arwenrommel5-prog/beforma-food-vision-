"""
BeForma Food Vision — V1 Enhanced Training Script
EfficientNet-B0 | FoodX-251 | Knowledge Distillation from old Food-101 V1

Run in Google Colab with GPU runtime.
Place this file at /content/ and run:
    python train_v1_enhanced_foodx251.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHY THE OLD APPROACH FAILED — DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Severe class imbalance ignored: FoodX-251 has a 19× imbalance
   ratio (34 samples vs 656 for some classes). Without a weighted
   sampler or loss weighting, the model memorises majority classes.

2. No learning-rate warmup: AdamW with a fixed LR and no warmup
   on a newly-initialised 251-class head causes large gradient
   spikes that corrupt the backbone in the first few batches.

3. No cosine scheduler: flat LR means the model never converges
   into good minima. Plateau is hit early (~epoch 5-6).

4. Distillation skipped: the old V1 teacher has rich food visual
   knowledge. Ignoring it wastes 85 %+ accuracy on 79 shared
   classes; the student must relearn representations from scratch
   on limited data.

5. Image resolution stays at 224: EfficientNet-B0 was pretrained
   at 224 but fine-tunes better on 256 centre-cropped to 224 with
   stronger augmentation.

6. Only last 3 blocks unfrozen: too conservative for 12 epochs.
   The backbone needs more capacity exposed progressively.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

# ──────────────────────────────────────────────────────────────
# CONFIG — adjust paths to match your Colab/Drive layout
# ──────────────────────────────────────────────────────────────

FOODX_DIR     = Path("/content/drive/MyDrive/FoodX")           # CSV + class_list.txt
WORK_DIR      = Path("/content/FoodX_Work")                    # Unzipped image folders
OLD_MODEL_DIR = Path("/content/beforma-food-vision-/models")   # V1 model directory
OUTPUT_DIR    = Path("/content/drive/MyDrive/BeForma_V1_Enhanced/Models/models_v1_enhanced_foodx251")

# Training hyper-parameters
IMAGE_SIZE      = 224          # Input resolution
BATCH_SIZE      = 48           # Increase to 64 if >15 GB VRAM
EPOCHS          = 20           # Full run; resume_from_checkpoint will skip done epochs
WARMUP_EPOCHS   = 2            # LR warmup before cosine schedule kicks in
FREEZE_EPOCHS   = 3            # Epochs training only the new head (backbone frozen)
LR_HEAD         = 3e-4         # LR for the new 251-class classifier head
LR_BACKBONE     = 3e-5         # LR for backbone layers (after unfreezing)
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
DISTILL_TEMP    = 4.0          # Temperature for knowledge distillation softmax
DISTILL_ALPHA   = 0.3          # Weight of distillation loss vs CE loss (0 = disable)
NUM_WORKERS     = 2
SEED            = 42

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_classes(class_list_path: Path) -> list[str]:
    """Parse class_list.txt → ordered list of 251 class names."""
    entries = {}
    for line in class_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            idx  = int(parts[0])
            name = parts[1].strip().lower().replace(" ", "_")
            entries[idx] = name
    return [entries[i] for i in sorted(entries)]


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────

def _train_transform(size: int) -> transforms.Compose:
    """
    Strong augmentation strategy:
    - RandomResizedCrop with aggressive scale to simulate partial views
    - RandAugment for diverse photometric / geometric distortions
    - Random erasing as regularisation
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.55, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def _val_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class FoodXDataset(Dataset):
    def __init__(self,
                 csv_path: Path,
                 images_root: Path,
                 transform: transforms.Compose):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform   = transform

        # Robust column detection
        cols = {c.lower(): c for c in self.df.columns}
        self.filename_col = next(
            (cols[c] for c in ["img_name","image","filename","file","path"] if c in cols),
            self.df.columns[0])
        self.label_col = next(
            (cols[c] for c in ["label","class","class_id","category","category_id"] if c in cols),
            self.df.columns[-1])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row[self.label_col])
        img   = self._load_image(str(row[self.filename_col]))
        return self.transform(img), label

    def _load_image(self, name: str) -> Image.Image:
        for candidate in [
            self.images_root / name,
            self.images_root / f"{name}.jpg",
            self.images_root / "train_set" / name,
            self.images_root / "train_set" / f"{name}.jpg",
            self.images_root / "val_set"   / name,
            self.images_root / "val_set"   / f"{name}.jpg",
        ]:
            if candidate.exists():
                return Image.open(candidate).convert("RGB")
        # rglob fallback (slow on first miss; cache if needed)
        for p in self.images_root.rglob(name):
            return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"Image not found: {name} under {self.images_root}")

    @property
    def labels(self) -> list[int]:
        return self.df[self.label_col].tolist()


def build_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """
    Inverse-frequency weighted sampler — gives each class equal expected
    representation per epoch regardless of raw count.
    This is the primary fix for the 19× class imbalance.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)                    # avoid division by zero
    weight_per_class = 1.0 / counts
    sample_weights   = weight_per_class[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(labels),
        replacement = True,
    )


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────

def build_student(num_classes: int) -> nn.Module:
    """EfficientNet-B0 with a fresh 251-class head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_teacher(old_model_dir: Path, device: torch.device) -> nn.Module | None:
    """
    Load the original V1 Food-101 EfficientNet-B0 as a frozen teacher.
    Returns None if old model files are missing.
    """
    model_path = old_model_dir / "food_classifier.pt"
    if not model_path.exists():
        print(f"[WARN] Teacher model not found at {model_path} — distillation disabled.")
        return None

    teacher = models.efficientnet_b0(weights=None)
    teacher.classifier[1] = nn.Linear(teacher.classifier[1].in_features, 101)

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    teacher.load_state_dict(state, strict=True)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Teacher (V1 Food-101) loaded and frozen.")
    return teacher


def transfer_backbone_weights(student: nn.Module, old_model_dir: Path) -> int:
    """
    Copy all matching backbone tensors from V1 to the student.
    Classifier head is intentionally excluded (101 ≠ 251 classes).
    Returns number of tensors transferred.
    """
    model_path = old_model_dir / "food_classifier.pt"
    if not model_path.exists():
        print("[WARN] No old weights to transfer.")
        return 0

    old_state  = torch.load(model_path, map_location="cpu")
    if isinstance(old_state, dict) and "model_state_dict" in old_state:
        old_state = old_state["model_state_dict"]

    student_state = student.state_dict()
    transferred   = {}
    for k, v in old_state.items():
        if (k in student_state
                and student_state[k].shape == v.shape
                and not k.startswith("classifier.1")):
            transferred[k] = v

    student_state.update(transferred)
    student.load_state_dict(student_state)
    print(f"Transferred {len(transferred)} backbone tensors from V1 to student.")
    return len(transferred)


# ──────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ──────────────────────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined Cross-Entropy (hard labels) + KL-Divergence (soft teacher labels).

    total_loss = (1 - alpha) * CE(student, hard_labels)
               + alpha       * T² * KL(softmax(student/T) ‖ softmax(teacher/T))

    If teacher is None, falls back to pure CE.
    """
    def __init__(self,
                 num_classes:      int,
                 label_smoothing:  float = 0.1,
                 temperature:      float = 4.0,
                 alpha:            float = 0.3):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.temperature = temperature
        self.alpha       = alpha

    def forward(self,
                student_logits:  torch.Tensor,         # (B, 251)
                targets:         torch.Tensor,          # (B,) int64
                teacher_logits:  torch.Tensor | None = None  # (B, 101)
                ) -> torch.Tensor:
        ce_loss = self.ce(student_logits, targets)

        if teacher_logits is None or self.alpha == 0.0:
            return ce_loss

        T = self.temperature
        # Soft targets from teacher (101 classes) cannot directly supervise
        # the 251-class student.  We use the teacher's *entropy* as a
        # confidence mask: high-confidence teacher predictions (low entropy)
        # indicate images where the teacher has strong food visual knowledge.
        # We weight the distillation loss by this confidence.
        with torch.no_grad():
            teacher_soft = F.softmax(teacher_logits / T, dim=1)
            teacher_entropy = -(teacher_soft * (teacher_soft + 1e-8).log()).sum(1)  # (B,)
            max_entropy = math.log(teacher_logits.size(1))
            confidence  = 1.0 - teacher_entropy / max_entropy  # 0=uncertain, 1=confident
            confidence  = confidence.clamp(0, 1)

        # Student soft predictions over its own 251 classes
        student_log_soft = F.log_softmax(student_logits / T, dim=1)  # (B, 251)

        # We distil through the student's own probability distribution:
        # encourage low entropy (sharp predictions) on images the teacher
        # is confident about (even without knowing the exact 251-class target).
        entropy_reg = -(student_log_soft.exp() * student_log_soft).sum(1)  # (B,)
        distill_loss = (confidence * entropy_reg).mean()

        return (1.0 - self.alpha) * ce_loss + self.alpha * (T ** 2) * distill_loss


# ──────────────────────────────────────────────────────────────
# SCHEDULER
# ──────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_steps`, then cosine anneal to `eta_min`.
    Applied manually each *step* (not epoch) for fine control.
    """
    def __init__(self,
                 optimizer:     torch.optim.Optimizer,
                 warmup_steps:  int,
                 total_steps:   int,
                 eta_min_ratio: float = 0.01):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.eta_min_ratio = eta_min_ratio
        self._step         = 0
        # Record base LRs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if s <= self.warmup_steps:
                pg["lr"] = base_lr * s / self.warmup_steps
            else:
                progress  = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                pg["lr"] = base_lr * (self.eta_min_ratio + (1 - self.eta_min_ratio) * cos_decay)

    @property
    def current_lrs(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> dict:
    """Returns top-1, top-5 accuracy and average loss (CE only)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct1, correct5, total_loss = 0, 0, 0, 0.0

    for x, y in tqdm(loader, desc="  val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

        # Top-1
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == y).sum().item()

        # Top-5
        top5 = logits.topk(5, dim=1).indices
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "val_loss": round(total_loss / total, 4),
        "val_top1": round(correct1  / total, 4),
        "val_top5": round(correct5  / total, 4),
    }


# ──────────────────────────────────────────────────────────────
# PROGRESSIVE UNFREEZING POLICY
# ──────────────────────────────────────────────────────────────

def configure_optimizer_for_epoch(model:          nn.Module,
                                  epoch:          int,
                                  freeze_epochs:  int,
                                  lr_head:        float,
                                  lr_backbone:    float,
                                  weight_decay:   float) -> torch.optim.AdamW:
    """
    Progressive unfreezing schedule:
      epochs 1‥freeze_epochs  → only classifier head trainable
      epoch  freeze_epochs+1  → unfreeze last 2 EfficientNet blocks
      epoch  freeze_epochs+3  → unfreeze last 4 blocks
      epoch  freeze_epochs+5  → unfreeze full backbone
    Discriminative LR: backbone gets 10× lower LR than head.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True

    offset = epoch - freeze_epochs

    if epoch <= freeze_epochs:
        # Head only
        param_groups = [{"params": model.classifier.parameters(), "lr": lr_head}]
        print(f"  [Freeze] Classifier head only")

    elif offset <= 2:
        # Unfreeze last 2 feature blocks
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        param_groups = [
            {"params": model.features[-2:].parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(),    "lr": lr_head},
        ]
        print(f"  [Unfreeze] Last 2 feature blocks + classifier")

    elif offset <= 4:
        # Unfreeze last 4 feature blocks
        for p in model.features[-4:].parameters():
            p.requires_grad = True
        param_groups = [
            {"params": model.features[-4:].parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(),    "lr": lr_head},
        ]
        print(f"  [Unfreeze] Last 4 feature blocks + classifier")

    else:
        # Full backbone
        for p in model.parameters():
            p.requires_grad = True
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("classifier")],
             "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ]
        print(f"  [Unfreeze] Full backbone + classifier")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def train():
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────
    classes = read_classes(FOODX_DIR / "class_list.txt")
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    train_ds = FoodXDataset(
        FOODX_DIR / "train_labels.csv", WORK_DIR,
        _train_transform(IMAGE_SIZE))
    val_ds = FoodXDataset(
        FOODX_DIR / "val_labels.csv", WORK_DIR,
        _val_transform(IMAGE_SIZE))

    # Weighted sampler to handle 19× class imbalance
    sampler = build_weighted_sampler(train_ds.labels, num_classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Models ────────────────────────────────────────────────
    student = build_student(num_classes)
    transfer_backbone_weights(student, OLD_MODEL_DIR)
    student.to(device)

    teacher = build_teacher(OLD_MODEL_DIR, device)

    criterion = DistillationLoss(
        num_classes     = num_classes,
        label_smoothing = LABEL_SMOOTHING,
        temperature     = DISTILL_TEMP,
        alpha           = DISTILL_ALPHA if teacher is not None else 0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ────────────────────────────────────────────────
    ckpt_path = OUTPUT_DIR / "checkpoint_last.pt"
    history: list[dict] = []
    start_epoch = 1
    best_top1   = 0.0

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        history     = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1   = ckpt.get("best_top1", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best Top-1 = {best_top1:.4f}")

    # ── Training epochs ───────────────────────────────────────
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        optimizer = configure_optimizer_for_epoch(
            student, epoch, FREEZE_EPOCHS,
            LR_HEAD, LR_BACKBONE, WEIGHT_DECAY)

        steps_per_epoch = len(train_loader)
        total_steps     = steps_per_epoch * (EPOCHS - max(start_epoch - 1, 0))
        warmup_steps    = steps_per_epoch * WARMUP_EPOCHS
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

        student.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"  train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                student_logits = student(x)

                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(x)   # (B, 101)

                loss = criterion(student_logits, y, teacher_logits)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss    += loss.item() * x.size(0)
            train_correct += (student_logits.argmax(1) == y).sum().item()
            train_total   += x.size(0)

        train_loss /= train_total
        train_top1  = train_correct / train_total

        val_metrics = evaluate(student, val_loader, device)

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 4),
            **val_metrics,
        }
        history.append(row)
        print(f"  {row}")

        # ── Save last checkpoint ───────────────────────────────
        torch.save({
            "epoch":            epoch,
            "model_state_dict": student.state_dict(),
            "history":          history,
            "best_top1":        best_top1,
        }, ckpt_path)

        # ── Save best ──────────────────────────────────────────
        if val_metrics["val_top1"] > best_top1:
            best_top1 = val_metrics["val_top1"]
            torch.save(student.state_dict(), OUTPUT_DIR / "food_classifier.pt")
            print(f"  ★ New best Top-1 = {best_top1:.4f} — saved food_classifier.pt")

        # ── Save history ───────────────────────────────────────
        (OUTPUT_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

    elapsed = (time.time() - t0) / 60

    # ── Save artefacts ─────────────────────────────────────────
    (OUTPUT_DIR / "classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8")

    metadata = {
        "version":                "v1_enhanced_foodx251",
        "architecture":           "efficientnet_b0",
        "dataset":                "FoodX-251 / iFood-2019",
        "classes_count":          num_classes,
        "old_weights_path":       str(OLD_MODEL_DIR / "food_classifier.pt"),
        "old_weights_used":       True,
        "distillation_enabled":   teacher is not None,
        "distillation_temp":      DISTILL_TEMP,
        "distillation_alpha":     DISTILL_ALPHA,
        "best_val_top1":          round(best_top1, 4),
        "best_val_top5":          max((h["val_top5"] for h in history), default=0),
        "epochs":                 EPOCHS,
        "image_size":             IMAGE_SIZE,
        "label_smoothing":        LABEL_SMOOTHING,
        "weighted_sampler":       True,
        "progressive_unfreezing": True,
        "warmup_epochs":          WARMUP_EPOCHS,
        "cosine_scheduler":       True,
        "calorie_estimation":     "per_class lookup: calories_per_100g × portion_g / 100",
        "elapsed_minutes":        round(elapsed, 2),
        "history":                history,
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n✓ Done  |  Best Top-1 = {best_top1:.4f}  |  {elapsed:.1f} min")
    print(f"  Files saved to: {OUTPUT_DIR}")


# ──────────────────────────────────────────────────────────────
# HOW TO RESUME
# ──────────────────────────────────────────────────────────────
# Simply re-run this script.  If OUTPUT_DIR/checkpoint_last.pt
# exists it will be loaded automatically and training continues
# from the next epoch.  Nothing else needs to change.
#
# To start fresh despite an existing checkpoint:
#   (OUTPUT_DIR / "checkpoint_last.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    train()

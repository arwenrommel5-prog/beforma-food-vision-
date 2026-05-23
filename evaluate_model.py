"""
BeForma V1 Enhanced — Evaluation & Deployment Decision Script

Computes Top-1, Top-5, per-class accuracy, confusion stats, and
prints a clear GO / NO-GO deployment recommendation.

Usage:
    python evaluate_model.py \
        --model_dir  /content/drive/MyDrive/BeForma_V1_Enhanced/Models/models_v1_enhanced_foodx251 \
        --val_csv    /content/drive/MyDrive/FoodX/val_labels.csv \
        --images_dir /content/FoodX_Work \
        [--device    cuda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ── Reuse val transform from training ─────────────────────────

def _val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_root, transform):
        self.df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in self.df.columns}
        self.fn_col  = next(cols[c] for c in ["img_name","image","filename"] if c in cols)
        self.lbl_col = next(cols[c] for c in ["label","class","class_id"] if c in cols)
        self.root    = Path(images_root)
        self.transform = transform

    def __len__(self):  return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row[self.lbl_col])
        name  = str(row[self.fn_col])
        for p in [self.root / name,
                  self.root / f"{name}.jpg",
                  self.root / "val_set" / name,
                  self.root / "val_set" / f"{name}.jpg"]:
            if p.exists():
                return self.transform(Image.open(p).convert("RGB")), label
        raise FileNotFoundError(name)


def load_model(model_dir: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(model_dir / "food_classifier.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_evaluation(model, loader, num_classes, device):
    total     = 0
    correct1  = 0
    correct5  = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # Top-1
        pred1 = logits.argmax(1)
        correct1 += (pred1 == y).sum().item()

        # Top-5
        top5 = logits.topk(5, dim=1).indices
        correct5 += (top5 == y.unsqueeze(1)).any(1).sum().item()

        total += x.size(0)

        # Per-class
        for gt, pr in zip(y.cpu().tolist(), pred1.cpu().tolist()):
            per_class_total[gt]   += 1
            per_class_correct[gt] += int(gt == pr)

    top1 = correct1 / total
    top5 = correct5 / total

    # Per-class accuracy
    per_class_acc = {
        cls: per_class_correct[cls] / max(per_class_total[cls], 1)
        for cls in range(num_classes)
    }
    worst5  = sorted(per_class_acc, key=per_class_acc.get)[:5]
    best5   = sorted(per_class_acc, key=per_class_acc.get, reverse=True)[:5]

    return {
        "total_samples": total,
        "top1":          round(top1, 4),
        "top5":          round(top5, 4),
        "per_class_acc": per_class_acc,
        "worst5_classes": worst5,
        "best5_classes":  best5,
    }


def deployment_decision(results: dict,
                        old_v1_top1: float,
                        classes: list[str]) -> str:
    """
    Decide whether V1 Enhanced is good enough to replace V1.

    Rules:
      GO if  val Top-1 ≥ 0.78 AND Top-5 ≥ 0.93
      PARTIAL if Top-1 ≥ 0.72 AND the team accepts reduced accuracy
              in exchange for 251-class coverage
      NO-GO otherwise — continue training or investigate data quality
    """
    t1 = results["top1"]
    t5 = results["top5"]
    worst = [classes[i] for i in results["worst5_classes"]]

    lines = [
        "━" * 60,
        "DEPLOYMENT DECISION",
        "━" * 60,
        f"  Top-1 accuracy : {t1 * 100:.2f}%",
        f"  Top-5 accuracy : {t5 * 100:.2f}%",
        f"  Old V1 Top-1   : {old_v1_top1 * 100:.2f}%  (Food-101, 101 classes)",
        "",
        f"  5 hardest classes : {worst}",
        "",
    ]

    if t1 >= 0.80:
        lines += [
            "  ✅  GO — exceeds 80% Top-1 target.",
            "  Replace current V1 with V1 Enhanced in production.",
        ]
    elif t1 >= 0.78:
        lines += [
            "  ✅  GO — meets target (≥78% Top-1, 251 classes).",
            "  Replace current V1 with V1 Enhanced.",
        ]
    elif t1 >= 0.72:
        lines += [
            "  ⚠️  PARTIAL — Top-1 is acceptable but below the 78% target.",
            "  Options:",
            "    • Continue training for more epochs (resume from checkpoint_last.pt).",
            "    • Increase EPOCHS to 25 and re-run.",
            "    • Deploy as beta alongside V1 for shadow-mode comparison.",
        ]
    else:
        lines += [
            "  ❌  NO-GO — Top-1 below 72%. Do NOT replace V1.",
            "  Likely causes:",
            "    • Data preprocessing mismatch — verify WORK_DIR image paths.",
            "    • Class imbalance not handled (verify WeightedRandomSampler ran).",
            "    • GPU memory too small — reduce BATCH_SIZE to 32.",
            "    • Not enough epochs — increase EPOCHS to 25.",
        ]

    lines.append("━" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--val_csv",    required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device    = torch.device(args.device)

    # Load metadata
    meta = json.loads((model_dir / "model_metadata.json").read_text())
    classes = json.loads((model_dir / "classes.json").read_text())
    num_classes = len(classes)
    old_top1    = 0.8552   # V1 Food-101 known accuracy

    print(f"Model : {meta.get('version', 'unknown')}")
    print(f"Classes: {num_classes}")

    ds     = _Dataset(args.val_csv, args.images_dir, _val_transform())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model   = load_model(model_dir, num_classes, device)
    results = run_evaluation(model, loader, num_classes, device)

    print(f"\nTop-1 : {results['top1'] * 100:.2f}%")
    print(f"Top-5 : {results['top5'] * 100:.2f}%")
    print(f"Samples: {results['total_samples']:,}")

    worst_names = [classes[i] for i in results["worst5_classes"]]
    best_names  = [classes[i] for i in results["best5_classes"]]
    print(f"\n5 worst classes : {worst_names}")
    print(f"5 best  classes : {best_names}")

    print()
    print(deployment_decision(results, old_top1, classes))

    # Save full per-class report
    report = {
        "top1": results["top1"],
        "top5": results["top5"],
        "total_samples": results["total_samples"],
        "per_class": {
            classes[i]: round(acc, 4)
            for i, acc in results["per_class_acc"].items()
        },
        "worst5": worst_names,
        "best5":  best_names,
    }
    out_path = model_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nPer-class report saved → {out_path}")


if __name__ == "__main__":
    main()

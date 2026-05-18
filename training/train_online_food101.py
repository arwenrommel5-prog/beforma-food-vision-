"""Download Food-101 from the internet and train BeForma food vision model.

This is the recommended MVP training pipeline because it downloads public online
Food-101 data automatically through torchvision.

Run locally or on Google Colab:
    python training/train_online_food101.py --data_root data --epochs 5 --output_dir models

Fast smoke test:
    python training/train_online_food101.py --data_root data --quick --max_images_per_class 40 --epochs 1

Outputs:
    models/food_classifier.pt
    models/classes.json
    models/model_metadata.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path


def _food101_label(dataset, raw_target) -> str:
    """Return a normalized Food-101 class label from torchvision target formats."""
    if isinstance(raw_target, str):
        return raw_target
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return str(classes[int(raw_target)])
    return str(raw_target)


def _limited_indices(dataset, max_images_per_class: int | None, max_classes: int | None, seed: int = 42) -> tuple[list[int], list[str]]:
    rng = random.Random(seed)
    labels = []
    for i in range(len(dataset)):
        # torchvision Food101 keeps labels in memory; this fallback avoids opening all images if possible.
        if hasattr(dataset, "_labels"):
            raw = dataset._labels[i]  # type: ignore[attr-defined]
        elif hasattr(dataset, "_samples"):
            raw = dataset._samples[i][1]  # type: ignore[attr-defined]
        else:
            _, raw = dataset[i]
        labels.append(_food101_label(dataset, raw))

    all_classes = sorted(set(labels))
    if max_classes:
        all_classes = all_classes[:max_classes]
    allowed = set(all_classes)

    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label in allowed:
            by_class[label].append(i)

    selected: list[int] = []
    for label in all_classes:
        idxs = by_class[label]
        rng.shuffle(idxs)
        selected.extend(idxs[:max_images_per_class] if max_images_per_class else idxs)
    rng.shuffle(selected)
    return selected, all_classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="Where Food-101 will be downloaded/cached")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--architecture", choices=["efficientnet_b0", "mobilenet_v3_small"], default="efficientnet_b0")
    parser.add_argument("--max_images_per_class", type=int, default=None, help="Useful for quick experiments")
    parser.add_argument("--max_classes", type=int, default=None, help="Train first N Food-101 classes for a fast test")
    parser.add_argument("--quick", action="store_true", help="Shortcut: 20 classes, 40 images/class, 1 epoch")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.max_classes = args.max_classes or 20
        args.max_images_per_class = args.max_images_per_class or 40
        args.epochs = min(args.epochs, 1)
        args.architecture = "mobilenet_v3_small"

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms
    from tqdm import tqdm

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Downloading/loading Food-101. First run may download ~5GB.")
    train_ds_full = datasets.Food101(root=args.data_root, split="train", download=True, transform=train_tfm)
    val_ds_full = datasets.Food101(root=args.data_root, split="test", download=True, transform=val_tfm)

    train_indices, classes = _limited_indices(train_ds_full, args.max_images_per_class, args.max_classes, args.seed)
    allowed = set(classes)
    val_indices, _ = _limited_indices(val_ds_full, max(1, (args.max_images_per_class or 250) // 4) if args.max_images_per_class else None, args.max_classes, args.seed)

    # Make a contiguous class index mapping after possible max_classes filtering.
    class_to_new_idx = {c: i for i, c in enumerate(classes)}

    class RelabelDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, j):
            img, raw_target = self.base[self.indices[j]]
            label = _food101_label(self.base, raw_target)
            return img, class_to_new_idx[label]

    train_ds = RelabelDataset(train_ds_full, train_indices)
    val_ds = RelabelDataset(val_ds_full, val_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | classes={len(classes)} | train={len(train_ds)} | val={len(val_ds)}")

    if args.architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()
        train_loss = total_loss / max(len(train_ds), 1)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_acc": round(val_acc, 4)})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "food_classifier.pt")

    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    metadata = {
        "dataset": "food101",
        "dataset_source": "torchvision.datasets.Food101(download=True)",
        "architecture": args.architecture,
        "classes_count": len(classes),
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "best_val_acc": round(best_acc, 4),
        "epochs": args.epochs,
        "quick_mode": bool(args.quick),
        "history": history,
        "elapsed_minutes": round((time.perf_counter() - started) / 60, 2),
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(f"- {output_dir / 'food_classifier.pt'}")
    print(f"- {output_dir / 'classes.json'}")
    print(f"- {output_dir / 'model_metadata.json'}")
    print(f"Best validation accuracy: {best_acc:.4f}")


def evaluate(model, loader, device: str) -> float:
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


if __name__ == "__main__":
    main()

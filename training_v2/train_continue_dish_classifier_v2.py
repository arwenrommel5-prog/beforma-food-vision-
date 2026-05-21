from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def load_classes(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_old_model(model: nn.Module, model_path: Path) -> nn.Module:
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old_model_dir = Path(args.old_model_dir)
    classes = load_classes(old_model_dir / "classes.json")
    assert len(classes) == 101, f"Expected 101 old classes, got {len(classes)}"

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.Food101(root=args.food101_root, split="train", download=True, transform=train_tf)
    val_ds = datasets.Food101(root=args.food101_root, split="test", download=True, transform=val_tf)
    assert list(train_ds.classes) == classes, "Food-101 class order differs from old classes.json"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_efficientnet_b0(len(classes))
    model = load_old_model(model, old_model_dir / "food_classifier.pt")
    model.to(device)

    if args.freeze_epochs > 0:
        for p in model.features.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler(enabled=(device == "cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - epoch + 1, 1))

        model.train()
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"dish epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        train_loss = loss_sum / max(len(train_loader), 1)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_acc": round(val_acc, 4)})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "food_classifier.pt")
            (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
            metadata = {
                "version": "dish_classifier_v2_continue_b0",
                "created_at": datetime.utcnow().isoformat(),
                "architecture": "efficientnet_b0",
                "classes_count": len(classes),
                "source_old_model_dir": str(old_model_dir),
                "best_val_acc": best_acc,
                "epochs": args.epochs,
                "image_size": args.image_size,
                "history": history,
                "note": "Continued fine-tuning from safe-v1 old Food-101 model.",
            }
            (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print("Saved best dish classifier:", out_dir)

    print("Best dish val_acc:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food101_root", default="data_v2/food101")
    parser.add_argument("--old_model_dir", default="models")
    parser.add_argument("--output_dir", default="models_v2/dish_classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze_epochs", type=int, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    main(parser.parse_args())

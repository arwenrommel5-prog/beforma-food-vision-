"""Train a simple food classifier from an ImageFolder dataset.

Expected data layout:
    data/train/chicken_breast_cooked/*.jpg
    data/train/rice_white_cooked/*.jpg
    data/val/chicken_breast_cooked/*.jpg
    data/val/rice_white_cooked/*.jpg

Class folder names should match food IDs or aliases in app/food_database.py.

Run:
    pip install -r requirements-ml.txt
    python training/train_food_classifier.py --data_dir data --epochs 8 --output_dir models
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tfm_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tfm_train)
    val_ds = datasets.ImageFolder(str(val_dir), transform=tfm_val) if val_dir.exists() else None
    classes = train_ds.classes
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if val_ds else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)
        acc = evaluate(model, val_loader, device) if val_loader else 0.0
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_dir / "food_classifier.pt")
    print(f"Saved model to {output_dir / 'food_classifier.pt'} best_acc={best_acc:.4f}")


def evaluate(model, loader, device: str) -> float:
    import torch
    if loader is None:
        return 0.0
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)

if __name__ == "__main__":
    main()

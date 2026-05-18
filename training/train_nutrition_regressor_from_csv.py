"""Optional advanced training script for direct calorie/macro regression.

Use this after preparing a CSV from Nutrition5k or your own labeled meals.
Expected CSV columns:
    image_path, calories, protein, carbs, fat

Run:
    python training/train_nutrition_regressor_from_csv.py --csv data/nutrition_train.csv --epochs 10 --output_dir models_regression

This is separate from the Food-101 classifier.  The classifier recognizes the dish;
this regressor predicts nutrition numbers directly when you have labels.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", default="models_regression")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models, transforms
    from PIL import Image
    from tqdm import tqdm

    class NutritionDataset(Dataset):
        def __init__(self, csv_path: str, transform):
            self.rows = []
            self.transform = transform
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.rows.append(row)
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            r = self.rows[idx]
            img = Image.open(r["image_path"]).convert("RGB")
            y = torch.tensor([float(r["calories"]), float(r["protein"]), float(r["carbs"]), float(r["fat"])], dtype=torch.float32)
            return self.transform(img), y

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = NutritionDataset(args.csv, tfm)
    val_n = max(1, int(len(ds) * 0.15))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_mae = 10**9
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0)
        mae = eval_mae(model, val_loader, device)
        print(f"epoch={epoch} train_loss={total_loss/max(train_n,1):.4f} val_mae_avg={mae:.2f}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out / "nutrition_regressor.pt")
    (out / "nutrition_regressor_metadata.json").write_text(json.dumps({"best_val_avg_mae": best_mae}, indent=2), encoding="utf-8")


def eval_mae(model, loader, device):
    import torch
    model.eval(); abs_sum = 0.0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            abs_sum += torch.abs(pred - y).mean().item() * x.size(0)
            n += x.size(0)
    return abs_sum / max(n, 1)


if __name__ == "__main__":
    main()

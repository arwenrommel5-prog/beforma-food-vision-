from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def find_image_mask_keys(example: dict[str, Any]) -> tuple[str, str]:
    image_candidates = ["image", "img", "rgb", "food_image"]
    mask_candidates = ["label", "mask", "annotation", "segmentation_mask", "semantic_mask"]
    image_key = next((k for k in image_candidates if k in example), None)
    mask_key = next((k for k in mask_candidates if k in example), None)
    if image_key is None:
        for k, v in example.items():
            if isinstance(v, Image.Image):
                image_key = k; break
    if mask_key is None:
        for k, v in example.items():
            if k != image_key and isinstance(v, Image.Image):
                mask_key = k; break
    if not image_key or not mask_key:
        raise RuntimeError(f"Could not detect image/mask keys. Keys: {list(example.keys())}")
    return image_key, mask_key


def build_classes(ds) -> list[str]:
    # FoodSeg103 masks commonly use pixel ids. We create generic names if label names are unavailable.
    # 0 is background.
    max_id = 0
    for split in ds.keys():
        for i in range(min(len(ds[split]), 50)):
            ex = ds[split][i]
            _, mask_key = find_image_mask_keys(ex)
            arr = np.array(ex[mask_key])
            max_id = max(max_id, int(arr.max()))
    return ["background"] + [f"foodseg103_class_{i}" for i in range(1, max_id + 1)]


class HFFoodSegDataset(Dataset):
    def __init__(self, hf_split, classes_count: int, image_size: int):
        self.ds = hf_split
        self.classes_count = classes_count
        self.image_size = image_size
        self.image_key, self.mask_key = find_image_mask_keys(self.ds[0])
        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex[self.image_key].convert("RGB")
        mask = ex[self.mask_key]
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask).astype("uint8"))
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_arr = np.array(mask).astype("int64")
        mask_arr = np.clip(mask_arr, 0, self.classes_count - 1)
        return self.img_tf(img), torch.from_numpy(mask_arr).long()


def build_model(num_classes: int):
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def mean_iou(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        union = (p | t).sum().item()
        if union == 0:
            continue
        inter = (p & t).sum().item()
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def evaluate(model, loader, device, num_classes):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)["out"]
            pred = out.argmax(1)
            scores.append(mean_iou(pred.cpu(), y.cpu(), num_classes))
    return float(np.mean(scores)) if scores else 0.0


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("EduardoPacheco/FoodSeg103")
    train_key = "train" if "train" in ds else list(ds.keys())[0]
    val_key = "validation" if "validation" in ds else "val" if "val" in ds else list(ds.keys())[-1]

    classes = build_classes(ds)
    num_classes = len(classes)
    print("FoodSeg103 splits:", {k: len(v) for k, v in ds.items()})
    print("Detected segmentation classes:", num_classes)

    train_ds = HFFoodSegDataset(ds[train_key], num_classes, args.image_size)
    val_ds = HFFoodSegDataset(ds[val_key], num_classes, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = GradScaler(enabled=(device == "cuda"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"seg epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                out = model(x)["out"]
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()
        val_miou = evaluate(model, val_loader, device, num_classes)
        train_loss = loss_sum / max(len(train_loader), 1)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_miou": round(val_miou, 4)})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_miou={val_miou:.4f}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), out_dir / "segmenter.pt")
            (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
            metadata = {
                "version": "ingredient_segmenter_v2_foodseg103",
                "created_at": datetime.utcnow().isoformat(),
                "architecture": "deeplabv3_resnet50",
                "dataset": "EduardoPacheco/FoodSeg103",
                "classes_count": num_classes,
                "best_val_miou": best_miou,
                "epochs": args.epochs,
                "image_size": args.image_size,
                "history": history,
            }
            (out_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print("Saved best segmenter:", out_dir)
    print("Best val_miou:", best_miou)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models_v2/ingredient_segmenter")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    main(parser.parse_args())

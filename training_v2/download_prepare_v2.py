from __future__ import annotations

import json
from pathlib import Path

from torchvision import datasets
from datasets import load_dataset

DATA_ROOT = Path("data_v2")
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_food101() -> None:
    print("Preparing Food-101 for dish classifier...")
    root = DATA_ROOT / "food101"
    train = datasets.Food101(root=str(root), split="train", download=True)
    val = datasets.Food101(root=str(root), split="test", download=True)
    write_json(DATA_ROOT / "food101_info.json", {
        "dataset": "Food-101",
        "task": "dish_classification",
        "train_images": len(train),
        "val_images": len(val),
        "classes_count": len(train.classes),
        "classes": list(train.classes),
    })
    print(f"Food-101 ready: train={len(train)} val={len(val)} classes={len(train.classes)}")


def prepare_foodseg103() -> None:
    print("Preparing FoodSeg103 from Hugging Face for ingredient segmentation...")
    ds = load_dataset("EduardoPacheco/FoodSeg103")
    out = DATA_ROOT / "foodseg103_hf"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset_info.json", {
        "dataset": "FoodSeg103",
        "task": "ingredient_semantic_segmentation",
        "source": "EduardoPacheco/FoodSeg103",
        "splits": list(ds.keys()),
        "sizes": {k: len(v) for k, v in ds.items()},
        "features": {k: str(v.features) for k, v in ds.items()},
        "note": "Loaded online through Hugging Face datasets. Training script detects image/mask columns automatically.",
    })
    print("FoodSeg103 ready:", {k: len(v) for k, v in ds.items()})


def try_prepare_uecfoodpixcomplete() -> None:
    """Optional: try an online HF mirror of UECFoodPixComplete.

    If unavailable or slow, training still works with FoodSeg103.
    """
    try:
        print("Trying optional UECFOODPIXCOMPLETE from Hugging Face...")
        ds = load_dataset("justinsiow/UECFOODPIXCOMPLETE")
        out = DATA_ROOT / "uecfoodpixcomplete_hf"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "dataset_info.json", {
            "dataset": "UECFOODPIXCOMPLETE",
            "task": "food_semantic_segmentation",
            "source": "justinsiow/UECFOODPIXCOMPLETE",
            "splits": list(ds.keys()),
            "sizes": {k: len(v) for k, v in ds.items()},
            "features": {k: str(v.features) for k, v in ds.items()},
            "note": "Optional dataset. Use only after FoodSeg103 training is working.",
        })
        print("UECFOODPIXCOMPLETE ready:", {k: len(v) for k, v in ds.items()})
    except Exception as exc:
        print("Optional UECFOODPIXCOMPLETE download skipped/failed:", repr(exc))


if __name__ == "__main__":
    prepare_food101()
    prepare_foodseg103()
    try_prepare_uecfoodpixcomplete()

"""
BeForma Food Vision V1 Enhanced — Inference Module
Plug this directly into the FastAPI model loader.

Usage:
    predictor = FoodPredictor(
        model_dir  = "path/to/models_v1_enhanced_foodx251",
        device     = "cpu",   # or "cuda"
    )
    result = predictor.predict(pil_image, portion_scale="medium")
    result = predictor.predict(pil_image, quantity_g=120)

Returned dict matches the documented API contract exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Transforms (must match val transform used during training)
# ──────────────────────────────────────────────────────────────

def _infer_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────────

PortionScale = Literal["small", "medium", "large"]

class FoodPredictor:
    """
    Loads V1-Enhanced model files from a directory and exposes a
    `predict` method returning the full documented API response.
    """

    def __init__(self,
                 model_dir: str | Path,
                 device:    str = "cpu",
                 image_size: int = 224):

        model_dir   = Path(model_dir)
        self.device = torch.device(device)

        # ── Classes ──────────────────────────────────────────
        classes_path = model_dir / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"classes.json not found in {model_dir}")
        self.classes: list[str] = json.loads(classes_path.read_text())
        self.num_classes        = len(self.classes)

        # ── Calorie lookup ───────────────────────────────────
        calorie_path = model_dir / "calorie_lookup.json"
        if not calorie_path.exists():
            raise FileNotFoundError(f"calorie_lookup.json not found in {model_dir}")
        # Keys are string indices "0" … "250"
        self._calorie_db: dict[str, dict] = json.loads(calorie_path.read_text())

        # ── Model ─────────────────────────────────────────────
        model_path = model_dir / "food_classifier.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"food_classifier.pt not found in {model_dir}")

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)

        state = torch.load(model_path, map_location="cpu")
        # Accept bare state_dict or wrapped checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = _infer_transform(image_size)
        print(f"FoodPredictor ready | {self.num_classes} classes | device={device}")

    # ── Public API ───────────────────────────────────────────

    def predict(self,
                image:         Image.Image | str | Path,
                portion_scale: PortionScale | None = "medium",
                quantity_g:    float | None = None,
                top_k:         int = 5) -> dict:
        """
        Classify a food image and estimate calories.

        Parameters
        ----------
        image         : PIL Image, or path to an image file.
        portion_scale : "small" | "medium" | "large"  (ignored if quantity_g given)
        quantity_g    : Override gram weight; takes priority over portion_scale.
        top_k         : Number of top predictions to return (default 5).

        Returns
        -------
        dict matching the documented API response contract.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"image must be PIL.Image, str, or Path — got {type(image)}")

        # ── Inference ─────────────────────────────────────────
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits      = self.model(tensor)          # (1, 251)
            probs       = F.softmax(logits, dim=1)[0] # (251,)

        # Top-K
        topk_probs, topk_idxs = probs.topk(top_k)
        topk_probs = topk_probs.cpu().tolist()
        topk_idxs  = topk_idxs.cpu().tolist()

        pred_idx    = topk_idxs[0]
        pred_class  = self.classes[pred_idx]
        confidence  = round(topk_probs[0], 4)

        top5 = [
            {"rank": i + 1,
             "class": self.classes[idx],
             "confidence": round(p, 4)}
            for i, (idx, p) in enumerate(zip(topk_idxs, topk_probs))
        ]

        # ── Calorie estimation ────────────────────────────────
        nutrition = self._estimate_nutrition(pred_idx, portion_scale, quantity_g)

        return {
            "predicted_food": pred_class,
            "confidence":     confidence,
            "top5":           top5,
            "portion": {
                "scale":           nutrition["portion_scale"],
                "estimated_grams": nutrition["estimated_grams"],
            },
            "nutrition_estimate": {
                "calories":         nutrition["calories"],
                "calories_per_100g": nutrition["calories_per_100g"],
                "protein_g":        nutrition["protein_g"],
                "carbs_g":          nutrition["carbs_g"],
                "fat_g":            nutrition["fat_g"],
                "calorie_source":   nutrition["calorie_source"],
            },
        }

    # ── Internal helpers ─────────────────────────────────────

    def _estimate_nutrition(self,
                            class_idx:     int,
                            portion_scale: PortionScale | None,
                            quantity_g:    float | None) -> dict:
        """
        Resolve gram weight → calculate calories + macros.

        Priority:
          1. quantity_g  (explicit override from caller)
          2. portion_scale (small / medium / large)
          3. default_serving_g from DB
        """
        entry = self._calorie_db.get(str(class_idx))

        if entry is None:
            # Should never happen if calorie_lookup.json is complete,
            # but handle gracefully.
            return {
                "portion_scale":    portion_scale or "medium",
                "estimated_grams":  200,
                "calories":         400,
                "calories_per_100g": 200,
                "protein_g":        None,
                "carbs_g":          None,
                "fat_g":            None,
                "calorie_source":   "fallback_generic",
            }

        kcal_per_100 = entry["calories_per_100g"]

        # Resolve grams
        if quantity_g is not None:
            grams        = float(quantity_g)
            scale_label  = "custom"
        elif portion_scale == "small":
            grams        = entry["small_serving_g"]
            scale_label  = "small"
        elif portion_scale == "large":
            grams        = entry["large_serving_g"]
            scale_label  = "large"
        else:
            # Default to medium (covers None and "medium")
            grams        = entry["medium_serving_g"]
            scale_label  = "medium"

        calories = round(kcal_per_100 * grams / 100)

        return {
            "portion_scale":    scale_label,
            "estimated_grams":  round(grams),
            "calories":         calories,
            "calories_per_100g": kcal_per_100,
            "protein_g":        round(entry["protein_per_100g"] * grams / 100, 1),
            "carbs_g":          round(entry["carbs_per_100g"]   * grams / 100, 1),
            "fat_g":            round(entry["fat_per_100g"]     * grams / 100, 1),
            "calorie_source":   entry["calorie_source"],
        }


# ──────────────────────────────────────────────────────────────
# FastAPI integration example
# ──────────────────────────────────────────────────────────────
#
#   from fastapi import FastAPI, UploadFile, File, Form
#   from PIL import Image
#   import io
#
#   app       = FastAPI()
#   predictor = FoodPredictor("models_v1_enhanced_foodx251", device="cpu")
#
#   @app.post("/predict")
#   async def predict_food(
#       file:          UploadFile = File(...),
#       portion_scale: str        = Form("medium"),
#       quantity_g:    float | None = Form(None),
#   ):
#       img_bytes = await file.read()
#       img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#       result    = predictor.predict(img, portion_scale=portion_scale,
#                                     quantity_g=quantity_g)
#       return result


# ──────────────────────────────────────────────────────────────
# Quick CLI test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python inference.py  <model_dir>  <image_path>  [small|medium|large]  [grams]")
        sys.exit(1)

    model_dir    = sys.argv[1]
    image_path   = sys.argv[2]
    portion      = sys.argv[3] if len(sys.argv) > 3 else "medium"
    grams        = float(sys.argv[4]) if len(sys.argv) > 4 else None

    predictor = FoodPredictor(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")
    result    = predictor.predict(image_path, portion_scale=portion, quantity_g=grams)

    print(json.dumps(result, indent=2))

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class IngredientSegmenter:
    """Optional V2 ingredient segmentation loader.

    Safe for Railway: if models_v2/ingredient_segmenter is missing, it returns no ingredients
    instead of breaking the API. Enable it later after training by setting ENABLE_V2_SEGMENTER=true.
    """

    def __init__(self, model_dir: str | Path | None = None):
        self.enabled = os.getenv("ENABLE_V2_SEGMENTER", "false").lower() == "true"
        self.model_dir = Path(model_dir or os.getenv("INGREDIENT_SEGMENTER_DIR", "models_v2/ingredient_segmenter"))
        self.model_path = self.model_dir / "segmenter.pt"
        self.classes_path = self.model_dir / "classes.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: nn.Module | None = None
        self.classes: list[str] = []
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.enabled:
            self.load()

    def load(self) -> bool:
        if not self.model_path.exists() or not self.classes_path.exists():
            self.model = None
            self.classes = []
            return False

        self.classes = json.loads(self.classes_path.read_text(encoding="utf-8"))
        num_classes = len(self.classes)
        model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self.model = model
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_loaded": self.model is not None,
            "classes_count": len(self.classes),
            "model_dir": str(self.model_dir),
        }

    def predict(self, image: Image.Image, min_area_pct: float = 1.0, top_k: int = 8) -> list[dict[str, Any]]:
        if not self.enabled or self.model is None:
            return []
        original_size = image.size
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)["out"][0]
            probs = torch.softmax(out, dim=0)
            mask = probs.argmax(dim=0).detach().cpu().numpy()
            conf_map = probs.max(dim=0).values.detach().cpu().numpy()

        total = float(mask.size)
        items: list[dict[str, Any]] = []
        for idx, name in enumerate(self.classes):
            if idx == 0 and name.lower() in {"background", "bg"}:
                continue
            area = float((mask == idx).sum())
            area_pct = area / total * 100.0
            if area_pct < min_area_pct:
                continue
            confidence = float(conf_map[mask == idx].mean()) if area > 0 else 0.0
            items.append({
                "name": name,
                "food_id": f"ingredient_{name.lower().replace(' ', '_')}",
                "confidence": round(confidence, 4),
                "area_pct": round(area_pct, 2),
                "source": "v2_ingredient_segmenter",
            })
        items.sort(key=lambda x: (x["area_pct"], x["confidence"]), reverse=True)
        return items[:top_k]

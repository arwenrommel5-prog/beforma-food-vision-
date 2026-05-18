"""Vision model adapter for meal image analysis.

Modes:
1. online_food101_model: load a PyTorch EfficientNet trained by
   training/train_online_food101.py on Food-101 downloaded from the internet.
2. torch_classifier: load a custom ImageFolder model where classes match local
   food IDs/aliases in app.food_database.
3. demo_detector: no trained model yet; use hints/filename/color fallback so the
   API and UI can still be tested.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image

from app.food_database import ALIAS_TO_ID, FOOD_BY_ID
from app.food101_classes import detection_from_food101, is_food101_label
from app.nutrition_logic import nutrition_for_quantity

PORTION_SCALE = {"small": 0.70, "medium": 1.0, "large": 1.35}


class FoodVisionModel:
    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "food_classifier.pt"
        self.classes_path = self.model_dir / "classes.json"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.mode = "demo_detector"
        self.classes: list[str] = []
        self.metadata: dict = {}
        self._torch_model = None
        self._try_load_torch_model()

    def _try_load_torch_model(self) -> None:
        if not self.model_path.exists() or not self.classes_path.exists():
            return
        try:
            import torch  # type: ignore
            from torchvision import models, transforms  # type: ignore

            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            arch = self.metadata.get("architecture", "efficientnet_b0")
            if arch == "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=None)
                model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.classes))
            else:
                model = models.efficientnet_b0(weights=None)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(self.classes))

            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self._torch_model = (
                model,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            )

            dataset_name = str(self.metadata.get("dataset", "")).lower()
            if dataset_name == "food101" or any(is_food101_label(c) for c in self.classes[:20]):
                self.mode = "online_food101_model"
            else:
                self.mode = "torch_classifier"
        except Exception as exc:  # keep API alive if model files are invalid
            self._torch_model = None
            self.metadata = {"load_error": str(exc)}
            self.mode = "demo_detector"

    def reload(self) -> dict:
        self._torch_model = None
        self.mode = "demo_detector"
        self.classes = []
        self.metadata = {}
        self._try_load_torch_model()
        return self.status()

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "model_path_exists": self.model_path.exists(),
            "classes_path_exists": self.classes_path.exists(),
            "classes_count": len(self.classes),
            "metadata": self.metadata,
        }

    def predict(self, image_path: str | Path, portion_scale: str = "medium", hints: Optional[str] = None) -> list[dict]:
        image_path = Path(image_path)
        if self._torch_model is not None:
            return self._predict_torch(image_path, portion_scale)
        return self._predict_demo(image_path, portion_scale, hints)

    def _predict_torch(self, image_path: Path, portion_scale: str) -> list[dict]:
        import torch  # type: ignore

        model, preprocess = self._torch_model
        img = Image.open(image_path).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        top = torch.topk(probs, k=min(5, len(self.classes)))
        results = []
        for p, idx in zip(top.values.tolist(), top.indices.tolist()):
            cls = self.classes[idx]

            # Online Food-101 model: class is a dish label, so map it to estimated nutrition.
            food101_det = detection_from_food101(cls, p, portion_scale)
            if food101_det:
                results.append(food101_det)
                continue

            # Custom local model: class should match food ID or alias.
            food_id = ALIAS_TO_ID.get(cls.lower(), cls.lower().replace(" ", "_"))
            food = FOOD_BY_ID.get(food_id)
            if not food:
                continue
            grams = food["typical_serving_g"] * PORTION_SCALE.get(portion_scale, 1.0)
            nut = nutrition_for_quantity(food, grams)
            results.append({
                "food_id": food["id"],
                "name_en": food["name_en"],
                "name_ar": food["name_ar"],
                "category": food["category"],
                "confidence": round(float(p), 3),
                "estimated_quantity_g": round(grams, 1),
                **nut,
                "source": "custom_torch_model",
                "image": food["image"],
            })
        return results

    def _predict_demo(self, image_path: Path, portion_scale: str, hints: Optional[str]) -> list[dict]:
        candidate_ids = self._ids_from_text(hints or "")
        if not candidate_ids:
            candidate_ids = self._ids_from_text(image_path.stem.replace("_", " ").replace("-", " "))
        if not candidate_ids:
            candidate_ids = self._color_fallback(image_path)
        scale = PORTION_SCALE.get(portion_scale, 1.0)
        results = []
        for i, fid in enumerate(candidate_ids[:5]):
            food = FOOD_BY_ID.get(fid)
            if not food:
                continue
            grams = food["typical_serving_g"] * scale
            nut = nutrition_for_quantity(food, grams)
            results.append({
                "food_id": food["id"],
                "name_en": food["name_en"],
                "name_ar": food["name_ar"],
                "category": food["category"],
                "confidence": round(0.88 - i * 0.07, 3),
                "estimated_quantity_g": round(grams, 1),
                **nut,
                "source": "demo_hint_or_filename",
                "image": food["image"],
            })
        return results

    def _ids_from_text(self, text: str) -> list[str]:
        text_l = (text or "").lower()
        if not text_l.strip():
            return []
        tokens = [t.strip() for t in re.split(r"[,;|]+", text_l) if t.strip()]
        found: list[str] = []
        for token in tokens:
            normalized = token.replace("_", " ").strip()
            fid = ALIAS_TO_ID.get(token) or ALIAS_TO_ID.get(normalized)
            if fid and fid not in found:
                found.append(fid)
        for alias, fid in sorted(ALIAS_TO_ID.items(), key=lambda x: -len(x[0])):
            if len(alias) >= 3 and alias in text_l and fid not in found:
                found.append(fid)
        return found

    def _color_fallback(self, image_path: Path) -> list[str]:
        try:
            img = Image.open(image_path).convert("RGB").resize((64, 64))
            pixels = list(img.getdata())
            r = sum(p[0] for p in pixels) / len(pixels)
            g = sum(p[1] for p in pixels) / len(pixels)
            b = sum(p[2] for p in pixels) / len(pixels)
            if g > r * 1.08 and g > b * 1.08:
                return ["salad_green", "broccoli_cooked"]
            if r > 150 and g > 110 and b < 100:
                return ["pasta_cooked", "tomato_raw"]
            if r > 130 and g > 95 and b > 70:
                return ["chicken_breast_cooked", "rice_white_cooked"]
            return ["rice_white_cooked", "chicken_breast_cooked"]
        except Exception:
            return ["rice_white_cooked", "chicken_breast_cooked"]

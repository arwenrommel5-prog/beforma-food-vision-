from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusionConfig:
    dish_confidence_threshold: float = 0.35
    ingredient_confidence_threshold: float = 0.20
    max_ingredients: int = 8


def _conf(x: dict[str, Any]) -> float:
    try:
        return float(x.get("confidence", 0.0))
    except Exception:
        return 0.0


def normalize_name(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def fuse_dish_and_ingredients(
    dish_predictions: list[dict[str, Any]],
    ingredient_predictions: list[dict[str, Any]] | None = None,
    config: FusionConfig | None = None,
) -> dict[str, Any]:
    """Merge dish-level classifier output with ingredient segmentation output.

    dish_predictions come from the old Food-101 EfficientNet classifier.
    ingredient_predictions come from the V2 segmentation model.
    """
    config = config or FusionConfig()
    ingredient_predictions = ingredient_predictions or []

    dish_predictions = sorted(dish_predictions, key=_conf, reverse=True)
    top_dish = dish_predictions[0] if dish_predictions else None

    ingredients = [
        item for item in ingredient_predictions
        if _conf(item) >= config.ingredient_confidence_threshold
    ]
    ingredients = sorted(ingredients, key=_conf, reverse=True)[: config.max_ingredients]

    if top_dish and ingredients:
        mode = "dish_plus_ingredients"
    elif top_dish:
        mode = "dish_only"
    elif ingredients:
        mode = "ingredients_only"
    else:
        mode = "no_confident_prediction"

    warnings: list[str] = []
    if top_dish and _conf(top_dish) < config.dish_confidence_threshold:
        warnings.append("Dish classifier confidence is low; rely more on ingredients.")
    if not ingredients:
        warnings.append("Ingredient segmentation is unavailable or found no confident ingredients.")

    return {
        "fusion_mode": mode,
        "top_dish": top_dish,
        "ingredients": ingredients,
        "dish_top_predictions": dish_predictions[:5],
        "warnings": warnings,
    }

"""Food-101 class support for online-data trained BeForma Food Vision models.

The online training pipeline trains on ETH Food-101 labels.  Food-101 labels are
mostly *dish names*, not raw ingredients, so this module translates a predicted
class into approximate nutrition and a user-facing detection item.

Nutrition values are estimates for a typical serving. They are intentionally
editable and should be improved with Nutrition5k/real app feedback later.
"""
from __future__ import annotations

from dataclasses import dataclass

FOOD101_CLASSES: list[str] = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
    "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog",
    "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
    "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup", "mussels",
    "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella",
    "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop",
    "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
    "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles",
]

# category preset: serving_g, kcal/100g, protein/100g, carbs/100g, fat/100g
PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "dessert": (140, 340, 5, 45, 16),
    "fried": (180, 300, 11, 28, 16),
    "salad": (220, 90, 5, 9, 5),
    "soup": (300, 65, 5, 7, 3),
    "seafood": (180, 160, 24, 2, 6),
    "meat": (200, 240, 25, 4, 14),
    "chicken": (220, 210, 23, 14, 8),
    "rice_dish": (300, 170, 8, 25, 5),
    "pasta": (300, 190, 9, 28, 7),
    "sandwich": (250, 250, 14, 27, 10),
    "snack": (160, 280, 8, 30, 12),
    "breakfast": (220, 230, 9, 30, 9),
}

CLASS_CATEGORY: dict[str, str] = {
    # desserts
    "apple_pie": "dessert", "baklava": "dessert", "bread_pudding": "dessert",
    "cannoli": "dessert", "carrot_cake": "dessert", "cheesecake": "dessert",
    "chocolate_cake": "dessert", "chocolate_mousse": "dessert", "churros": "dessert",
    "creme_brulee": "dessert", "cup_cakes": "dessert", "donuts": "dessert",
    "frozen_yogurt": "dessert", "ice_cream": "dessert", "macarons": "dessert",
    "panna_cotta": "dessert", "red_velvet_cake": "dessert",
    "strawberry_shortcake": "dessert", "tiramisu": "dessert", "waffles": "breakfast",
    "pancakes": "breakfast", "french_toast": "breakfast", "beignets": "dessert",
    # salads / veg
    "beet_salad": "salad", "caesar_salad": "salad", "caprese_salad": "salad",
    "greek_salad": "salad", "seaweed_salad": "salad", "edamame": "salad",
    "guacamole": "snack", "hummus": "snack", "bruschetta": "snack",
    # soups
    "clam_chowder": "soup", "french_onion_soup": "soup", "hot_and_sour_soup": "soup",
    "lobster_bisque": "soup", "miso_soup": "soup", "pho": "soup", "ramen": "soup",
    # meat/chicken/seafood
    "baby_back_ribs": "meat", "beef_carpaccio": "meat", "beef_tartare": "meat",
    "filet_mignon": "meat", "foie_gras": "meat", "pork_chop": "meat",
    "prime_rib": "meat", "steak": "meat", "peking_duck": "meat",
    "chicken_curry": "chicken", "chicken_quesadilla": "chicken", "chicken_wings": "chicken",
    "ceviche": "seafood", "crab_cakes": "seafood", "fish_and_chips": "fried",
    "fried_calamari": "fried", "grilled_salmon": "seafood", "lobster_roll_sandwich": "sandwich",
    "mussels": "seafood", "oysters": "seafood", "sashimi": "seafood", "scallops": "seafood",
    "shrimp_and_grits": "seafood", "sushi": "seafood", "takoyaki": "seafood", "tuna_tartare": "seafood",
    # rice/pasta/starch/dishes
    "bibimbap": "rice_dish", "fried_rice": "rice_dish", "paella": "rice_dish",
    "risotto": "rice_dish", "gnocchi": "pasta", "lasagna": "pasta", "macaroni_and_cheese": "pasta",
    "pad_thai": "pasta", "ravioli": "pasta", "spaghetti_bolognese": "pasta",
    "spaghetti_carbonara": "pasta", "dumplings": "snack", "falafel": "snack",
    "gyoza": "snack", "nachos": "snack", "onion_rings": "fried", "samosa": "fried",
    "spring_rolls": "snack", "french_fries": "fried", "poutine": "fried",
    # sandwiches / fast food
    "breakfast_burrito": "sandwich", "club_sandwich": "sandwich", "croque_madame": "sandwich",
    "grilled_cheese_sandwich": "sandwich", "hamburger": "sandwich", "hot_dog": "sandwich",
    "pulled_pork_sandwich": "sandwich", "tacos": "sandwich", "pizza": "snack",
    "garlic_bread": "snack", "cheese_plate": "snack", "deviled_eggs": "breakfast",
    "eggs_benedict": "breakfast", "huevos_rancheros": "breakfast", "omelette": "breakfast",
}

# class-specific overrides: serving_g, kcal/100g, protein/100g, carbs/100g, fat/100g
OVERRIDES: dict[str, tuple[float, float, float, float, float]] = {
    "grilled_salmon": (180, 208, 20.4, 0, 13.4),
    "sashimi": (150, 140, 24, 1, 4),
    "sushi": (220, 145, 7, 27, 2),
    "pizza": (180, 266, 11, 33, 10),
    "hamburger": (230, 295, 17, 24, 14),
    "french_fries": (150, 312, 3.4, 41, 15),
    "omelette": (180, 154, 11, 1, 12),
    "greek_salad": (250, 95, 4, 7, 6),
    "caesar_salad": (260, 150, 8, 8, 10),
    "chicken_wings": (220, 290, 22, 3, 21),
    "steak": (200, 250, 26, 0, 16),
    "spaghetti_bolognese": (320, 170, 9, 25, 5),
    "spaghetti_carbonara": (320, 210, 9, 24, 10),
    "lasagna": (300, 180, 10, 18, 8),
    "fried_rice": (300, 170, 6, 27, 5),
    "ice_cream": (120, 207, 3.5, 24, 11),
    "donuts": (90, 452, 5, 51, 25),
    "pancakes": (220, 227, 6, 28, 10),
    "waffles": (200, 291, 8, 33, 14),
    "hummus": (120, 166, 8, 14, 10),
    "falafel": (150, 333, 13, 32, 18),
    "miso_soup": (300, 35, 3, 4, 1),
    "ramen": (450, 95, 4, 14, 3),
    "pho": (450, 75, 6, 10, 2),
}

AR_NAMES: dict[str, str] = {
    "apple_pie": "فطيرة تفاح", "pizza": "بيتزا", "hamburger": "برجر", "fried_rice": "أرز مقلي",
    "grilled_salmon": "سالمون مشوي", "sushi": "سوشي", "steak": "ستيك", "omelette": "أومليت",
    "caesar_salad": "سلطة سيزر", "greek_salad": "سلطة يوناني", "chicken_wings": "أجنحة دجاج",
    "spaghetti_bolognese": "مكرونة بولونيز", "spaghetti_carbonara": "مكرونة كاربونارا",
    "french_fries": "بطاطس مقلية", "ice_cream": "آيس كريم", "donuts": "دونات",
}

PORTION_SCALE = {"small": 0.70, "medium": 1.0, "large": 1.35}


def normalize_food101_label(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_").replace("-", "_")


def is_food101_label(label: str) -> bool:
    return normalize_food101_label(label) in set(FOOD101_CLASSES)


def class_to_nutrition(label: str, portion_scale: str = "medium") -> dict | None:
    cls = normalize_food101_label(label)
    if cls not in FOOD101_CLASSES:
        return None
    serving, kcal100, p100, c100, f100 = OVERRIDES.get(cls, PRESETS[CLASS_CATEGORY.get(cls, "snack")])
    grams = serving * PORTION_SCALE.get(portion_scale, 1.0)
    mult = grams / 100
    return {
        "food_id": f"food101_{cls}",
        "name_en": cls.replace("_", " ").title(),
        "name_ar": AR_NAMES.get(cls, cls.replace("_", " ")),
        "category": CLASS_CATEGORY.get(cls, "mixed_common_foods"),
        "estimated_quantity_g": round(grams, 1),
        "calories": round(kcal100 * mult, 1),
        "protein": round(p100 * mult, 1),
        "carbs": round(c100 * mult, 1),
        "fat": round(f100 * mult, 1),
        "image": "",
    }


def detection_from_food101(label: str, confidence: float, portion_scale: str = "medium") -> dict | None:
    nut = class_to_nutrition(label, portion_scale)
    if not nut:
        return None
    return {
        **nut,
        "confidence": round(float(confidence), 3),
        "source": "online_food101_model",
    }

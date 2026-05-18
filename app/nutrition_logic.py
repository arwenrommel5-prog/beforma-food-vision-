from __future__ import annotations
from app.food_database import FoodNutrition, FOODS


def normalize_goal(goal: str) -> str:
    v = (goal or "maintain").strip().lower().replace("-", "_").replace(" ", "_")
    if v in {"lose", "loss", "weight_loss", "fat_loss", "cut", "cutting"}:
        return "lose"
    if v in {"gain", "muscle_gain", "bulk", "bulking", "build_muscle"}:
        return "gain"
    return "maintain"


def nutrition_for_quantity(food: FoodNutrition, grams: float) -> dict:
    m = max(grams, 0) / 100.0
    return {
        "calories": round(food["calories_per_100g"] * m, 1),
        "protein": round(food["protein_per_100g"] * m, 1),
        "carbs": round(food["carbs_per_100g"] * m, 1),
        "fat": round(food["fat_per_100g"] * m, 1),
    }


def sum_foods(items: list[dict]) -> dict:
    return {
        "calories": round(sum(i.get("calories", 0) for i in items), 1),
        "protein": round(sum(i.get("protein", 0) for i in items), 1),
        "carbs": round(sum(i.get("carbs", 0) for i in items), 1),
        "fat": round(sum(i.get("fat", 0) for i in items), 1),
    }


def compare_to_daily_target(daily_target: float, calories_so_far: float, meal_calories: float) -> dict:
    daily_target = max(float(daily_target), 1.0)
    calories_so_far = max(float(calories_so_far or 0), 0.0)
    remaining_before = max(daily_target - calories_so_far, 0.0)
    after = calories_so_far + meal_calories
    remaining_after = daily_target - after
    ratio = meal_calories / max(remaining_before, 1.0)
    if remaining_after < -100:
        status = "over"
        msg_en = "This meal will push you over your remaining daily calories. Reduce portions or replace calorie-dense foods."
        msg_ar = "الوجبة دي هتزودك عن السعرات المتبقية في اليوم. قلل الكمية أو بدّل الأصناف العالية في السعرات."
    elif 0.65 <= ratio <= 1.05:
        status = "good"
        msg_en = "This meal fits well within your remaining daily calories."
        msg_ar = "الوجبة مناسبة جدًا للسعرات المتبقية عندك في اليوم."
    else:
        status = "under"
        msg_en = "This meal is below your remaining calories. You may need to add a suitable side or snack."
        msg_ar = "الوجبة أقل من السعرات المتبقية. ممكن تزود صنف مناسب أو سناك حسب هدفك."
    return {
        "daily_calorie_target": round(daily_target, 1),
        "calories_before_meal": round(calories_so_far, 1),
        "remaining_before_meal": round(remaining_before, 1),
        "meal_calories": round(meal_calories, 1),
        "calories_after_meal": round(after, 1),
        "remaining_after_meal": round(remaining_after, 1),
        "status": status,
        "message_en": msg_en,
        "message_ar": msg_ar,
    }


def _allowed(food: FoodNutrition, dietary_preference: str) -> bool:
    pref = (dietary_preference or "normal").lower().replace("-", "_").replace(" ", "_")
    if pref in {"normal", "none", "all"}:
        return True
    tags = set(food.get("dietary_tags", []))
    category = food["category"]
    if pref == "vegan":
        return "vegan" in tags and category not in {"red_meat", "white_meat", "fish_seafood", "dairy", "eggs"}
    if pref == "vegetarian":
        return category not in {"red_meat", "white_meat", "fish_seafood"} and ("vegetarian" in tags or "vegan" in tags)
    if pref == "pescatarian":
        return category not in {"red_meat", "white_meat"}
    if pref == "dairy_free":
        return category != "dairy"
    if pref == "gluten_free":
        return "gluten_free" in tags
    return True


def suggest_food_changes(goal: str, dietary_preference: str, comparison: dict, totals: dict, recognized_ids: set[str]) -> dict:
    goal_norm = normalize_goal(goal)
    allowed = [f for f in FOODS if _allowed(f, dietary_preference) and f["id"] not in recognized_ids]
    lean_protein = sorted(allowed, key=lambda f: (-(f["protein_per_100g"] / max(f["calories_per_100g"], 1)), f["calories_per_100g"]))[:6]
    low_cal = sorted(allowed, key=lambda f: f["calories_per_100g"])[:6]
    dense = sorted(allowed, key=lambda f: -f["calories_per_100g"])[:6]
    balanced = sorted(allowed, key=lambda f: abs(f["calories_per_100g"] - 140) + abs(f["protein_per_100g"] - 12))[:6]

    if comparison["status"] == "over":
        better = low_cal if goal_norm == "lose" else balanced
        reason_ar = "بديل أخف في السعرات ويحافظ على الشبع"
        reason_en = "Lower-calorie alternative that helps control the meal total"
        action_ar = "قلل النشويات/الدهون أو استبدل جزء منها بخضار وبروتين أخف."
        action_en = "Reduce starches/fats or replace part of them with vegetables and lean protein."
    elif comparison["status"] == "under":
        better = dense if goal_norm == "gain" else lean_protein if goal_norm == "lose" else balanced
        reason_ar = "يساعدك تكمل السعرات أو البروتين الناقص حسب هدفك"
        reason_en = "Helps complete missing calories/protein based on your goal"
        action_ar = "زود صنف مناسب بدل ما تسيب عجز كبير في سعرات اليوم."
        action_en = "Add a suitable item instead of leaving a large calorie gap."
    else:
        better = lean_protein if goal_norm == "lose" else balanced
        reason_ar = "اختيار مناسب كبديل صحي لو عايز تنوع"
        reason_en = "Good healthy alternative if you want variety"
        action_ar = "الوجبة مناسبة، حافظ على الكمية أو بدّل بصنف مشابه."
        action_en = "The meal is suitable; keep portions or swap with similar options."

    items = [{
        "food_id": f["id"],
        "name_en": f["name_en"],
        "name_ar": f["name_ar"],
        "reason_en": reason_en,
        "reason_ar": reason_ar,
        "image": f["image"],
    } for f in better[:5]]

    return {
        "action_en": action_en,
        "action_ar": action_ar,
        "better_foods": items,
        "macro_note": _macro_note(totals),
    }


def _macro_note(totals: dict) -> dict:
    cal = max(totals.get("calories", 1), 1)
    p_pct = (totals.get("protein", 0) * 4) / cal
    c_pct = (totals.get("carbs", 0) * 4) / cal
    f_pct = (totals.get("fat", 0) * 9) / cal
    if p_pct < 0.18:
        ar = "البروتين في الوجبة قليل نسبيًا؛ حاول تزود مصدر بروتين."
        en = "Protein is relatively low; consider adding a protein source."
    elif f_pct > 0.45:
        ar = "نسبة الدهون عالية؛ راقب الزيوت والمكسرات والمقليات."
        en = "Fat ratio is high; watch oils, nuts, and fried foods."
    elif c_pct > 0.65:
        ar = "الكربوهيدرات عالية؛ حاول توازنها ببروتين أو خضار."
        en = "Carbs are high; balance with protein or vegetables."
    else:
        ar = "توزيع الماكروز مقبول كبداية."
        en = "Macro distribution looks acceptable as a starting point."
    return {"protein_pct": round(p_pct * 100, 1), "carbs_pct": round(c_pct * 100, 1), "fat_pct": round(f_pct * 100, 1), "message_ar": ar, "message_en": en}

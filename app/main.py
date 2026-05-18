from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.food_database import FOODS, categories_summary, get_food, search_foods
from app.nutrition_logic import compare_to_daily_target, nutrition_for_quantity, sum_foods, suggest_food_changes, normalize_goal
from app.schemas import ManualCorrectionRequest
from app.vision_model import FoodVisionModel

API_VERSION = "0.2.0-online-food101-ready"
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "sample_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="BeForma Food Vision API",
    version=API_VERSION,
    description="Standalone meal image calorie estimator. Upload a plate image, detect foods, estimate calories, compare with BeForma daily calorie target, and suggest better options.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision_model = FoodVisionModel(BASE_DIR / "models")

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.get("/")
def root() -> dict:
    return {
        "service": "BeForma Food Vision API",
        "version": API_VERSION,
        "docs": "/docs",
        "frontend": "/frontend/index.html",
        "main_endpoint": "POST /analyze-meal-image",
        "foods_count": len(FOODS),
        "model_mode": vision_model.mode,
        "model_status": "/model/status",
    }

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "beforma-food-vision", "version": API_VERSION, "model_mode": vision_model.mode}

@app.get("/model/status")
def model_status() -> dict:
    """Show whether the API is using demo mode or a trained online Food-101 model."""
    return vision_model.status()

@app.post("/model/reload")
def model_reload() -> dict:
    """Reload model files from /models after training/copying new weights."""
    return vision_model.reload()

@app.get("/foods")
def foods(search: str = "", limit: int = 50) -> dict:
    items = search_foods(search, limit=max(1, min(limit, 100)))
    return {"total": len(items), "items": items}

@app.get("/foods/categories")
def food_categories() -> dict:
    return {"categories": categories_summary()}

@app.post("/analyze-meal-image")
async def analyze_meal_image(
    image: UploadFile = File(...),
    daily_calorie_target: float = Form(...),
    calories_consumed_so_far: float = Form(0),
    goal: str = Form("maintain"),
    dietary_preference: str = Form("normal"),
    portion_scale: str = Form("medium"),
    food_hints: Optional[str] = Form(None),
) -> dict:
    """Analyze a meal image and compare it with a daily calorie target.

    In demo mode, pass food_hints like: "chicken, rice, salad".
    With a trained model in /models, hints become optional.
    """
    started = time.perf_counter()
    if portion_scale not in {"small", "medium", "large"}:
        raise HTTPException(status_code=400, detail="portion_scale must be small, medium, or large")
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(image.filename or "meal.jpg").suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with save_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    recognized = vision_model.predict(save_path, portion_scale=portion_scale, hints=food_hints)
    warnings: list[str] = []
    if not recognized:
        warnings.append("No food was recognized. Add food_hints in demo mode or upload trained model weights.")
    if vision_model.mode == "demo_detector":
        warnings.append("Demo mode is active. Train with training/train_online_food101.py to use online Food-101 image recognition.")
    elif vision_model.mode == "online_food101_model":
        warnings.append("Online Food-101 model is active. Food classes are dish-level estimates; portion size is still approximate from one image.")
    warnings.append("Calories are estimates. Single-image portion size is approximate without a reference object or depth data.")

    totals = sum_foods(recognized)
    comparison = compare_to_daily_target(daily_calorie_target, calories_consumed_so_far, totals["calories"])
    recs = suggest_food_changes(goal, dietary_preference, comparison, totals, {r["food_id"] for r in recognized})

    return {
        "status": "success",
        "recognized_foods": recognized,
        "meal_totals": totals,
        "daily_comparison": comparison,
        "recommendations": recs,
        "model_info": {
            "mode": vision_model.mode,
            "goal_normalized": normalize_goal(goal),
            "portion_scale": portion_scale,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "uploaded_filename": image.filename,
        },
        "warnings": warnings,
    }

@app.post("/manual-correction")
def manual_correction(payload: ManualCorrectionRequest) -> dict:
    """Let the frontend correct detected foods/grams and recalculate calories."""
    items = []
    for item in payload.corrected_items:
        food = get_food(item.food_id)
        if not food:
            raise HTTPException(status_code=404, detail={"error": "FOOD_NOT_FOUND", "food_id": item.food_id})
        nut = nutrition_for_quantity(food, item.quantity_g)
        items.append({
            "food_id": food["id"], "name_en": food["name_en"], "name_ar": food["name_ar"],
            "category": food["category"], "confidence": 1.0, "estimated_quantity_g": item.quantity_g,
            **nut, "source": "manual_correction", "image": food["image"],
        })
    totals = sum_foods(items)
    comparison = compare_to_daily_target(payload.daily_calorie_target, payload.calories_consumed_so_far, totals["calories"])
    recs = suggest_food_changes(payload.goal, payload.dietary_preference, comparison, totals, {r["food_id"] for r in items})
    return {"status": "success", "recognized_foods": items, "meal_totals": totals, "daily_comparison": comparison, "recommendations": recs}

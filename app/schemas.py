from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional

Goal = Literal["lose", "maintain", "gain", "fat_loss", "weight_loss", "bulk", "bulking", "muscle_gain", "build_muscle"]
PortionScale = Literal["small", "medium", "large"]

class RecognizedFood(BaseModel):
    food_id: str
    name_en: str
    name_ar: str
    category: str
    confidence: float
    estimated_quantity_g: float
    calories: float
    protein: float
    carbs: float
    fat: float
    source: str
    image: str

class MealTotals(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float

class DailyComparison(BaseModel):
    daily_calorie_target: float
    calories_before_meal: float
    remaining_before_meal: float
    meal_calories: float
    calories_after_meal: float
    remaining_after_meal: float
    status: Literal["under", "good", "over"]
    message_en: str
    message_ar: str

class RecommendationItem(BaseModel):
    food_id: str
    name_en: str
    name_ar: str
    reason_en: str
    reason_ar: str
    image: str

class MealImageAnalysisResponse(BaseModel):
    status: str
    recognized_foods: list[RecognizedFood]
    meal_totals: MealTotals
    daily_comparison: DailyComparison
    recommendations: dict
    model_info: dict
    warnings: list[str] = []

class ManualCorrectionItem(BaseModel):
    food_id: str
    quantity_g: float = Field(..., gt=0)

class ManualCorrectionRequest(BaseModel):
    daily_calorie_target: float = Field(..., gt=0)
    calories_consumed_so_far: float = Field(0, ge=0)
    goal: Goal = "maintain"
    dietary_preference: str = "normal"
    corrected_items: list[ManualCorrectionItem]

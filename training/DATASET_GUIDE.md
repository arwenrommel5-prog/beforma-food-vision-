# Dataset Guide for Real Food Recognition

The API works immediately in demo mode, but real image recognition requires training data and model weights.

Recommended dataset path for this project:

## Option A — Fast MVP classifier
Use Food-101 or a custom folder of common foods.

Folder format:

```text
data/
  train/
    chicken_breast_cooked/
    rice_white_cooked/
    pasta_cooked/
  val/
    chicken_breast_cooked/
    rice_white_cooked/
    pasta_cooked/
```

Train:

```bash
pip install -r requirements-ml.txt
python training/train_food_classifier.py --data_dir data --epochs 8 --output_dir models
```

Then run:

```bash
uvicorn app.main:app --reload
```

## Option B — Better multi-food plates
Use detection/segmentation datasets like FoodSeg103 or UECFOOD-256, then replace `FoodVisionModel.predict()` with YOLO/segmentation output.

Recommended production architecture:
1. Object detection or segmentation model detects food regions.
2. Classifier labels each region.
3. Portion estimator estimates grams using plate/reference object/user correction.
4. Nutrition database calculates calories/macros.

## Important limitation
Calories from a single image are always an estimate unless the system knows portion size/weight. For better accuracy, ask the user for one of:
- portion scale: small/medium/large
- plate diameter
- reference object
- manual correction of grams

# BeForma Food Vision API — Final GitHub/API Version

Standalone microservice for analyzing a meal image, estimating calories/macros, and comparing the meal against the user's BeForma daily calorie target.

## Main features

- Upload a meal/plate image.
- Detect likely food items.
- Estimate calories, protein, carbs, and fat.
- Compare the meal with daily calorie target and consumed calories so far.
- Tell the user whether the meal is suitable, too high, or too low.
- Suggest better food choices based on goal and dietary preference.
- Manual correction endpoint for food/quantity edits.
- Frontend test page included.
- Online Food-101 training pipeline included.
- Railway/Docker/GitHub ready.

## Important accuracy note

Calories from a single 2D image are estimates. Accurate portion sizing needs grams, reference objects, depth, or user correction. This MVP uses typical serving sizes plus `portion_scale` (`small`, `medium`, `large`).

## Project structure

```text
beforma_food_vision_final/
├── app/
│   ├── main.py
│   ├── vision_model.py
│   ├── food_database.py
│   ├── food101_classes.py
│   ├── nutrition_logic.py
│   └── schemas.py
├── frontend/
│   └── index.html
├── training/
│   ├── train_online_food101.py
│   ├── train_food_classifier.py
│   ├── train_nutrition_regressor_from_csv.py
│   └── ONLINE_DATASET_GUIDE.md
├── models/
├── sample_uploads/
├── requirements.txt
├── requirements-inference.txt
├── requirements-ml.txt
├── Dockerfile
├── Dockerfile.ml
├── railway.json
├── postman_collection.json
├── RUN_LOCAL.md
├── GITHUB_RAILWAY_DEPLOY.md
└── MODEL_FINAL_NOTE.md
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/frontend/index.html
```

## Main endpoint

```http
POST /analyze-meal-image
```

Form-data fields:

| Field | Type | Required | Example |
|---|---:|---:|---|
| image | file | yes | plate.jpg |
| daily_calorie_target | number | yes | 2200 |
| calories_consumed_so_far | number | no | 900 |
| goal | string | no | lose / maintain / gain |
| dietary_preference | string | no | normal / vegan / vegetarian / pescatarian |
| portion_scale | string | no | small / medium / large |
| food_hints | string | no | chicken, rice, salad |

Before training, use `food_hints` to test the full API flow.

## API endpoints

- `GET /`
- `GET /health`
- `GET /model/status`
- `POST /model/reload`
- `GET /foods`
- `GET /foods/categories`
- `POST /analyze-meal-image`
- `POST /manual-correction`
- `GET /docs`
- `GET /frontend/index.html`

## Train online Food-101 model

Quick test:

```powershell
python -m pip install -r requirements-ml.txt
python training/train_online_food101.py --data_root data --quick --output_dir models
```

Full training:

```powershell
python training/train_online_food101.py --data_root data --epochs 5 --batch_size 32 --output_dir models
```

Then run API with inference dependencies:

```powershell
python -m pip install -r requirements-inference.txt
python -m uvicorn app.main:app --reload
```

Check:

```text
http://127.0.0.1:8000/model/status
```

Expected trained mode:

```json
{
  "mode": "online_food101_model"
}
```

## Deploy

See:

- `GITHUB_RAILWAY_DEPLOY.md`
- `RUN_LOCAL.md`
- `MODEL_FINAL_NOTE.md`

## Postman

Import:

```text
postman_collection.json
```

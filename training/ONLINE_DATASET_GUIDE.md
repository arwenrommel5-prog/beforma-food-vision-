# Online Dataset Training Guide

This project is now prepared to train the food-recognition model using online data.

## Recommended MVP dataset: Food-101

Food-101 is used first because it can be downloaded automatically by `torchvision.datasets.Food101(download=True)`.

It trains a dish classifier with labels like:

- pizza
- hamburger
- grilled_salmon
- fried_rice
- spaghetti_bolognese
- sushi
- omelette
- steak
- caesar_salad

After prediction, the API maps the Food-101 dish label to estimated calories/macros using `app/food101_classes.py`.

### Fast test training

```bash
python -m pip install -r requirements-ml.txt
python training/train_online_food101.py --data_root data --quick --output_dir models
```

### Full training

```bash
python -m pip install -r requirements-ml.txt
python training/train_online_food101.py --data_root data --epochs 5 --batch_size 32 --output_dir models
```

Outputs:

```text
models/food_classifier.pt
models/classes.json
models/model_metadata.json
```

Then run the API:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Check:

```text
http://127.0.0.1:8000/model/status
```

If mode is `online_food101_model`, the API is using the trained model.

## Important limitation

Food-101 is a classification dataset, not a precise nutrition dataset. It identifies dish type, then calories are estimated from typical serving sizes.

For more accurate calorie/macro prediction, train a second-stage regression model on Nutrition5k, because Nutrition5k contains dish images with nutritional annotations.

## Better next datasets

- Nutrition5k: direct visual nutrition data for realistic plates.
- UECFOOD-256: food localization with bounding boxes.
- FoodSeg103/UECFoodPix: segmentation masks for ingredient/food regions.

## Suggested roadmap

1. MVP: Food-101 classifier.
2. Add manual correction in frontend to fix class/grams.
3. Store corrected examples.
4. Fine-tune the model with user corrections.
5. Add segmentation/detection dataset to identify multiple foods in one plate.
6. Add Nutrition5k regression for direct calorie/macros estimation.

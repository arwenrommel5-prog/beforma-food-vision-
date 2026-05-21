# BeForma Food Vision V2 — Multitask Dish + Ingredient System

## Goal
V2 keeps the current safe Food-101 dish classifier and adds a second ingredient segmentation model.

- `models/` remains the safe V1 model.
- `models_v2/dish_classifier/` is a continued fine-tuned dish classifier.
- `models_v2/ingredient_segmenter/` is a segmentation model trained on FoodSeg103.
- Railway should not require the segmenter until `ENABLE_V2_SEGMENTER=true` and model files exist.

## Online datasets
- Food-101 via `torchvision.datasets.Food101(download=True)`
- FoodSeg103 via Hugging Face: `EduardoPacheco/FoodSeg103`
- Optional UECFOODPIXCOMPLETE via Hugging Face mirror if available.

## Colab setup
```python
!git clone https://github.com/arwenrommel5-prog/beforma-food-vision-.git
%cd beforma-food-vision-
!git checkout train/food-vision-v2-multitask
!pip install -r requirements-v2.txt
!python training_v2/download_prepare_v2.py
```

## Continue dish classifier from old model
```python
!python training_v2/train_continue_dish_classifier_v2.py \
  --old_model_dir models \
  --food101_root data_v2/food101 \
  --output_dir models_v2/dish_classifier \
  --epochs 8 \
  --batch_size 32 \
  --image_size 224 \
  --lr 5e-5
```

## Train ingredient segmenter
```python
!python training_v2/train_segmenter_foodseg103_v2.py \
  --output_dir models_v2/ingredient_segmenter \
  --epochs 20 \
  --batch_size 4 \
  --image_size 384 \
  --lr 1e-4
```

## Railway note
Keep V1 API stable. V2 ingredient model is optional:

```env
ENABLE_V2_SEGMENTER=false
```

After segmenter is trained and uploaded:

```env
ENABLE_V2_SEGMENTER=true
INGREDIENT_SEGMENTER_DIR=models_v2/ingredient_segmenter
```

If the segmenter files are missing, API should return dish-only output instead of crashing.

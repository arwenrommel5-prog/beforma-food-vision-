# Final model note

This repo is API-ready and training-ready. It does not include trained `.pt` weights because Food-101 is large and training requires internet/GPU time.

The production flow is:

1. Push this repo to GitHub.
2. Run `training/train_online_food101.py` locally or on Colab.
3. Copy generated files into `models/`:
   - `food_classifier.pt`
   - `classes.json`
   - `model_metadata.json`
4. Run API with `requirements-inference.txt`.
5. Confirm `/model/status` returns `online_food101_model`.

Before training, the API still works in demo mode using `food_hints` so frontend/backend integration can be tested immediately.

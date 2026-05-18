# Run locally

## 1) Lightweight API demo mode

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/frontend/index.html
http://127.0.0.1:8000/model/status
```

## 2) Train the online Food-101 model

```powershell
python -m pip install -r requirements-ml.txt
python training/train_online_food101.py --data_root data --quick --output_dir models
```

For full training:

```powershell
python training/train_online_food101.py --data_root data --epochs 5 --batch_size 32 --output_dir models
```

## 3) Run API with trained PyTorch model

```powershell
python -m pip install -r requirements-inference.txt
python -m uvicorn app.main:app --reload
```

Then check:

```text
http://127.0.0.1:8000/model/status
```

Expected after training:

```json
{"mode": "online_food101_model"}
```

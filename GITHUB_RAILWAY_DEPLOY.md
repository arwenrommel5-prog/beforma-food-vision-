# GitHub + Railway Deploy

## Push to GitHub

```powershell
git init
git add .
git commit -m "Add BeForma Food Vision API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/beforma-food-vision-api.git
git push -u origin main
```

## Deploy on Railway

1. Railway → New Project.
2. Deploy from GitHub Repo.
3. Choose this repo.
4. Railway will use `Dockerfile` by default.
5. Add variable:

```env
ALLOWED_ORIGINS=*
```

6. Generate public domain from Settings → Networking.
7. Test:

```text
https://YOUR-DOMAIN.up.railway.app/health
https://YOUR-DOMAIN.up.railway.app/docs
```

## Deploy with trained PyTorch model

The default `Dockerfile` is lightweight and runs demo mode unless torch is installed. If you train the model and commit/copy model files to the deployment environment, use `Dockerfile.ml` or change Railway config to point to it.

Required files after training:

```text
models/food_classifier.pt
models/classes.json
models/model_metadata.json
```

Important: model weights can be large. For GitHub, use Git LFS or upload weights to cloud storage and copy them during deployment.

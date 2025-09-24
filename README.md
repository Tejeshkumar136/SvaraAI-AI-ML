# SvaraAI – Reply Classification Assignment

## Structure
- `data/`: CSV dataset (`test.csv` provided)
- `notebooks/notebook.ipynb`: Training (baseline + transformer), evaluation, and model saving
- `api/app.py`: FastAPI service with `/predict` and `/health`
- `models/`: Saved models (`baseline.pkl` or `transformer/`)

## Setup
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # on Windows PowerShell
pip install -r requirements.txt
```

## Train the model
1) Open `notebooks/notebook.ipynb`.
2) Run the cells (baseline first, then transformer if you want better accuracy).

When it finishes, you’ll have:
- Baseline model at `models/baseline.pkl`
- Transformer model under `models/transformer/`

Tip: If you’re low on disk space, the notebook uses a local cache under `models/hf_cache` and can switch to a lighter model automatically.

## Run the API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Try a prediction:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Looking forward to the demo!"}'
```

## Docker (optional)
```bash
docker build -t svara-reply-clf .
docker run -p 8000:8000 svara-reply-clf
```

## Notes & tips
- The provided settings are kept small for quick runs. You can bump epochs, batch size, or switch models in the notebook to trade speed for accuracy.
- If no trained transformer is found, the API will use the baseline model. If neither is available, it falls back to a simple dummy predictor.
- Keep your CSV in `data/` with columns `text` and `sentiment` (the notebook can normalize common label names).

Have questions or want to tune further? Open the notebook and tweak away.


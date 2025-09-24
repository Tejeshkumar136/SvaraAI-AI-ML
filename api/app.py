import os
import json
import pickle
from typing import Optional, Dict, Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class PredictRequest(BaseModel):
    text: str


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _load_temperature(model_dir: str) -> Optional[float]:
    temp_path = os.path.join(model_dir, "temperature.json")
    if os.path.isfile(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = float(data.get("temperature", 1.0))
            if t > 0:
                return t
        except Exception:
            pass
    return None


def load_model() -> Dict[str, Any]:
    """Load the best available model from models/.

    Priority:
      1) Transformer saved via Hugging Face in models/transformer/
      2) Sklearn pipeline saved as models/baseline.pkl
    """
    transformer_path = os.path.abspath(os.path.join(MODEL_DIR, "transformer"))
    baseline_path = os.path.abspath(os.path.join(MODEL_DIR, "baseline.pkl"))

    # Try transformer first
    if os.path.isdir(transformer_path):
        try:
            from transformers import pipeline

            text_classifier = pipeline(
                "text-classification",
                model=transformer_path,
                tokenizer=transformer_path,
                top_k=None,
                function_to_apply="softmax",
                truncation=True,
            )
            labels = None  # labels come from model config
            temperature = _load_temperature(transformer_path)
            return {"type": "transformer", "model": text_classifier, "labels": labels, "temperature": temperature}
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load transformer model: {exc}")

    # Fallback to sklearn
    if os.path.isfile(baseline_path):
        with open(baseline_path, "rb") as f:
            skl = pickle.load(f)
        labels = None
        if hasattr(skl, "classes_"):
            labels = list(skl.classes_)
        return {"type": "sklearn", "model": skl, "labels": labels, "temperature": None}

    # If nothing is available, create a dummy that always returns neutral
    print("Warning: No model found in models/. Using a dummy classifier.")
    labels = ["negative", "neutral", "positive"]

    class Dummy:
        def predict(self, X):  # noqa: N802
            return ["neutral" for _ in X]

        def predict_proba(self, X):  # noqa: N802
            probs = np.zeros((len(X), 3), dtype=float)
            probs[:, 1] = 1.0  # all neutral
            return probs

    return {"type": "dummy", "model": Dummy(), "labels": labels}


app = FastAPI(title="SvaraAI Reply Classifier", version="1.0.0")

# Allow local tools and browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL: Dict[str, Any] = load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Service running. Use /predict (POST) or /docs for Swagger UI."}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    text = req.text.strip() if req.text else ""
    if not text:
        return {"label": "neutral", "confidence": 0.0}

    if MODEL["type"] == "transformer":
        classifier = MODEL["model"]
        outputs = classifier(text, truncation=True)
        # HF pipeline with top_k=None returns list of dicts per class
        candidates = outputs[0] if isinstance(outputs, list) else outputs
        # Normalize labels to lower-case without model-specific prefixes
        def normalize(label: str) -> str:
            label = label.lower()
            for prefix in ("label_", "labels_", "__label__"):
                if label.startswith(prefix):
                    label = label[len(prefix):]
            return label

        # Optional temperature scaling when available: p_i(T) ∝ p_i^{1/T}
        temp = MODEL.get("temperature") or 1.0
        if temp != 1.0 and temp > 0:
            alpha = 1.0 / float(temp)
            scores = np.array([max(1e-12, float(c.get("score", 0.0))) for c in candidates], dtype=float)
            scores = scores ** alpha
            denom = scores.sum()
            if denom > 0:
                scores = scores / denom
                for i, c in enumerate(candidates):
                    c["score"] = float(scores[i])

        best = max(candidates, key=lambda d: d.get("score", 0.0))
        return {"label": normalize(best.get("label", "neutral")), "confidence": float(best.get("score", 0.0))}

    if MODEL["type"] in {"sklearn", "dummy"}:
        model = MODEL["model"]
        labels = MODEL["labels"] or ["negative", "neutral", "positive"]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
        elif hasattr(model, "decision_function"):
            logits = model.decision_function([text])
            probs = _softmax(np.array(logits))
            probs = probs[0]
        else:
            pred = model.predict([text])[0]
            probs = np.zeros((len(labels),), dtype=float)
            if pred in labels:
                probs[labels.index(pred)] = 1.0
        idx = int(np.argmax(probs))
        return {"label": str(labels[idx]).lower(), "confidence": float(probs[idx])}

    return {"label": "neutral", "confidence": 0.0}


if __name__ == "__main__":
    import uvicorn
    # Bind to localhost for local testing and pass the app directly
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



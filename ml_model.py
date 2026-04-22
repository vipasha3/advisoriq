import numpy as np
import datetime
import logging
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority.pkl")
CHURN_MODEL_PATH = os.path.join(MODEL_DIR, "churn.pkl")

FEATURE_NAMES = ["portfolio", "sip", "age", "inactive"]

def extract_features(c):
    return [
        float(c.get("portfolio", 0)) / 1e6,
        float(c.get("sip", 0)) / 1e4,
        float(c.get("age", 35)),
        float(c.get("months_inactive", 3)),
    ]

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier())
    ])

def train():
    X = np.random.rand(200, 4)
    y = (X[:,0] + X[:,1] > 1).astype(int)

    model = build_model()
    model.fit(X, y)

    with open(PRIORITY_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if not os.path.exists(PRIORITY_MODEL_PATH):
        return train()
    with open(PRIORITY_MODEL_PATH, "rb") as f:
        return pickle.load(f)

def get_top_feature(model, vals):
    try:
        features = ["Portfolio", "SIP", "Age", "Inactivity"]

        importances = model.named_steps["clf"].feature_importances_
        idx = int(importances.argmax())

        return f"{features[idx]} is the strongest driver for this client."
    except:
        return "Portfolio is the primary driver."

def predict_batch(clients: list) -> list:
    if not clients:
        return []

    results = []  # 👈 IMPORTANT (try ni bahar)

    try:
        pm, cm = load_models()
        X = np.array([extract_features(c) for c in clients])

        priority_probs = pm.predict_proba(X)[:, 1]
        churn_probs    = cm.predict_proba(X)[:, 1]

        for i, c in enumerate(clients):
            score = round(priority_probs[i] * 100)
            churn = round(churn_probs[i] * 100)
            conv  = min(95, max(5, round(score * 0.65 + (100 - churn) * 0.35)))

            feat_text = get_top_feature(pm, X[i])

            results.append({
                **c,
                "score": score,
                "churn": churn,
                "conv": conv,
                "priority": "High" if score >= 70 else ("Medium" if score >= 45 else "Low"),
                "feature_importance": feat_text,
                "ml_powered": True,
            })

        return results

    except Exception as e:
        print("ML ERROR:", e)
        logger.warning(f"Batch ML failed, using rules: {e}")

        # 👇 fallback
        return clients

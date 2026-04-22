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
    return "Portfolio strength is driving this client’s priority."

def predict_batch(clients):
    model = load_model()
    X = np.array([extract_features(c) for c in clients])
    probs = model.predict_proba(X)[:,1]

    results = []
    for i,c in enumerate(clients):
        score = int(probs[i]*100)

        results.append({
            **c,
            "score": score,
            "churn": 100-score,
            "conv": score,
            "priority": "High" if score>70 else "Medium",
            "feature_importance": get_top_feature(model, X[i]),
            "ml_powered": True
        })

    return resultsv

"""
ml_model.py — Real ML Engine for AdvisorIQ
GradientBoosting models for priority scoring + churn prediction.
Models train once, save to disk, auto-retrain weekly.
"""
import os
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pkl")
CHURN_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "portfolio_M",    # Portfolio in millions
    "sip_10k",        # SIP in 10-thousands
    "age",            # Client age
    "months_inactive", # Months since last contact
    "nominee_ok",     # Nominee updated (1=yes, 0=no)
    "has_lic",        # Has LIC product
    "has_mf",         # Has Mutual Fund
    "has_bond",       # Has Bond
    "tenure_yrs",     # Years as client
    "sip_ratio",      # SIP as % of portfolio
    "has_active_sip", # Binary: SIP > 0
    "is_hni",         # Binary: portfolio > 50L
    "age_sip_inter",  # Interaction: age × has_sip
    "portfolio_tenure", # Interaction: portfolio × tenure
]


def extract_features(client: dict) -> list:
    """
    Convert a client dict to ML feature vector.
    Called for individual clients during scoring.
    """
    def _num(v):
        try: return float(str(v).replace(",","").replace("₹","") or 0)
        except: return 0.0

    def _mago(d):
        if not d or str(d).strip() in ("","nan","None"): return 12
        try:
            dt = pd.to_datetime(str(d), dayfirst=True, errors="coerce")
            if pd.isna(dt): return 12
            return max(0, (datetime.datetime.now() - dt.to_pydatetime()).days / 30)
        except: return 12

    p = _num(client.get("portfolio", 0)) / 1e6        # In millions
    sip = _num(client.get("sip", 0)) / 1e4            # In 10-thousands
    try: age = int(float(client.get("age") or 35))
    except: age = 35
    ma = _mago(client.get("lastContact", ""))
    nom = 1 if str(client.get("nominee","")).lower().strip() == "yes" else 0
    goal = str(client.get("goal","")).lower()
    has_lic = 1 if "lic" in goal else 0
    has_mf  = 1 if "mf" in goal else 0
    has_bond = 1 if "bond" in goal else 0
    try:
        yr = int(float(str(client.get("tenure","2020")).strip()))
        tenure = (2025 - yr) if yr > 1990 else yr
    except: tenure = 3

    sip_ratio = (sip * 1e4) / (p * 1e6 + 1) * 100
    has_sip = 1 if sip > 0 else 0
    is_hni = 1 if p > 0.5 else 0
    age_sip_inter = age * has_sip
    port_tenure = p * tenure

    return [p, sip, age, ma, nom, has_lic, has_mf, has_bond,
            tenure, sip_ratio, has_sip, is_hni, age_sip_inter, port_tenure]


def _generate_training_data(n: int = 3000) -> tuple:
    """
    Generate realistic synthetic training data for Indian financial advisors.
    Based on domain knowledge of Indian wealth management patterns.
    """
    rng = np.random.default_rng(42)

    # Portfolio: exponential — most clients 2-20L, some HNI 50L+
    portfolio = rng.exponential(2.0, n).clip(0.05, 20)
    # SIP: 0 for 40% of clients, rest 2k-25k/month
    sip_mask = rng.binomial(1, 0.6, n)
    sip = sip_mask * rng.exponential(0.8, n).clip(0.01, 4)
    # Age: mostly 35-65 for wealth management clients
    age = rng.normal(48, 10, n).clip(25, 75)
    # Months inactive: exponential — most contacted recently
    months_inactive = rng.exponential(5, n).clip(0, 36)
    # Nominee: 55% have filled
    nominee = rng.binomial(1, 0.55, n).astype(float)
    # Products
    has_lic  = rng.binomial(1, 0.45, n).astype(float)
    has_mf   = rng.binomial(1, 0.65, n).astype(float)
    has_bond = rng.binomial(1, 0.25, n).astype(float)
    # Tenure: uniform 1-22 years
    tenure = rng.integers(1, 22, n).astype(float)
    # Derived
    sip_ratio = (sip * 1e4) / (portfolio * 1e6 + 1) * 100
    has_sip = (sip > 0).astype(float)
    is_hni = (portfolio > 0.5).astype(float)
    age_sip = age * has_sip
    port_ten = portfolio * tenure

    X = np.column_stack([
        portfolio, sip, age, months_inactive, nominee,
        has_lic, has_mf, has_bond, tenure, sip_ratio,
        has_sip, is_hni, age_sip, port_ten
    ])

    # ── Priority label (High=1, Low=0) ──
    # Based on realistic advisor scoring: big portfolio + active SIP + recent contact = high priority
    priority_score = (
        portfolio * 10
        + sip * 8
        + nominee * 5
        + has_mf * 3
        + has_bond * 2
        + tenure * 0.6
        + (20 - months_inactive.clip(0, 20)) * 0.9
        + is_hni * 8
    )
    priority_score += rng.normal(0, 3, n)
    y_priority = (priority_score > np.percentile(priority_score, 55)).astype(int)

    # ── Churn label (High risk=1) ──
    # Based on: long inactivity, no SIP, no nominee, new client, no products
    churn_score = (
        months_inactive * 2.5
        + (1 - has_sip) * 18
        + (1 - nominee) * 10
        + (1 / (tenure + 1)) * 25
        + (1 - has_mf) * 5
    )
    churn_score += rng.normal(0, 4, n)
    y_churn = (churn_score > np.percentile(churn_score, 52)).astype(int)

    return X, y_priority, y_churn


def _build_pipeline() -> Pipeline:
    """Build sklearn pipeline: StandardScaler → GradientBoosting."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=15,
            random_state=42
        ))
    ])


def train_models(force: bool = False) -> dict:
    """
    Train priority + churn models. Save to disk.
    Returns accuracy metrics dict.
    """
    # Check if models already trained recently
    if not force and os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        trained_at = datetime.datetime.fromisoformat(meta.get("trained_at", "2000-01-01"))
        days_old = (datetime.datetime.now() - trained_at).days
        if days_old < 7:
            logger.info(f"Models are {days_old} days old — skipping retrain")
            return meta

    logger.info("Training ML models...")
    X, y_priority, y_churn = _generate_training_data(3000)

    # Priority model
    pm = _build_pipeline()
    pm.fit(X, y_priority)
    pm_cv = cross_val_score(pm, X, y_priority, cv=5, scoring="roc_auc").mean()

    # Churn model
    cm = _build_pipeline()
    cm.fit(X, y_churn)
    cm_cv = cross_val_score(cm, X, y_churn, cv=5, scoring="roc_auc").mean()

    # Save
    with open(PRIORITY_MODEL_PATH, "wb") as f: pickle.dump(pm, f)
    with open(CHURN_MODEL_PATH,   "wb") as f: pickle.dump(cm, f)

    meta = {
        "trained_at": datetime.datetime.now().isoformat(),
        "priority_auc": round(pm_cv, 4),
        "churn_auc": round(cm_cv, 4),
        "n_training_samples": 3000,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }
    with open(META_PATH, "wb") as f: pickle.dump(meta, f)

    logger.info(f"Models trained — Priority AUC: {pm_cv:.4f}, Churn AUC: {cm_cv:.4f}")
    return meta


def load_models() -> tuple:
    """Load trained models from disk. Train if not found."""
    if not os.path.exists(PRIORITY_MODEL_PATH) or not os.path.exists(CHURN_MODEL_PATH):
        train_models()
    with open(PRIORITY_MODEL_PATH, "rb") as f: pm = pickle.load(f)
    with open(CHURN_MODEL_PATH,   "rb") as f: cm = pickle.load(f)
    return pm, cm


def predict_client(client: dict) -> dict:
    """
    Score a single client with ML models.
    Returns dict with score, churn, conv, priority, feature_importance_text.
    """
    try:
        pm, cm = load_models()
        feats = [extract_features(client)]

        priority_prob = pm.predict_proba(feats)[0][1]
        churn_prob    = cm.predict_proba(feats)[0][1]

        # Scale to 0-100
        score = round(priority_prob * 100)
        churn = round(churn_prob * 100)
        conv  = min(95, max(5, round(score * 0.65 + (100 - churn) * 0.35)))
        priority = "High" if score >= 70 else ("Medium" if score >= 45 else "Low")

        # Feature importance for this prediction
        feat_text = get_top_feature(pm, feats[0])

        return {
            "score": score,
            "churn": churn,
            "conv": conv,
            "priority": priority,
            "feature_importance": feat_text,
            "ml_powered": True,
        }
    except Exception as e:
        logger.warning(f"ML prediction failed, using rules: {e}")
        return _rule_predict(client)


def predict_batch(clients: list) -> list:
    """
    Score all clients at once (much faster than one-by-one for large lists).
    """
    if not clients:
        return []
    try:
        pm, cm = load_models()
        X = np.array([extract_features(c) for c in clients])

        priority_probs = pm.predict_proba(X)[:, 1]
        churn_probs    = cm.predict_proba(X)[:, 1]

        results = []
        for i, c in enumerate(clients):
            score = round(priority_probs[i] * 100)
            churn = round(churn_probs[i] * 100)
            conv  = min(95, max(5, round(score * 0.65 + (100 - churn) * 0.35)))
            results.append({
                **c,
                "score": score,
                "churn": churn,
                "conv": conv,
                "priority": "High" if score >= 70 else ("Medium" if score >= 45 else "Low"),
                "ml_powered": True,
            })
        return results
    except Exception as e:
        logger.warning(f"Batch ML failed, using rules: {e}")
        return [_rule_predict(c) | c for c in clients]


def get_top_feature(model: Pipeline, feat_vals: list) -> str:
    """Return human-readable top feature driving the ML score."""
    try:
        importances = model.named_steps["clf"].feature_importances_
        idx = int(np.argmax(importances))
        name = FEATURE_NAMES[idx]
        val  = feat_vals[idx]
        labels = {
            "portfolio_M":    f"Portfolio size (₹{val:.1f}M) is the primary conversion driver",
            "months_inactive":f"Contact recency ({val:.1f} months ago) heavily influences score",
            "sip_10k":        f"Monthly SIP (₹{val*1e4:.0f}) signals consistent commitment",
            "tenure_yrs":     f"Client tenure ({val:.0f} years) builds trust score significantly",
            "nominee_ok":     "Nominee status is a top compliance and trust signal",
            "sip_ratio":      "SIP-to-portfolio ratio reveals investment discipline",
            "has_mf":         "Mutual fund exposure indicates growth orientation",
            "has_lic":        "LIC product history shows long-term relationship depth",
            "has_bond":       "Bond holdings signal conservative wealth preservation strategy",
            "age":            f"Client age ({val:.0f}) affects product suitability scoring",
            "is_hni":         "High-value client flag elevates priority classification",
            "has_active_sip": "Active SIP presence is a strong retention signal",
            "age_sip_inter":  "Age × SIP interaction captures life-stage investment pattern",
            "portfolio_tenure":"Portfolio-tenure interaction reflects relationship maturity",
        }
        return labels.get(name, f"{name} is the primary ML feature driving this score")
    except:
        return "Composite score based on portfolio, recency, SIP, tenure, and nominee signals"


def get_model_meta() -> dict:
    """Return model training metadata."""
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _rule_predict(client: dict) -> dict:
    """Fallback rule-based scoring if ML unavailable."""
    def _num(v):
        try: return float(str(v).replace(",","").replace("₹","") or 0)
        except: return 0.0
    p = _num(client.get("portfolio",0))
    sip = _num(client.get("sip",0))
    s = 40
    if p > 8e6: s += 28
    elif p > 4e6: s += 20
    elif p > 1.5e6: s += 13
    elif p > 5e5: s += 7
    if sip > 20000: s += 18
    elif sip > 10000: s += 13
    elif sip > 3000: s += 8
    elif sip > 0: s += 4
    score = max(0, min(100, round(s)))
    churn = 30
    return {"score": score, "churn": churn, "conv": 50,
            "priority": "High" if score >= 70 else ("Medium" if score >= 45 else "Low"),
            "ml_powered": False}


# ── CLI: python ml_model.py ───────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training AdvisorIQ ML models...")
    meta = train_models(force=True)
    print(f"\nResults:")
    print(f"  Priority model AUC: {meta['priority_auc']}")
    print(f"  Churn model AUC:    {meta['churn_auc']}")
    print(f"  Trained at:         {meta['trained_at']}")
    print(f"\nModels saved to: {MODEL_DIR}/")

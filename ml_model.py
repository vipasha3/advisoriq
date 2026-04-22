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
    """
    Return client-specific human insight — not generic model importance.
    Uses model importance × client actual value to find the real driver
    for THIS specific client (not the same answer for everyone).
    """
    try:
        importances = model.named_steps["clf"].feature_importances_

        # Normalize feature values to 0-1 range (approximate)
        feat_arr = np.array(feat_vals, dtype=float)
        feat_norms = np.abs(feat_arr) / (np.abs(feat_arr).max() + 1e-9)

        # Client-specific score = global importance × this client's value contribution
        client_scores = importances * feat_norms
        idx = int(np.argmax(client_scores))

        name = FEATURE_NAMES[idx]
        val  = feat_vals[idx]

        # Build insight based on actual client value — sounds human, not template
        def _inr(v_m): # v in millions
            n = v_m * 1e6
            if n >= 1e7: return f"₹{n/1e7:.1f}Cr"
            if n >= 1e5: return f"₹{n/1e5:.1f}L"
            return f"₹{n/1e3:.0f}K"

        # Each case reads actual value and generates a meaningful sentence
        if name == "portfolio_M":
            size = _inr(val)
            if val > 5: return f"Portfolio of {size} places this client in the top tier — large AUM is the strongest conversion signal the model sees here."
            if val > 1: return f"Mid-range portfolio of {size} is the primary driver — enough value to prioritise, room to grow."
            return f"Portfolio of {size} is the key factor — smaller base means higher effort needed to convert."

        elif name == "months_inactive":
            mo = round(val, 1)
            if mo < 1: return f"Contacted less than a month ago — recency is the top signal here. Strike while engagement is fresh."
            if mo < 3: return f"Last contact {mo} months ago keeps this client in the active zone — the model weights this recency heavily."
            if mo < 6: return f"Contact gap of {mo} months is starting to show — the model flags this as an early warning signal."
            return f"{mo} months without contact is the dominant risk factor — this is the single biggest drag on this client's score."

        elif name == "sip_10k":
            sip_amt = val * 1e4
            if sip_amt > 15000: return f"Monthly SIP of ₹{sip_amt:,.0f} is exceptionally strong — systematic commitment is the model's top signal for this client."
            if sip_amt > 5000: return f"Active SIP of ₹{sip_amt:,.0f}/month signals disciplined investing — the model treats this as a strong loyalty indicator."
            if sip_amt > 0: return f"Small but active SIP of ₹{sip_amt:,.0f} is the key positive signal — even modest systematic investment builds retention score."
            return f"No active SIP despite having a portfolio — the model identifies this absence as the primary opportunity gap."

        elif name == "tenure_yrs":
            yrs = round(val, 0)
            if yrs > 12: return f"{yrs:.0f} years as a client — deep relationship tenure is what's driving the score here. Long-term clients are statistically far less likely to churn."
            if yrs > 5: return f"{yrs:.0f} years of relationship history is the top signal — enough tenure to show loyalty, still active enough to grow."
            return f"Relatively new client at {yrs:.0f} years — tenure is the key factor to watch. Early-stage relationships need consistent touchpoints."

        elif name == "nominee_ok":
            if val == 1: return f"Nominee is updated — the model treats this as a strong compliance and trust signal, lifting this client's priority score."
            return f"Nominee form not filled — this is the top risk flag for this client. Fixing this single item would measurably improve their score."

        elif name == "has_active_sip":
            if val == 1: return f"Active SIP in place — the model sees systematic investment as the clearest loyalty signal for this client profile."
            return f"No SIP running — for a client at this portfolio level, the absence of systematic investment is the strongest opportunity signal the model detects."

        elif name == "is_hni":
            if val == 1: return f"High-value client (₹50L+ portfolio) — the HNI flag is the dominant classification driver here. Warrants priority attention."
            return f"Portfolio below HNI threshold — crossing ₹50L would significantly shift this client's score in the model."

        elif name == "sip_ratio":
            ratio = round(val, 2)
            if ratio > 3: return f"SIP-to-portfolio ratio of {ratio}% is high — the model reads this as strong investment discipline relative to portfolio size."
            if ratio > 0: return f"SIP commitment at {ratio}% of portfolio — the model uses this ratio to assess proportional engagement. Below-average here."
            return f"Zero SIP-to-portfolio ratio is the primary signal — no systematic commitment relative to existing wealth."

        elif name == "has_mf":
            if val == 1: return f"Mutual fund exposure is the top driver — MF clients show higher engagement patterns in the training data, lifting this score."
            return f"No mutual fund products yet — the model sees MF absence as an opportunity gap for this client's profile and age."

        elif name == "has_lic":
            if val == 1: return f"LIC product history signals a long-term relationship orientation — the model weights this as a strong retention indicator."
            return f"No LIC product — at this client's age and portfolio level, the model flags LIC absence as an underserved need."

        elif name == "age":
            ag = round(val, 0)
            if ag > 58: return f"Age {ag:.0f} — senior client profile means the model prioritises estate planning and LIC maturity needs heavily in the scoring."
            if ag > 45: return f"Age {ag:.0f} is in the peak wealth-building zone — the model sees high conversion potential for growth products at this life stage."
            return f"Younger client at {ag:.0f} — the model factors in longer time horizon and higher SIP upsell potential for this age group."

        elif name == "age_sip_inter":
            if val > 0: return f"Age-SIP interaction is the top driver — this client's combination of age and active SIP creates a strong systematic investment signal."
            return f"Age-SIP interaction score is low — no active SIP at this life stage is the primary model concern."

        elif name == "portfolio_tenure":
            if val > 10: return f"High portfolio-tenure product is the key signal — large portfolio combined with long relationship indicates a deeply embedded client."
            return f"Portfolio-tenure combination is the top driver — building either AUM or relationship length would shift this score significantly."

        return f"Composite score from portfolio, recency, SIP, and tenure — no single dominant factor for this client profile."

    except Exception as e:
        return "Composite score based on portfolio size, contact recency, SIP consistency, and tenure signals."


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

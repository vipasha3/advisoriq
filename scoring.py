"""
scoring.py — Shared client scoring + processing logic
Used by app.py, api.py, and scheduler.py — single source of truth.
"""
import logging
import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

COLUMN_HINTS = {
    "name":        ["name","client","naam","clientname"],
    "age":         ["age","umur","ayu"],
    "portfolio":   ["portfolio","aum","value","investment","amount","total"],
    "sip":         ["sip","monthly","sipamount"],
    "lastContact": ["last","date","meeting","contact","interaction","lastdate"],
    "goal":        ["product","goal","scheme","type"],
    "tenure":      ["since","tenure","year","startyear","clientsince"],
    "nominee":     ["nominee","nomination"],
    "phone":       ["phone","mobile","contact","number","mob"],
}


def auto_map_columns(columns: list) -> dict:
    """Auto-detect which DataFrame column maps to which field."""
    mapping = {}
    for field, hints in COLUMN_HINTS.items():
        for col in columns:
            cl = col.lower().replace(" ","").replace("_","")
            for hint in hints:
                if hint in cl:
                    mapping[field] = col
                    break
            if field in mapping:
                break
    return mapping


def clean_number(v) -> str:
    try:
        return str(float(str(v).replace(",","").replace("₹","").strip()))
    except:
        return "0"


def clean_phone(v) -> str:
    if not v: return ""
    digits = "".join(filter(str.isdigit, str(v)))
    return ("91" + digits) if len(digits) == 10 else digits


def months_ago(d) -> float:
    if not d or str(d).strip() in ("","nan","None","NaT"): return 12
    try:
        dt = pd.to_datetime(str(d), dayfirst=True, errors="coerce")
        if pd.isna(dt): return 12
        return max(0, (datetime.datetime.now() - dt.to_pydatetime()).days / 30)
    except: return 12


def fmt_inr(v) -> str:
    try: n = float(str(v).replace(",","").replace("₹","") or 0)
    except: n = 0
    if n >= 1e7: return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5: return f"₹{n/1e5:.1f}L"
    if n >= 1e3: return f"₹{n/1e3:.0f}K"
    return f"₹{int(n)}"


def build_flags(client: dict) -> list:
    """Build alert flags for a client."""
    flags = []
    p   = float(str(client.get("portfolio","0")).replace(",","") or 0)
    sip = float(str(client.get("sip","0")).replace(",","") or 0)
    ma  = months_ago(client.get("lastContact",""))
    nom = str(client.get("nominee","")).lower().strip()

    if p > 5e6:        flags.append("High Value")
    if ma > 6:         flags.append("Inactive 6m+")
    if sip == 0 and p > 5e5: flags.append("No SIP")
    if nom == "no":    flags.append("No Nominee")
    return flags


def smart_dedup(clients: list) -> tuple[list, int]:
    """Merge duplicate clients by phone or name."""
    seen_phones = {}
    seen_names  = {}
    out = []
    merged = 0

    for c in clients:
        ph = c.get("phone","").strip()
        nm = c.get("name","").strip().lower()
        p  = float(str(c.get("portfolio","0")).replace(",","") or 0)

        if ph and len(ph) >= 10 and ph in seen_phones:
            ex = seen_phones[ph]
            ex_p = float(str(ex.get("portfolio","0")).replace(",","") or 0)
            if p > ex_p:
                idx = out.index(ex)
                out[idx] = c
                seen_phones[ph] = c
            merged += 1
        elif nm and nm in seen_names:
            ex = seen_names[nm]
            ex_p = float(str(ex.get("portfolio","0")).replace(",","") or 0)
            if p > ex_p:
                idx = out.index(ex)
                out[idx] = c
                seen_names[nm] = c
            merged += 1
        else:
            out.append(c)
            if ph and len(ph) >= 10: seen_phones[ph] = c
            if nm: seen_names[nm] = c

    return out, merged


def process_dataframe(df: pd.DataFrame, mapping: dict) -> list:
    """
    Convert a DataFrame to list of scored client dicts.
    Uses ML model for scoring, falls back to rules.
    """
    defaults = {
        "name":"", "age":"", "portfolio":"0", "sip":"0",
        "lastContact":"", "goal":"", "tenure":"2020",
        "nominee":"", "phone":""
    }

    raw_clients = []
    for _, row in df.iterrows():
        c = dict(defaults)
        for field, col in mapping.items():
            if col and col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() not in ("","nan","None"):
                    if field in ("portfolio","sip"):
                        c[field] = clean_number(val)
                    elif field == "phone":
                        c[field] = clean_phone(val)
                    else:
                        c[field] = str(val).strip()
        raw_clients.append(c)

    # Score with ML (batch — fast)
    try:
        from ml_model import predict_batch
        scored = predict_batch(raw_clients)
    except Exception as e:
        logger.warning(f"ML batch scoring failed: {e}. Using rules.")
        scored = [_rule_score_client(c) for c in raw_clients]

    # Add flags and finalize
    final = []
    for c in scored:
        c["flags"] = build_flags(c)
        # Churn risk flag
        if c.get("churn", 0) > 55:
            if "Leaving Risk" not in c["flags"]:
                c["flags"].append("Leaving Risk")
        final.append(c)

    # Deduplicate
    final, merged = smart_dedup(final)
    final.sort(key=lambda x: x.get("score", 0), reverse=True)

    logger.info(f"Processed {len(final)} clients ({merged} duplicates merged)")
    return final


def _rule_score_client(client: dict) -> dict:
    """Pure rule-based fallback scoring."""
    p   = float(str(client.get("portfolio","0")).replace(",","") or 0)
    sip = float(str(client.get("sip","0")).replace(",","") or 0)
    try: age = int(float(client.get("age") or 35))
    except: age = 35
    try:
        yr = int(float(str(client.get("tenure","2020")).strip()))
        ty = (2025 - yr) if yr > 1990 else yr
    except: ty = 3
    ma  = months_ago(client.get("lastContact",""))
    nom = str(client.get("nominee","")).lower().strip()

    s = 40
    if p > 8e6:s+=28
    elif p > 4e6:s+=20
    elif p > 1.5e6:s+=13
    elif p > 5e5:s+=7
    if sip > 20000:s+=18
    elif sip > 10000:s+=13
    elif sip > 3000:s+=8
    elif sip > 0:s+=4
    if ma < 1:s+=15
    elif ma < 3:s+=10
    elif ma < 6:s+=5
    elif ma > 12:s-=18
    elif ma > 6:s-=10
    if ty > 15:s+=15
    elif ty > 8:s+=10
    elif ty > 3:s+=5
    if nom == "no":s-=8

    score = max(0, min(100, round(s)))
    churn = min(100, max(0, round(
        (ma > 12) * 40 + (ma > 6) * 25 + (sip == 0) * 20 + (nom == "no") * 15
    )))
    conv  = min(95, max(5, round(score * 0.65 + (100 - churn) * 0.35)))

    return {**client, "score": score, "churn": churn, "conv": conv,
            "priority": "High" if score >= 70 else ("Medium" if score >= 45 else "Low")}

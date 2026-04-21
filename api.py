"""
api.py — FastAPI backend for AdvisorIQ
Runs alongside Streamlit as a separate service on port 8000.
Enables mobile app, WhatsApp bot, and CRM integrations in future.

Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import jwt
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn pydantic python-jose")

from config import cfg
import database as db
import ml_model as ml

if not FASTAPI_AVAILABLE:
    print("FastAPI not available. Install with: pip install 'fastapi[all]'")
    sys.exit(1)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AdvisorIQ API",
    description="REST API for AdvisorIQ — Financial advisor intelligence platform",
    version=cfg.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str
    company: str
    role: str = "advisor"

class PredictRequest(BaseModel):
    portfolio: float = 0
    sip: float = 0
    age: int = 40
    months_inactive: float = 3
    nominee: str = "Yes"
    goal: str = "MF"
    tenure: int = 2018

class SheetsRequest(BaseModel):
    sheets_url: str

class WhatsAppRequest(BaseModel):
    client_name: str
    phone: str
    template: str = "checkin"

class UpgradePlanRequest(BaseModel):
    plan: str
    razorpay_order_id: str = ""
    razorpay_payment_id: str = ""
    razorpay_signature: str = ""


# ── Auth ──────────────────────────────────────────────────────────────────────
def create_token(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7),
    }
    return jwt.encode(payload, cfg.API_SECRET_KEY, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, cfg.API_SECRET_KEY, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    user = db.get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app": cfg.APP_NAME,
        "version": cfg.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}


@app.post("/auth/register")
def register(req: RegisterRequest):
    ok, msg = db.create_user(req.username, req.password, req.full_name, req.company, req.role)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg, "username": req.username}


@app.post("/auth/login")
def login(req: LoginRequest):
    user = db.login_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(user["id"], user["username"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "company": user["company"],
            "role": user["role"],
            "plan": user["plan"],
        }
    }


@app.get("/clients")
def get_clients(
    search: str = "",
    priority: str = "",
    min_aum: float = 0,
    max_aum: float = 999999999,
    user = Depends(get_current_user)
):
    clients = db.load_clients(user["id"])
    # Apply filters
    if search:
        sq = search.lower()
        clients = [c for c in clients if sq in c.get("name","").lower() or sq in c.get("goal","").lower()]
    if priority:
        clients = [c for c in clients if c.get("priority","").lower() == priority.lower()]
    if min_aum > 0 or max_aum < 999999999:
        clients = [c for c in clients
                   if min_aum <= float(str(c.get("portfolio","0")).replace(",","") or 0) <= max_aum]
    return {"clients": clients, "total": len(clients)}


@app.get("/clients/summary")
def get_summary(user = Depends(get_current_user)):
    clients = db.load_clients(user["id"])
    total_aum = sum(float(str(c.get("portfolio","0")).replace(",","") or 0) for c in clients)
    return {
        "total_clients": len(clients),
        "total_aum": total_aum,
        "high_priority": sum(1 for c in clients if c.get("priority") == "High"),
        "churn_risk": sum(1 for c in clients if c.get("churn",0) > 50),
        "no_sip": sum(1 for c in clients if "No SIP" in c.get("flags",[])),
        "no_nominee": sum(1 for c in clients if "No Nominee" in c.get("flags",[])),
        "hni_count": sum(1 for c in clients if "High Value" in c.get("flags",[])),
    }


@app.post("/predict")
def predict_single(req: PredictRequest, user = Depends(get_current_user)):
    """Score a single client with ML model."""
    client = {
        "portfolio": req.portfolio,
        "sip": req.sip,
        "age": req.age,
        "lastContact": "",
        "nominee": req.nominee,
        "goal": req.goal,
        "tenure": req.tenure,
    }
    result = ml.predict_client(client)
    return result


@app.post("/sheets/connect")
def connect_sheets(req: SheetsRequest, user = Depends(get_current_user)):
    """Connect Google Sheets URL for auto-sync."""
    from sheets_sync import validate_sheets_url
    valid, msg = validate_sheets_url(req.sheets_url)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    db.update_sheets_url(user["id"], req.sheets_url)
    return {"message": msg, "sheets_url": req.sheets_url}


@app.post("/sheets/sync")
def trigger_sync(user = Depends(get_current_user)):
    """Manually trigger a Google Sheets sync."""
    from sheets_sync import sync_user_sheets
    import ml_model as ml
    status = sync_user_sheets(user["id"], user.get("sheets_url",""), db, ml)
    return status


@app.post("/whatsapp/send")
def send_whatsapp_msg(req: WhatsAppRequest, user = Depends(get_current_user)):
    """Send a WhatsApp message to a client."""
    from subscription import can_use_whatsapp
    if not can_use_whatsapp(user.get("plan","free")):
        raise HTTPException(status_code=403, detail="WhatsApp requires Growth or Firm plan.")
    from whatsapp import send_whatsapp, build_message, TEMPLATES
    client = {"name": req.client_name, "phone": req.phone}
    msg = build_message(req.template, client, user)
    result = send_whatsapp(req.phone, msg, dry_run=False)
    return result


@app.get("/model/info")
def model_info(user = Depends(get_current_user)):
    """Return ML model metadata."""
    meta = ml.get_model_meta()
    return meta


@app.post("/subscription/upgrade")
def upgrade_subscription(req: UpgradePlanRequest, user = Depends(get_current_user)):
    """Upgrade user plan after payment."""
    valid_plans = ["starter", "growth", "firm"]
    if req.plan not in valid_plans:
        raise HTTPException(status_code=400, detail=f"Invalid plan. Choose: {valid_plans}")
    db.upgrade_plan(user["id"], req.plan, req.razorpay_order_id)
    return {"message": f"Upgraded to {req.plan}", "plan": req.plan}


@app.get("/subscription/plans")
def get_plans():
    """Return all available plans."""
    from config import cfg
    return cfg.PLAN_LIMITS


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    db.init_db()
    ml.train_models()
    logger.info(f"Starting AdvisorIQ API on {cfg.API_HOST}:{cfg.API_PORT}")
    uvicorn.run("api:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=cfg.DEBUG)

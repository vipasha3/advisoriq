"""
config.py — Central configuration for AdvisorIQ
All secrets come from environment variables (never hardcode!)
"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///advisoriq.db"          # SQLite fallback for local dev
        # Production: "postgresql://user:pass@host:5432/advisoriq"
    )
    
    # ── Google Sheets ─────────────────────────────────────────
    GOOGLE_CREDENTIALS_FILE: str = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    SHEETS_SYNC_INTERVAL_MIN: int = int(os.getenv("SHEETS_SYNC_INTERVAL_MIN", "5"))
    
    # ── WhatsApp (Twilio) ─────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    
    # ── FastAPI ───────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "change-this-in-production-please")
    
    # ── ML Model ──────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/advisoriq_model.pkl")
    MODEL_RETRAIN_DAYS: int = int(os.getenv("MODEL_RETRAIN_DAYS", "7"))
    
    # ── Subscription Limits ───────────────────────────────────
    PLAN_LIMITS: dict = None
    
    # ── App ───────────────────────────────────────────────────
    APP_NAME: str = "AdvisorIQ"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    def __post_init__(self):
        self.PLAN_LIMITS = {
            "free":    {"clients": 25,    "price_inr": 0,     "whatsapp": False, "api": False},
            "starter": {"clients": 100,   "price_inr": 1999,  "whatsapp": False, "api": False},
            "growth":  {"clients": 500,   "price_inr": 4999,  "whatsapp": True,  "api": False},
            "firm":    {"clients": 99999, "price_inr": 12999, "whatsapp": True,  "api": True},
        }

# Singleton
cfg = Config()

"""
database.py — Database layer for AdvisorIQ
Supports PostgreSQL (production) and SQLite (local dev).
Uses raw SQL — no ORM overhead, easy to understand and debug.
"""
import sqlite3
import json
import datetime
import logging
import os
import bcrypt
from contextlib import contextmanager
from config import cfg

logger = logging.getLogger(__name__)

# ── Detect backend ────────────────────────────────────────────────────────────
USE_POSTGRES = cfg.DATABASE_URL.startswith("postgresql")

if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        logger.info("Using PostgreSQL backend")
    except ImportError:
        logger.warning("psycopg2 not installed — falling back to SQLite")
        USE_POSTGRES = False

SQLITE_PATH = cfg.DATABASE_URL.replace("sqlite:///", "") if not USE_POSTGRES else "advisoriq.db"


# ── Connection ────────────────────────────────────────────────────────────────
@contextmanager
def get_conn():
    """Get a database connection (PostgreSQL or SQLite)."""
    if USE_POSTGRES:
        conn = psycopg2.connect(cfg.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name     TEXT,
    company       TEXT,
    role          TEXT DEFAULT 'advisor',
    plan          TEXT DEFAULT 'free',
    sheets_url    TEXT,
    sheets_last_synced TEXT,
    created_at    TEXT,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS clients (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT,
    age           TEXT,
    portfolio     TEXT DEFAULT '0',
    sip           TEXT DEFAULT '0',
    last_contact  TEXT,
    goal          TEXT,
    tenure        TEXT,
    nominee       TEXT,
    phone         TEXT,
    score         INTEGER DEFAULT 0,
    churn         INTEGER DEFAULT 0,
    conv          INTEGER DEFAULT 50,
    priority      TEXT DEFAULT 'Low',
    flags         TEXT DEFAULT '[]',
    source        TEXT DEFAULT 'upload',
    uploaded_at   TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    plan          TEXT NOT NULL,
    status        TEXT DEFAULT 'active',
    starts_at     TEXT,
    expires_at    TEXT,
    razorpay_id   TEXT,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER,
    action        TEXT,
    detail        TEXT,
    ip_address    TEXT,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS whatsapp_log (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER,
    client_name   TEXT,
    phone         TEXT,
    message_type  TEXT,
    status        TEXT,
    sent_at       TEXT
);
"""

# SQLite uses INTEGER PRIMARY KEY instead of SERIAL
SCHEMA_SQLITE = SCHEMA_SQL.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")


def init_db():
    """Initialize database schema with safe migrations."""
    schema = SCHEMA_SQL if USE_POSTGRES else SCHEMA_SQLITE
    with get_conn() as conn:
        c = conn.cursor()
        for statement in schema.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                try:
                    c.execute(stmt)
                except Exception as e:
                    logger.debug(f"Schema statement skipped: {e}")

        # ── Safe migrations: add missing columns to existing tables ──
        migrations = [
            "ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free'",
            "ALTER TABLE users ADD COLUMN sheets_url TEXT",
            "ALTER TABLE users ADD COLUMN sheets_last_synced TEXT",
            "ALTER TABLE users ADD COLUMN last_login TEXT",
            "ALTER TABLE clients ADD COLUMN conv INTEGER DEFAULT 50",
            "ALTER TABLE clients ADD COLUMN source TEXT DEFAULT 'upload'",
            "ALTER TABLE clients ADD COLUMN updated_at TEXT",
        ]
        for m in migrations:
            try:
                c.execute(m)
                logger.info(f"Migration applied: {m}")
            except Exception:
                pass  # Column already exists — skip silently

    logger.info(f"Database initialized ({'PostgreSQL' if USE_POSTGRES else 'SQLite'})")


# ── Password ──────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        # Migration fallback for old sha256 hashes
        import hashlib
        return hashlib.sha256(pw.encode()).hexdigest() == hashed


# ── Users ─────────────────────────────────────────────────────────────────────
def create_user(username: str, password: str, full_name: str, company: str, role: str = "advisor") -> tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users(username,password_hash,full_name,company,role,plan,created_at) VALUES(?,?,?,?,?,?,?)"
                if not USE_POSTGRES else
                "INSERT INTO users(username,password_hash,full_name,company,role,plan,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (username.strip(), hash_password(password), full_name.strip(),
                 company.strip(), role, "free", datetime.datetime.now().isoformat())
            )
        logger.info(f"User created: {username}")
        return True, "Account created successfully."
    except Exception as e:
        if "UNIQUE" in str(e) or "unique" in str(e).lower():
            return False, "Username already taken. Choose a different one."
        logger.error(f"User creation error: {e}")
        return False, f"Error: {e}"


def login_user(username: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict or None."""
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"SELECT * FROM users WHERE username={ph}", (username,))
        row = c.fetchone()
        if row and verify_password(password, row["password_hash"]):
            # Update last login
            c.execute(
                f"UPDATE users SET last_login={ph} WHERE id={ph}",
                (datetime.datetime.now().isoformat(), row["id"])
            )
            return dict(row)
    return None


def get_user(user_id: int) -> dict | None:
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"SELECT * FROM users WHERE id={ph}", (user_id,))
        row = c.fetchone()
        return dict(row) if row else None


def update_sheets_url(user_id: int, sheets_url: str):
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"UPDATE users SET sheets_url={ph} WHERE id={ph}", (sheets_url, user_id))

def save_session_token(user_id, token):
    pass

def get_user_by_token(token):
    pass


# ── Clients ───────────────────────────────────────────────────────────────────
def save_clients(user_id: int, clients: list, source: str = "upload"):
    """Save/replace all clients for a user."""
    ph = "?" if not USE_POSTGRES else "%s"
    now = datetime.datetime.now().isoformat()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"DELETE FROM clients WHERE user_id={ph}", (user_id,))
        for cl in clients:
            c.execute(
                f"""INSERT INTO clients(user_id,name,age,portfolio,sip,last_contact,goal,tenure,
                nominee,phone,score,churn,conv,priority,flags,source,uploaded_at,updated_at)
                VALUES({','.join([ph]*18)})""",
                (user_id, cl.get("name",""), cl.get("age",""),
                 cl.get("portfolio","0"), cl.get("sip","0"),
                 cl.get("lastContact",""), cl.get("goal",""),
                 cl.get("tenure",""), cl.get("nominee",""), cl.get("phone",""),
                 cl.get("score",0), cl.get("churn",0), cl.get("conv",50),
                 cl.get("priority","Low"), json.dumps(cl.get("flags",[])),
                 source, now, now)
            )
    logger.info(f"Saved {len(clients)} clients for user {user_id}")


def load_clients(user_id: int) -> list:
    """Load all clients for a user, sorted by score."""
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            f"SELECT * FROM clients WHERE user_id={ph} ORDER BY score DESC",
            (user_id,)
        )
        rows = c.fetchall()
    return [
        {**dict(r), "lastContact": r["last_contact"],
         "flags": json.loads(r["flags"]) if r["flags"] else []}
        for r in rows
    ]


def get_client_count(user_id: int) -> int:
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) as cnt FROM clients WHERE user_id={ph}", (user_id,))
        return c.fetchone()["cnt"]


# ── Subscriptions ─────────────────────────────────────────────────────────────
def get_user_plan(user_id: int) -> str:
    """Return current plan name for user."""
    user = get_user(user_id)
    return user["plan"] if user else "free"


def upgrade_plan(user_id: int, plan: str, razorpay_id: str = ""):
    """Upgrade user plan (called after payment confirmation)."""
    ph = "?" if not USE_POSTGRES else "%s"
    now = datetime.datetime.now()
    expires = (now + datetime.timedelta(days=30)).isoformat()
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(f"UPDATE users SET plan={ph} WHERE id={ph}", (plan, user_id))
        c.execute(
            f"""INSERT INTO subscriptions(user_id,plan,status,starts_at,expires_at,razorpay_id,created_at)
            VALUES({','.join([ph]*7)})""",
            (user_id, plan, "active", now.isoformat(), expires, razorpay_id, now.isoformat())
        )
    logger.info(f"User {user_id} upgraded to {plan}")


# ── Audit Log ─────────────────────────────────────────────────────────────────
def log_action(user_id: int, action: str, detail: str = "", ip: str = ""):
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            f"INSERT INTO audit_log(user_id,action,detail,ip_address,created_at) VALUES({','.join([ph]*5)})",
            (user_id, action, detail, ip, datetime.datetime.now().isoformat())
        )


# ── WhatsApp Log ──────────────────────────────────────────────────────────────
def log_whatsapp(user_id: int, client_name: str, phone: str, msg_type: str, status: str):
    ph = "?" if not USE_POSTGRES else "%s"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            f"INSERT INTO whatsapp_log(user_id,client_name,phone,message_type,status,sent_at) VALUES({','.join([ph]*6)})",
            (user_id, client_name, phone, msg_type, status, datetime.datetime.now().isoformat())
        )

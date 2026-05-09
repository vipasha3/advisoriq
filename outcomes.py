"""
outcomes.py — Call outcome tracking for AdvisorIQ
Tracks what happened after each client interaction.
This data trains the real ML model over time.
"""
import sqlite3
import datetime
import json
import logging

logger = logging.getLogger(__name__)

DB = "advisoriq.db"

OUTCOME_TYPES = {
    "called_invested":   {"label": "Called — Invested ✓",     "color": "#3fb950", "weight": 1.0},
    "called_interested": {"label": "Called — Interested",      "color": "#58a6ff", "weight": 0.6},
    "called_no_action":  {"label": "Called — No action",       "color": "#d29922", "weight": 0.2},
    "no_answer":         {"label": "No answer",                "color": "#8b949e", "weight": 0.0},
    "called_not_now":    {"label": "Called — Not now",         "color": "#d29922", "weight": 0.1},
    "called_refused":    {"label": "Called — Not interested",  "color": "#f85149", "weight": -0.3},
}


def init_outcome_tables():
    """Add outcome tracking tables to existing DB."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS call_logs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        client_name  TEXT,
        client_phone TEXT,
        outcome      TEXT,
        note         TEXT,
        score_at_call INTEGER,
        portfolio_at_call TEXT,
        called_at    TEXT,
        created_at   TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS client_outcomes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        client_name  TEXT,
        total_calls  INTEGER DEFAULT 0,
        conversions  INTEGER DEFAULT 0,
        last_outcome TEXT,
        last_called  TEXT,
        conversion_rate REAL DEFAULT 0.0,
        updated_at   TEXT
    )""")

    conn.commit()
    conn.close()
    logger.info("Outcome tables ready")


def log_outcome(user_id: int, client_name: str, phone: str,
                outcome: str, note: str = "", score: int = 0, portfolio: str = "0"):
    """Save a call outcome."""
    now = datetime.datetime.now().isoformat()
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    # Log the call
    c.execute("""INSERT INTO call_logs
        (user_id,client_name,client_phone,outcome,note,score_at_call,portfolio_at_call,called_at,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (user_id, client_name, phone, outcome, note, score, portfolio, now, now))

    # Update summary
    c.execute("SELECT * FROM client_outcomes WHERE user_id=? AND client_name=?",
              (user_id, client_name))
    existing = c.fetchone()

    is_conversion = outcome == "called_invested"

    if existing:
        total = existing[4] + 1
        convs = existing[5] + (1 if is_conversion else 0)
        rate = round(convs / total, 3)
        c.execute("""UPDATE client_outcomes
            SET total_calls=?, conversions=?, last_outcome=?, last_called=?,
                conversion_rate=?, updated_at=?
            WHERE user_id=? AND client_name=?""",
            (total, convs, outcome, now, rate, now, user_id, client_name))
    else:
        c.execute("""INSERT INTO client_outcomes
            (user_id,client_name,total_calls,conversions,last_outcome,last_called,conversion_rate,updated_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (user_id, client_name, 1, 1 if is_conversion else 0,
             outcome, now, 1.0 if is_conversion else 0.0, now))

    conn.commit()
    conn.close()
    logger.info(f"Outcome logged: {client_name} → {outcome}")


def get_client_history(user_id: int, client_name: str) -> dict:
    """Get call history summary for one client."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM client_outcomes WHERE user_id=? AND client_name=?",
              (user_id, client_name))
    summary = c.fetchone()

    c.execute("""SELECT outcome, note, called_at FROM call_logs
               WHERE user_id=? AND client_name=?
               ORDER BY called_at DESC LIMIT 5""",
              (user_id, client_name))
    logs = c.fetchall()
    conn.close()

    if not summary:
        return {"total_calls": 0, "conversions": 0, "conversion_rate": 0.0,
                "last_outcome": None, "last_called": None, "history": []}

    return {
        "total_calls": summary["total_calls"],
        "conversions": summary["conversions"],
        "conversion_rate": summary["conversion_rate"],
        "last_outcome": summary["last_outcome"],
        "last_called": summary["last_called"],
        "history": [{"outcome": r["outcome"], "note": r["note"],
                     "called_at": r["called_at"]} for r in logs]
    }


def get_outcome_adjusted_score(base_score: int, client_name: str, user_id: int) -> int:
    """
    Adjust ML score based on real outcome history.
    Client jo past ma invest kari chhe → score boost.
    Client jo refused chhe → score reduce.
    """
    history = get_client_history(user_id, client_name)
    if history["total_calls"] == 0:
        return base_score

    rate = history["conversion_rate"]
    last = history["last_outcome"]

    adjustment = 0

    # Conversion rate boost
    if rate > 0.5:   adjustment += 15
    elif rate > 0.2: adjustment += 8
    elif rate == 0 and history["total_calls"] > 2: adjustment -= 10

    # Last outcome signal
    if last == "called_invested":   adjustment += 10
    elif last == "called_refused":  adjustment -= 12
    elif last == "called_interested": adjustment += 5
    elif last == "no_answer":        adjustment -= 3

    return max(0, min(100, base_score + adjustment))


def get_all_outcomes(user_id: int) -> list:
    """Get all outcome summaries for a user (for ML retraining)."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM client_outcomes WHERE user_id=? ORDER BY updated_at DESC",
              (user_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_outcome_stats(user_id: int) -> dict:
    """Summary stats for dashboard."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as cnt FROM call_logs WHERE user_id=?", (user_id,))
    total_calls = c.fetchone()["cnt"]

    c.execute("SELECT COUNT(*) as cnt FROM call_logs WHERE user_id=? AND outcome='called_invested'",
              (user_id,))
    total_converted = c.fetchone()["cnt"]

    c.execute("""SELECT outcome, COUNT(*) as cnt FROM call_logs
               WHERE user_id=? GROUP BY outcome""", (user_id,))
    breakdown = {r["outcome"]: r["cnt"] for r in c.fetchall()}

    conn.close()

    return {
        "total_calls": total_calls,
        "total_converted": total_converted,
        "conversion_rate": round(total_converted / max(total_calls, 1) * 100, 1),
        "breakdown": breakdown,
        "has_enough_data": total_calls >= 20,
    }


def get_best_calling_patterns(user_id: int) -> str:
    """
    Analyze outcomes and return a human insight.
    Once enough data exists, this becomes genuinely intelligent.
    """
    stats = get_outcome_stats(user_id)

    if stats["total_calls"] < 5:
        return ""

    if stats["total_calls"] < 20:
        return (f"Early data: {stats['total_calls']} calls logged, "
                f"{stats['total_converted']} converted. "
                f"Keep tracking — insights will sharpen at 20+ calls.")

    rate = stats["conversion_rate"]
    breakdown = stats["breakdown"]
    invested = breakdown.get("called_invested", 0)
    refused = breakdown.get("called_refused", 0)
    no_ans = breakdown.get("no_answer", 0)

    if rate > 30:
        return (f"Strong conversion rate of {rate}% across {stats['total_calls']} calls. "
                f"{invested} clients invested after contact. Your follow-up timing is working.")
    elif rate > 15:
        return (f"Moderate conversion at {rate}%. "
                f"{no_ans} calls went unanswered — try different times for these clients.")
    else:
        return (f"Low conversion rate of {rate}% — "
                f"{refused} clients declined. Consider revisiting messaging approach "
                f"or focus on higher-score clients first.")

"""
subscription.py — Subscription management for AdvisorIQ
Handles plan limits, upgrade flows, and Razorpay integration.
"""
import logging
import datetime
from config import cfg

logger = logging.getLogger(__name__)

PLANS = cfg.PLAN_LIMITS  # Loaded from config


def get_plan_info(plan: str) -> dict:
    return PLANS.get(plan, PLANS["free"])


def check_client_limit(user_id: int, current_count: int, plan: str) -> tuple[bool, str]:
    """
    Check if user can add more clients.
    Returns (allowed, message).
    """
    limit = PLANS.get(plan, PLANS["free"])["clients"]
    if current_count >= limit:
        return False, (
            f"You have reached the {plan.title()} plan limit of {limit} clients. "
            f"Upgrade to add more."
        )
    return True, ""


def can_use_whatsapp(plan: str) -> bool:
    return PLANS.get(plan, PLANS["free"]).get("whatsapp", False)


def can_use_api(plan: str) -> bool:
    return PLANS.get(plan, PLANS["free"]).get("api", False)


def get_plan_badge_html(plan: str) -> str:
    """Return HTML badge for a plan."""
    colors = {
        "free":    ("#6e7681", "#21262d"),
        "starter": ("#58a6ff", "#1c2d41"),
        "growth":  ("#3fb950", "#1a2f1e"),
        "firm":    ("#d29922", "#2e2008"),
    }
    tc, bg = colors.get(plan, colors["free"])
    return (
        f'<span style="font-size:11px;padding:2px 9px;border-radius:10px;'
        f'background:{bg};color:{tc};border:1px solid {tc}44;'
        f'font-family:JetBrains Mono,monospace;font-weight:600">'
        f'{plan.upper()}</span>'
    )


def get_upgrade_prompt(current_plan: str, feature: str) -> str:
    """Return upgrade message for a locked feature."""
    next_plan = {
        "free": "Starter (₹1,999/mo)",
        "starter": "Growth (₹4,999/mo)",
        "growth": "Firm (₹12,999/mo)",
    }.get(current_plan, "a higher plan")

    messages = {
        "whatsapp":  f"WhatsApp automation requires {next_plan}.",
        "api":       f"API access requires Firm plan (₹12,999/mo).",
        "clients":   f"More clients require {next_plan}.",
        "sheets":    f"Google Sheets sync is available on all paid plans.",
        "export":    f"Export is available on Starter and above.",
    }
    return messages.get(feature, f"This feature requires {next_plan}.")


# ── Razorpay integration ──────────────────────────────────────────────────────
def create_razorpay_order(plan: str, user_id: int) -> dict:
    """
    Create a Razorpay order for plan upgrade.
    Returns order dict with order_id for frontend.
    """
    try:
        import razorpay
        import os
        key_id = os.getenv("RAZORPAY_KEY_ID", "")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
        if not key_id or not key_secret:
            return {"error": "Razorpay not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."}

        client = razorpay.Client(auth=(key_id, key_secret))
        amount_inr = PLANS.get(plan, PLANS["free"])["price_inr"]
        order = client.order.create({
            "amount": amount_inr * 100,  # Razorpay uses paise
            "currency": "INR",
            "receipt": f"advisoriq_{user_id}_{plan}_{datetime.datetime.now().strftime('%Y%m%d')}",
            "notes": {"user_id": str(user_id), "plan": plan},
        })
        logger.info(f"Razorpay order created: {order['id']} for user {user_id} ({plan})")
        return {"order_id": order["id"], "amount": amount_inr, "plan": plan}
    except ImportError:
        return {"error": "razorpay not installed. Run: pip install razorpay"}
    except Exception as e:
        logger.error(f"Razorpay error: {e}")
        return {"error": str(e)}


def verify_razorpay_payment(order_id: str, payment_id: str, signature: str) -> bool:
    """Verify Razorpay payment signature."""
    try:
        import razorpay, os, hmac, hashlib
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
        generated = hmac.new(
            key_secret.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256
        ).hexdigest()
        return generated == signature
    except Exception as e:
        logger.error(f"Payment verification error: {e}")
        return False


def plan_comparison_html() -> str:
    """Return HTML table comparing all plans."""
    rows = ""
    features = [
        ("Max clients", lambda p: str(PLANS[p]["clients"]) if PLANS[p]["clients"] < 99999 else "Unlimited"),
        ("Price/month", lambda p: f"₹{PLANS[p]['price_inr']:,}" if PLANS[p]['price_inr'] > 0 else "Free"),
        ("ML scoring", lambda p: "✓"),
        ("Priority rankings", lambda p: "✓"),
        ("Event intelligence", lambda p: "✓"),
        ("Google Sheets sync", lambda p: "✓" if p != "free" else "—"),
        ("Excel export", lambda p: "✓" if p != "free" else "—"),
        ("WhatsApp automation", lambda p: "✓" if PLANS[p]["whatsapp"] else "—"),
        ("API access", lambda p: "✓" if PLANS[p]["api"] else "—"),
        ("Background jobs", lambda p: "✓" if p in ("growth","firm") else "—"),
    ]
    for label, fn in rows:
        cells = "".join(f"<td>{fn(p)}</td>" for p in ["free","starter","growth","firm"])
        rows += f"<tr><td>{label}</td>{cells}</tr>"
    return f"""
    <table style="width:100%;font-size:13px;border-collapse:collapse">
      <thead><tr>
        <th style="text-align:left;padding:8px;border-bottom:1px solid #30363d">Feature</th>
        <th style="padding:8px;border-bottom:1px solid #30363d;color:#6e7681">Free</th>
        <th style="padding:8px;border-bottom:1px solid #30363d;color:#58a6ff">Starter</th>
        <th style="padding:8px;border-bottom:1px solid #30363d;color:#3fb950">Growth</th>
        <th style="padding:8px;border-bottom:1px solid #30363d;color:#d29922">Firm</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""

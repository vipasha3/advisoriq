"""
whatsapp.py — Real WhatsApp automation for AdvisorIQ
Uses Twilio WhatsApp Business API.

Setup:
1. Create Twilio account at twilio.com
2. Enable WhatsApp sandbox (free) or Business API (paid)
3. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM in .env
"""
import logging
import datetime
from config import cfg

logger = logging.getLogger(__name__)

# Message templates — human sounding, not generic
TEMPLATES = {
    "checkin": (
        "Hi {first_name}! Hope you're doing well. "
        "I've been reviewing your portfolio and there are a couple of developments "
        "I'd like to walk you through. Would a quick 20-minute call this week work for you? "
        "— {advisor_name}, {company}"
    ),
    "sip_proposal": (
        "Hi {first_name}! Based on your current portfolio of {portfolio}, "
        "I've prepared a personalised SIP projection that could make a real difference "
        "over the next 10 years. The numbers are quite compelling — "
        "can we find 15 minutes to go through it? "
        "— {advisor_name}, {company}"
    ),
    "portfolio_review": (
        "Hi {first_name}! Your annual portfolio review is due. "
        "Given the current market conditions, I want to make sure your investments "
        "are positioned right for the year ahead. I've already done the analysis. "
        "When works best for a quick call? "
        "— {advisor_name}, {company}"
    ),
    "nominee_update": (
        "Hi {first_name}! As part of our annual client care review, "
        "I noticed your nominee details may need updating. "
        "This is important to protect your family's interests and it takes under 10 minutes. "
        "Can I help you with this? "
        "— {advisor_name}, {company}"
    ),
    "inactivity_alert": (
        "Hi {first_name}! It's been a while since we connected. "
        "I wanted to personally check in — markets have moved and "
        "I want to make sure your portfolio is positioned well. "
        "Would love to reconnect. "
        "— {advisor_name}, {company}"
    ),
    "churn_save": (
        "Hi {first_name}! Hope all is well. "
        "I was thinking about your financial goals and wanted to touch base. "
        "No agenda — just a quick portfolio health check to make sure "
        "everything is on track. Would that work for you? "
        "— {advisor_name}, {company}"
    ),
}


def _get_twilio_client():
    """Return authenticated Twilio client."""
    try:
        from twilio.rest import Client
        if not cfg.TWILIO_ACCOUNT_SID or not cfg.TWILIO_AUTH_TOKEN:
            raise ValueError(
                "Twilio credentials not set.\n"
                "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your .env file."
            )
        return Client(cfg.TWILIO_ACCOUNT_SID, cfg.TWILIO_AUTH_TOKEN)
    except ImportError:
        raise ImportError("twilio not installed. Run: pip install twilio")


def format_phone(phone: str) -> str:
    """Convert phone to WhatsApp format: whatsapp:+91XXXXXXXXXX"""
    if not phone:
        return ""
    digits = "".join(filter(str.isdigit, str(phone)))
    if len(digits) == 10:
        digits = "91" + digits
    if not digits.startswith("91"):
        digits = "91" + digits
    return f"whatsapp:+{digits}"


def build_message(template_key: str, client: dict, advisor: dict) -> str:
    """Build personalised message from template."""
    first_name = (client.get("name", "") or "").split()[0] or "there"
    portfolio_str = ""
    try:
        p = float(str(client.get("portfolio", 0)).replace(",","").replace("₹",""))
        if p >= 1e7:   portfolio_str = f"₹{p/1e7:.1f}Cr"
        elif p >= 1e5: portfolio_str = f"₹{p/1e5:.1f}L"
        else:          portfolio_str = f"₹{int(p)}"
    except: portfolio_str = "your portfolio"

    template = TEMPLATES.get(template_key, TEMPLATES["checkin"])
    return template.format(
        first_name=first_name,
        portfolio=portfolio_str,
        advisor_name=advisor.get("full_name", "Your Advisor"),
        company=advisor.get("company", ""),
    )


def send_whatsapp(phone: str, message: str, dry_run: bool = False) -> dict:
    """
    Send a WhatsApp message via Twilio.
    dry_run=True: simulate without actually sending (for testing).
    Returns dict with status info.
    """
    wa_phone = format_phone(phone)
    if not wa_phone:
        return {"success": False, "error": "Invalid phone number", "sid": ""}

    if dry_run:
        logger.info(f"[DRY RUN] Would send to {wa_phone}: {message[:60]}...")
        return {
            "success": True,
            "sid": "DRY_RUN_SID",
            "status": "simulated",
            "to": wa_phone,
            "dry_run": True,
        }

    try:
        client = _get_twilio_client()
        msg = client.messages.create(
            from_=cfg.TWILIO_WHATSAPP_FROM,
            to=wa_phone,
            body=message,
        )
        logger.info(f"WhatsApp sent to {wa_phone} — SID: {msg.sid}")
        return {
            "success": True,
            "sid": msg.sid,
            "status": msg.status,
            "to": wa_phone,
        }
    except Exception as e:
        logger.error(f"WhatsApp send failed to {wa_phone}: {e}")
        return {"success": False, "error": str(e), "sid": ""}


def send_bulk(clients: list, template_key: str, advisor: dict,
              dry_run: bool = True) -> list:
    """
    Send WhatsApp messages to a list of clients.
    Always dry_run=True by default — set False only for live sends.
    Returns list of result dicts.
    """
    results = []
    for c in clients:
        phone = c.get("phone", "")
        if not phone:
            results.append({
                "name": c.get("name", ""),
                "success": False,
                "error": "No phone number",
            })
            continue
        msg = build_message(template_key, c, advisor)
        result = send_whatsapp(phone, msg, dry_run=dry_run)
        result["name"] = c.get("name", "")
        result["message"] = msg
        results.append(result)

        # Log to database
        try:
            from database import log_whatsapp
            log_whatsapp(
                advisor.get("id", 0),
                c.get("name", ""),
                phone,
                template_key,
                "sent" if result["success"] else "failed"
            )
        except Exception:
            pass

    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"Bulk WhatsApp: {success_count}/{len(results)} sent (dry_run={dry_run})")
    return results


def get_whatsapp_link(phone: str, message: str) -> str:
    """Generate wa.me link for manual send (no API needed)."""
    digits = "".join(filter(str.isdigit, str(phone)))
    if len(digits) == 10:
        digits = "91" + digits
    encoded = message.replace("\n", "%0A").replace(" ", "%20")
    return f"https://wa.me/{digits}?text={encoded}"


def check_twilio_configured() -> tuple[bool, str]:
    """Check if Twilio is properly configured."""
    if not cfg.TWILIO_ACCOUNT_SID:
        return False, "TWILIO_ACCOUNT_SID not set in environment"
    if not cfg.TWILIO_AUTH_TOKEN:
        return False, "TWILIO_AUTH_TOKEN not set in environment"
    try:
        _get_twilio_client()
        return True, "Twilio configured correctly"
    except Exception as e:
        return False, str(e)

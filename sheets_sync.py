"""
sheets_sync.py — Google Sheets real-time sync for AdvisorIQ
Kartik updates his Google Sheet → app detects → insights refresh automatically.

Setup:
1. Google Cloud Console → Enable Google Sheets API
2. Create Service Account → Download credentials.json
3. Share your Google Sheet with the service account email
4. Set GOOGLE_CREDENTIALS_FILE=credentials.json in .env
"""
import logging
import datetime
import hashlib
import json
import os
import pandas as pd
from config import cfg

logger = logging.getLogger(__name__)


def _get_gsheet_client():
    """Return authenticated gspread client."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds_file = cfg.GOOGLE_CREDENTIALS_FILE
        if not os.path.exists(creds_file):
            raise FileNotFoundError(
                f"Google credentials not found: {creds_file}\n"
                "Download from Google Cloud Console → Service Accounts → Keys"
            )
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
        return gspread.authorize(creds)
    except ImportError:
        raise ImportError(
            "gspread not installed. Run: pip install gspread google-auth"
        )


def validate_sheets_url(url: str) -> tuple[bool, str]:
    """Check if a Google Sheets URL is valid and accessible."""
    if not url or not url.strip():
        return False, "URL cannot be empty."
    if "docs.google.com/spreadsheets" not in url:
        return False, "Please enter a valid Google Sheets URL."
    try:
        gc = _get_gsheet_client()
        sh = gc.open_by_url(url.strip())
        ws = sh.get_worksheet(0)
        rows = ws.get_all_records()
        if not rows:
            return False, "Sheet is empty or has no data rows."
        return True, f"Connected! Found {len(rows)} records in '{sh.title}'."
    except FileNotFoundError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Could not access sheet: {e}"


def fetch_sheet_data(sheets_url: str) -> tuple[pd.DataFrame | None, str]:
    """
    Fetch data from Google Sheets as DataFrame.
    Returns (dataframe, error_message).
    """
    try:
        gc = _get_gsheet_client()
        sh = gc.open_by_url(sheets_url.strip())
        ws = sh.get_worksheet(0)
        records = ws.get_all_records()
        if not records:
            return None, "Sheet has no data."
        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} rows from Google Sheets: {sh.title}")
        return df, ""
    except Exception as e:
        logger.error(f"Sheets fetch error: {e}")
        return None, str(e)


def compute_sheet_hash(sheets_url: str) -> str:
    """
    Compute a hash of current sheet content.
    Used to detect if data has changed since last sync.
    """
    try:
        gc = _get_gsheet_client()
        sh = gc.open_by_url(sheets_url.strip())
        ws = sh.get_worksheet(0)
        content = json.dumps(ws.get_all_values(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute sheet hash: {e}")
        return ""


def has_sheet_changed(user_id: int, sheets_url: str, db_module) -> bool:
    """
    Check if the Google Sheet has been updated since last sync.
    Returns True if data has changed.
    """
    current_hash = compute_sheet_hash(sheets_url)
    if not current_hash:
        return False

    # Store hash in user record (we use a simple file-based approach)
    hash_file = f"models/sheet_hash_{user_id}.txt"
    os.makedirs("models", exist_ok=True)

    old_hash = ""
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            old_hash = f.read().strip()

    if current_hash != old_hash:
        with open(hash_file, "w") as f:
            f.write(current_hash)
        logger.info(f"Sheet changed for user {user_id}")
        return True

    logger.debug(f"No sheet change for user {user_id}")
    return False


def sync_user_sheets(user_id: int, sheets_url: str, db_module, ml_module) -> dict:
    """
    Full sync cycle:
    1. Check if sheet changed
    2. Fetch new data
    3. Auto-detect columns
    4. Score with ML
    5. Save to database
    Returns dict with sync status.
    """
    from scoring import auto_map_columns, process_dataframe

    result = {
        "synced": False,
        "changed": False,
        "rows": 0,
        "error": "",
        "timestamp": datetime.datetime.now().isoformat(),
    }

    try:
        # Check if changed
        changed = has_sheet_changed(user_id, sheets_url, db_module)
        result["changed"] = changed

        if not changed:
            result["synced"] = True
            return result

        # Fetch data
        df, err = fetch_sheet_data(sheets_url)
        if err or df is None:
            result["error"] = err
            return result

        # Auto-detect column mapping
        mapping = auto_map_columns(df.columns.tolist())

        # Process + score
        clients = process_dataframe(df, mapping)

        # Save
        db_module.save_clients(user_id, clients, source="sheets_sync")

        # Update sync timestamp
        from database import get_conn
        ph = "?" 
        with get_conn() as conn:
            c = conn.cursor()
            c.execute(
                f"UPDATE users SET sheets_last_synced={ph} WHERE id={ph}",
                (datetime.datetime.now().isoformat(), user_id)
            )

        result["synced"] = True
        result["rows"] = len(clients)
        logger.info(f"Synced {len(clients)} clients for user {user_id}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Sync error for user {user_id}: {e}")

    return result


def get_sync_status(user_id: int) -> dict:
    """Return when the sheet was last synced for a user."""
    from database import get_user
    user = get_user(user_id)
    if not user:
        return {"has_sheets": False}
    return {
        "has_sheets": bool(user.get("sheets_url")),
        "sheets_url": user.get("sheets_url", ""),
        "last_synced": user.get("sheets_last_synced", "Never"),
        "sync_interval_min": cfg.SHEETS_SYNC_INTERVAL_MIN,
    }

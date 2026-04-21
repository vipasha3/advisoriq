"""
scheduler.py — Background intelligence jobs for AdvisorIQ
Runs alongside the Streamlit app as a separate process.

Jobs:
  Every 5 min  — Sync Google Sheets for all active users
  Every 1 hour — Re-score clients whose data changed
  Every day    — Send inactivity alerts, churn risk warnings
  Every week   — Retrain ML models with latest data

Run: python scheduler.py
"""
import logging
import datetime
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scheduler.log"),
    ]
)
logger = logging.getLogger("scheduler")
os.makedirs("logs", exist_ok=True)

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed. Run: pip install apscheduler")


# ── Jobs ──────────────────────────────────────────────────────────────────────

def job_sync_all_sheets():
    """Sync Google Sheets for every user who has set up a sheets URL."""
    logger.info("JOB: Syncing Google Sheets for all users...")
    try:
        import database as db
        import ml_model as ml
        import sheets_sync as ss
        from scoring import auto_map_columns, process_dataframe

        with db.get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT id, sheets_url FROM users WHERE sheets_url IS NOT NULL AND sheets_url != ''")
            users = c.fetchall()

        for user in users:
            user_id = user["id"]
            sheets_url = user["sheets_url"]
            try:
                result = ss.sync_user_sheets(user_id, sheets_url, db, ml)
                if result["changed"]:
                    logger.info(f"User {user_id}: synced {result['rows']} clients")
                else:
                    logger.debug(f"User {user_id}: no sheet changes")
            except Exception as e:
                logger.error(f"Sync failed for user {user_id}: {e}")

    except Exception as e:
        logger.error(f"Sheets sync job error: {e}")


def job_send_inactivity_alerts():
    """
    Daily job: find clients inactive 90+ days and queue WhatsApp alert.
    Only sends to users with WhatsApp enabled (growth/firm plan).
    """
    logger.info("JOB: Checking inactivity alerts...")
    try:
        import database as db
        import whatsapp as wa

        with db.get_conn() as conn:
            c = conn.cursor()
            # Get all advisors with whatsapp-enabled plans
            c.execute("""
                SELECT u.id, u.full_name, u.company, u.plan
                FROM users u
                WHERE u.plan IN ('growth', 'firm')
            """)
            advisors = c.fetchall()

        for advisor in advisors:
            advisor_dict = dict(advisor)
            user_id = advisor_dict["id"]

            # Get inactive clients (90+ days no contact)
            clients = db.load_clients(user_id)
            inactive = []
            for c in clients:
                churn = c.get("churn", 0)
                phone = c.get("phone", "")
                if churn > 60 and phone:
                    inactive.append(c)

            if inactive:
                logger.info(f"Advisor {user_id}: {len(inactive)} inactive high-risk clients")
                # Queue messages (dry_run=True until advisor explicitly enables live sends)
                results = wa.send_bulk(
                    inactive[:5],  # Max 5 per day to avoid spam
                    "inactivity_alert",
                    advisor_dict,
                    dry_run=True   # Change to False after Twilio setup
                )
                success = sum(1 for r in results if r.get("success"))
                logger.info(f"Advisor {user_id}: {success}/{len(results)} alerts queued")

    except Exception as e:
        logger.error(f"Inactivity alert job error: {e}")


def job_churn_prevention():
    """
    Daily job: identify top churn risk clients and log action items.
    """
    logger.info("JOB: Running churn prevention analysis...")
    try:
        import database as db

        with db.get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT DISTINCT user_id FROM clients")
            user_ids = [r["user_id"] for r in c.fetchall()]

        for user_id in user_ids:
            clients = db.load_clients(user_id)
            high_risk = [c for c in clients if c.get("churn", 0) > 70]
            if high_risk:
                names = [c.get("name","") for c in high_risk[:3]]
                db.log_action(
                    user_id,
                    "churn_alert",
                    f"High churn risk: {', '.join(names)} — {len(high_risk)} total"
                )
                logger.info(f"User {user_id}: {len(high_risk)} high churn risk clients logged")

    except Exception as e:
        logger.error(f"Churn prevention job error: {e}")


def job_retrain_models():
    """
    Weekly job: retrain ML models with any new data patterns.
    """
    logger.info("JOB: Retraining ML models...")
    try:
        import ml_model as ml
        meta = ml.train_models(force=True)
        logger.info(f"Models retrained — Priority AUC: {meta['priority_auc']}, Churn AUC: {meta['churn_auc']}")
    except Exception as e:
        logger.error(f"Model retrain job error: {e}")


def job_rescore_clients():
    """
    Hourly job: rescore clients if sheet was synced recently.
    """
    logger.info("JOB: Rescoring recently synced clients...")
    try:
        import database as db
        import ml_model as ml

        with db.get_conn() as conn:
            c = conn.cursor()
            # Users whose sheets were synced in last 1 hour
            one_hour_ago = (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
            c.execute(
                "SELECT id FROM users WHERE sheets_last_synced > ?",
                (one_hour_ago,)
            )
            users = c.fetchall()

        for user in users:
            user_id = user["id"]
            clients = db.load_clients(user_id)
            if clients:
                updated = ml.predict_batch(clients)
                db.save_clients(user_id, updated, source="rescore")
                logger.info(f"User {user_id}: rescored {len(updated)} clients")

    except Exception as e:
        logger.error(f"Rescore job error: {e}")


def job_daily_digest():
    """
    Daily: log a summary for each advisor (useful for future email/push notifications).
    """
    logger.info("JOB: Generating daily digest...")
    try:
        import database as db

        with db.get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT id, full_name FROM users")
            users = c.fetchall()

        for user in users:
            user_id = user["id"]
            clients = db.load_clients(user_id)
            if not clients: continue

            high = sum(1 for c in clients if c.get("priority") == "High")
            risk = sum(1 for c in clients if c.get("churn", 0) > 50)
            today = datetime.datetime.now().strftime("%Y-%m-%d")

            db.log_action(
                user_id,
                "daily_digest",
                f"Date: {today} | Total: {len(clients)} | High priority: {high} | Churn risk: {risk}"
            )

    except Exception as e:
        logger.error(f"Daily digest job error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    if not SCHEDULER_AVAILABLE:
        logger.error("APScheduler not available. Install it: pip install apscheduler")
        return

    # Ensure DB is initialized
    try:
        import database as db
        db.init_db()
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

    # Ensure models are trained
    try:
        import ml_model as ml
        ml.train_models()
    except Exception as e:
        logger.warning(f"Model init warning: {e}")

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # ── Schedule all jobs ──
    scheduler.add_job(
        job_sync_all_sheets,
        IntervalTrigger(minutes=5),
        id="sheets_sync",
        name="Google Sheets sync",
        replace_existing=True,
    )
    scheduler.add_job(
        job_rescore_clients,
        IntervalTrigger(hours=1),
        id="rescore",
        name="Client rescoring",
        replace_existing=True,
    )
    scheduler.add_job(
        job_send_inactivity_alerts,
        CronTrigger(hour=9, minute=0, timezone="Asia/Kolkata"),  # 9 AM IST
        id="inactivity_alerts",
        name="Inactivity alerts",
        replace_existing=True,
    )
    scheduler.add_job(
        job_churn_prevention,
        CronTrigger(hour=10, minute=0, timezone="Asia/Kolkata"),  # 10 AM IST
        id="churn_prevention",
        name="Churn prevention",
        replace_existing=True,
    )
    scheduler.add_job(
        job_daily_digest,
        CronTrigger(hour=20, minute=0, timezone="Asia/Kolkata"),  # 8 PM IST
        id="daily_digest",
        name="Daily digest",
        replace_existing=True,
    )
    scheduler.add_job(
        job_retrain_models,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone="Asia/Kolkata"),  # Sunday 2 AM
        id="model_retrain",
        name="ML model retrain",
        replace_existing=True,
    )

    logger.info("=" * 60)
    logger.info("AdvisorIQ Background Scheduler Started")
    logger.info("Jobs:")
    logger.info("  Every 5 min  — Google Sheets sync")
    logger.info("  Every 1 hour — Client rescoring")
    logger.info("  9:00 AM IST  — Inactivity alerts")
    logger.info("  10:00 AM IST — Churn prevention")
    logger.info("  8:00 PM IST  — Daily digest")
    logger.info("  Sunday 2 AM  — ML model retrain")
    logger.info("=" * 60)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    run()

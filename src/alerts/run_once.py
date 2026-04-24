"""
src/alerts/run_once.py
───────────────────────
Runs ONE polling cycle and exits.

Use this for:
  - Testing the alert system
  - Scheduled runs via Windows Task Scheduler
  - Cloud deployment (Railway, Render cron jobs)

Run: python -m src.alerts.run_once
"""

from src.alerts.live_poller import LivePoller


if __name__ == "__main__":
    poller = LivePoller(capital=500.0, poll_minutes=15)
    poller.poll_once()
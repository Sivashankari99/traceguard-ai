"""
Lightweight feedback logging, shared by:
  - the main app's inline Helpful / Not Helpful buttons (per-answer rating)
  - the dedicated Feedback page (suggestions, bugs, feature requests)

Both write into the same CSV so the log doubles as the seed of a real
evaluation dataset over time, exactly as intended.

Honest limitation, not hidden: Streamlit Community Cloud's filesystem is
EPHEMERAL. This CSV persists across reruns and across different visitors
to the same running deployment, but is wiped on every redeploy/restart.
Treat it as a rolling log you periodically export (see the download
button on the Feedback page), not a durable database.
"""

import csv
from pathlib import Path
from datetime import datetime

import pandas as pd

FEEDBACK_COLUMNS = ["timestamp", "source", "type", "query", "workflow", "rating_reason", "message"]


def _feedback_path(repo_root: Path) -> Path:
    return repo_root / "data" / "feedback_log.csv"


def append_feedback(repo_root: Path, source: str, type_: str, query: str = "",
                     workflow: str = "", rating_reason: str = "", message: str = ""):
    """Append one feedback row. Creates the file (with header) on first use."""
    path = _feedback_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FEEDBACK_COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            source, type_, query, workflow, rating_reason, message,
        ])


def load_feedback_df(repo_root: Path) -> pd.DataFrame:
    path = _feedback_path(repo_root)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(path)

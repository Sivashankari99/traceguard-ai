"""
Feedback logging, backed by Google Sheets -- split across two tabs
within the same spreadsheet:
  - "Ratings"  -- 👍 / 👎 left on an answer on the main page
  - "Feedback" -- suggestions / bugs / feature requests from the
                  dedicated Feedback page's form

Kept as two tabs rather than one flat sheet because the two row types
answer different questions (was this specific answer good? vs. what
should change about the app?) and mixing them made the sheet harder to
actually use for either purpose.

Streamlit Community Cloud's filesystem is ephemeral (wiped on every
redeploy/restart), so anything written to local disk was never durable
-- a Google Sheet, written to via a service account, actually persists.

Auth: expects a `[gcp_service_account]` table in st.secrets (the full
service-account JSON, same shape Google gives you when you create a
key), plus an optional `gsheet_name` (defaults to "TraceGuard Feedback
Log" if not set). This module only ever reads secrets via st.secrets,
never hardcodes or logs the key material itself.

`repo_root` is kept as a parameter on the public functions purely for
call-site compatibility with app.py / pages/3_Feedback.py -- it is not
actually used, since nothing here touches local disk.
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2.service_account import Credentials

_COLUMNS = ["timestamp", "source", "type", "query", "workflow", "rating_reason", "message"]
_DEFAULT_SHEET_NAME = "TraceGuard Feedback Log"
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

RATINGS_TAB = "Ratings"
FEEDBACK_TAB = "Feedback"


@st.cache_resource(show_spinner=False)
def _get_spreadsheet():
    """Authenticates once per app process (cached, same principle as the
    engine load) and returns the whole spreadsheet object."""
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_info, scopes=_SCOPES)
    client = gspread.authorize(creds)
    sheet_name = st.secrets.get("gsheet_name", _DEFAULT_SHEET_NAME)
    return client.open(sheet_name)


@st.cache_resource(show_spinner=False)
def _get_worksheet(tab_name: str):
    """Returns the named tab, creating it (with the header row) if it
    doesn't exist yet -- so a fresh spreadsheet with only the default
    "Sheet1" still works the first time this runs."""
    spreadsheet = _get_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=len(_COLUMNS))

    header = worksheet.row_values(1)
    if header != _COLUMNS:
        worksheet.update("A1", [_COLUMNS])

    return worksheet


def _append_row(tab_name: str, source: str, type_: str, query: str,
                 workflow: str, rating_reason: str, message: str):
    """Shared row-append, used by both public functions below. Never
    raises into the caller -- a Sheets outage shouldn't break the main
    app, it just means this one row silently didn't get logged
    (surfaced as a warning in the UI instead of a crash)."""
    try:
        worksheet = _get_worksheet(tab_name)
        worksheet.append_row([
            datetime.now().isoformat(timespec="seconds"),
            source, type_, query, workflow, rating_reason, message,
        ])
    except Exception as exc:
        st.warning(f"Couldn't save to Google Sheets ({tab_name} tab): {exc}")


def append_rating(repo_root, query: str = "", workflow: str = "",
                   rating_reason: str = "", type_: str = "helpful", message: str = ""):
    """Logs a 👍/👎 from the main page's answer. type_ is "helpful" or
    "not_helpful"; rating_reason and message are only ever set for
    "not_helpful" (message is optional free text, most useful when
    rating_reason is "Other" and the category alone says nothing)."""
    _append_row(RATINGS_TAB, source="main_page", type_=type_, query=query,
                workflow=workflow, rating_reason=rating_reason, message=message)


def append_feedback(repo_root, source: str, type_: str, query: str = "",
                     workflow: str = "", rating_reason: str = "", message: str = ""):
    """Logs a submission from the Feedback page's form (suggestion, bug,
    feature request, etc.)."""
    _append_row(FEEDBACK_TAB, source=source, type_=type_, query=query,
                workflow=workflow, rating_reason=rating_reason, message=message)


def load_df(repo_root, tab_name: str) -> pd.DataFrame:
    """Reads a tab back as a DataFrame. Not wired into any page's UI --
    the app intentionally never displays feedback/ratings publicly --
    but kept as a utility for a private admin script or notebook."""
    try:
        worksheet = _get_worksheet(tab_name)
        records = worksheet.get_all_records()
        if not records:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame(records)
    except Exception as exc:
        st.warning(f"Couldn't load from Google Sheets ({tab_name} tab): {exc}")
        return pd.DataFrame(columns=_COLUMNS)

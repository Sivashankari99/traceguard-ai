"""
A tiny, deliberately dumb module-level flag: has the TraceGuard engine
finished loading at least once in this running server process?

Why this exists: Dataset Explorer's "search dataset before querying"
picker needs to know whether it's safe to send someone to the main page
expecting an instant answer, or whether doing so would trigger a fresh
2-5 minute cold start. It must be able to check this WITHOUT importing
or calling load_orchestrator() itself (that would defeat the point --
merely checking would trigger the very thing being checked for).

This works because Streamlit's multipage apps run every page in the
SAME Python process, and modules under src/ are only ever imported once
per process (standard sys.modules caching) -- so a plain module-level
global set by app.py after a successful load is visible to every page
that imports this module, for the lifetime of the server process.

Deliberately NOT using st.session_state: that's per-browser-session,
but this needs to reflect the engine's real, server-wide, cached state
(shared across every visitor), matching how @st.cache_resource itself
behaves.
"""

_engine_ready = False


def mark_ready():
    global _engine_ready
    _engine_ready = True


def is_ready() -> bool:
    return _engine_ready

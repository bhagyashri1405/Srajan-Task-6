import os
import sqlite3
import json
import hashlib
import secrets
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent  # noqa: E402  (import after load_dotenv so env vars are set)

st.set_page_config(page_title="Research & Competitor Intelligence Agent", page_icon="🛰️")

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            trace TEXT NOT NULL,
            search_queries TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """
    Logs in an existing user, or creates a new account on first use.
    Returns (success, message).
    """
    conn = _get_conn()
    row = conn.execute("SELECT salt, password_hash FROM users WHERE username = ?", (username,)).fetchone()

    if row is None:
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
            (username, salt, password_hash),
        )
        conn.commit()
        conn.close()
        return True, "Account created and logged in."

    salt, stored_hash = row
    conn.close()
    if _hash_password(password, salt) == stored_hash:
        return True, "Logged in."
    return False, "Incorrect password for that username."


def save_session(username: str, query: str, result: dict):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (username, query, answer, trace, search_queries, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            username,
            query,
            result["answer"],
            json.dumps(result["trace"]),
            json.dumps(result["search_queries"]),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    conn.close()


def get_history(username: str):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, query, created_at FROM sessions WHERE username = ? ORDER BY id DESC",
        (username,),
    ).fetchall()
    conn.close()
    return rows


def get_session(session_id: int):
    conn = _get_conn()
    row = conn.execute(
        "SELECT query, answer, trace, search_queries, created_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    query, answer, trace_json, search_queries_json, created_at = row
    return {
        "query": query,
        "answer": answer,
        "trace": json.loads(trace_json),
        "search_queries": json.loads(search_queries_json),
        "created_at": created_at,
    }


def build_long_term_context(username: str, limit: int = 3) -> str:
    """
    This is the agent's long-term memory: a short summary of this user's most
    recent past searches, pulled from persistent storage (SQLite) and handed
    to the Research Agent so it can note connections across sessions.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT query, answer, created_at FROM sessions WHERE username = ? ORDER BY id DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    conn.close()

    if not rows:
        return ""

    lines = []
    for query, answer, created_at in rows:
        summary = answer[:200].replace("\n", " ").strip()
        lines.append(f"- On {created_at}, researched \"{query}\": {summary}...")
    return "\n".join(lines)


# ---------- Session state ----------
if "username" not in st.session_state:
    st.session_state.username = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None


# ---------- Sidebar: login + history ----------
with st.sidebar:
    st.header("👤 Account")
    if st.session_state.username is None:
        name_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        st.caption("New username? An account is created automatically on first login.")
        if st.button("Log in / Sign up", use_container_width=True):
            if not name_input.strip() or not password_input:
                st.error("Enter both a username and a password.")
            else:
                success, message = authenticate(name_input.strip(), password_input)
                if success:
                    st.session_state.username = name_input.strip()
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.success(f"Logged in as **{st.session_state.username}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.username = None
            st.session_state.active_session_id = None
            st.rerun()

        st.divider()
        st.header("🕘 Search History")
        history_rows = get_history(st.session_state.username)
        if not history_rows:
            st.caption("No past searches yet.")
        else:
            for session_id, query, created_at in history_rows:
                label = f"{query[:40]}{'...' if len(query) > 40 else ''}"
                if st.button(label, key=f"hist_{session_id}", use_container_width=True, help=created_at):
                    st.session_state.active_session_id = session_id
                    st.rerun()

        if st.button("➕ New search", use_container_width=True):
            st.session_state.active_session_id = None
            st.rerun()

        st.divider()
        st.header("🧪 Adversarial Demo Mode")
        simulate_failure = st.checkbox(
            "Force tool failures this run",
            key="simulate_failure_toggle",
            help="For live demo use: forces every tool call's primary attempt to fail, "
            "so you can show the fallback/recovery path happening in real time.",
        )
        if simulate_failure:
            st.warning("Demo mode ON — tool calls will fail and fall back on purpose.")


# ---------- Main area ----------
st.title("🛰️ Research & Competitor Intelligence Agent")
st.caption(
    "Give it a topic, technology area, or competitor name. A **LangGraph pipeline** "
    "plans what to investigate, runs web + arXiv searches (in parallel where possible), "
    "self-checks for gaps or conflicting evidence, replans if needed, and hands off to "
    "an Analyst step that synthesizes the final briefing."
)

if st.session_state.username is None:
    st.info("Log in from the sidebar to start searching and to save your search history.")
    st.stop()

_ICONS = {
    "context": "📚",
    "plan": "🗺️",
    "action": "🔧",
    "tool_failure": "⚠️",
    "tool_fallback": "🔁",
    "observation": "📄",
    "evaluator": "🧪",
    "research_final": "🔬",
    "handoff": "➡️",
    "analyst_final": "📊",
    "checkpoint": "💾",
}
_LABELS = {
    "context": "Long-Term Memory Loaded",
    "plan": "Planner",
    "action": "Tool Call",
    "tool_failure": "Tool Failure",
    "tool_fallback": "Fallback Triggered",
    "observation": "Observation",
    "evaluator": "Evaluator (self-check / conflict check)",
    "research_final": "Research Compiled",
    "handoff": "Handoff",
    "analyst_final": "Analyst Agent — Final Briefing",
    "checkpoint": "Checkpoint",
}


def render_trace(trace: list):
    with st.expander(f"🧠 View reasoning trace ({len(trace)} steps)", expanded=False):
        for step in trace:
            icon = _ICONS.get(step["type"], "•")
            label = _LABELS.get(step["type"], step["type"])
            st.markdown(f"**{icon} {label}**")
            if step["type"] == "observation":
                st.text(step["content"][:800] + ("..." if len(step["content"]) > 800 else ""))
            else:
                st.markdown(step["content"])
            st.divider()


def render_result(query: str, result: dict):
    st.subheader(f"🔎 {query}")
    st.markdown(result["answer"])
    if result["search_queries"]:
        st.info(f"🔍 Ran {len(result['search_queries'])} tool call(s): " + ", ".join(result["search_queries"]))
    else:
        st.success("🧠 Answered from existing knowledge — no tools needed")
    render_trace(result["trace"])


# Viewing a past session from history
if st.session_state.active_session_id is not None:
    past = get_session(st.session_state.active_session_id)
    if past:
        st.caption(f"Viewing saved search from {past['created_at']}")
        render_result(past["query"], past)
    if st.button("← Back to new search"):
        st.session_state.active_session_id = None
        st.rerun()

else:
    user_input = st.chat_input("Enter a topic, technology area, or competitor to monitor...")
    if user_input:
        with st.spinner("Planner → Tools (parallel) → Evaluator → Analyst..."):
            try:
                long_term_context = build_long_term_context(st.session_state.username)
                thread_id = f"{st.session_state.username}-{uuid.uuid4().hex[:8]}"
                result = run_agent(
                    user_input,
                    long_term_context=long_term_context,
                    simulate_failure=st.session_state.get("simulate_failure_toggle", False),
                    thread_id=thread_id,
                )
            except Exception as exc:
                st.error(
                    f"Something went wrong talking to the Gemini API: {exc}\n\n"
                    "If this says RESOURCE_EXHAUSTED / 429, you've hit your API quota — "
                    "wait a bit, or check your usage at ai.dev/rate-limit."
                )
                st.stop()
        save_session(st.session_state.username, user_input, result)
        render_result(user_input, result)

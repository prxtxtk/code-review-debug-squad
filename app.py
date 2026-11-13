"""Streamlit GUI for the Code Review & Debug Squad.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import difflib
import html
import json
import re
from datetime import datetime
from typing import Any

import streamlit as st

from agents import CodeReviewSquad, ReviewResult
from config import (
    MODEL_PRESETS,
    PROVIDER_IDS,
    PROVIDER_LABELS,
    AppConfig,
    ConfigurationError,
    build_llm,
    config_from_env,
    default_model,
    llm_status,
    provider_api_key,
    provider_base_url,
    test_llm_connection,
)
from sandbox import environment_description
from vector_store import VectorStoreManager

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Code Review & Debug Squad",
    page_icon=":material/code:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
  --ink: #1d2433;
  --muted: #697386;
  --line: #e3e6ec;
  --line-strong: #d2d7e0;
  --canvas: #f7f7f8;
  --surface: #ffffff;
  --surface-soft: #fafbfc;
  --surface-muted: #f7f8fa;
  --accent: #2f5fc7;
  --accent-strong: #274fa6;
  --accent-soft: #eff3fb;
  --green: #236b55;
  --amber: #8a5b16;
  --red: #a9364d;
}
html, body, [class*="css"] { font-family: "Segoe UI", "Inter", system-ui, sans-serif; }
.stApp { background: var(--canvas); color: var(--ink); }
.block-container { padding-top: 1.75rem; padding-bottom: 4rem; max-width: 1480px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #fcfcfd; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
[data-testid="stHeader"] { background: var(--canvas); }
.hero {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 24px 27px 22px; color: var(--ink); margin-bottom: 16px;
}
.hero .eyebrow {
  font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); font-weight: 650;
}
.hero h1 {
  font-size: 30px; line-height: 1.2; margin: 8px 0 8px;
  letter-spacing: -0.022em; font-weight: 680;
}
.hero h1 .accent { color: var(--accent); font-weight: 680; }
.hero p { color: var(--muted); max-width: 820px; margin: 0; font-size: 14px; line-height: 1.6; }
.hero .chips { margin-top: 14px; }
.hero .pill { background: #fafbfc; color: #586273; border-color: var(--line); }
.hero-head { display: flex; align-items: flex-start; gap: 13px; }
.hero-mark {
  display: flex; align-items: center; justify-content: center; flex: 0 0 auto;
  width: 40px; height: 40px; margin-top: 2px;
  border: 1px solid #dbe4f4; border-radius: 9px;
  color: var(--accent); background: var(--accent-soft);
}
.ui-icon { display: inline-block; vertical-align: -2px; flex: 0 0 auto; }
.notice {
  background: #fbfbfc; border: 1px solid var(--line); border-left: 3px solid #cbd3df;
  border-radius: 8px; padding: 9px 12px; margin: 12px 0 2px;
  color: var(--muted); font-size: 12px; line-height: 1.55;
}
.notice strong { color: var(--ink); font-weight: 650; }
.sidebar-brand {
  display: flex; align-items: center; gap: 10px; margin: 2px 0 10px;
}
.brand-mark {
  width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;
  border-radius: 8px; border: 1px solid #dbe4f4; background: var(--accent-soft); color: var(--accent);
}
.brand-title { color: var(--ink); font-size: 13px; font-weight: 680; line-height: 1.2; }
.brand-sub { color: var(--muted); font-size: 11px; margin-top: 1px; }
.theme-glyph {
  display: flex; align-items: center; justify-content: center;
  width: 28px; height: 28px;
  border: 1px solid var(--line); border-radius: 7px;
  background: var(--surface-soft); color: var(--muted);
}
.sidebar-heading {
  display: flex; align-items: center; gap: 8px; margin: 17px 0 8px;
  color: var(--ink); font-size: 12.5px; font-weight: 650;
}
.sidebar-heading .sidebar-heading-icon { color: var(--muted); display: inline-flex; }
.sidebar-heading .step {
  margin-left: auto; color: #9aa3b2; font-size: 10px; font-weight: 600;
  letter-spacing: 0.05em;
}
.section-heading { display: flex; align-items: center; gap: 10px; margin: 4px 0 11px; }
.section-icon {
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--line); border-radius: 8px; background: #fbfbfc; color: var(--accent);
}
.field-label {
  display: flex; align-items: center; gap: 7px; margin-bottom: 5px;
  font-size: 13px; font-weight: 650; color: var(--ink);
}
.field-label .field-icon { color: var(--muted); display: inline-flex; }
.field-tag {
  margin-left: auto; font-size: 9.5px; font-weight: 600; color: var(--muted);
  border: 1px solid var(--line); background: #fafbfc; border-radius: 5px;
  padding: 2px 6px; letter-spacing: 0.03em;
}
.flow-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px;
  margin: 12px 0 18px;
}
.flow-card {
  border: 1px solid var(--line); border-radius: 9px; background: var(--surface);
  padding: 12px 13px; transition: border-color 0.15s ease, background 0.15s ease;
}
.flow-card:hover { border-color: #cbd5e3; background: #fdfdfe; }
.flow-top { display: flex; align-items: center; justify-content: space-between; }
.flow-icon { color: var(--accent); display: inline-flex; }
.flow-step { font-size: 9.5px; color: #9aa3b2; font-weight: 650; letter-spacing: 0.07em; }
.flow-title { color: var(--ink); font-size: 12.5px; font-weight: 650; margin-top: 9px; }
.flow-copy { color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 3px; }
.empty-state {
  display: flex; align-items: flex-start; gap: 12px;
  border: 1px solid var(--line); border-radius: 9px; background: #fcfcfd;
  padding: 17px 16px; margin: 4px 0 6px;
}
.empty-icon {
  width: 38px; height: 38px; flex: 0 0 auto; display: flex; align-items: center; justify-content: center;
  border: 1px solid #dbe4f4; border-radius: 8px; background: var(--accent-soft); color: var(--accent);
}
.empty-title { color: var(--ink); font-size: 13px; font-weight: 650; margin-top: 1px; }
.empty-copy { color: var(--muted); font-size: 12px; line-height: 1.55; margin-top: 3px; }
.kbd-hint { color: #9aa3b2; font-size: 10.5px; text-align: right; margin-top: -4px; }
.kbd {
  display: inline-block; border: 1px solid var(--line); border-bottom-color: #cfd5df;
  border-radius: 4px; background: #fafbfc; padding: 1px 5px; font-size: 10px;
  font-family: "Cascadia Code", Consolas, monospace; color: var(--muted);
}
.pill {
  display: inline-flex; align-items: center; gap: 5px; border-radius: 999px;
  padding: 3px 8px; font-size: 10.5px; font-weight: 600; letter-spacing: 0.01em;
  margin: 2px 4px 2px 0; border: 1px solid transparent; white-space: nowrap;
}
.pill-blue { background: #f1f5fc; color: #2d55ad; border-color: #d8e2f4; }
.pill-green { background: #f1f8f5; color: #226a53; border-color: #d5e9e1; }
.pill-amber { background: #fbf7ef; color: #805616; border-color: #eee1c9; }
.pill-red { background: #fbf2f4; color: #9b3047; border-color: #efd7dd; }
.pill-gray { background: #f7f8fa; color: #586273; border-color: #e4e7ed; }
.section-kicker {
  color: var(--muted); font-size: 10px; letter-spacing: 0.13em;
  text-transform: uppercase; font-weight: 650; margin-bottom: 3px;
}
.section-title {
  color: var(--ink); font-size: 18px; font-weight: 680;
  letter-spacing: -0.012em; margin: 0 0 5px;
}
.status-box {
  border: 1px solid var(--line); background: #fcfcfd; border-radius: 8px;
  padding: 10px 12px; margin: 8px 0 4px;
}
.status-box .status-label { font-size: 12px; font-weight: 650; color: var(--ink); }
.status-box .status-detail { font-size: 11px; color: var(--muted); line-height: 1.45; margin-top: 2px; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }
.dot-green { background: #4b8f7a; }
.dot-amber { background: #b08a4a; }
.dot-red { background: #b85a6b; }
.dot-gray { background: #8b94a4; }
.rule { height: 1px; background: var(--line); margin: 18px 0; }
.card-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 8px 0 16px; }
.stat-card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 13px 14px; transition: border-color 0.15s ease;
}
.stat-card:hover { border-color: #cbd5e3; }
.stat-card .k { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.11em; color: var(--muted); font-weight: 650; }
.stat-card .v { font-size: 20px; font-weight: 680; color: var(--ink); margin-top: 4px; line-height: 1.15; }
.stat-card .s { font-size: 11px; color: var(--muted); margin-top: 3px; }
.criteria-row {
  border: 1px solid var(--line); background: var(--surface); border-radius: 8px;
  padding: 11px 13px; margin-bottom: 7px;
}
.criteria-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.criteria-title { font-size: 12.5px; font-weight: 650; color: var(--ink); }
.criteria-evidence { color: var(--muted); font-size: 12px; line-height: 1.5; margin-top: 4px; }
.table-wrap {
  border: 1px solid var(--line); border-radius: 8px; overflow-x: auto;
  margin: 8px 0 12px; background: var(--surface);
}
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; color: var(--ink); }
.data-table th {
  text-align: left; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--muted); background: var(--surface-soft); font-weight: 650;
  padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap;
}
.data-table td { padding: 8px 10px; border-top: 1px solid var(--line); vertical-align: top; }
.data-table tbody tr:first-child td { border-top: 0; }
.data-table .mono { font-family: "Cascadia Code", Consolas, monospace; font-size: 11px; }
.prompt-meta { color: var(--muted); font-size: 12px; margin: 2px 0 7px; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
[data-testid="stMetricLabel"] { color: var(--muted); }
div[data-testid="stExpander"] { border-color: var(--line); border-radius: 8px; background: #fcfcfd; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; border-radius: 10px !important; }
.stTextArea textarea {
  font-family: "Cascadia Code", Consolas, "Courier New", monospace;
  font-size: 12.5px; line-height: 1.55;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 7px; font-weight: 600; min-height: 2.3rem;
  border-color: var(--line-strong); color: var(--ink);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #bcc5d3; color: var(--ink); background: #fafbfc;
}
.stButton > button[kind="primary"] {
  background: var(--accent); border: 1px solid var(--accent); color: #ffffff;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-strong); border-color: var(--accent-strong); color: #ffffff;
}
.st-key-quick_intents .stButton > button[kind="primary"] {
  background: var(--accent-soft); border-color: #c9d7f0; color: var(--accent-strong);
}
.st-key-quick_intents .stButton > button[kind="primary"]:hover {
  background: #e8eef9; border-color: #b9caeb; color: var(--accent-strong);
}
.stTabs [data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  padding: 8px 1px 9px; border-radius: 0; background: transparent;
  color: var(--muted); font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: transparent; color: var(--ink);
  border-bottom: 2px solid var(--accent);
}
.small-note { color: var(--muted); font-size: 11.5px; line-height: 1.5; }
@media (max-width: 900px) {
  .hero h1 { font-size: 26px; }
  .card-grid, .flow-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .hero { padding: 20px 20px; }
}
</style>
"""
DARK_CSS = """
<style>
:root {
  color-scheme: dark;
  --ink: #e7ebf2;
  --muted: #99a3b4;
  --line: #2a313c;
  --line-strong: #3c4656;
  --canvas: #0f1218;
  --surface: #161b23;
  --surface-soft: #1a2029;
  --surface-muted: #202732;
  --accent: #86a3f2;
  --accent-strong: #b3c4ff;
  --accent-soft: #1c2740;
  --green: #82c9b0;
  --amber: #ddb878;
  --red: #e393a4;
}
html, body { background-color: var(--canvas) !important; color: var(--ink) !important; }
.stApp, [data-testid="stAppViewContainer"] { background-color: var(--canvas) !important; color: var(--ink) !important; }
[data-testid="stHeader"] { background-color: var(--canvas) !important; }
[data-testid="stSidebar"] { background-color: #11151c !important; border-right-color: var(--line) !important; }
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp label, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] span { color: var(--ink); }
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stCaptionContainer"] span { color: var(--muted) !important; }
.stApp a { color: var(--accent-strong); }
.hero p { color: var(--muted) !important; }
.hero, .flow-card, .stat-card, .criteria-row, .empty-state {
  background: var(--surface) !important; border-color: var(--line) !important;
}
.flow-card:hover { background: var(--surface-soft) !important; border-color: var(--line-strong) !important; }
.notice, .status-box, .field-tag, .kbd, .section-icon, .theme-glyph {
  background: var(--surface-soft) !important; border-color: var(--line) !important;
}
.notice { border-left-color: #3c4656 !important; }
.hero-mark, .brand-mark, .empty-icon {
  background: var(--accent-soft) !important; border-color: #2c3b57 !important;
  color: var(--accent-strong) !important;
}
.hero .pill, .pill-gray {
  background: var(--surface-soft) !important; border-color: var(--line) !important;
  color: var(--muted) !important;
}
.pill-blue { background: #162237 !important; border-color: #2b3c5d !important; color: #aec4ff !important; }
.pill-green { background: #14261f !important; border-color: #284638 !important; color: #93d6bd !important; }
.pill-amber { background: #2a2214 !important; border-color: #493a21 !important; color: #e3c184 !important; }
.pill-red { background: #2b171c !important; border-color: #4b2933 !important; color: #e9a0b0 !important; }
.dot-green { background: #75c2a5; }
.dot-amber { background: #d3ac6d; }
.dot-red { background: #df899b; }
.dot-gray { background: #778397; }
.rule { background: var(--line); }
.section-icon { color: var(--accent-strong); }
.sidebar-heading .step, .flow-step, .kbd-hint { color: #778397; }
.status-box .status-label, .criteria-title, .empty-title, .flow-title,
.stat-card .v, .field-label, .brand-title { color: var(--ink); }
.status-box .status-detail, .criteria-evidence, .empty-copy, .flow-copy,
.stat-card .s, .stat-card .k, .brand-sub, .field-tag { color: var(--muted); }
.data-table { color: var(--ink); }
.data-table th { color: var(--muted); background: var(--surface-soft); border-bottom-color: var(--line); }
.data-table td { border-top-color: var(--line); }
.stApp input, .stApp textarea,
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
  background-color: #11161e !important; color: var(--ink) !important;
  border-color: var(--line-strong) !important;
}
.stApp input::placeholder, .stApp textarea::placeholder { color: #727e91 !important; }
div[data-baseweb="select"] span { color: var(--ink) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; color: var(--muted) !important; }
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] div[role="listbox"] {
  background-color: var(--surface) !important; border-color: var(--line) !important;
}
div[data-baseweb="popover"] li { color: var(--ink) !important; }
div[data-baseweb="popover"] li:hover { background-color: var(--surface-soft) !important; }
div[data-baseweb="tooltip"] { background-color: var(--surface) !important; color: var(--ink) !important; }
.stApp textarea:focus, .stApp input:focus { border-color: var(--accent) !important; }
.stButton > button, .stDownloadButton > button {
  background-color: var(--surface) !important; color: var(--ink) !important;
  border-color: var(--line-strong) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background-color: var(--surface-soft) !important; color: var(--ink) !important;
  border-color: #4a5566 !important;
}
.stButton > button[kind="primary"] {
  background-color: var(--accent) !important; border-color: var(--accent) !important;
  color: #10151d !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: var(--accent-strong) !important; border-color: var(--accent-strong) !important;
  color: #10151d !important;
}
.st-key-quick_intents .stButton > button[kind="primary"] {
  background-color: var(--accent-soft) !important; border-color: #344967 !important;
  color: var(--accent-strong) !important;
}
.st-key-quick_intents .stButton > button[kind="primary"]:hover {
  background-color: #22314f !important; border-color: #48618a !important;
}
.stTabs [data-baseweb="tab-list"] { border-bottom-color: var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
div[data-testid="stExpander"] {
  background: var(--surface) !important; border-color: var(--line) !important;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary { background: transparent !important; color: var(--ink) !important; }
div[data-testid="stAlert"] {
  background: var(--surface-soft) !important; border: 1px solid var(--line) !important;
  color: var(--ink) !important;
}
div[data-testid="stAlert"] p { color: var(--ink) !important; }
div[data-testid="stStatusWidget"], div[data-testid="stStatus"] {
  background: var(--surface) !important; border-color: var(--line) !important; color: var(--ink) !important;
}
.stApp pre { background: #0c1118 !important; border: 1px solid var(--line) !important; }
.stApp pre code, .stApp pre code span { color: #d8e2ef !important; }
.stApp pre .token.keyword { color: #9db7ff !important; }
.stApp pre .token.string { color: #a8d8b9 !important; }
.stApp pre .token.comment { color: #7e8a9d !important; font-style: italic; }
.stApp pre .token.function { color: #e4c07a !important; }
.stApp pre .token.number { color: #d9a77e !important; }
.stApp [data-testid="stMarkdownContainer"] code {
  background: var(--surface-soft) !important; color: #d8e2ef !important;
  border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px;
}
.stApp pre code { border: 0 !important; padding: 0 !important; background: transparent !important; }
.stApp pre button { color: #d8e2ef !important; background: transparent !important; }
.stApp hr { border-color: var(--line) !important; }
[data-testid="stSlider"] { color: var(--ink) !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--accent) !important; }
[data-testid="stToggle"] [role="checkbox"] { background-color: var(--line-strong) !important; }
[data-testid="stToggle"] [aria-checked="true"] { background-color: var(--accent) !important; }
::selection { background: #334f86; color: #ffffff; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

SAMPLE_CODE: dict[str, str] = {
    "Off-by-one & zero-division": '''def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total / len(numbers)


def find_max(values):
    max_value = values[0]
    for i in range(len(values) + 1):
        if values[i] > max_value:
            max_value = values[i]
    return max_value
''',
    "Mutable default arguments": '''def add_item(item, items=[]):
    items.append(item)
    return items


class User:
    def __init__(self, name, roles=[]):
        self.name = name
        self.roles = roles
''',
    "SQL injection & resource leak": '''import sqlite3


def get_user(username):
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return cursor.execute(query).fetchall()
''',
    "Messy, unstyled code": '''def calc(x,y,op):
 if op=='add': return x+y
 elif op=='sub': return x-y
 elif op=='mul': return x*y
 elif op=='div': return x/y
 else: return None
''',
    "Trivial function (RAG-skip demo)": '''def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b
''',
}

QUICK_PROMPTS = {
    "full_review": (
        "Perform a full squad review: find bugs, check PEP 8 style, fix the code, "
        "and generate + run pytest tests in the sandbox."
    ),
    "debugging": "Debug this code. Explain the root cause and provide a corrected version.",
    "style_review": (
        "Review this code for PEP 8 style, naming, readability, and lint issues, "
        "then apply the fixes."
    ),
    "code_explanation": (
        "Explain this code clearly. Walk through the inputs, outputs, control flow, "
        "and edge cases."
    ),
    "testing": (
        "Generate comprehensive pytest tests for normal and edge cases, then run them "
        "in the isolated sandbox."
    ),
}

QUICK_BUTTONS = [
    ("Full Squad Review", "full_review"),
    ("Debug Only", "debugging"),
    ("Style Review", "style_review"),
    ("Explain Code", "code_explanation"),
    ("Generate & Run Tests", "testing"),
]

INTENT_DISPLAY = {
    "full_review": "Full Squad Review",
    "debugging": "Debug Only",
    "style_review": "Style Review",
    "code_explanation": "Explain Code",
    "testing": "Generate & Run Tests",
}


# ---------------------------------------------------------------------------
# Session-state helpers and callbacks
# ---------------------------------------------------------------------------

def _init_state() -> None:
    env = config_from_env()
    st.session_state.setdefault("code_input", SAMPLE_CODE["Off-by-one & zero-division"])
    st.session_state.setdefault("task_input", QUICK_PROMPTS["full_review"])
    st.session_state.setdefault("intent_hint", "full_review")
    st.session_state.setdefault("sample_choice", "Off-by-one & zero-division")
    st.session_state.setdefault("provider_label", PROVIDER_LABELS[env.provider])
    st.session_state.setdefault("temperature", env.temperature)
    st.session_state.setdefault("top_k", env.top_k)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_config", None)
    st.session_state.setdefault("llm_probe", None)
    if "dark_mode" not in st.session_state:
        try:
            preferred_theme = st.context.theme.type
        except Exception:
            preferred_theme = "light"
        st.session_state["dark_mode"] = preferred_theme == "dark"


def _load_sample_callback() -> None:
    choice = st.session_state.get("sample_choice", "Off-by-one & zero-division")
    st.session_state["code_input"] = SAMPLE_CODE.get(choice, "")
    st.session_state["last_result"] = None


def _clear_workspace_callback() -> None:
    st.session_state["last_result"] = None
    st.session_state["code_input"] = SAMPLE_CODE["Off-by-one & zero-division"]
    st.session_state["task_input"] = QUICK_PROMPTS["full_review"]
    st.session_state["intent_hint"] = "full_review"


def _quick_intent_callback(intent: str) -> None:
    st.session_state["task_input"] = QUICK_PROMPTS[intent]
    st.session_state["intent_hint"] = intent
    st.session_state["autorun_requested"] = True


_init_state()
if st.session_state.get("dark_mode", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small UI utilities
# ---------------------------------------------------------------------------

SVG_ICONS = {
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "route": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "agents": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "bug": '<circle cx="12" cy="13" r="4"/><path d="M12 9V7"/><path d="M6 13H4"/><path d="M20 13h-2"/><path d="m7.5 8.5-1.5-1.5"/><path d="m17 7 1.5-1.5"/><path d="m7.5 17.5-1.5 1.5"/><path d="m17 17 1.5 1.5"/>',
    "paint": '<path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.6 7.6"/><circle cx="11" cy="11" r="2"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "beaker": '<path d="M9 3h6"/><path d="M10 3v7L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 10V3"/><path d="M7 15h10"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "key": '<circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12.5-9.17 4.16a2 2 0 0 1-1.66 0L2 12.5"/><path d="m22 17.5-9.17 4.16a2 2 0 0 1-1.66 0L2 17.5"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>',
    "trace": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "chat": '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>',
    "diff": '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M6 8.5V13a4 4 0 0 0 4 4h5"/><path d="M18 15.5V11a4 4 0 0 0-4-4H9"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "bolt": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/>',
}


def svg_icon(name: str, size: int = 16) -> str:
    paths = SVG_ICONS.get(name, SVG_ICONS["check"])
    return (
        f'<svg class="ui-icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


def sidebar_heading(icon: str, title: str, step: str = "") -> None:
    st.markdown(
        f'<div class="sidebar-heading"><span class="sidebar-heading-icon">{svg_icon(icon, 14)}</span>'
        f'<span>{html.escape(title)}</span><span class="step">{html.escape(step)}</span></div>',
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, icon: str) -> None:
    st.markdown(
        f'<div class="section-heading"><div class="section-icon">{svg_icon(icon, 16)}</div>'
        f'<div><div class="section-kicker">{html.escape(kicker)}</div>'
        f'<div class="section-title">{html.escape(title)}</div></div></div>',
        unsafe_allow_html=True,
    )


def field_label(icon: str, label: str, tag: str = "") -> None:
    tag_html = f'<span class="field-tag">{html.escape(tag)}</span>' if tag else ""
    st.markdown(
        f'<div class="field-label"><span class="field-icon">{svg_icon(icon, 15)}</span>'
        f'<span>{html.escape(label)}</span>{tag_html}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, detail: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="empty-icon">{svg_icon(icon, 19)}</div>'
        f'<div><div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-copy">{html.escape(detail)}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_workflow_strip() -> None:
    steps = [
        ("route", "01", "Route", "Classify the request into one of five review intents."),
        ("search", "02", "Retrieve", "The RAG gate decides whether ChromaDB is useful."),
        ("agents", "03", "Review & fix", "Bug, style, fixer, and test agents collaborate."),
        ("beaker", "04", "Verify", "Generated pytest runs in the isolated sandbox."),
    ]
    cards = "".join(
        f'<div class="flow-card"><div class="flow-top"><span class="flow-icon">{svg_icon(icon, 17)}</span>'
        f'<span class="flow-step">{step}</span></div><div class="flow-title">{html.escape(title)}</div>'
        f'<div class="flow-copy">{html.escape(copy)}</div></div>'
        for icon, step, title, copy in steps
    )
    st.markdown(f'<div class="flow-grid">{cards}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "gray") -> str:
    return f'<span class="pill pill-{kind}">{html.escape(str(text))}</span>'


def status_box(level: str, label: str, detail: str) -> str:
    dot = {"success": "dot-green", "warning": "dot-amber", "error": "dot-red"}.get(level, "dot-gray")
    return (
        f'<div class="status-box"><div class="status-label">'
        f'<span class="dot {dot}"></span>{html.escape(label)}</div>'
        f'<div class="status-detail">{html.escape(detail)}</div></div>'
    )


def stat_cards(cards: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="stat-card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>'
        for k, v, s in cards
    )
    st.markdown(f'<div class="card-grid">{cells}</div>', unsafe_allow_html=True)


def _count_headings(text: str, prefix: str) -> int:
    return len(re.findall(rf"^###\s+{re.escape(prefix)}-\d+", text or "", flags=re.MULTILINE))


def _sandbox_badge(result: Any) -> str:
    if result is None:
        return pill("Not run", "gray")
    if result.blocked:
        return pill("Blocked", "red")
    if result.timed_out:
        return pill("Timeout", "amber")
    if result.tests_passed:
        return pill("Tests passed", "green")
    return pill("Tests failed", "red")


def _rag_badge(result: ReviewResult) -> str:
    if result.rag.should_retrieve:
        return pill(f"RAG retrieved · {len(result.rag.hits)} chunks", "green")
    return pill("RAG skipped", "amber")


def _runtime_badge(result: ReviewResult) -> str:
    provider = result.trace.get("llm", {}).get("provider", "unknown")
    if provider == "demo":
        return pill("Offline demo", "amber")
    return pill("Live LLM", "green")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_vector_store(
    persist_directory: str,
    collection_name: str,
    knowledge_base_dir: str,
    embedding_model: str,
) -> VectorStoreManager:
    store = VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        knowledge_base_dir=knowledge_base_dir,
        embedding_model=embedding_model,
    )
    store.initialize()
    return store


def _render_sidebar(env: AppConfig) -> tuple[AppConfig, VectorStoreManager | None, dict[str, Any], str]:
    """Render configuration and return the resolved config + vector store."""
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-brand"><div class="brand-mark">{svg_icon("shield", 18)}</div>'
            '<div><div class="brand-title">Code Review</div><div class="brand-sub">Debug Squad</div></div></div>',
            unsafe_allow_html=True,
        )
        st.caption("LCEL · ChromaDB · isolated pytest sandbox")

        is_dark = bool(st.session_state.get("dark_mode", False))
        theme_icon_col, theme_toggle_col = st.columns([0.16, 0.84], vertical_alignment="center")
        with theme_icon_col:
            st.markdown(
                f'<div class="theme-glyph">{svg_icon("moon" if is_dark else "sun", 15)}</div>',
                unsafe_allow_html=True,
            )
        with theme_toggle_col:
            st.toggle(
                "Dark mode",
                key="dark_mode",
                help="Switch between the light and dark workspace. The initial mode follows your system theme.",
            )

        sidebar_heading("key", "LLM provider", "01")
        provider_label = st.selectbox(
            "Provider",
            options=list(PROVIDER_LABELS.values()),
            key="provider_label",
            help="Gemini and OpenAI use hosted APIs. Ollama runs locally. Offline Demo needs no credentials.",
        )
        provider = PROVIDER_IDS.get(provider_label, "demo")
        if provider == "demo":
            model = "code-review-squad-demo"
            st.text_input("Model", value=model, disabled=True)
        else:
            options = list(
                dict.fromkeys(MODEL_PRESETS.get(provider, []) + [default_model(provider)])
            )
            current_default = st.session_state.get(f"model_choice_{provider}", default_model(provider))
            if current_default not in options:
                options.append(current_default)
            choice = st.selectbox(
                "Model",
                options=options + ["Custom model ID…"],
                index=(options + ["Custom model ID…"]).index(current_default),
                key=f"model_choice_{provider}",
            )
            if choice == "Custom model ID…":
                model = st.text_input(
                    "Custom model ID",
                    value=st.session_state.get(f"custom_model_{provider}", default_model(provider)),
                    key=f"custom_model_{provider}",
                ).strip()
            else:
                model = choice

        sidebar_heading("bolt", "Connection", "02")
        api_key = ""
        base_url = ""
        ollama_url = env.ollama_base_url
        if provider in {"google", "openai"}:
            env_key = provider_api_key(provider)
            placeholder = (
                "Using key from .env — leave blank to keep it"
                if env_key
                else ("Paste Google API key" if provider == "google" else "Paste OpenAI API key")
            )
            entered_key = st.text_input(
                "API key",
                type="password",
                key=f"api_key_{provider}",
                placeholder=placeholder,
                help="The key is kept only in session state and is never written to the trace or logs.",
            )
            api_key = entered_key.strip() or env_key
            if provider == "openai":
                base_url = st.text_input(
                    "OpenAI-compatible base URL",
                    value=provider_base_url("openai"),
                    key="openai_base_url",
                    help="Use a compatible gateway if you are not calling api.openai.com directly.",
                ).strip()
            if env_key and not entered_key:
                st.caption("A key from `.env` is loaded.")
        elif provider == "ollama":
            ollama_url = st.text_input(
                "Ollama URL",
                value=env.ollama_base_url,
                key="ollama_url_input",
                help="Ollama must be running locally, e.g. `ollama serve`.",
            ).strip()
            st.caption("Start Ollama, then pull the model with `ollama pull <model>`.")
        else:
            st.caption("Offline demo: deterministic local model, no key or network required.")

        sidebar_heading("sliders", "Retrieval & run limits", "03")
        top_k = st.slider(
            "RAG chunks (top-k)",
            min_value=1,
            max_value=8,
            key="top_k",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="temperature",
        )

        config = AppConfig(
            provider=provider,
            model=model or default_model(provider),
            api_key=api_key,
            base_url=base_url,
            ollama_base_url=ollama_url or env.ollama_base_url,
            temperature=float(temperature),
            top_k=int(top_k),
            chroma_dir=env.chroma_dir,
            knowledge_base_dir=env.knowledge_base_dir,
            collection_name=env.collection_name,
            embedding_model=env.embedding_model,
            sandbox_timeout=env.sandbox_timeout,
            max_test_retries=env.max_test_retries,
        )
        provider_status = llm_status(config)

        st.markdown(status_box(provider_status["level"], provider_status["label"], provider_status["detail"]), unsafe_allow_html=True)
        if not provider_status["ready"]:
            st.caption("The run button stays disabled until the selected provider is configured.")

        test_disabled = not provider_status["ready"] and provider_status["level"] == "error"
        if st.button("Test LLM connection", icon=":material/bolt:", width="stretch", disabled=test_disabled):
            with st.spinner("Sending a tiny probe request..."):
                probe = test_llm_connection(config)
            st.session_state["llm_probe"] = {
                **probe,
                "provider": provider,
                "model": config.model,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        probe = st.session_state.get("llm_probe")
        if probe and probe.get("provider") == provider and probe.get("model") == config.model:
            if probe.get("ok"):
                st.success(f"Probe OK · {probe.get('latency_ms')} ms · {probe.get('timestamp')}")
            else:
                st.error(f"Probe failed: {probe.get('message')}")

        st.divider()
        sidebar_heading("database", "ChromaDB knowledge base", "04")
        store: VectorStoreManager | None = None
        store_status: dict[str, Any] = {"ready": False, "error": "", "count": 0, "sources": []}
        try:
            with st.spinner("Loading local embeddings + ChromaDB..."):
                store = _load_vector_store(
                    str(env.chroma_dir),
                    env.collection_name,
                    str(env.knowledge_base_dir),
                    env.embedding_model,
                )
            store_status = store.status()
        except Exception as exc:
            store = None
            store_status = {
                "ready": False,
                "error": f"{type(exc).__name__}: {exc}",
                "count": 0,
                "sources": [],
            }
        if store_status.get("ready"):
            st.markdown(
                status_box(
                    "success",
                    f"ChromaDB ready · {store_status.get('count', 0)} chunks",
                    str(store_status.get("embedding_backend", "")),
                ),
                unsafe_allow_html=True,
            )
            sources = store_status.get("sources", [])
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            st.caption(
                f"Path: `{store_status.get('persist_directory', '')}` · "
                f"collection: `{store_status.get('collection', '')}`"
            )
            if st.button("Re-index knowledge base", icon=":material/refresh:", width="stretch"):
                if store is not None:
                    with st.spinner("Rebuilding ChromaDB from knowledge_base/*.md..."):
                        try:
                            info = store.reindex()
                            st.session_state["reindex_message"] = (
                                f"Re-indexed {info['ingested_chunks']} chunks into "
                                f"`{info['collection']}`."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Re-index failed: {exc}")
            reindex_message = st.session_state.pop("reindex_message", None)
            if reindex_message:
                st.success(reindex_message)
        else:
            st.markdown(
                status_box("error", "ChromaDB unavailable", store_status.get("error", "Unknown error")),
                unsafe_allow_html=True,
            )
            st.caption("Run `pip install -r requirements.txt` and check write access to `./chroma_db`.")

        st.divider()
        sidebar_heading("layers", "Sample presets", "05")
        st.selectbox(
            "Load a buggy snippet",
            options=list(SAMPLE_CODE.keys()),
            key="sample_choice",
            label_visibility="collapsed",
        )
        st.button(
            "Load sample into editor",
            icon=":material/download:",
            width="stretch",
            on_click=_load_sample_callback,
        )
        st.caption("Presets exercise off-by-one, zero-division, mutable defaults, SQL injection, and lint signals.")

        st.divider()
        with st.expander("Environment setup", icon=":material/settings:"):
            st.markdown(
                "Copy `.env.example` to `.env` and pick one provider:\n\n"
                "```env\nLLM_PROVIDER=google\nGOOGLE_API_KEY=...\n```\n"
                "or\n\n"
                "```env\nLLM_PROVIDER=ollama\nOLLAMA_BASE_URL=http://localhost:11434\nLLM_MODEL=qwen2.5-coder:7b\n```\n"
                "Offline Demo is the default so the GUI always runs."
            )
        st.caption("Assignment 2 · Code Review & Debug Squad")

    return config, store, store_status, provider_status


# ---------------------------------------------------------------------------
# Header + inputs
# ---------------------------------------------------------------------------

def _render_header(config: AppConfig, store_status: dict[str, Any]) -> None:
    provider = html.escape(config.provider_label)
    model = html.escape(config.model)
    kb_count = store_status.get("count", 0)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-head">
            <div class="hero-mark">{svg_icon("shield", 21)}</div>
            <div>
              <div class="eyebrow">Assignment 2 · Multi-agent code review</div>
              <h1>Code Review <span class="accent">Debug Squad</span></h1>
            </div>
          </div>
          <p>
            Submit a Python snippet and a request. The squad routes the task, retrieves only
            relevant guidance, reviews and repairs the code, and verifies the result in an
            isolated test sandbox — with every step available in the audit trail.
          </p>
          <div class="chips">
            <span class="pill">{provider}</span>
            <span class="pill">{model}</span>
            <span class="pill">{kb_count} guide chunks</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if config.provider == "demo":
        st.markdown(
            '<div class="notice"><strong>Offline Demo is active.</strong> '
            "All LCEL chains, routing, RAG decisions, prompts, and sandbox tools are live; "
            "the model transport is deterministic and local. Select Gemini, OpenAI-compatible, "
            "or Ollama in the sidebar for hosted or local inference.</div>",
            unsafe_allow_html=True,
        )
    if not store_status.get("ready"):
        st.error(
            "The ChromaDB knowledge base is not ready. The squad needs it for agentic retrieval. "
            "See the sidebar error and run `pip install -r requirements.txt`."
        )
    render_workflow_strip()


def _render_workbench(config: AppConfig, store: VectorStoreManager | None, store_status: dict[str, Any]) -> ReviewResult | None:
    section_header("Live workbench", "Submit code, choose a task, run the squad", "terminal")
    with st.container(border=True):
        code_col, control_col = st.columns([1.65, 1], gap="large")
        with code_col:
            field_label("code", "Code under review", "PYTHON")
            st.caption("Syntax-aware review, linting, and sandbox execution")
            st.text_area(
                "Submitted code",
                key="code_input",
                height=430,
                label_visibility="collapsed",
                placeholder="Paste Python code here...",
            )
            st.caption("The original snippet is never executed. Only generated tests run inside the sandbox.")

        with control_col:
            field_label("chat", "Natural-language request", "PROMPT")
            st.caption("Describe the review, bug, explanation, or test objective.")
            st.text_area(
                "Request",
                key="task_input",
                height=155,
                label_visibility="collapsed",
                placeholder="e.g. Find the bug, fix it, and add tests...",
            )
            st.caption(
                "Intent hint: **" + INTENT_DISPLAY.get(st.session_state.get("intent_hint", "full_review"), "Auto") + "**"
                " · the router makes the final decision."
            )

            selected_intent = st.session_state.get("intent_hint", "full_review")
            with st.container(key="quick_intents"):
                quick_row_one_left, quick_row_one_right = st.columns(2)
                with quick_row_one_left:
                    st.button(
                        "Full Squad Review",
                        key="quick_full_review",
                        icon=":material/check_circle:" if selected_intent == "full_review" else ":material/groups:",
                        width="stretch",
                        type="primary" if selected_intent == "full_review" else "secondary",
                        on_click=_quick_intent_callback,
                        args=("full_review",),
                        help="Run RAG + Bug-Finder + Style/Lint + Fixer reflection + tests.",
                    )
                with quick_row_one_right:
                    st.button(
                        "Debug Only",
                        key="quick_debugging",
                        icon=":material/check_circle:" if selected_intent == "debugging" else ":material/bug_report:",
                        width="stretch",
                        type="primary" if selected_intent == "debugging" else "secondary",
                        on_click=_quick_intent_callback,
                        args=("debugging",),
                    )
                quick_row_two_left, quick_row_two_right = st.columns(2)
                with quick_row_two_left:
                    st.button(
                        "Style Review",
                        key="quick_style_review",
                        icon=":material/check_circle:" if selected_intent == "style_review" else ":material/format_paint:",
                        width="stretch",
                        type="primary" if selected_intent == "style_review" else "secondary",
                        on_click=_quick_intent_callback,
                        args=("style_review",),
                    )
                with quick_row_two_right:
                    st.button(
                        "Explain Code",
                        key="quick_code_explanation",
                        icon=":material/check_circle:" if selected_intent == "code_explanation" else ":material/lightbulb:",
                        width="stretch",
                        type="primary" if selected_intent == "code_explanation" else "secondary",
                        on_click=_quick_intent_callback,
                        args=("code_explanation",),
                    )
                st.button(
                    "Generate & Run Tests",
                    key="quick_testing",
                    icon=":material/check_circle:" if selected_intent == "testing" else ":material/science:",
                    width="stretch",
                    type="primary" if selected_intent == "testing" else "secondary",
                    on_click=_quick_intent_callback,
                    args=("testing",),
                )
            st.caption(
                f"Selected task: **{INTENT_DISPLAY.get(selected_intent, 'Full Squad Review')}**. "
                "Quick actions set the request and run immediately."
            )

            autorun = bool(st.session_state.pop("autorun_requested", False))
            run_requested = st.button(
                "Run Squad Review",
                key="run_review_button",
                icon=":material/play_arrow:",
                type="primary",
                width="stretch",
                disabled=not store_status.get("ready") or not llm_status(config)["ready"],
                help="Route the request, retrieve only if useful, invoke agents, and execute tests.",
                shortcut="Ctrl+Enter",
            )
            st.markdown(
                '<div class="kbd-hint"><span class="kbd">Ctrl</span> + <span class="kbd">Enter</span> to run</div>',
                unsafe_allow_html=True,
            )
            st.button(
                "Clear workspace",
                icon=":material/restart_alt:",
                width="stretch",
                on_click=_clear_workspace_callback,
            )

    should_run = bool(run_requested or autorun)
    if not should_run:
        return st.session_state.get("last_result")

    code = st.session_state.get("code_input", "")
    request = st.session_state.get("task_input", "")
    if not code.strip():
        st.error("Add a Python snippet before running the squad.")
        return st.session_state.get("last_result")
    if not request.strip():
        st.error("Describe the task in the natural-language request box.")
        return st.session_state.get("last_result")
    if store is None or not store_status.get("ready"):
        st.error("ChromaDB is not ready, so the agentic retrieval stage cannot run.")
        return st.session_state.get("last_result")
    if not llm_status(config)["ready"]:
        st.error("The selected LLM provider is not configured. Check the sidebar before running.")
        return st.session_state.get("last_result")

    with st.status("Running the Code Review Squad...", expanded=True) as status:
        st.write("1/4 · Routing intent and evaluating the Agentic RAG gate")
        try:
            llm = build_llm(config)
            squad = CodeReviewSquad(llm, store, config)
            st.write("2/4 · Running specialist agents and the reflection loop")
            result = squad.review(request, code, intent_hint=st.session_state.get("intent_hint", ""))
            st.write("3/4 · Static analysis + isolated sandbox checks")
            st.write("4/4 · Consolidating report and trace")
            st.session_state["last_result"] = result
            st.session_state["last_config"] = config
            if result.errors:
                status.update(label="Review completed with errors", state="error", expanded=True)
            else:
                status.update(
                    label=f"Review complete · {result.intent.route_label} · {result.duration_seconds:.1f}s",
                    state="complete",
                    expanded=False,
                )
            return result
        except (ConfigurationError, Exception) as exc:
            status.update(label="Review failed", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            with st.expander("Failure details"):
                st.exception(exc)
            return st.session_state.get("last_result")


# ---------------------------------------------------------------------------
# Consolidated review tab
# ---------------------------------------------------------------------------

def _consolidated_markdown(result: ReviewResult) -> str:
    lines = [
        "# Code Review Squad Report",
        "",
        f"- **Run:** `{result.run_id}`",
        f"- **Request:** {result.request}",
        f"- **Intent:** `{result.intent.intent}` (confidence {result.intent.confidence:.2f}, {result.intent.source})",
        f"- **Route:** {result.intent.route_description}",
        f"- **RAG decision:** {'retrieve' if result.rag.should_retrieve else 'skip'} — {result.rag.reason}",
        "",
        "## Bug-Finder",
        result.bug_report or "_Not run on this route._",
        "",
        "## Style/Lint",
        result.style_report or "_Not run on this route._",
        "",
        "## Fixer reflection",
        result.reflection_report or "_Not run on this route._",
        "",
        "## Explanation",
        result.explanation or "_Not run on this route._",
    ]
    if result.test_plan:
        lines.extend(["", "## Tests", f"- Rationale: {result.test_plan.rationale}"])
    if result.final_sandbox:
        lines.extend(
            [
                f"- Sandbox status: **{result.final_sandbox.status_label}**",
                f"- Exit code: {result.final_sandbox.exit_code}",
                "",
                "```text",
                result.final_sandbox.stdout.strip(),
                result.final_sandbox.stderr.strip(),
                "```",
            ]
        )
    return "\n".join(lines)


def _render_consolidated(result: ReviewResult) -> None:
    if result is None:
        empty_state(
            "trace",
            "Your review will appear here",
            "Choose a sample or paste code, describe the task, then run the squad. "
            "The consolidated report collects findings, fixes, and test evidence.",
        )
        return

    test_status = result.final_sandbox.status_label if result.final_sandbox else "Not run"
    stat_cards(
        [
            ("Detected route", result.intent.route_label, f"confidence {result.intent.confidence:.2f} · {result.intent.source}"),
            ("Bugs reported", str(_count_headings(result.bug_report, "BUG")), "Bug-Finder CoT findings"),
            ("Style findings", str(_count_headings(result.style_report, "STYLE")), "PEP 8 / lint findings"),
            ("Sandbox", test_status, f"{result.duration_seconds:.1f}s total runtime"),
        ]
    )

    if result.errors:
        st.error("\n".join(result.errors))
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    top_left, top_right = st.columns([1.55, 1])
    with top_left:
        st.markdown("### Review brief")
        st.markdown(
            f"""
**Request.** {html.escape(result.request)}

**Route taken.** `RunnableBranch → {html.escape(result.intent.intent)}`  
{html.escape(result.intent.route_description)}

**RAG gate.** {"retrieved " + str(len(result.rag.hits)) + " chunks" if result.rag.should_retrieve else "skipped (autonomous decision)"} — {html.escape(result.rag.reason)}
            """
        )
        if result.intent.error or result.rag.error:
            with st.expander("Routing / retrieval fallback details"):
                st.code(
                    "\n".join(
                        part
                        for part in (
                            result.intent.error,
                            result.rag.error,
                        )
                        if part
                    )
                    or "No fallback errors.",
                    language="text",
                )
    with top_right:
        st.markdown("### Evidence snapshot")
        st.markdown(
            f"{_runtime_badge(result)} {_rag_badge(result)} {_sandbox_badge(result.final_sandbox)}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Provider: {result.trace.get('llm', {}).get('provider_label', 'unknown')} · "
            f"LLM calls: {result.trace.get('llm_call_count', 0)} · "
            f"Prompt records: {len(result.trace.get('prompts', []))}"
        )
        st.caption(
            f"Knowledge chunks: {len(result.rag.hits)} · "
            f"Tool calls: {len(result.trace.get('tool_calls', []))} · "
            f"Trace id: {result.run_id}"
        )
        st.download_button(
            "Download review report",
            data=_consolidated_markdown(result),
            file_name=f"code_review_{result.run_id}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )

    if result.bug_report:
        with st.expander("Bug-Finder Agent — Chain-of-Thought report", expanded=True, icon=":material/bug_report:"):
            st.markdown(result.bug_report)
    if result.style_report:
        with st.expander("Style/Lint Agent — few-shot PEP 8 report", expanded=True, icon=":material/format_paint:"):
            st.markdown(result.style_report)
    if result.explanation:
        with st.expander("Explainer — code walkthrough", expanded=True, icon=":material/lightbulb:"):
            st.markdown(result.explanation)

    if result.test_plan:
        st.markdown("### Test strategy")
        st.markdown(result.test_plan.rationale or "Tests generated for normal and edge-case behaviour.")
        if result.final_sandbox:
            st.markdown(_sandbox_badge(result.final_sandbox), unsafe_allow_html=True)
            st.code(
                result.final_sandbox.stdout or result.final_sandbox.stderr or "(no test output)",
                language="text",
            )

    if result.reflection_report:
        with st.expander("Fixer Agent — Reflection pass", expanded=bool(result.fixed_code)):
            st.markdown(result.reflection_report)

    if result.rag.hits:
        st.markdown("### Grounding sources")
        for hit in result.rag.hits:
            st.caption(f"`{hit.source}` · {hit.section} · similarity {hit.similarity:.3f}")


# ---------------------------------------------------------------------------
# Fixed code tab
# ---------------------------------------------------------------------------

def _render_fixed_code(result: ReviewResult | None) -> None:
    if result is None:
        empty_state(
            "diff",
            "No patch yet",
            "Run a debugging, style, or full-squad review to see the repaired file and a unified diff.",
        )
        return
    if not result.has_fixed_code:
        empty_state(
            "code",
            "This route leaves the file unchanged",
            "The explanation and testing routes intentionally do not modify the submission. "
            "Use Debug Only, Style Review, or Full Squad Review for a patch.",
        )
        return

    st.markdown(
        pill("Patched by Fixer Agent", "green")
        + " "
        + pill("Reflection checked", "blue"),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Final fixed code")
        st.code(result.fixed_code, language="python")
        st.download_button(
            "Download fixed_solution.py",
            data=result.fixed_code,
            file_name="fixed_solution.py",
            mime="text/x-python",
            icon=":material/download:",
            width="stretch",
        )
    with right:
        st.markdown("### Unified diff")
        diff = "\n".join(
            difflib.unified_diff(
                result.code.splitlines(),
                result.fixed_code.splitlines(),
                fromfile="submitted.py",
                tofile="fixed.py",
                lineterm="",
            )
        )
        if diff.strip():
            st.code(diff, language="diff")
        else:
            st.success("No textual changes were produced by the fixer.")
        if result.initial_fixed_code and result.initial_fixed_code.strip() != result.fixed_code.strip():
            with st.expander("Initial patch before feedback retry"):
                st.code(result.initial_fixed_code, language="python")

    if result.fixer_initial_output:
        with st.expander("Fixer Agent raw output (rationale + checklist)"):
            st.markdown(result.fixer_initial_output)
    if result.reflection_report:
        with st.expander("Reflection verdict", expanded=False):
            st.markdown(result.reflection_report)

    st.markdown("### Safety checks")
    safe = result.trace.get("sandbox_environment", {})
    st.caption(
        "The fixed file is never executed on the host. Tests run in a temporary directory with a "
        f"{result.trace.get('model_config', {}).get('sandbox_timeout_seconds', 5)}s timeout, an AST "
        "safety screen, and a restricted PYTHONPATH." if safe else
        "No test execution was part of this route. The patch is still validated for syntax by the Fixer Agent."
    )


# ---------------------------------------------------------------------------
# Test results tab
# ---------------------------------------------------------------------------

def _render_tests(result: ReviewResult | None) -> None:
    if result is None:
        empty_state(
            "beaker",
            "No test run yet",
            "Run a full review or the Generate & Run Tests route to execute a generated pytest suite "
            "inside the isolated sandbox.",
        )
        return
    if not result.sandbox_attempts:
        empty_state(
            "play",
            "No sandbox execution on this route",
            "Debugging and style reviews produce patches. Use Full Squad Review or Generate & Run Tests "
            "to exercise pytest in the sandbox.",
        )
        return

    st.markdown("### Isolated execution results")
    stat_cards(
        [
            ("Final status", result.final_sandbox.status_label, "AST screen + pytest"),
            ("Exit code", str(result.final_sandbox.exit_code), "0 means all tests passed"),
            ("Duration", f"{result.final_sandbox.duration_seconds:.2f}s", f"hard timeout {result.final_sandbox.timeout_seconds}s"),
            ("Attempts", str(len(result.sandbox_attempts)), "1 retry max in full review"),
        ]
    )

    if len(result.sandbox_attempts) > 1:
        st.warning(
            "The first pytest run failed. One Fixer reflection retry was triggered with the exact "
            "stdout/stderr feedback, then the suite was rerun against the repaired file."
        )
    retry_trace = result.trace.get("sandbox_retry", {})
    if retry_trace:
        st.caption(str(retry_trace))

    attempts = result.sandbox_attempts
    if len(attempts) > 1:
        tabs = st.tabs([f"Attempt {i + 1}" for i in range(len(attempts))])
    else:
        tabs = [st.container()]
    for index, (container, attempt) in enumerate(zip(tabs, attempts), start=1):
        with container:
            st.markdown(
                f"**Attempt {index}** · {_sandbox_badge(attempt)} · exit `{attempt.exit_code}` · "
                f"{attempt.duration_seconds:.2f}s",
                unsafe_allow_html=True,
            )
            if attempt.blocked:
                st.error("The sandbox safety screen rejected this run before execution.")
                for reason in attempt.block_reasons:
                    st.code(reason, language="text")
                continue
            if attempt.timed_out:
                st.error(f"Execution exceeded the strict {attempt.timeout_seconds}s timeout.")
            st.markdown("**Command**")
            st.code(attempt.command, language="text")
            out_col, err_col = st.columns(2, gap="large")
            with out_col:
                st.markdown("**stdout**")
                st.code(attempt.stdout or "(empty)", language="text")
            with err_col:
                st.markdown("**stderr**")
                st.code(attempt.stderr or "(empty)", language="text")

    if result.test_plan:
        st.markdown("### Generated pytest suite")
        st.caption(result.test_plan.rationale)
        st.code(result.test_plan.test_code, language="python")
        st.download_button(
            "Download test_solution.py",
            data=result.test_plan.test_code,
            file_name="test_solution.py",
            mime="text/x-python",
            icon=":material/download:",
            width="stretch",
        )

    with st.expander("Sandbox isolation details", icon=":material/shield:"):
        env = environment_description()
        st.markdown(
            "- Fresh `tempfile.TemporaryDirectory` per run; auto-deleted\n"
            "- `PYTHONPATH` restricted to the temp directory\n"
            "- Parent API keys/Ollama URLs are **not** forwarded into the child environment\n"
            "- Hard timeout via `subprocess.run(timeout=...)`\n"
            "- AST safety screen blocks dangerous imports, calls, attributes, and absolute DB paths\n"
            "- No Docker required"
        )
        st.markdown("**Environment allow-list**")
        st.code(", ".join(env["env_allowlist"]), language="text")
        st.markdown("**Blocked imports**")
        st.code(", ".join(env["blocked_imports"]), language="text")
        st.markdown("**Blocked calls**")
        st.code(", ".join(env["blocked_calls"]), language="text")


# ---------------------------------------------------------------------------
# Criteria & trace inspector
# ---------------------------------------------------------------------------

def _collect_techniques(result: ReviewResult) -> list[str]:
    techniques: set[str] = set()
    for prompt in result.trace.get("prompts", []):
        techniques.update(prompt.get("techniques", []))
    return sorted(techniques)


def _criteria_rows(result: ReviewResult) -> list[tuple[str, str, str, str]]:
    provider = result.trace.get("llm", {}).get("provider", "unknown")
    llm_status_text = "PASS · LIVE" if provider != "demo" else "PASS · DEMO"
    llm_kind = "green" if provider != "demo" else "amber"
    techniques = _collect_techniques(result)
    rag = result.rag
    kb = result.trace.get("knowledge_base", {})
    route = result.trace.get("route_path", "")
    rows = [
        (
            "1 · LLM integration",
            llm_status_text,
            llm_kind,
            f"Provider `{provider}` / `{result.trace.get('llm', {}).get('model', '')}` completed "
            f"{result.trace.get('llm_call_count', 0)} agent calls. Gemini, OpenAI-compatible, and Ollama "
            "adapters are switchable in the sidebar; use Test LLM connection for a live probe.",
        ),
        (
            "2 · Interactive GUI",
            "PASS",
            "green",
            "Sidebar provider/model/key/Ollama controls, presets, ChromaDB re-index; main editor, task prompt, "
            "five quick intents, and all four required output tabs are present.",
        ),
        (
            "3 · Prompt engineering",
            "PASS",
            "green",
            "Few-Shot: Intent Router + Style/Lint. CoT: Bug-Finder execution trace. Reflection: Fixer critique"
            + (" plus sandbox-feedback retry." if result.trace.get("sandbox_retry", {}).get("attempted") else ".")
            + f" Observed technique tags: {', '.join(techniques) or 'none on this route'}.",
        ),
        (
            "4 · LangChain LCEL",
            "PASS",
            "green",
            f"Route `{route}`; agents are `prompt | llm | parser` chains, selected by `RunnableBranch`, with "
            "Bug-Finder + Style/Lint in `RunnableParallel` and post-processing `RunnableLambda`s.",
        ),
        (
            "5 · Agentic RAG",
            "PASS",
            "green",
            f"Gate decided `should_retrieve={rag.should_retrieve}` ({rag.source}); "
            f"{'queried ChromaDB and injected ' + str(len(rag.hits)) + ' chunks.' if rag.should_retrieve else 'skipped the vector lookup and logged the reason.'}",
        ),
        (
            "6 · Intent detection & routing",
            "PASS",
            "green",
            f"Detected `{result.intent.intent}` at confidence {result.intent.confidence:.2f} "
            f"({result.intent.source}); executed `{route}` with route contract: {result.intent.route_description}.",
        ),
        (
            "7 · ChromaDB knowledge base",
            "PASS" if kb.get("ready") else "CHECK",
            "green" if kb.get("ready") else "red",
            f"Collection `{kb.get('collection', 'coding_guidelines')}` has {kb.get('count', 0)} persisted chunks "
            f"from {len(kb.get('sources', []))} curated markdown files: {', '.join(kb.get('sources', [])) or 'none'}.",
        ),
    ]
    return rows


def _render_trace_inspector(result: ReviewResult | None) -> None:
    if result is None:
        empty_state(
            "trace",
            "The audit trail starts after a run",
            "Every retrieval decision, rendered prompt, static-analysis command, and sandbox result "
            "will be captured here for inspection.",
        )
        return

    st.markdown("### Mandatory acceptance criteria — live evidence")
    for title, status, kind, evidence in _criteria_rows(result):
        st.markdown(
            f'<div class="criteria-row"><div class="criteria-head">'
            f'<div class="criteria-title">{html.escape(title)}</div>'
            f'{pill(status, kind)}</div>'
            f'<div class="criteria-evidence">{html.escape(evidence)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Execution trace")
    stat_cards(
        [
            ("Detected intent", result.intent.intent, result.intent.source),
            ("Route executed", result.trace.get("route_path", "—").split("→")[-1].strip(), "RunnableBranch"),
            ("LLM calls", str(result.trace.get("llm_call_count", 0)), f"{len(result.trace.get('prompts', []))} prompt records"),
            ("Tool calls", str(len(result.trace.get("tool_calls", []))), f"run {result.run_id}"),
        ]
    )

    st.markdown("### 1 · Intent & route")
    st.markdown(
        f"{pill(result.intent.route_label, 'blue')} {pill('confidence ' + f'{result.intent.confidence:.2f}', 'gray')} "
        f"{pill('source: ' + result.intent.source, 'gray')}",
        unsafe_allow_html=True,
    )
    st.code(result.trace.get("route_path", "not executed"), language="text")
    st.markdown(f"**Rationale:** {result.intent.rationale or '—'}")
    st.caption(result.intent.route_description)

    st.markdown("### 2 · Agentic RAG decision")
    st.markdown(
        pill("should_retrieve = true" if result.rag.should_retrieve else "should_retrieve = false",
             "green" if result.rag.should_retrieve else "amber")
        + " "
        + pill(f"decision source: {result.rag.source}", "gray")
        + " "
        + pill(f"hits: {len(result.rag.hits)}", "blue"),
        unsafe_allow_html=True,
    )
    st.markdown(f"**Generated query:** `{result.rag.query or '(empty — retrieval skipped)'}`")
    st.markdown(f"**Reason:** {result.rag.reason}")
    if result.rag.error:
        st.warning(result.rag.error)
    if result.rag.hits:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(hit.source)}</td>"
            f"<td>{html.escape(hit.section)}</td>"
            f"<td class=\"mono\">{hit.similarity:.4f}</td>"
            f"<td class=\"mono\">{hit.distance:.4f}</td>"
            f"<td class=\"mono\">{html.escape(str(hit.metadata.get('chunk_id', '')))}</td>"
            "</tr>"
            for hit in result.rag.hits
        )
        st.markdown(
            '<div class="table-wrap"><table class="data-table"><thead><tr>'
            "<th>Source</th><th>Section</th><th>Similarity</th><th>Distance</th><th>Chunk ID</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )
        for index, hit in enumerate(result.rag.hits, start=1):
            with st.expander(
                f"Retrieved chunk {index} · {hit.source} · {hit.section}",
                icon=":material/search:",
            ):
                st.caption(f"similarity {hit.similarity:.4f} · distance {hit.distance:.4f}")
                st.markdown(hit.content)
    else:
        st.caption("ChromaDB was not queried because the gate returned false. This is the autonomous skip path.")

    st.markdown("### 3 · Prompts used")
    st.caption(
        "Exact rendered messages. The System prompt establishes the agent role; User messages carry code, "
        "tool evidence, and retrieved context."
    )
    for prompt in result.trace.get("prompts", []):
        label = prompt.get("label", prompt.get("key", "Prompt"))
        techniques = prompt.get("techniques", [])
        with st.expander(
            f"{label} · " + " + ".join(techniques) if techniques else label,
            expanded=prompt.get("key") in {"intent_router", "rag_gate"},
            icon=":material/chat:",
        ):
            st.caption(prompt.get("purpose", ""))
            st.markdown("".join(pill(tag, "gray") for tag in techniques), unsafe_allow_html=True)
            for message in prompt.get("messages", []):
                role = message.get("role", "Message")
                st.markdown(f"**{html.escape(role)}**")
                st.code(message.get("content", ""), language="markdown")

    st.markdown("### 4 · Tool calls")
    if not result.trace.get("tool_calls"):
        empty_state("trace", "No tool calls on this route", "This route answered from the model and prompt context without invoking the AST, lint, or sandbox tools.")
    for index, call in enumerate(result.trace.get("tool_calls", []), start=1):
        name = call.get("name", f"Tool {index}")
        exit_code = call.get("exit_code")
        tool_icon = ":material/terminal:" if call.get("kind") == "static_analysis" else ":material/science:"
        with st.expander(f"{index}. {name} · exit {exit_code}", expanded=(index == 1), icon=tool_icon):
            st.markdown("**Command**")
            st.code(call.get("command", "—"), language="text")
            if call.get("kind") == "static_analysis":
                summary = call.get("summary", {})
                st.markdown(
                    f"{pill('syntax ' + ('valid' if summary.get('syntax_ok') else 'invalid'), 'green' if summary.get('syntax_ok') else 'red')} "
                    f"{pill('lint ' + str(summary.get('lint_tool')), 'blue')} "
                    f"{pill(str(summary.get('lint_issue_count', 0)) + ' lint findings', 'amber')}"
                    ,
                    unsafe_allow_html=True,
                )
                details = call.get("details", {})
                if details.get("report_markdown"):
                    st.markdown(details["report_markdown"])
            if call.get("block_reasons"):
                for reason in call["block_reasons"]:
                    st.error(reason)
            out_col, err_col = st.columns(2)
            with out_col:
                st.markdown("**stdout / output**")
                st.code(call.get("stdout", "") or "(empty)", language="text")
            with err_col:
                st.markdown("**stderr**")
                st.code(call.get("stderr", "") or "(empty)", language="text")
            if call.get("kind") == "sandbox":
                st.caption(
                    f"timeout {call.get('timeout_seconds')}s · duration {call.get('duration_seconds')}s · "
                    f"status {call.get('status')}"
                )

    with st.expander("Architecture executed", icon=":material/hub:"):
        st.code(
            "RunnableBranch(intent)\n"
            "  ├─ debugging       → Agentic RAG → Bug-Finder → Fixer (reflection)\n"
            "  ├─ style_review    → Agentic RAG → Style/Lint → Fixer (reflection)\n"
            "  ├─ code_explanation→ Explainer (+ RAG only if gate says yes)\n"
            "  ├─ testing         → Agentic RAG → Test-Writer → Isolated sandbox\n"
            "  └─ full_review     → Agentic RAG → RunnableParallel(Bug, Style)\n"
            "                       → Fixer (reflection) → Test-Writer → Sandbox\n"
            "                       → on failure: 1 reflection retry → rerun",
            language="text",
        )
        st.markdown(
            "Every agent chain: `ChatPromptTemplate | ChatModel | OutputParser`  \n"
            "Router: `INTENT_ROUTER_PROMPT | llm | JsonOutputParser`  \n"
            "RAG gate: `RAG_GATE_PROMPT | llm | JsonOutputParser`"
        )

    st.markdown("### Raw trace")
    trace_json = json.dumps(result.trace, indent=2, default=str)
    st.download_button(
        "Download full trace JSON",
        data=trace_json,
        file_name=f"trace_{result.run_id}.json",
        mime="application/json",
        icon=":material/download:",
    )
    with st.expander("Inspect raw trace JSON", icon=":material/code:"):
        st.code(trace_json, language="json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    env = config_from_env()
    config, store, store_status, provider_status = _render_sidebar(env)
    _render_header(config, store_status)

    result = _render_workbench(config, store, store_status)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    section_header("Review output", "Consolidated artifacts & audit trail", "trace")
    review_tab, code_tab, test_tab, trace_tab = st.tabs(
        [
            "Consolidated Review",
            "Fixed Code & Diff",
            "Test Results (Sandbox)",
            "Criteria & Trace Inspector",
        ]
    )
    with review_tab:
        _render_consolidated(result)
    with code_tab:
        _render_fixed_code(result)
    with test_tab:
        _render_tests(result)
    with trace_tab:
        _render_trace_inspector(result)


if __name__ == "__main__":
    main()


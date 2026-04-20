"""

### Main Dashboard ###

# Purpose: App.py is the the main web application.  
           Run this file to start the dashboard: python app.py
           Then open  http://localhost:8050  in local browser.

# Layout : Seperate Tabs for Intercompany Reconciliation & Bank Reconciliation 
           Each layout consists of Data Ingestion, Rules Configuration, Results and AI insights
           
"""

###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries

import numpy as np
import pandas as pd
import io
import json
import base64
import traceback
import time

# Dashboard Library used : Plotly Dash

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, no_update
import dash_bootstrap_components as dbc


###----------------------------------------------------------------------------------------------------------------###
# Import Local Modules from pipeline, engine, inferences and AI config and Agent py files

from config import(
    APP_TITLE, APP_PORT, APP_DEBUG, APP_HOST,
    COLOR, RULES_CONFIG, AI_PROVIDERS, AI_DEFAULT_CONFIG,
    AI_MODEL_SUGGESTIONS, AI_SYSTEM_PROMPT,
    DATE_TOLERANCE, AMOUNT_TOLERANCE, FUZZY_THRESHOLD,
)

from data_ingestion import load_file, preprocess, get_column_options, df_to_store, store_to_df
from engine import run_full_reconciliation, get_unreconciled, get_manual_review_items
from inferences import compute_totals, compute_reconciliation_summary, compute_currency_summary, compute_amount_diff_summary, build_ai_context
from ai_agent import AIAgent


###----------------------------------------------------------------------------------------------------------------###
# Application Initialisation

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions= True,
    prevent_initial_callbacks="initial_duplicate",
    title= APP_TITLE,
    )
#Expose Flask server for deployment
server = app.server 

###----------------------------------------------------------------------------------------------------------------###
# Layout Design (in HTML)

def card(children, style=None, className="mb-3 p-3"):
    base = {"background": COLOR["card"], "borderRadius": "8px",
            "border": f"1px solid {COLOR['border']}", "boxShadow": "0 1px 4px rgba(0,0,0,.07)"}
    if style:
        base.update(style)
    return html.Div(children, style=base, className=className)


def kpi_card(title, value, color=COLOR["accent"], icon=""):
    """Single KPI badge card used in the Results tab header row."""
    return dbc.Col(card([
        html.Div(f"{icon} {title}", style={"fontSize": "0.78rem", "color": COLOR["muted"], "marginBottom": "4px"}),
        html.Div(str(value), style={"fontSize": "1.6rem", "fontWeight": "700", "color": color}),
    ], style={"textAlign": "center", "minWidth": "130px"}), width="auto")


def section_header(text):
    """Bold section label used inside tabs."""
    return html.Div(text, style={
        "fontSize": "0.9rem", "fontWeight": "700",
        "color": COLOR["primary"], "marginBottom": "8px",
        "borderBottom": f"2px solid {COLOR['primary']}", "paddingBottom": "4px",
    })


def table_from_df(df, table_id, page_size=15):
    """Build a styled dash_table.DataTable from a DataFrame."""
    if df is None or df.empty:
        return html.Div("No data available.", style={"color": COLOR["muted"], "padding": "20px"})

    # Drop internal helper columns (start with _) before displaying
    display_cols = [c for c in df.columns if not str(c).startswith("_")]
    display_df   = df[display_cols].copy()

    return dash_table.DataTable(
        id          = table_id,
        columns     = [{"name": c, "id": c} for c in display_cols],
        data        = display_df.head(2000).to_dict("records"),   # cap at 2000 rows for speed
        page_size   = page_size,
        page_action = "native",
        sort_action = "native",
        filter_action="native",
        style_table = {"overflowX": "auto"},
        style_header = {
            "backgroundColor": COLOR["primary"],
            "color": "white", "fontWeight": "600",
            "fontSize": "0.8rem", "padding": "8px 10px",
        },
        style_cell  = {
            "fontSize": "0.78rem", "padding": "6px 10px",
            "border": f"1px solid {COLOR['border']}",
            "fontFamily": "monospace",
            "textAlign": "left",
        },
        style_data_conditional=[
            {
                "if"             : {"filter_query": "{Matched} eq 'True'"},
                "backgroundColor": "#dcfce7",
            },
            {
                "if"             : {"filter_query": "{Matched} eq 'False'"},
                "backgroundColor": "#fef2f2",
            },
        ],
    )


###----------------------------------------------------------------------------------------------------------------###
# HELP File

help_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Help & Credits"), close_button=True),
    dbc.ModalBody([
        dcc.Markdown(id="help-content", children="Loading help file...",
                     style={"fontSize": "0.9rem"}),
    ]),
    dbc.ModalFooter(dbc.Button("Close", id="help-close", color="secondary")),
], id="help-modal", size="lg", is_open=False)

# Default help/credits text (edit assets/help.md to customise)
HELP_TEXT = """
## Reconciliation Modules — Help & Credits

### How to Use
1. **Data Ingestion tab** — Upload your Ledger and Bank Statement files (CSV / XLS / XLSX).
   Map the **Date**, **Amount**, and **Narration** columns for each file.
   Click **Proceed to Rules**.

2. **Rules & Configuration tab** — Select which reconciliation rules to apply.
   Tune the **Date Tolerance**, **Amount Tolerance**, and **Fuzzy Threshold** KPIs.
   Click **Run Reconciliation**.

3. **Results tab** — View matched/unmatched transactions, rule breakdowns,
   donut chart, and download the reconciled sheets as Excel or CSV.

4. **AI Agent tab** — Configure your LLM provider and API key, test the connection,
   then ask questions about your reconciliation results in plain English.

---

### Reconciliation Rules (in priority order)
| Rule | Description |
|------|-------------|
| Duplicate Detection | Removes exact duplicate rows within each dataset |
| Narration Exact | Exact narration text match |
| Narration Exact/Fuzzy | Exact OR fuzzy text match (score ≥ threshold) |
| Narration + Date Exact | Exact narration + same date |
| Narration + Date Range | Exact narration + date within tolerance |
| Narration Fuzzy + Date Range | Fuzzy narration + date within tolerance |
| Date Exact | Same date (any narration) |
| Date Range | Date within tolerance (widest net) |
| Many-to-One | Several ledger lines sum to one bank line |
| One-to-Many | One ledger line equals several bank lines summed |

---

### Supported LLM Providers
OpenAI · Anthropic (Claude) · Groq · Google Gemini

---

### Credits & Project Information

**Project:** Reconciliation Automation Platform
**Organisation:** ISB 
**Collaboration:** EY (Ernst & Young)

*Add your team credits here — edit assets/help.md*

---
*Version 1.0 — Built with Plotly Dash*
"""

###----------------------------------------------------------------------------------------------------------------###
# Data Ingestion Tab

def upload_block(title, upload_id, preview_id):
    """Reusable upload + preview block for ledger OR bank file."""
    return card([
        section_header(title),
        dcc.Upload(
            id=upload_id,
            children=html.Div([
                html.I(className="fa fa-cloud-upload", style={"fontSize": "2rem", "color": COLOR["accent"]}),
                html.Br(),
                "Drag & Drop or ",
                html.A("Browse File", style={"color": COLOR["accent"], "fontWeight": "600"}),
                html.Br(),
                html.Small("Supports CSV · XLS · XLSX", style={"color": COLOR["muted"]}),
            ], style={"textAlign": "center", "padding": "20px"}),
            style={
                "border": f"2px dashed {COLOR['accent']}", "borderRadius": "8px",
                "backgroundColor": "#f0f7ff", "cursor": "pointer",
            },
            multiple=False,
        ),
        html.Div(id=preview_id, style={"marginTop": "12px"}),
    ])


def mapping_block(source, date_id, amount_id, narration_id):
    """Column mapping dropdowns for one source file."""
    return card([
        section_header(f"{source} — Column Mapping"),
        dbc.Row([
            dbc.Col([
                html.Label("📅 Date Column *", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Dropdown(id=date_id, placeholder="Select date column...",
                             style={"fontSize": "0.82rem"}),
            ], width=4),
            dbc.Col([
                html.Label("💰 Amount Column *", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Dropdown(id=amount_id, placeholder="Select amount column...",
                             style={"fontSize": "0.82rem"}),
            ], width=4),
            dbc.Col([
                html.Label("📝 Narration Column(s) * (select one or more)",
                           style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Dropdown(id=narration_id, multi=True,
                             placeholder="Select narration column(s)...",
                             style={"fontSize": "0.82rem"}),
            ], width=4),
        ]),
    ])


tab_data_ingestion = html.Div([
    dbc.Row([
        dbc.Col(upload_block("📁 Ledger File",         "upload-ledger", "preview-ledger"), width=6),
        dbc.Col(upload_block("📁 Bank Statement File", "upload-bank",   "preview-bank"),   width=6),
    ]),
    mapping_block("Ledger",         "ledger-date",   "ledger-amount",   "ledger-narration"),
    mapping_block("Bank Statement", "bank-date",     "bank-amount",     "bank-narration"),
    html.Div(id="mapping-status", style={"marginBottom": "8px"}),
    dbc.Button("✅ Confirm Mapping & Proceed to Rules →",
               id="btn-proceed-rules", color="primary", size="lg",
               className="w-100", disabled=True),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# Rules and Reconciliation Tab

rule_checkboxes = [
    dbc.Checklist(
        id="rules-checklist",
        options=[{"label": r["label"], "value": r["key"]} for r in RULES_CONFIG],
        value=[r["key"] for r in RULES_CONFIG if r["enabled"]],   # all ON by default
        style={"fontSize": "0.86rem", "lineHeight": "2"},
    )
]

tab_rules = html.Div([
    dbc.Row([
        # ── Left: KPI sliders ─────────────────────────────────────────────
        dbc.Col(card([
            section_header("⚙️ Reconciliation Parameters"),
            html.Label("📅 Date Tolerance (days)", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Slider(id="kpi-date-tol", min=0, max=120, step=1, value=DATE_TOLERANCE,
                       marks={i: str(i) for i in range(0, 121, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            html.Label("💰 Amount Tolerance", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Input(id="kpi-amount-tol", type="number", value=AMOUNT_TOLERANCE,
                      min=0, step=0.01, debounce=True,
                      style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                             "border": f"1px solid {COLOR['border']}"}),
            html.Br(), html.Br(),
            html.Label("🔍 Fuzzy Match Threshold (0–100)", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Slider(id="kpi-fuzzy-tol", min=0, max=100, step=5, value=FUZZY_THRESHOLD,
                       marks={i: str(i) for i in range(0, 101, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]), width=5),

        # ── Right: Rule checkboxes ─────────────────────────────────────────
        dbc.Col(card([
            section_header("📋 Reconciliation Rules"),
            dbc.Row([
                dbc.Col(dbc.Button("✅ Enable All",  id="btn-enable-all",  color="success",
                                   size="sm", outline=True, className="me-2"), width="auto"),
                dbc.Col(dbc.Button("⬜ Disable All", id="btn-disable-all", color="secondary",
                                   size="sm", outline=True), width="auto"),
            ], className="mb-3"),
            *rule_checkboxes,
        ]), width=7),
    ]),
    html.Div(id="rules-status", className="mb-2"),
    dbc.Button("▶  Run Reconciliation", id="btn-run-recon", color="primary",
               size="lg", className="w-100", disabled = True),
    dbc.Progress(id="recon-progress", 
                 value=0, 
                 animated = True, striped = True, style={"height": "10px", "marginTop": "10px",
                                                        "transition": "width 0.4s ease"},
                 className = "mb-2",
                ),
    html.Div(id = "progress-label", 
             style={"fontSize": "0.78rem", "color": COLOR["muted"],
                "textAlign": "center", "marginBottom": "8px"}),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# Results Tab

tab_results = html.Div([
    # KPI row (populated by callback)
    dbc.Row(id="results-kpi-row", className="mb-3 g-2"),

    # Sub-navigation within Results
    dbc.Tabs(id="results-subtabs", children=[

        dbc.Tab(label="✅ Reconciled Ledger", tab_id="tab-recon-ledger", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("⬇ Download Excel", id="btn-dl-ledger",
                                       color="success", size="sm", outline=True), width="auto"),
                    dbc.Col(dbc.Button("⬇ Download CSV", id="btn-dl-ledger-csv",
                                       color="secondary", size="sm", outline=True), width="auto"),
                ], className="mb-2"),
                dcc.Download(id="dl-ledger"),
                dcc.Download(id="dl-ledger-csv"),
                html.Div(id="table-recon-ledger"),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="🏦 Reconciled Bank", tab_id="tab-recon-bank", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("⬇ Download Excel", id="btn-dl-bank",
                                       color="success", size="sm", outline=True), width="auto"),
                    dbc.Col(dbc.Button("⬇ Download CSV", id="btn-dl-bank-csv",
                                       color="secondary", size="sm", outline=True), width="auto"),
                ], className="mb-2"),
                dcc.Download(id="dl-bank"),
                dcc.Download(id="dl-bank-csv"),
                html.Div(id="table-recon-bank"),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="❌ Unreconciled Items", tab_id="tab-unrecon", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("⬇ Download Unreconciled (Excel)", id="btn-dl-unrecon",
                                       color="danger", size="sm", outline=True), width="auto"),
                ], className="mb-2"),
                dcc.Download(id="dl-unrecon"),
                html.Div(id="table-unrecon"),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="📊 Rule Summary & Chart", tab_id="tab-rule-chart", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id="table-rule-summary"), width=5),
                    dbc.Col(dcc.Graph(id="chart-donut", config={"displayModeBar": False}), width=7),
                ]),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="🔍 Manual Review", tab_id="tab-review", children=[
            html.Div([
                html.Div(
                    "These transactions were matched via Group rules (Many-to-One / One-to-Many) "
                    "and should be reviewed before final sign-off.",
                    style={"fontSize": "0.84rem", "color": COLOR["warning"],
                           "marginBottom": "12px", "padding": "8px",
                           "background": "#fffbeb", "borderRadius": "6px",
                           "border": f"1px solid {COLOR['warning']}"},
                ),
                html.Div(id="table-review"),
            ], style={"padding": "12px"}),
        ]),

    ], active_tab="tab-recon-ledger"),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# AI Agent Tab

tab_ai_agent = html.Div([
    # LLM provider configuration panel
    card([
        section_header("🔧 AI Agent Configuration"),
        dbc.Row([
            dbc.Col([
                html.Label("Provider", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Dropdown(id="ai-provider", options=[{"label": p, "value": p} for p in AI_PROVIDERS],
                             value=AI_DEFAULT_CONFIG["provider"],
                             clearable=False, style={"fontSize": "0.82rem"}),
            ], width=3),
            dbc.Col([
                html.Label("Model", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ai-model", value=AI_DEFAULT_CONFIG["model"],
                          debounce=True, style={"width": "100%", "padding": "6px",
                                                "borderRadius": "4px",
                                                "border": f"1px solid {COLOR['border']}"}),
                html.Small(id="ai-model-hint", style={"color": COLOR["muted"], "fontSize": "0.75rem"}),
            ], width=3),
            dbc.Col([
                html.Label("API Key", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ai-apikey", type="password", placeholder="Paste API key here…",
                          debounce=True, style={"width": "100%", "padding": "6px",
                                                "borderRadius": "4px",
                                                "border": f"1px solid {COLOR['border']}"}),
            ], width=3),
            dbc.Col([
                html.Label("Max Tokens", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ai-max-tokens", type="number", value=AI_DEFAULT_CONFIG["max_tokens"],
                          min=100, max=4096, step=100, debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
            ], width=2),
            dbc.Col([
                html.Label("Temperature", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ai-temperature", type="number",
                          value=AI_DEFAULT_CONFIG["temperature"],
                          min=0.0, max=1.0, step=0.05, debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
            ], width=1),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("💾 Save & Test Connection", id="btn-test-ai",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(html.Div(id="ai-conn-status",
                             style={"fontSize": "0.84rem", "paddingTop": "6px",
                                    "fontWeight": "600"}), width="auto"),
        ]),
    ]),

    # Chat Interface
    card([
        section_header("💬 Chat with Reconciliation Data"),

        # Scrollable chat history
        html.Div(id="chat-history",
                 style={"height": "380px", "overflowY": "auto",
                        "border": f"1px solid {COLOR['border']}",
                        "borderRadius": "6px", "padding": "12px",
                        "background": COLOR["background"],
                        "fontSize": "0.86rem", "lineHeight": "1.6",
                        "marginBottom": "10px"}),

        # Input row
        dbc.Row([
            dbc.Col(dcc.Input(id="chat-input", placeholder="Ask a question about your reconciliation…",
                              debounce=False, n_submit=0,
                              style={"width": "100%", "padding": "8px 12px",
                                     "borderRadius": "6px",
                                     "border": f"1px solid {COLOR['accent']}",
                                     "fontSize": "0.86rem"}), width=9),
            dbc.Col(dbc.Button("➤ Send", id="btn-send-chat", color="primary"), width=2),
            dbc.Col(dbc.Button("🗑 Clear", id="btn-clear-chat", color="secondary",
                               outline=True), width=1),
        ], className="g-2"),
    ]),

    # Option to export the chat results
    dbc.Row([
        dbc.Col(dbc.Button("⬇ Export Chat (TXT)", id="btn-export-chat",
                           color="secondary", size="sm", outline=True), width="auto"),
        dcc.Download(id="dl-chat"),
    ], className="mt-2"),

], style={"padding": "16px"})

###----------------------------------------------------------------------------------------------------------------###
# MAIN LAYOUT & TABS

app.layout = html.Div([

    # Top Header
    html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span("📊 ", style={"fontSize": "1.5rem"}),
                html.Span(APP_TITLE, style={"fontSize": "1.5rem", "fontWeight": "800",
                                            "color": "white", "letterSpacing": "-0.3px"}),
            ]), width="auto"),
            dbc.Col(
                dbc.Button("? Help", id="btn-help", color="light", outline=True, size="sm"),
                width="auto", style={"marginLeft": "auto"},
            ),
        ], align="center"),
    ], style={
        "background": f"linear-gradient(135deg, {COLOR['primary']} 0%, #2563eb 100%)",
        "padding": "14px 24px",
    }),

    help_modal,

    # Main Module Tabs (Initial screen on dashboard)
    dcc.Tabs(id="main-tabs", value="tab-bank", children=[

        # Intercompany Reconciliation Tab
        dcc.Tab(label="🏢  Inter Company Reconciliation", value="tab-ic",
                style={"fontWeight": "600", "padding": "12px 20px"},
                selected_style={"fontWeight": "700", "color": COLOR["primary"],
                                "borderTop": f"3px solid {COLOR['primary']}",
                                "padding": "12px 20px"},
                children=[
                    html.Div([
                        html.Div("🚧  Inter Company Reconciliation module is coming soon.",
                                 style={"textAlign": "center", "padding": "80px 0",
                                        "fontSize": "1.1rem", "color": COLOR["muted"]}),
                    ])
                ]),

        # Bank Reconciliation Tab
        dcc.Tab(label="🏦  Bank Reconciliation", value="tab-bank",
                style={"fontWeight": "600", "padding": "12px 20px"},
                selected_style={"fontWeight": "700", "color": COLOR["primary"],
                                "borderTop": f"3px solid {COLOR['primary']}",
                                "padding": "12px 20px"},
                children=[
                    html.Div([
                        dbc.Tabs(id="bank-subtabs", active_tab ="subtab-ingestion", children=[
                            dbc.Tab(label="📁  Data Ingestion",      tab_id="subtab-ingestion",
                                    children=tab_data_ingestion),
                            dbc.Tab(label="⚙️  Rules & Config",      tab_id="subtab-rules",
                                    children=tab_rules),
                            dbc.Tab(label="📊  Results",              tab_id="subtab-results",
                                    children=tab_results),
                            dbc.Tab(label="🤖  AI Agent",             tab_id="subtab-agent",
                                    children=tab_ai_agent),
                        ], className="mt-0"),
                    ]),
                ]),

    ], style={"fontFamily": "Inter, Segoe UI, sans-serif"}),

    # Stores all the details from dataframe via json for active session
    dcc.Store(id="store-ledger-raw"),                 # raw loaded ledger (list of records)
    dcc.Store(id="store-bank-raw"),                   # raw loaded bank
    dcc.Store(id="store-ledger-cols"),                # column names list for ledger dropdowns
    dcc.Store(id="store-bank-cols"),                  # column names list for bank dropdowns
    dcc.Store(id="store-ledger-clean"),               # preprocessed ledger after mapping
    dcc.Store(id="store-bank-clean"),                 # preprocessed bank after mapping
    dcc.Store(id="store-recon-running", data=False),  # For progress bar           
    dcc.Interval(id="interval-progress", interval=400, n_intervals=0, disabled=True),       
    dcc.Store(id="store-results-ledger"),  # reconciled ledger
    dcc.Store(id="store-results-bank"),    # reconciled bank
    dcc.Store(id="store-rule-summary"),    # rule summary DataFrame
    dcc.Store(id="store-ai-config"),       # AI agent config dict
    dcc.Store(id="store-chat-history"),    # list of {role, content} dicts
    dcc.Store(id="store-ai-context"),      # built AI context string

], style={"fontFamily": "Inter, Segoe UI, sans-serif",
          "background": COLOR["background"], "minHeight": "100vh"})

 
# Help Call back

@app.callback(
    Output("help-modal",   "is_open"),
    Output("help-content", "children"),
    Input("btn-help",   "n_clicks"),
    Input("help-close", "n_clicks"),
    State("help-modal",   "is_open"),
    prevent_initial_call=True,
)
def toggle_help(open_clicks, close_clicks, is_open):
    """Open or close the help modal; load help text from HELP_TEXT constant."""
    return not is_open, HELP_TEXT


# Option for File uploads

def _parse_upload(contents, filename):
    """Helper used by both ledger and bank upload callbacks."""
    if contents is None:
        return None, None, [], "No file uploaded."
    df, err = load_file(contents, filename)
    if err:
        return None, None, [], f"❌ {err}"
    cols    = get_column_options(df)
    records = df_to_store(df)
    return records, cols, cols, f"✅ Loaded **{filename}** — {len(df):,} rows × {len(df.columns)} columns"


@app.callback(
    Output("store-ledger-raw",  "data"),
    Output("store-ledger-cols", "data"),
    Output("ledger-date",       "options"),
    Output("ledger-amount",     "options"),
    Output("ledger-narration",  "options"),
    Output("preview-ledger",    "children"),
    Input("upload-ledger", "contents"),
    State("upload-ledger", "filename"),
    prevent_initial_call=True,
)
def upload_ledger(contents, filename):
    records, cols, _, msg = _parse_upload(contents, filename)
    preview = dbc.Alert(msg, color="success" if records else "danger",
                        style={"fontSize": "0.82rem", "padding": "8px 12px"})
    return records, cols, cols or [], cols or [], cols or [], preview


@app.callback(
    Output("store-bank-raw",  "data"),
    Output("store-bank-cols", "data"),
    Output("bank-date",       "options"),
    Output("bank-amount",     "options"),
    Output("bank-narration",  "options"),
    Output("preview-bank",    "children"),
    Input("upload-bank", "contents"),
    State("upload-bank", "filename"),
    prevent_initial_call=True,
)
def upload_bank(contents, filename):
    records, cols, _, msg = _parse_upload(contents, filename)
    preview = dbc.Alert(msg, color="success" if records else "danger",
                        style={"fontSize": "0.82rem", "padding": "8px 12px"})
    return records, cols, cols or [], cols or [], cols or [], preview


# Proceed button for confirming the mapping of columns

@app.callback(
    Output("btn-proceed-rules", "disabled"),
    Output("mapping-status",    "children"),
    Input("store-ledger-raw",  "data"),
    Input("store-bank-raw",    "data"),
    Input("ledger-date",       "value"),
    Input("ledger-amount",     "value"),
    Input("ledger-narration",  "value"),
    Input("bank-date",         "value"),
    Input("bank-amount",       "value"),
    Input("bank-narration",    "value"),
    prevent_initial_call=True,
)
def check_mapping_complete(l_raw, b_raw, l_date, l_amt, l_nar, b_date, b_amt, b_nar):
    """Enable the 'Proceed' button only when both files are loaded and all 3 mandatory
    columns are mapped for each file."""
    checks = [l_raw, b_raw, l_date, l_amt, l_nar, b_date, b_amt, b_nar]
    if all(c for c in checks):
        msg = dbc.Alert("✅ All columns mapped. Ready to proceed to Rules.",
                        color="success", style={"fontSize": "0.82rem", "padding": "8px 12px"})
        return False, msg
    missing = []
    if not l_raw:   missing.append("Upload Ledger file")
    if not b_raw:   missing.append("Upload Bank file")
    if not l_date:  missing.append("Map Ledger Date column")
    if not l_amt:   missing.append("Map Ledger Amount column")
    if not l_nar:   missing.append("Map Ledger Narration column(s)")
    if not b_date:  missing.append("Map Bank Date column")
    if not b_amt:   missing.append("Map Bank Amount column")
    if not b_nar:   missing.append("Map Bank Narration column(s)")
    msg = dbc.Alert("⚠️ Remaining: " + " · ".join(missing),
                    color="warning", style={"fontSize": "0.82rem", "padding": "8px 12px"})
    return True, msg


# Rules selection

@app.callback(
    Output("store-ledger-clean",  "data"),
    Output("store-bank-clean",    "data"),
    Output("bank-subtabs",        "active_tab"),
    Output("btn-run-recon",       "disabled"),
    Input("btn-proceed-rules", "n_clicks"),
    State("store-ledger-raw",  "data"),
    State("store-bank-raw",    "data"),
    State("ledger-date",       "value"),
    State("ledger-amount",     "value"),
    State("ledger-narration",  "value"),
    State("bank-date",         "value"),
    State("bank-amount",       "value"),
    State("bank-narration",    "value"),
    prevent_initial_call=True,
)
def proceed_to_rules(n, l_raw, b_raw, l_date, l_amt, l_nar, b_date, b_amt, b_nar):
    """Preprocess both DataFrames with the user's column mappings, then
    switch to the Rules & Config tab."""
    if not n:
        return no_update, no_update, no_update, True

    l_df = store_to_df(l_raw)
    b_df = store_to_df(b_raw)

    l_mapping = {"date_col": l_date, "amount_col": l_amt, "narration_cols": l_nar or []}
    b_mapping = {"date_col": b_date, "amount_col": b_amt, "narration_cols": b_nar or []}

    l_clean, l_err = preprocess(l_df, l_mapping, "Ledger")
    b_clean, b_err = preprocess(b_df, b_mapping, "Bank")
    print(f"[DEBUG] l_err = {l_err}")  
    print(f"[DEBUG] b_err = {b_err}")       
           
    if l_err or b_err:
        # If preprocessing fails, stay on current tab — errors shown elsewhere
        return no_update, no_update, "subtab-ingestion", True

    return df_to_store(l_clean), df_to_store(b_clean), "subtab-rules", False


# Selection of Rules (Enable/Disable)

@app.callback(
    Output("rules-checklist", "value"),
    Input("btn-enable-all",  "n_clicks"),
    Input("btn-disable-all", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_all_rules(enable_n, disable_n):
    all_keys = [r["key"] for r in RULES_CONFIG]
    triggered = ctx.triggered_id
    if triggered == "btn-enable-all":
        return all_keys
    return []


# Run Reconciliation
# Callback for progress of reconciliation
@app.callback(
    Output("recon-progress", "value", allow_duplicate=True),
    Input("store-recon-running", "data"),
    prevent_initial_call=True,
)
def reset_progress(running):
    if running:
        return 10
    return no_update
           
# Callback for actual reconciliation          
@app.callback(
    Output("store-results-ledger", "data"),
    Output("store-results-bank",   "data"),
    Output("store-rule-summary",   "data"),
    Output("store-ai-context",     "data"),
    Output("bank-subtabs",         "active_tab", allow_duplicate=True),
    Output("rules-status",         "children"),
    Output("recon-progress", "value", allow_duplicate=True),  
    Output("store-recon-running", "data"),     
    Input("btn-run-recon", "n_clicks"),
    State("store-ledger-clean",  "data"),
    State("store-bank-clean",    "data"),
    State("rules-checklist",     "value"),
    State("kpi-date-tol",        "value"),
    State("kpi-amount-tol",      "value"),
    State("kpi-fuzzy-tol",       "value"),
    prevent_initial_call=True,
)
def run_reconciliation(n_clicks, l_clean, b_clean, enabled_rules,
                       date_tol, amount_tol, fuzzy_tol):
    """
    Main reconciliation callback.
    Reads cleaned DataFrames from stores, runs the engine, saves results back.
    """
    if not n_clicks:
               return no_update, no_update, no_update, no_update, no_update, no_update, True
    if l_clean is None or b_clean is None:
               warn = dbc.Alert(
                          "⚠️ No preprocessed data found. Go to Data Ingestion tab and click Confirm Mapping first.",
        color="warning", style={"fontSize": "0.84rem", "padding": "8px 12px"}
               )
               return no_update, no_update, no_update, no_update, no_update, warn, 0, False                        
    try:
        l_df = store_to_df(l_clean)
        b_df = store_to_df(b_clean)

        params = {
            "date_tolerance"  : int(date_tol   or DATE_TOLERANCE),
            "amount_tolerance": float(amount_tol or AMOUNT_TOLERANCE),
            "fuzzy_threshold" : int(fuzzy_tol   or FUZZY_THRESHOLD),
        }
        time.sleep(0.5)  # For each rule (effective for online deployment)   
        l_out, b_out, _match_log, rule_summary = run_full_reconciliation(
            l_df, b_df, params, enabled_rules or []
        )
        time.sleep(0.5)  # For each rule (effective for online deployment)   
        # Build AI context string now so it's ready when user opens AI tab
        ai_context = build_ai_context(l_out, b_out, rule_summary, params)

        matched_l = int(l_out["Matched"].sum())
        matched_b = int(b_out["Matched"].sum())
        total_l   = len(l_out)
        total_b   = len(b_out)

        status_msg = dbc.Alert(
            f"✅ Reconciliation complete — "
            f"Ledger: {matched_l}/{total_l} matched | "
            f"Bank: {matched_b}/{total_b} matched",
            color="success", style={"fontSize": "0.84rem", "padding": "8px 12px"},
        )

        return (df_to_store(l_out), df_to_store(b_out),
                rule_summary.to_dict("records"),
                ai_context,
                "subtab-results",
                status_msg,
                100, 
                False)

    except Exception as exc:
        err_msg = dbc.Alert(f"❌ Error: {str(exc)}", color="danger",
                            style={"fontSize": "0.84rem", "padding": "8px 12px"})
        return no_update, no_update, no_update, no_update, no_update, err_msg, 0, False


# RESULTS TAB

@app.callback(
    Output("results-kpi-row",     "children"),
    Output("table-recon-ledger",  "children"),
    Output("table-recon-bank",    "children"),
    Output("table-unrecon",       "children"),
    Output("table-rule-summary",  "children"),
    Output("chart-donut",         "figure"),
    Output("table-review",        "children"),
    Input("store-results-ledger", "data"),
    Input("store-results-bank",   "data"),
    Input("store-rule-summary",   "data"),
    prevent_initial_call=True,
)
def populate_results(l_data, b_data, rule_sum_data):
    """Populate all results sub-tabs whenever reconciliation output is updated."""
    if not l_data or not b_data:
        empty = html.Div("Run reconciliation first.", style={"color": COLOR["muted"], "padding": "20px"})
        empty_fig = go.Figure()
        return [], empty, empty, empty, empty, empty_fig, empty

    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)

    # Key Performance Indicators (KPI cards)
    summary = compute_reconciliation_summary(l_df, b_df)
    totals  = compute_totals(l_df, b_df)

    kpi_row = dbc.Row([
        kpi_card("Ledger Rows",        summary.get("ledger_total", 0),    COLOR["accent"],   "📄"),
        kpi_card("Ledger Matched",     summary.get("ledger_matched", 0),  COLOR["success"],  "✅"),
        kpi_card("Ledger Unmatched",   summary.get("ledger_unmatched",0), COLOR["danger"],   "❌"),
        kpi_card("Match %",           f"{summary.get('ledger_match_pct',0)}%",
                 COLOR["success"], "📈"),
        kpi_card("Bank Rows",          summary.get("bank_total", 0),      COLOR["accent"],   "🏦"),
        kpi_card("Bank Matched",       summary.get("bank_matched", 0),    COLOR["success"],  "✅"),
        kpi_card("Recon Groups",       summary.get("total_group_ids", 0), COLOR["warning"],  "🔗"),
        kpi_card("Net Difference",
                 f"{totals.get('overall_difference', 0):,.2f}",
                 COLOR["danger"] if abs(totals.get("overall_difference", 0)) > 0 else COLOR["success"],
                 "⚖️"),
    ], className="g-2")

    # Excel Tables
    tbl_ledger = table_from_df(l_df, "dt-ledger")
    tbl_bank   = table_from_df(b_df, "dt-bank")

    unmatched_l, unmatched_b = get_unreconciled(l_df, b_df)
    unmatched_combined = pd.concat(
        [unmatched_l.assign(Source="Ledger"), unmatched_b.assign(Source="Bank")],
        ignore_index=True
    )
    tbl_unrecon = table_from_df(unmatched_combined, "dt-unrecon")

    # Rule Summaries
    rule_df = pd.DataFrame(rule_sum_data) if rule_sum_data else pd.DataFrame()
    tbl_rule = table_from_df(rule_df, "dt-rule-summary", page_size=15)

    # Donut Chart
    if rule_df is not None and not rule_df.empty:
        fig_donut = px.pie(
            rule_df,
            names  = "Rule",
            values = "Transactions_Reconciled",
            hole   = 0.45,
            title  = "Transactions Reconciled by Rule",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_donut.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(font=dict(size=11)),
            title_font=dict(size=13),
        )
    else:
        fig_donut = go.Figure()
        fig_donut.update_layout(title="No data to chart")

    # Manual Review Table
    review_df = get_manual_review_items(l_df, b_df)
    tbl_review = table_from_df(review_df, "dt-review") if not review_df.empty else \
        html.Div("No group-match items requiring manual review.", style={"color": COLOR["muted"]})

    return kpi_row, tbl_ledger, tbl_bank, tbl_unrecon, tbl_rule, fig_donut, tbl_review


# DOWNLOAD OPTIONS

def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to Excel bytes using openpyxl."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out = df[[c for c in df.columns if not str(c).startswith("_")]]
        df_out.to_excel(writer, index=False, sheet_name="Reconciliation")
    return buf.getvalue()


@app.callback(
    Output("dl-ledger", "data"),
    Input("btn-dl-ledger", "n_clicks"),
    State("store-results-ledger", "data"),
    prevent_initial_call=True,
)
def download_ledger_excel(n, data):
    if not data: return no_update
    df = store_to_df(data)
    return dcc.send_bytes(_df_to_excel_bytes(df), "reconciled_ledger.xlsx")


@app.callback(
    Output("dl-ledger-csv", "data"),
    Input("btn-dl-ledger-csv", "n_clicks"),
    State("store-results-ledger", "data"),
    prevent_initial_call=True,
)
def download_ledger_csv(n, data):
    if not data: return no_update
    df = store_to_df(data)
    display_df = df[[c for c in df.columns if not str(c).startswith("_")]]
    return dcc.send_data_frame(display_df.to_csv, "reconciled_ledger.csv", index=False)


@app.callback(
    Output("dl-bank", "data"),
    Input("btn-dl-bank", "n_clicks"),
    State("store-results-bank", "data"),
    prevent_initial_call=True,
)
def download_bank_excel(n, data):
    if not data: return no_update
    df = store_to_df(data)
    return dcc.send_bytes(_df_to_excel_bytes(df), "reconciled_bank.xlsx")


@app.callback(
    Output("dl-bank-csv", "data"),
    Input("btn-dl-bank-csv", "n_clicks"),
    State("store-results-bank", "data"),
    prevent_initial_call=True,
)
def download_bank_csv(n, data):
    if not data: return no_update
    df = store_to_df(data)
    display_df = df[[c for c in df.columns if not str(c).startswith("_")]]
    return dcc.send_data_frame(display_df.to_csv, "reconciled_bank.csv", index=False)


@app.callback(
    Output("dl-unrecon", "data"),
    Input("btn-dl-unrecon", "n_clicks"),
    State("store-results-ledger", "data"),
    State("store-results-bank",   "data"),
    prevent_initial_call=True,
)
def download_unrecon(n, l_data, b_data):
    if not l_data or not b_data: return no_update
    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)
    ul, ub = get_unreconciled(l_df, b_df)
    combined = pd.concat([ul.assign(Source="Ledger"), ub.assign(Source="Bank")], ignore_index=True)
    return dcc.send_bytes(_df_to_excel_bytes(combined), "unreconciled_items.xlsx")


# Model selection and saving the configuration
@app.callback(
    Output("ai-model-hint", "children"),
    Input("ai-provider", "value"),
    prevent_initial_call=True,
)
def update_model_hint(provider):
    """Show model name suggestions when the user changes provider."""
    hints = AI_MODEL_SUGGESTIONS.get(provider, [])
    if hints:
        return "Suggestions: " + " · ".join(hints)
    return ""


@app.callback(
    Output("ai-conn-status", "children"),
    Output("store-ai-config",  "data"),
    Input("btn-test-ai", "n_clicks"),
    State("ai-provider",     "value"),
    State("ai-apikey",       "value"),
    State("ai-model",        "value"),
    State("ai-max-tokens",   "value"),
    State("ai-temperature",  "value"),
    prevent_initial_call=True,
)
def save_and_test_ai(n, provider, api_key, model, max_tokens, temperature):
    """Save AI config to store and test connection."""
    if not n: return no_update, no_update

    config = {
        "provider"     : provider or AI_DEFAULT_CONFIG["provider"],
        "api_key"      : api_key  or "",
        "model"        : model    or AI_DEFAULT_CONFIG["model"],
        "max_tokens"   : int(max_tokens  or 1000),
        "temperature"  : float(temperature or 0.3),
        "system_prompt": AI_SYSTEM_PROMPT,
    }

    agent   = AIAgent(config)
    ok, msg = agent.test_connection()

    color  = COLOR["success"] if ok else COLOR["danger"]
    status = html.Span(msg, style={"color": color})

    return status, config if ok else no_update


# AI Agentc Chat call backs
@app.callback(
    Output("chat-history",      "children"),
    Output("store-chat-history","data"),
    Output("chat-input",        "value"),
    Input("btn-send-chat",  "n_clicks"),
    Input("btn-clear-chat", "n_clicks"),
    Input("chat-input",     "n_submit"),
    State("chat-input",          "value"),
    State("store-chat-history",  "data"),
    State("store-ai-config",     "data"),
    State("store-ai-context",    "data"),
    prevent_initial_call=True,
)
def handle_chat(send_n, clear_n, submit_n, user_input, history, ai_config, ai_context):
    """Send user message to LLM and update chat history display."""
    triggered = ctx.triggered_id

    if triggered == "btn-clear-chat":
        return [], [], ""

    if not user_input or not user_input.strip():
        return no_update, no_update, ""

    # Initialise history if first message
    if not history:
        history = []

    # Add user message
    history.append({"role": "user", "content": user_input.strip()})

    # Call LLM
    if ai_config:
        try:
            agent = AIAgent(ai_config)
            reply = agent.chat(history, context=ai_context or "")
        except Exception as exc:
            reply = f"⚠️ Error: {str(exc)}"
    else:
        reply = "⚠️ AI Agent not configured. Please set your API key in the Agent tab and save."

    history.append({"role": "assistant", "content": reply})

    # Build chat HTML display
    chat_bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubble  = html.Div([
            html.Div(
                "You" if is_user else "🤖 Agent",
                style={"fontSize": "0.72rem", "fontWeight": "700",
                       "color": COLOR["accent"] if is_user else COLOR["primary"],
                       "marginBottom": "2px"},
            ),
            html.Div(
                msg["content"],
                style={
                    "background": "#dbeafe" if is_user else "#f0fdf4",
                    "borderRadius": "8px",
                    "padding": "8px 12px",
                    "maxWidth": "85%",
                    "alignSelf": "flex-end" if is_user else "flex-start",
                    "fontSize": "0.84rem",
                    "lineHeight": "1.55",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ], style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "flex-end" if is_user else "flex-start",
            "marginBottom": "12px",
        })
        chat_bubbles.append(bubble)

    return chat_bubbles, history, ""

# Export chat as text
@app.callback(
    Output("dl-chat", "data"),
    Input("btn-export-chat", "n_clicks"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def export_chat(n, history):
    if not history: return no_update
    lines = []
    for msg in history:
        role = "YOU" if msg["role"] == "user" else "AGENT"
        lines.append(f"[{role}]\n{msg['content']}\n")
    text = "\n".join(lines)
    return dcc.send_string(text, "reconciliation_chat.txt")


###----------------------------------------------------------------------------------------------------------------###
# ENTRY POINT TO SERVER

if __name__ == "__main__":
    app.run(
        debug = APP_DEBUG,
        host  = APP_HOST,
        port  = APP_PORT,
    )

















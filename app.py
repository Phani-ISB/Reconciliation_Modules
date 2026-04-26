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
import os
import json
import base64
import traceback
import time

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, no_update, ALL
import dash_bootstrap_components as dbc


###----------------------------------------------------------------------------------------------------------------###
# Import Local Modules

from config import(
    APP_TITLE, APP_PORT, APP_DEBUG, APP_HOST,
    COLOR, RULES_CONFIG, AI_PROVIDERS, AI_DEFAULT_CONFIG,
    AI_MODEL_SUGGESTIONS, AI_SYSTEM_PROMPT, AI_IC_SYSTEM_PROMPT,
    DATE_TOLERANCE, AMOUNT_TOLERANCE, FUZZY_THRESHOLD,
)

from data_ingestion import (
    load_file, preprocess, preprocess_ic, get_column_options,
    df_to_store, store_to_df, df_to_store_ic, store_to_df_ic
)
from engine import (
    run_full_reconciliation, get_unreconciled, get_manual_review_items,
    run_ic_reconciliation, get_ic_unmatched, get_ic_review_items, get_ic_entity_matrix
)
from inferences import (
    compute_totals, compute_reconciliation_summary, compute_currency_summary,
    compute_amount_diff_summary, build_ai_context, build_ic_ai_context
)
from ai_agent import AIAgent


###----------------------------------------------------------------------------------------------------------------###
# Application Initialisation

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate",
    title=APP_TITLE,
)
server = app.server


###----------------------------------------------------------------------------------------------------------------###
# Layout Helper Functions

def card(children, style=None, className="mb-3 p-3"):
    base = {"background": COLOR["card"], "borderRadius": "8px",
            "border": f"1px solid {COLOR['border']}", "boxShadow": "0 1px 4px rgba(0,0,0,.07)"}
    if style:
        base.update(style)
    return html.Div(children, style=base, className=className)


def kpi_card(title, value, color=COLOR["accent"], icon=""):
    return dbc.Col(card([
        html.Div(f"{icon} {title}", style={"fontSize": "0.78rem", "color": COLOR["muted"], "marginBottom": "4px"}),
        html.Div(str(value), style={"fontSize": "1.6rem", "fontWeight": "700", "color": color}),
    ], style={"textAlign": "center", "minWidth": "130px"}), width="auto")


def section_header(text):
    return html.Div(text, style={
        "fontSize": "0.9rem", "fontWeight": "700",
        "color": COLOR["primary"], "marginBottom": "8px",
        "borderBottom": f"2px solid {COLOR['primary']}", "paddingBottom": "4px",
    })


def table_from_df(df, table_id, page_size=15):
    if df is None or df.empty:
        return html.Div("No data available.", style={"color": COLOR["muted"], "padding": "20px"})
    display_cols = [c for c in df.columns if not str(c).startswith("_")]
    display_df   = df[display_cols].copy()
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in display_cols],
        data=display_df.head(2000).to_dict("records"),
        page_size=page_size,
        page_action="native",
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLOR["primary"],
            "color": "white", "fontWeight": "600",
            "fontSize": "0.8rem", "padding": "8px 10px",
        },
        style_cell={
            "fontSize": "0.78rem", "padding": "6px 10px",
            "border": f"1px solid {COLOR['border']}",
            "fontFamily": "monospace", "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Matched} eq 'True'"},  "backgroundColor": "#dcfce7"},
            {"if": {"filter_query": "{Matched} eq 'False'"}, "backgroundColor": "#fef2f2"},
        ],
    )


###----------------------------------------------------------------------------------------------------------------###
# Manual Review Helper Functions

def _mini_table(df):
    """Compact read-only table for use inside review group cards."""
    if df is None or df.empty:
        return html.Div("—", style={"color": COLOR["muted"], "fontSize": "0.78rem"})
    display_cols = [c for c in df.columns if not str(c).startswith("_") and c not in ("Source",)]
    display_df   = df[display_cols].head(20).copy()
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in display_df.columns],
        data=display_df.to_dict("records"),
        page_action="none",
        style_table={"overflowX": "auto", "fontSize": "0.72rem"},
        style_header={
            "backgroundColor": COLOR["primary"], "color": "white",
            "fontWeight": "600", "fontSize": "0.72rem", "padding": "4px 8px",
        },
        style_cell={
            "fontSize": "0.71rem", "padding": "3px 8px",
            "border": f"1px solid {COLOR['border']}", "fontFamily": "monospace",
        },
    )


def _gid_int(gid):
    """Return GroupID as a clean integer string for display."""
    try:
        return str(int(float(gid)))
    except Exception:
        return str(gid)


def build_bank_review_cards(l_df, b_df):
    """
    Build group-wise review cards for Bank Reconciliation Manual Review tab.
    Each card shows one GroupID's ledger rows, bank rows, and Accept/Reject radio.
    """
    review_items = get_manual_review_items(l_df, b_df)
    if review_items is None or review_items.empty:
        return html.Div(
            "No group-match items requiring manual review.",
            style={"color": COLOR["muted"], "padding": "20px", "fontSize": "0.86rem"},
        )

    review_items = review_items.dropna(subset=["GroupID"])
    cards = []

    for gid, group in review_items.groupby("GroupID"):
        ledger_rows = group[group["Source"] == "Ledger"].drop(columns=["Source"], errors="ignore")
        bank_rows   = group[group["Source"] == "Bank"].drop(columns=["Source"],  errors="ignore")
        rule_name   = group["Rule"].iloc[0] if "Rule" in group.columns else ""

        # Compute group total amount diff
        amt_col = "AmountDiff"
        total_diff = round(group[amt_col].sum(), 4) if amt_col in group.columns else "—"

        card_el = dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(
                        html.Span(f"Group {_gid_int(gid)}",
                                  style={"fontWeight": "700", "fontSize": "0.88rem", "color": COLOR["primary"]}),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(f"Rule: {rule_name}",
                                  style={"color": COLOR["muted"], "fontSize": "0.78rem"}),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(f"Amt Diff: {total_diff}",
                                  style={"color": COLOR["warning"], "fontSize": "0.78rem"}),
                        width="auto",
                    ),
                ], align="center", className="g-3"),
                style={"padding": "6px 14px", "background": "#f8fafc",
                       "borderBottom": f"1px solid {COLOR['border']}"},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Small("📄 Ledger",
                                   style={"fontWeight": "600", "color": COLOR["primary"],
                                          "display": "block", "marginBottom": "4px"}),
                        _mini_table(ledger_rows),
                    ], width=6),
                    dbc.Col([
                        html.Small("🏦 Bank",
                                   style={"fontWeight": "600", "color": COLOR["accent"],
                                          "display": "block", "marginBottom": "4px"}),
                        _mini_table(bank_rows),
                    ], width=6),
                ], className="mb-3"),
                dbc.RadioItems(
                    id={"type": "bank-review-decision", "group": str(gid)},
                    options=[
                        {"label": "  ✅ Accept — keep as Matched (Reviewed)", "value": "accepted"},
                        {"label": "  ❌ Reject — return to Unmatched",        "value": "rejected"},
                    ],
                    value=None,
                    inline=True,
                    style={"fontSize": "0.82rem"},
                ),
            ], style={"padding": "10px 14px"}),
        ], className="mb-3",
           style={"border": f"1px solid {COLOR['border']}", "borderRadius": "8px",
                  "boxShadow": "0 1px 3px rgba(0,0,0,.06)"})

        cards.append(card_el)

    return html.Div(cards)


def build_ic_review_cards(review_df):
    """
    Build group-wise review cards for IC Reconciliation Manual Review tab.
    Each card shows one Recon_ID group with Accept/Reject radio.
    """
    if review_df is None or review_df.empty:
        return html.Div(
            "No IC transactions flagged for manual review.",
            style={"color": COLOR["muted"], "padding": "20px", "fontSize": "0.86rem"},
        )

    cards = []
    for recon_id, group in review_df.groupby("Recon_ID"):
        rule_name   = group["Rule_Applied"].iloc[0] if "Rule_Applied" in group.columns else ""
        n_rows      = len(group)
        display_cols = [c for c in group.columns if not str(c).startswith("_")]
        display_df  = group[display_cols].copy()

        card_el = dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(
                        html.Span(f"Recon ID: {recon_id}",
                                  style={"fontWeight": "700", "fontSize": "0.88rem", "color": COLOR["primary"]}),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(f"Rule: {rule_name}",
                                  style={"color": COLOR["muted"], "fontSize": "0.78rem"}),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(f"{n_rows} transactions",
                                  style={"color": COLOR["accent"], "fontSize": "0.78rem"}),
                        width="auto",
                    ),
                ], align="center", className="g-3"),
                style={"padding": "6px 14px", "background": "#f8fafc",
                       "borderBottom": f"1px solid {COLOR['border']}"},
            ),
            dbc.CardBody([
                _mini_table(display_df),
                html.Div(style={"height": "10px"}),
                dbc.RadioItems(
                    id={"type": "ic-review-decision", "group": str(recon_id)},
                    options=[
                        {"label": "  ✅ Accept — confirm as Matched (Reviewed)", "value": "accepted"},
                        {"label": "  ❌ Reject — return to Unmatched",           "value": "rejected"},
                    ],
                    value=None,
                    inline=True,
                    style={"fontSize": "0.82rem"},
                ),
            ], style={"padding": "10px 14px"}),
        ], className="mb-3",
           style={"border": f"1px solid {COLOR['border']}", "borderRadius": "8px",
                  "boxShadow": "0 1px 3px rgba(0,0,0,.06)"})

        cards.append(card_el)

    return html.Div(cards)


def _df_to_assign_table(df, table_id):
    """
    Row-selectable DataTable for the Manual Assign panels.
    Adds a hidden 'OrigIdx' column so callbacks can locate rows in the store DF.
    """
    if df is None or df.empty:
        return html.Div("No unmatched rows available.",
                        style={"color": COLOR["muted"], "fontSize": "0.84rem", "padding": "12px"})

    display_cols = [c for c in df.columns if not str(c).startswith("_")]
    display_df   = df[display_cols].copy()
    display_df.insert(0, "OrigIdx", df.index.tolist())   # store original DF index

    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in display_df.columns],
        data=display_df.reset_index(drop=True).to_dict("records"),
        row_selectable="multi",
        selected_rows=[],
        page_size=15,
        page_action="native",
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLOR["primary"], "color": "white",
            "fontWeight": "600", "fontSize": "0.78rem", "padding": "6px 10px",
        },
        style_cell={
            "fontSize": "0.76rem", "padding": "4px 8px",
            "border": f"1px solid {COLOR['border']}", "fontFamily": "monospace",
        },
        style_data_conditional=[
            {"if": {"column_id": "OrigIdx"},
             "color": COLOR["muted"], "fontSize": "0.7rem", "width": "60px"},
        ],
    )


###----------------------------------------------------------------------------------------------------------------###
# HELP File
# Import help section from help.md file text
def load_help_text():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "help.md")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"⚠️ Unable to load help file:\n\n{str(e)}"
               
help_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Help & Credits"), close_button=True),
    dbc.ModalBody([
        dcc.Markdown(id="help-content", children="Loading help file...",
                     style={"fontSize": "0.9rem"}),
    ]),
    dbc.ModalFooter(dbc.Button("Close", id="help-close", color="secondary")),
], id="help-modal", size="lg", is_open=False)

@app.callback(
    Output("help-content", "children", allow_duplicate=True),
    Input("help-modal", "is_open"),
    prevent_initial_call=True,
)
def update_help_content(is_open):
    if is_open:
        return load_help_text()
    return no_update

###----------------------------------------------------------------------------------------------------------------###
# Data Ingestion Tab

def upload_block(title, upload_id, preview_id):
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
    mapping_block("Ledger",         "ledger-date", "ledger-amount", "ledger-narration"),
    mapping_block("Bank Statement", "bank-date",   "bank-amount",   "bank-narration"),
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
        value=[r["key"] for r in RULES_CONFIG if r["enabled"]],
        style={"fontSize": "0.86rem", "lineHeight": "2"},
    )
]

tab_rules = html.Div([
    dbc.Row([
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
               size="lg", className="w-100", disabled=True),
    dbc.Progress(id="recon-progress",
                 value=0, animated=True, striped=True,
                 style={"height": "10px", "marginTop": "10px", "transition": "width 0.4s ease"},
                 className="mb-2"),
    html.Div(id="progress-label",
             style={"fontSize": "0.78rem", "color": COLOR["muted"],
                    "textAlign": "center", "marginBottom": "8px"}),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# Results Tab 

tab_results = html.Div([
    dbc.Row(id="results-kpi-row", className="mb-3 g-2"),

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

        # ── Manual Review tab — group-wise with sub-tabs ──────────────────────
        dbc.Tab(label="🔍 Manual Review", tab_id="tab-review", children=[
            dbc.Tabs(id="bank-review-subtabs", active_tab="bank-tab-review-groups",
                     className="mt-2", children=[

                # ─ Sub-tab 1: Review Groups ───────────────────────────────────
                dbc.Tab(label="📋 Review Groups", tab_id="bank-tab-review-groups", children=[
                    html.Div([
                        html.Div(
                            "Review each matched group below. "
                            "Accept to keep as Matched (Reviewed) or Reject to return rows to Unmatched. "
                            "Click Apply to commit all decisions.",
                            style={"fontSize": "0.84rem", "color": COLOR["warning"],
                                   "marginBottom": "12px", "padding": "8px",
                                   "background": "#fffbeb", "borderRadius": "6px",
                                   "border": f"1px solid {COLOR['warning']}"},
                        ),
                        # ── Accept All shortcut ──────────────────────────
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("☑️ Accept All Groups", id="btn-bank-accept-all",
                                           color="success", size="sm", outline=True),
                                width="auto",
                            ),
                            dbc.Col(
                                html.Small("Ticks every group as Accepted",
                                           style={"color": COLOR["muted"]}),
                                width="auto",
                            ),
                        ], align="center", className="mb-3"),
                        # Group cards rendered by populate_results callback
                        html.Div(id="table-review"),
                        html.Div(style={"height": "14px"}),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("✅ Apply All Decisions", id="btn-apply-bank-decisions",
                                           color="primary", size="sm"),
                                width="auto",
                            ),
                            dbc.Col(html.Div(id="bank-review-status"), width="auto"),
                        ], align="center", className="mt-2"),
                    ], style={"padding": "12px"}),
                ]),

                # ─ Sub-tab 2: Manual Assign ──────────────────────────────────
                dbc.Tab(label="➕ Manual Assign", tab_id="bank-tab-manual-assign", children=[
                    html.Div([
                        html.Div(
                            "Tick one or more rows from the Unmatched Ledger table AND the Unmatched Bank table, "
                            "then click Assign. They will be grouped together with the next available GroupID "
                            "and Rule = 'Manual Assignment'.",
                            style={"fontSize": "0.84rem", "color": COLOR["accent"],
                                   "marginBottom": "14px", "padding": "8px",
                                   "background": "#eff6ff", "borderRadius": "6px",
                                   "border": f"1px solid {COLOR['accent']}"},
                        ),
                        dbc.Row([
                            dbc.Col([
                                section_header("📄 Unmatched Ledger"),
                                html.Div(id="dt-assign-ledger"),
                            ], width=6),
                            dbc.Col([
                                section_header("🏦 Unmatched Bank"),
                                html.Div(id="dt-assign-bank"),
                            ], width=6),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("🔗 Assign Selected as Group",
                                           id="btn-bank-manual-assign",
                                           color="success", size="sm"),
                                width="auto",
                            ),
                            dbc.Col(html.Div(id="bank-assign-status"), width="auto"),
                        ], align="center"),
                    ], style={"padding": "12px"}),
                ]),
            ]),
        ]),

    ], active_tab="tab-recon-ledger"),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# AI Agent Tab

tab_ai_agent = html.Div([
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
                dcc.Input(id="ai-model", value=AI_DEFAULT_CONFIG["model"], debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
                html.Small(id="ai-model-hint", style={"color": COLOR["muted"], "fontSize": "0.75rem"}),
            ], width=3),
            dbc.Col([
                html.Label("API Key", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ai-apikey", type="password", placeholder="Paste API key here…",
                          debounce=True, style={"width": "100%", "padding": "6px", "borderRadius": "4px",
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
                dcc.Input(id="ai-temperature", type="number", value=AI_DEFAULT_CONFIG["temperature"],
                          min=0.0, max=1.0, step=0.05, debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
            ], width=1),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("💾 Save & Test Connection", id="btn-test-ai",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(html.Div(id="ai-conn-status",
                             style={"fontSize": "0.84rem", "paddingTop": "6px", "fontWeight": "600"}),
                    width="auto"),
        ]),
    ]),
    card([
        section_header("💬 Chat with Reconciliation Data"),
        html.Div(id="chat-history",
                 style={"height": "380px", "overflowY": "auto",
                        "border": f"1px solid {COLOR['border']}",
                        "borderRadius": "6px", "padding": "12px",
                        "background": COLOR["background"],
                        "fontSize": "0.86rem", "lineHeight": "1.6", "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(dcc.Input(id="chat-input", placeholder="Ask a question about your reconciliation…",
                              debounce=False, n_submit=0,
                              style={"width": "100%", "padding": "8px 12px", "borderRadius": "6px",
                                     "border": f"1px solid {COLOR['accent']}", "fontSize": "0.86rem"}),
                    width=9),
            dbc.Col(dbc.Button("➤ Send",  id="btn-send-chat",  color="primary"), width=2),
            dbc.Col(dbc.Button("🗑 Clear", id="btn-clear-chat", color="secondary", outline=True), width=1),
        ], className="g-2"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("⬇ Export Chat (TXT)", id="btn-export-chat",
                           color="secondary", size="sm", outline=True), width="auto"),
        dcc.Download(id="dl-chat"),
    ], className="mt-2"),
], style={"padding": "16px"})


###----------------------------------------------------------------------------------------------------------------###
# IC AI Agent Tab

tab_ic_ai_agent = html.Div([
    card([
        section_header("🔧 IC AI Agent Configuration"),
        dbc.Row([
            dbc.Col([
                html.Label("Provider", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Dropdown(id="ic-ai-provider", options=[{"label": p, "value": p} for p in AI_PROVIDERS],
                             value=AI_DEFAULT_CONFIG["provider"],
                             clearable=False, style={"fontSize": "0.82rem"}),
            ], width=3),
            dbc.Col([
                html.Label("Model", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ic-ai-model", value=AI_DEFAULT_CONFIG["model"], debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
                html.Small(id="ic-ai-model-hint", style={"color": COLOR["muted"], "fontSize": "0.75rem"}),
            ], width=3),
            dbc.Col([
                html.Label("API Key", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ic-ai-apikey", type="password", placeholder="Paste API key here…",
                          debounce=True, style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                                "border": f"1px solid {COLOR['border']}"}),
            ], width=3),
            dbc.Col([
                html.Label("Max Tokens", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ic-ai-max-tokens", type="number", value=AI_DEFAULT_CONFIG["max_tokens"],
                          min=100, max=4096, step=100, debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
            ], width=2),
            dbc.Col([
                html.Label("Temperature", style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dcc.Input(id="ic-ai-temperature", type="number", value=AI_DEFAULT_CONFIG["temperature"],
                          min=0.0, max=1.0, step=0.05, debounce=True,
                          style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                                 "border": f"1px solid {COLOR['border']}"}),
            ], width=1),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("💾 Save & Test Connection", id="ic-btn-test-ai",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(html.Div(id="ic-ai-conn-status",
                             style={"fontSize": "0.84rem", "paddingTop": "6px", "fontWeight": "600"}),
                    width="auto"),
        ]),
    ]),
    card([
        section_header("💬 Chat with IC Reconciliation Data"),
        html.Div(id="ic-chat-history",
                 style={"height": "380px", "overflowY": "auto",
                        "border": f"1px solid {COLOR['border']}",
                        "borderRadius": "6px", "padding": "12px",
                        "background": COLOR["background"],
                        "fontSize": "0.86rem", "lineHeight": "1.6", "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(dcc.Input(id="ic-chat-input",
                              placeholder="Ask a question about your IC reconciliation…",
                              debounce=False, n_submit=0,
                              style={"width": "100%", "padding": "8px 12px", "borderRadius": "6px",
                                     "border": f"1px solid {COLOR['accent']}", "fontSize": "0.86rem"}),
                    width=9),
            dbc.Col(dbc.Button("➤ Send",  id="ic-btn-send-chat",  color="primary"), width=2),
            dbc.Col(dbc.Button("🗑 Clear", id="ic-btn-clear-chat", color="secondary", outline=True), width=1),
        ], className="g-2"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("⬇ Export Chat (TXT)", id="ic-btn-export-chat",
                           color="secondary", size="sm", outline=True), width="auto"),
        dcc.Download(id="ic-dl-chat"),
    ], className="mt-2"),
], style={"padding": "16px"})


###################################################################################################################
# INTER-COMPANY RECONCILIATION TABS

tab_data_ingestion_ic = html.Div([
    dbc.Row([
        dbc.Col(upload_block("📄 IC Master Data File", "upload-ic", "preview-ic"), width=12),
    ]),
    mapping_block("IC Data", "ic-date", "ic-amount", "ic-narration"),
    dbc.Row([
        dbc.Col([
            html.Label("🏢 Entity Column *", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Dropdown(id="ic-entity", placeholder="Select entity column...",
                         style={"fontSize": "0.82rem"}),
        ], width=6),
        dbc.Col([
            html.Label("🤝 Partner Entity Column *", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Dropdown(id="ic-partner", placeholder="Select partner entity column...",
                         style={"fontSize": "0.82rem"}),
        ], width=6),
    ]),
    html.Div(id="ic-mapping-status", style={"marginBottom": "8px"}),
    dbc.Button("✅ Confirm Mapping & Proceed to Rules →",
               id="btn-ic-proceed-rules", color="primary", size="lg",
               className="w-100", disabled=True),
], style={"padding": "16px"})


ic_rule_names = [
    "Narration Exact + Date Exact + Amount Offset",
    "Narration Fuzzy + Date Exact + Amount Offset",
    "Narration Exact + Date Range + Amount Offset",
    "Narration Fuzzy + Date Range + Amount Offset",
    "Date Exact + Amount Offset",
    "Amount Offset Only",
    "Within Company Reversal + Amount Offset",
    "Multiple Matchings (Manual Review)",
]

ic_rule_checklist = dbc.Checklist(
    id="ic-rules-checklist",
    options=[{"label": rule, "value": rule} for rule in ic_rule_names],
    value=ic_rule_names,
    style={"fontSize": "0.86rem", "lineHeight": "2"},
)

tab_rules_ic = html.Div([
    dbc.Row([
        dbc.Col(card([
            section_header("⚙️ IC Reconciliation Parameters"),
            html.Label("📅 Date Tolerance (days)", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Slider(id="ic-kpi-date-tol", min=0, max=30, step=1, value=DATE_TOLERANCE,
                       marks={i: str(i) for i in range(0, 31, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            html.Label("💰 Amount Tolerance (Offset)", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Input(id="ic-kpi-amount-tol", type="number", value=AMOUNT_TOLERANCE,
                      min=0, step=0.01, debounce=True,
                      style={"width": "100%", "padding": "6px", "borderRadius": "4px",
                             "border": f"1px solid {COLOR['border']}"}),
            html.Br(), html.Br(),
            html.Label("🔍 Fuzzy Match Threshold (0–100)", style={"fontSize": "0.82rem", "fontWeight": "600"}),
            dcc.Slider(id="ic-kpi-fuzzy-tol", min=0, max=100, step=5, value=FUZZY_THRESHOLD,
                       marks={i: str(i) for i in range(0, 101, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]), width=5),
        dbc.Col(card([
            section_header("📋 IC Reconciliation Rules"),
            dbc.Row([
                dbc.Col(dbc.Button("✅ Enable All",  id="btn-ic-enable-all",  color="success",
                                   size="sm", outline=True, className="me-2"), width="auto"),
                dbc.Col(dbc.Button("⬜ Disable All", id="btn-ic-disable-all", color="secondary",
                                   size="sm", outline=True), width="auto"),
            ], className="mb-3"),
            ic_rule_checklist,
        ]), width=7),
    ]),
    html.Div(id="ic-rules-status", className="mb-2"),
    dbc.Button("▶  Run IC Reconciliation", id="btn-ic-run-recon", color="primary",
               size="lg", className="w-100", disabled=True),
    dbc.Progress(id="ic-recon-progress", value=0, style={"height": "6px", "marginTop": "10px"}),
], style={"padding": "16px"})


#IC Results Tab

tab_results_ic = html.Div([
    dbc.Row(id="ic-results-kpi-row", className="mb-3 g-2"),
    dbc.Tabs(id="ic-results-subtabs", children=[

        dbc.Tab(label="✅ Matched Transactions", tab_id="tab-ic-matched", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("⬇ Download Excel", id="btn-ic-dl-matched",
                                       color="success", size="sm", outline=True), width="auto"),
                    dbc.Col(dbc.Button("⬇ Download CSV", id="btn-ic-dl-matched-csv",
                                       color="secondary", size="sm", outline=True), width="auto"),
                ], className="mb-2"),
                dcc.Download(id="dl-ic-matched"),
                dcc.Download(id="dl-ic-matched-csv"),
                html.Div(id="table-ic-matched"),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="❌ Unmatched Transactions", tab_id="tab-ic-unmatched", children=[
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Button("⬇ Download Excel", id="btn-ic-dl-unmatched",
                                       color="danger", size="sm", outline=True), width="auto"),
                ], className="mb-2"),
                dcc.Download(id="dl-ic-unmatched"),
                html.Div(id="table-ic-unmatched"),
            ], style={"padding": "12px"}),
        ]),

        # ── IC Manual Review — group-wise sub-tabs ─────────────────────────
        dbc.Tab(label="👁️ For Manual Review", tab_id="tab-ic-review", children=[
            dbc.Tabs(id="ic-review-subtabs", active_tab="ic-tab-review-groups",
                     className="mt-2", children=[

                # ─ Sub-tab 1: Review Groups ──────────────────────────────
                dbc.Tab(label="📋 Review Groups", tab_id="ic-tab-review-groups", children=[
                    html.Div([
                        html.Div(
                            "Review each IC transaction group. "
                            "Accept to confirm as Matched (Reviewed) or Reject to return to Unmatched. "
                            "Click Apply to commit decisions.",
                            style={"fontSize": "0.84rem", "color": COLOR["warning"],
                                   "marginBottom": "12px", "padding": "8px",
                                   "background": "#fffbeb", "borderRadius": "6px",
                                   "border": f"1px solid {COLOR['warning']}"},
                        ),
                        # ── Accept All shortcut ──────────────────────────
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("☑️ Accept All Groups", id="btn-ic-accept-all",
                                           color="success", size="sm", outline=True),
                                width="auto",
                            ),
                            dbc.Col(
                                html.Small("Ticks every group as Accepted",
                                           style={"color": COLOR["muted"]}),
                                width="auto",
                            ),
                        ], align="center", className="mb-3"),
                        # IC group cards rendered by populate_ic_results callback
                        html.Div(id="table-ic-review"),
                        html.Div(style={"height": "14px"}),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("✅ Apply All Decisions", id="btn-apply-ic-decisions",
                                           color="primary", size="sm"),
                                width="auto",
                            ),
                            dbc.Col(html.Div(id="ic-review-status"), width="auto"),
                        ], align="center", className="mt-2"),
                    ], style={"padding": "12px"}),
                ]),

                # ─ Sub-tab 2: Manual Assign ──────────────────────────────
                dbc.Tab(label="➕ Manual Assign", tab_id="ic-tab-manual-assign", children=[
                    html.Div([
                        html.Div(
                            "Select IC transactions from the Unmatched list below to manually group them. "
                            "Assign creates a new Recon ID with Rule = 'Manual Assignment'.",
                            style={"fontSize": "0.84rem", "color": COLOR["accent"],
                                   "marginBottom": "14px", "padding": "8px",
                                   "background": "#eff6ff", "borderRadius": "6px",
                                   "border": f"1px solid {COLOR['accent']}"},
                        ),
                        section_header("📄 Unmatched IC Transactions"),
                        html.Div(id="dt-assign-ic"),
                        html.Div(style={"height": "12px"}),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("🔗 Assign Selected as Group",
                                           id="btn-ic-manual-assign",
                                           color="success", size="sm"),
                                width="auto",
                            ),
                            dbc.Col(html.Div(id="ic-assign-status"), width="auto"),
                        ], align="center"),
                    ], style={"padding": "12px"}),
                ]),
            ]),
        ]),

        dbc.Tab(label="🗺️ Entity Matrix", tab_id="tab-ic-matrix", children=[
            html.Div([
                html.Div("Summary of reconciled amounts between entity pairs.",
                         style={"fontSize": "0.84rem", "color": COLOR["muted"], "marginBottom": "12px"}),
                html.Div(id="table-ic-matrix"),
            ], style={"padding": "12px"}),
        ]),

        dbc.Tab(label="📊 Rule Summary", tab_id="tab-ic-rules", children=[
            html.Div([html.Div(id="table-ic-rule-summary")], style={"padding": "12px"}),
        ]),

    ], active_tab="tab-ic-matched"),
], style={"padding": "16px"})


###################################################################################################################
# MAIN LAYOUT & STORES

app.layout = html.Div([

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

    dcc.Tabs(id="main-tabs", value="tab-bank", children=[

        dcc.Tab(label="🏢  Inter Company Reconciliation", value="tab-ic",
                style={"fontWeight": "600", "padding": "12px 20px"},
                selected_style={"fontWeight": "700", "color": COLOR["primary"],
                                "borderTop": f"3px solid {COLOR['primary']}", "padding": "12px 20px"},
                children=[
                    html.Div([
                        dbc.Tabs(id="ic-subtabs", active_tab="subtab-ic-ingestion", children=[
                            dbc.Tab(label="📁  Data Ingestion", tab_id="subtab-ic-ingestion",
                                    children=tab_data_ingestion_ic),
                            dbc.Tab(label="⚙️  Rules & Config", tab_id="subtab-ic-rules",
                                    children=tab_rules_ic),
                            dbc.Tab(label="📊  Results",         tab_id="subtab-ic-results",
                                    children=tab_results_ic),
                            dbc.Tab(label="🤖  AI Agent",         tab_id="subtab-ic-agent",
                                    children=tab_ic_ai_agent),
                        ], className="mt-0"),
                    ]),
                ]),

        dcc.Tab(label="🏦  Bank Reconciliation", value="tab-bank",
                style={"fontWeight": "600", "padding": "12px 20px"},
                selected_style={"fontWeight": "700", "color": COLOR["primary"],
                                "borderTop": f"3px solid {COLOR['primary']}", "padding": "12px 20px"},
                children=[
                    html.Div([
                        dbc.Tabs(id="bank-subtabs", active_tab="subtab-ingestion", children=[
                            dbc.Tab(label="📁  Data Ingestion", tab_id="subtab-ingestion",
                                    children=tab_data_ingestion),
                            dbc.Tab(label="⚙️  Rules & Config", tab_id="subtab-rules",
                                    children=tab_rules),
                            dbc.Tab(label="📊  Results",         tab_id="subtab-results",
                                    children=tab_results),
                            dbc.Tab(label="🤖  AI Agent",         tab_id="subtab-agent",
                                    children=tab_ai_agent),
                        ], className="mt-0"),
                    ]),
                ]),

    ], style={"fontFamily": "Inter, Segoe UI, sans-serif"}),

    # ── dcc.Store registry ──────────────────────────────────────────────────
    dcc.Store(id="store-ledger-raw"),
    dcc.Store(id="store-bank-raw"),
    dcc.Store(id="store-ledger-cols"),
    dcc.Store(id="store-bank-cols"),
    dcc.Store(id="store-ledger-clean"),
    dcc.Store(id="store-bank-clean"),
    dcc.Store(id="store-recon-running", data=False),
    dcc.Interval(id="interval-progress", interval=400, n_intervals=0, disabled=True),
    dcc.Store(id="store-results-ledger"),
    dcc.Store(id="store-results-bank"),
    dcc.Store(id="store-rule-summary"),
    dcc.Store(id="store-ai-config"),
    dcc.Store(id="store-chat-history"),
    dcc.Store(id="store-ai-context"),

    # IC stores
    dcc.Store(id="store-ic-raw"),
    dcc.Store(id="store-ic-cols"),
    dcc.Store(id="store-ic-clean"),
    dcc.Store(id="store-ic-results"),
    dcc.Store(id="store-ic-matched"),
    dcc.Store(id="store-ic-unmatched"),
    dcc.Store(id="store-ic-review"),
    dcc.Store(id="store-ic-rule-summary"),
    dcc.Store(id="store-ic-ai-config"),
    dcc.Store(id="store-ic-chat-history"),
    dcc.Store(id="store-ic-ai-context"),

], style={"fontFamily": "Inter, Segoe UI, sans-serif",
          "background": COLOR["background"], "minHeight": "100vh"})


###################################################################################################################
# HELP CALLBACK

@app.callback(
    Output("help-modal",   "is_open"),
    Output("help-content", "children"),
    Input("btn-help",   "n_clicks"),
    Input("help-close", "n_clicks"),
    State("help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_help(open_clicks, close_clicks, is_open):
    return not is_open, load_help_text()


###################################################################################################################
# SHARED UPLOAD HELPER

def _parse_upload(contents, filename):
    if contents is None:
        return None, None, [], "No file uploaded."
    df, err = load_file(contents, filename)
    if err:
        return None, None, [], f"❌ {err}"
    cols    = get_column_options(df)
    records = df_to_store(df)
    return records, cols, cols, f"✅ Loaded **{filename}** — {len(df):,} rows × {len(df.columns)} columns"


###################################################################################################################
# INTER-COMPANY RECONCILIATION CALLBACKS

@app.callback(
    Output("store-ic-raw",  "data"),
    Output("store-ic-cols", "data"),
    Output("ic-date",       "options"),
    Output("ic-amount",     "options"),
    Output("ic-narration",  "options"),
    Output("ic-entity",     "options"),
    Output("ic-partner",    "options"),
    Output("preview-ic",    "children"),
    Input("upload-ic", "contents"),
    State("upload-ic", "filename"),
    prevent_initial_call=True,
)
def upload_ic(contents, filename):
    records, cols, _, msg = _parse_upload(contents, filename)
    preview = dbc.Alert(msg, color="success" if records else "danger",
                        style={"fontSize": "0.82rem", "padding": "8px 12px"})
    col_options = cols or []
    return records, cols, col_options, col_options, col_options, col_options, col_options, preview


@app.callback(
    Output("btn-ic-proceed-rules", "disabled"),
    Output("ic-mapping-status",    "children"),
    Input("store-ic-raw",  "data"),
    Input("ic-date",       "value"),
    Input("ic-amount",     "value"),
    Input("ic-narration",  "value"),
    Input("ic-entity",     "value"),
    Input("ic-partner",    "value"),
    prevent_initial_call=True,
)
def check_ic_mapping_complete(raw, date_col, amt_col, nar_cols, entity_col, partner_col):
    checks = [raw, date_col, amt_col, nar_cols, entity_col, partner_col]
    if all(c for c in checks):
        msg = dbc.Alert("✅ All columns mapped. Ready to proceed to Rules.",
                        color="success", style={"fontSize": "0.82rem", "padding": "8px 12px"})
        return False, msg
    missing = []
    if not raw:         missing.append("Upload IC file")
    if not date_col:    missing.append("Map Date column")
    if not amt_col:     missing.append("Map Amount column")
    if not nar_cols:    missing.append("Map Narration column(s)")
    if not entity_col:  missing.append("Map Entity column")
    if not partner_col: missing.append("Map Partner Entity column")
    msg = dbc.Alert("⚠️ Remaining: " + " · ".join(missing),
                    color="warning", style={"fontSize": "0.82rem", "padding": "8px 12px"})
    return True, msg


@app.callback(
    Output("store-ic-clean",   "data"),
    Output("ic-subtabs",       "active_tab"),
    Output("btn-ic-run-recon", "disabled"),
    Input("btn-ic-proceed-rules", "n_clicks"),
    State("store-ic-raw",  "data"),
    State("ic-date",       "value"),
    State("ic-amount",     "value"),
    State("ic-narration",  "value"),
    State("ic-entity",     "value"),
    State("ic-partner",    "value"),
    prevent_initial_call=True,
)
def proceed_to_ic_rules(n, ic_raw, date_col, amt_col, nar_cols, entity_col, partner_col):
    if not n:
        return None, no_update, True
    ic_df = store_to_df_ic(ic_raw)
    if ic_df is None:
        return None, no_update, True
    mapping = {
        "date_col": date_col, "amount_col": amt_col,
        "narration_cols": nar_cols or [],
        "entity_col": entity_col, "partner_entity_col": partner_col,
    }
    ic_clean, err = preprocess_ic(ic_df, mapping, "IC Data")
    if err:
        return None, no_update, True
    return df_to_store_ic(ic_clean), "subtab-ic-rules", False


@app.callback(
    Output("ic-rules-checklist", "value"),
    Input("btn-ic-enable-all",  "n_clicks"),
    Input("btn-ic-disable-all", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_all_ic_rules(enable_n, disable_n):
    return ic_rule_names if ctx.triggered_id == "btn-ic-enable-all" else []


@app.callback(
    Output("store-ic-results",      "data"),
    Output("store-ic-matched",      "data"),
    Output("store-ic-unmatched",    "data"),
    Output("store-ic-review",       "data"),
    Output("store-ic-rule-summary", "data"),
    Output("store-ic-ai-context",   "data"),
    Output("ic-subtabs",            "active_tab", allow_duplicate=True),
    Output("ic-rules-status",       "children"),
    Input("btn-ic-run-recon", "n_clicks"),
    State("store-ic-clean",     "data"),
    State("ic-rules-checklist", "value"),
    State("ic-kpi-date-tol",    "value"),
    State("ic-kpi-amount-tol",  "value"),
    State("ic-kpi-fuzzy-tol",   "value"),
    prevent_initial_call=True,
)
def run_ic_recon(n_clicks, ic_clean, enabled_rules, date_tol, amt_tol, fuzzy_tol):
    if not n_clicks or ic_clean is None:
        err_msg = dbc.Alert("❌ No clean IC data. Please complete data ingestion.",
                            color="danger", style={"fontSize": "0.84rem"})
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, err_msg
    try:
        ic_df  = store_to_df_ic(ic_clean)
        params = {"date_tolerance": date_tol, "amount_tolerance": amt_tol, "fuzzy_threshold": fuzzy_tol}
        recon_df, matched_df, unmatched_df, review_df, rule_summary = run_ic_reconciliation(
            ic_df, params, enabled_rules
        )
        ic_context = build_ic_ai_context(matched_df, unmatched_df, review_df, rule_summary, params)
        success_msg = dbc.Alert(
            f"✅ IC Reconciliation complete! {len(matched_df)} matched, "
            f"{len(unmatched_df)} unmatched, {len(review_df)} for review.",
            color="success", style={"fontSize": "0.84rem", "padding": "8px 12px"},
        )
        return (
            df_to_store_ic(recon_df),
            df_to_store_ic(matched_df),
            df_to_store_ic(unmatched_df),
            df_to_store_ic(review_df),
            rule_summary.to_dict("records") if rule_summary is not None else [],
            ic_context,
            "subtab-ic-results",
            success_msg,
        )
    except Exception as exc:
        err_msg = dbc.Alert(f"❌ Error: {str(exc)}", color="danger",
                            style={"fontSize": "0.84rem", "padding": "8px 12px"})
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, err_msg


@app.callback(
    Output("ic-results-kpi-row",    "children"),
    Output("table-ic-matched",      "children"),
    Output("table-ic-unmatched",    "children"),
    Output("table-ic-review",       "children"),   # ← now renders group cards
    Output("table-ic-matrix",       "children"),
    Output("table-ic-rule-summary", "children"),
    Input("store-ic-matched",       "data"),
    Input("store-ic-unmatched",     "data"),
    Input("store-ic-review",        "data"),
    Input("store-ic-results",       "data"),
    Input("store-ic-rule-summary",  "data"),
    prevent_initial_call=True,
)
def populate_ic_results(matched_data, unmatched_data, review_data, results_data, rule_sum_data):
    if not results_data:
        empty = html.Div("Run reconciliation first.", style={"color": COLOR["muted"], "padding": "20px"})
        return [], empty, empty, empty, empty, empty

    recon_df     = store_to_df_ic(results_data)
    matched_df   = store_to_df_ic(matched_data)
    unmatched_df = store_to_df_ic(unmatched_data)
    review_df    = store_to_df_ic(review_data)

    total_tx       = len(recon_df)    if recon_df    is not None else 0
    matched_count  = len(matched_df)  if matched_df  is not None else 0
    unmatched_count= len(unmatched_df)if unmatched_df is not None else 0
    review_count   = len(review_df)   if review_df   is not None else 0
    match_pct      = round(100 * matched_count / total_tx, 1) if total_tx > 0 else 0

    kpi_row = dbc.Row([
        kpi_card("Total Transactions", total_tx,        COLOR["accent"],  "📝"),
        kpi_card("Matched",            matched_count,   COLOR["success"], "✅"),
        kpi_card("Unmatched",          unmatched_count, COLOR["danger"],  "❌"),
        kpi_card("For Review",         review_count,    COLOR["warning"], "👁️"),
        kpi_card("Match %",            f"{match_pct}%", COLOR["success"], "📈"),
    ], className="mb-2 g-2")

    matched_table   = table_from_df(matched_df,   "table-ic-matched-dt")   if matched_df   is not None else html.Div("No matched transactions.")
    unmatched_table = table_from_df(unmatched_df, "table-ic-unmatched-dt") if unmatched_df is not None else html.Div("No unmatched transactions.")

    # Group wise review
    review_ui = build_ic_review_cards(review_df)

    entity_matrix = get_ic_entity_matrix(recon_df, "Matched")
    matrix_table  = table_from_df(entity_matrix.reset_index(), "table-ic-entity-matrix-dt") \
                    if entity_matrix is not None else html.Div("No entity matrix data.")

    if rule_sum_data:
        rule_df    = pd.DataFrame(rule_sum_data)
        rule_table = table_from_df(rule_df, "table-ic-rule-sum-dt")
    else:
        rule_table = html.Div("No rule summary.")

    return kpi_row, matched_table, unmatched_table, review_ui, matrix_table, rule_table


# IC Manual Review — Apply Decisions 

@app.callback(
    Output("store-ic-results",   "data",     allow_duplicate=True),
    Output("store-ic-matched",   "data",     allow_duplicate=True),
    Output("store-ic-unmatched", "data",     allow_duplicate=True),
    Output("store-ic-review",    "data",     allow_duplicate=True),
    Output("ic-review-status",   "children"),
    Input("btn-apply-ic-decisions", "n_clicks"),
    State({"type": "ic-review-decision", "group": ALL}, "value"),
    State({"type": "ic-review-decision", "group": ALL}, "id"),
    State("store-ic-results",  "data"),
    prevent_initial_call=True,
)
def apply_ic_review_decisions(n_clicks, values, ids, results_data):
    """Accept or Reject IC review groups and update all IC stores."""
    if not n_clicks or not results_data:
        return no_update, no_update, no_update, no_update, no_update

    decisions = {id_dict["group"]: val for val, id_dict in zip(values, ids) if val}
    if not decisions:
        msg = dbc.Alert("⚠️ No decisions selected.", color="warning",
                        style={"fontSize": "0.82rem", "padding": "6px 12px"})
        return no_update, no_update, no_update, no_update, msg

    ic_df    = store_to_df_ic(results_data)
    accepted = 0
    rejected = 0

    for recon_id_str, decision in decisions.items():
        mask = ic_df["Recon_ID"].astype(str) == recon_id_str
        if not mask.any():
            continue
        if decision == "accepted":
            ic_df.loc[mask, "Recon_Status"] = "Matched"
            ic_df.loc[mask, "Rule_Applied"] = "Matched (Reviewed)"
            accepted += 1
        elif decision == "rejected":
            ic_df.loc[mask, "Recon_Status"] = "Unmatched"
            ic_df.loc[mask, "Rule_Applied"] = None
            ic_df.loc[mask, "Recon_ID"]     = None
            rejected += 1

    matched_df   = ic_df[ic_df["Recon_Status"] == "Matched"].copy()
    unmatched_df = ic_df[ic_df["Recon_Status"] == "Unmatched"].copy()
    review_df    = ic_df[ic_df["Recon_Status"] == "Review"].copy()

    msg = dbc.Alert(
        f"✅ {accepted} group(s) accepted as Matched (Reviewed), {rejected} group(s) returned to Unmatched.",
        color="success", style={"fontSize": "0.82rem", "padding": "6px 12px"},
    )
    return (df_to_store_ic(ic_df), df_to_store_ic(matched_df),
            df_to_store_ic(unmatched_df), df_to_store_ic(review_df), msg)


# IC manual review — group-wise assign

@app.callback(
    Output("dt-assign-ic", "children"),
    Input("store-ic-results", "data"),
    prevent_initial_call=True,
)
def populate_ic_assign_table(results_data):
    """Render selectable unmatched IC table for Manual Assign."""
    if not results_data:
        return html.Div("Run reconciliation first.", style={"color": COLOR["muted"]})
    ic_df       = store_to_df_ic(results_data)
    unmatched   = ic_df[ic_df["Recon_Status"] == "Unmatched"].copy()
    return _df_to_assign_table(unmatched, "dt-assign-ic-table")


# IC Manual assign commit

@app.callback(
    Output("store-ic-results",   "data",     allow_duplicate=True),
    Output("store-ic-matched",   "data",     allow_duplicate=True),
    Output("store-ic-unmatched", "data",     allow_duplicate=True),
    Output("ic-assign-status",   "children"),
    Input("btn-ic-manual-assign",  "n_clicks"),
    State("dt-assign-ic-table",    "selected_rows"),
    State("dt-assign-ic-table",    "data"),
    State("store-ic-results",      "data"),
    prevent_initial_call=True,
)
def ic_manual_assign(n_clicks, selected_rows, table_data, results_data):
    """Group selected unmatched IC rows into a new Recon ID."""
    if not n_clicks or not results_data:
        return no_update, no_update, no_update, no_update
    if not selected_rows:
        msg = dbc.Alert("⚠️ Select at least one IC transaction row first.",
                        color="warning", style={"fontSize": "0.82rem", "padding": "6px 12px"})
        return no_update, no_update, no_update, msg

    ic_df = store_to_df_ic(results_data)

    # Determine next Recon_ID number
    existing_ids = ic_df["Recon_ID"].dropna().astype(str)
    nums = []
    for rid in existing_ids:
        try:
            nums.append(int(rid.replace("REC_", "")))
        except Exception:
            pass
    next_num  = (max(nums) + 1) if nums else 1
    new_recon = f"REC_{next_num}"

    # Identify selected rows via OrigIdx column in table data
    orig_indices = [table_data[i]["OrigIdx"] for i in selected_rows]

    ic_df.loc[orig_indices, "Recon_Status"] = "Matched"
    ic_df.loc[orig_indices, "Rule_Applied"] = "Manual Assignment"
    ic_df.loc[orig_indices, "Recon_ID"]     = new_recon
    ic_df.loc[orig_indices, "Amt_Diff"]     = ic_df.loc[orig_indices, "_Amount"].sum()

    matched_df   = ic_df[ic_df["Recon_Status"] == "Matched"].copy()
    unmatched_df = ic_df[ic_df["Recon_Status"] == "Unmatched"].copy()

    msg = dbc.Alert(
        f"✅ {len(orig_indices)} transaction(s) assigned as group {new_recon}.",
        color="success", style={"fontSize": "0.82rem", "padding": "6px 12px"},
    )
    return df_to_store_ic(ic_df), df_to_store_ic(matched_df), df_to_store_ic(unmatched_df), msg


# IC Download results

@app.callback(
    Output("dl-ic-matched", "data"),
    Input("btn-ic-dl-matched", "n_clicks"),
    State("store-ic-matched",  "data"),
    prevent_initial_call=True,
)
def download_ic_matched_excel(n_clicks, data):
    if not data: return no_update
    return dcc.send_data_frame(store_to_df_ic(data).to_excel, "IC_Matched_Transactions.xlsx", index=False)


@app.callback(
    Output("dl-ic-matched-csv", "data"),
    Input("btn-ic-dl-matched-csv", "n_clicks"),
    State("store-ic-matched",      "data"),
    prevent_initial_call=True,
)
def download_ic_matched_csv(n_clicks, data):
    if not data: return no_update
    return dcc.send_data_frame(store_to_df_ic(data).to_csv, "IC_Matched_Transactions.csv", index=False)


@app.callback(
    Output("dl-ic-unmatched", "data"),
    Input("btn-ic-dl-unmatched", "n_clicks"),
    State("store-ic-unmatched",  "data"),
    prevent_initial_call=True,
)
def download_ic_unmatched_excel(n_clicks, data):
    if not data: return no_update
    return dcc.send_data_frame(store_to_df_ic(data).to_excel, "IC_Unmatched_Transactions.xlsx", index=False)


###################################################################################################################
# BANK RECONCILIATION CALLBACKS

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
    checks = [l_raw, b_raw, l_date, l_amt, l_nar, b_date, b_amt, b_nar]
    if all(c for c in checks):
        return False, dbc.Alert("✅ All columns mapped. Ready to proceed to Rules.",
                                color="success", style={"fontSize": "0.82rem", "padding": "8px 12px"})
    missing = []
    if not l_raw:  missing.append("Upload Ledger file")
    if not b_raw:  missing.append("Upload Bank file")
    if not l_date: missing.append("Map Ledger Date")
    if not l_amt:  missing.append("Map Ledger Amount")
    if not l_nar:  missing.append("Map Ledger Narration")
    if not b_date: missing.append("Map Bank Date")
    if not b_amt:  missing.append("Map Bank Amount")
    if not b_nar:  missing.append("Map Bank Narration")
    return True, dbc.Alert("⚠️ Remaining: " + " · ".join(missing),
                           color="warning", style={"fontSize": "0.82rem", "padding": "8px 12px"})


@app.callback(
    Output("store-ledger-clean", "data"),
    Output("store-bank-clean",   "data"),
    Output("bank-subtabs",       "active_tab"),
    Output("btn-run-recon",      "disabled"),
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
        return no_update, no_update, "subtab-ingestion", True
    return df_to_store(l_clean), df_to_store(b_clean), "subtab-rules", False


@app.callback(
    Output("rules-checklist", "value"),
    Input("btn-enable-all",  "n_clicks"),
    Input("btn-disable-all", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_all_rules(enable_n, disable_n):
    all_keys = [r["key"] for r in RULES_CONFIG]
    return all_keys if ctx.triggered_id == "btn-enable-all" else []


@app.callback(
    Output("recon-progress", "value", allow_duplicate=True),
    Input("store-recon-running", "data"),
    prevent_initial_call=True,
)
def reset_progress(running):
    return 10 if running else no_update


@app.callback(
    Output("store-results-ledger", "data"),
    Output("store-results-bank",   "data"),
    Output("store-rule-summary",   "data"),
    Output("store-ai-context",     "data"),
    Output("bank-subtabs",         "active_tab", allow_duplicate=True),
    Output("rules-status",         "children"),
    Output("recon-progress",       "value",      allow_duplicate=True),
    Output("store-recon-running",  "data"),
    Input("btn-run-recon", "n_clicks"),
    State("store-ledger-clean", "data"),
    State("store-bank-clean",   "data"),
    State("rules-checklist",    "value"),
    State("kpi-date-tol",       "value"),
    State("kpi-amount-tol",     "value"),
    State("kpi-fuzzy-tol",      "value"),
    prevent_initial_call=True,
)
def run_reconciliation(n_clicks, l_clean, b_clean, enabled_rules, date_tol, amount_tol, fuzzy_tol):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, True, False
    if l_clean is None or b_clean is None:
        warn = dbc.Alert("⚠️ No preprocessed data found. Go to Data Ingestion and click Confirm Mapping.",
                         color="warning", style={"fontSize": "0.84rem", "padding": "8px 12px"})
        return no_update, no_update, no_update, no_update, no_update, warn, 0, False
    try:
        l_df = store_to_df(l_clean)
        b_df = store_to_df(b_clean)
        params = {
            "date_tolerance":   int(date_tol   or DATE_TOLERANCE),
            "amount_tolerance": float(amount_tol or AMOUNT_TOLERANCE),
            "fuzzy_threshold":  int(fuzzy_tol   or FUZZY_THRESHOLD),
        }
        time.sleep(0.5)
        l_out, b_out, _match_log, rule_summary = run_full_reconciliation(
            l_df, b_df, params, enabled_rules or []
        )
        time.sleep(0.5)
        ai_context = build_ai_context(l_out, b_out, rule_summary, params)
        matched_l  = int(l_out["Matched"].sum())
        matched_b  = int(b_out["Matched"].sum())
        status_msg = dbc.Alert(
            f"✅ Reconciliation complete — Ledger: {matched_l}/{len(l_out)} matched | "
            f"Bank: {matched_b}/{len(b_out)} matched",
            color="success", style={"fontSize": "0.84rem", "padding": "8px 12px"},
        )
        return (df_to_store(l_out), df_to_store(b_out),
                rule_summary.to_dict("records"), ai_context,
                "subtab-results", status_msg, 100, False)
    except Exception as exc:
        err_msg = dbc.Alert(f"❌ Error: {str(exc)}", color="danger",
                            style={"fontSize": "0.84rem", "padding": "8px 12px"})
        return no_update, no_update, no_update, no_update, no_update, err_msg, 0, False


# Bank Results display

@app.callback(
    Output("results-kpi-row",    "children"),
    Output("table-recon-ledger", "children"),
    Output("table-recon-bank",   "children"),
    Output("table-unrecon",      "children"),
    Output("table-rule-summary", "children"),
    Output("chart-donut",        "figure"),
    Output("table-review",       "children"),   # ← now renders group cards
    Input("store-results-ledger", "data"),
    Input("store-results-bank",   "data"),
    Input("store-rule-summary",   "data"),
    prevent_initial_call=True,
)
def populate_results(l_data, b_data, rule_sum_data):
    if not l_data or not b_data:
        empty     = html.Div("Run reconciliation first.", style={"color": COLOR["muted"], "padding": "20px"})
        empty_fig = go.Figure()
        return [], empty, empty, empty, empty, empty_fig, empty

    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)

    summary = compute_reconciliation_summary(l_df, b_df)
    totals  = compute_totals(l_df, b_df)

    kpi_row = dbc.Row([
        kpi_card("Ledger Rows",      summary.get("ledger_total", 0),     COLOR["accent"],   "📄"),
        kpi_card("Ledger Matched",   summary.get("ledger_matched", 0),   COLOR["success"],  "✅"),
        kpi_card("Ledger Unmatched", summary.get("ledger_unmatched", 0), COLOR["danger"],   "❌"),
        kpi_card("Match %",          f"{summary.get('ledger_match_pct', 0)}%", COLOR["success"], "📈"),
        kpi_card("Bank Rows",        summary.get("bank_total", 0),       COLOR["accent"],   "🏦"),
        kpi_card("Bank Matched",     summary.get("bank_matched", 0),     COLOR["success"],  "✅"),
        kpi_card("Recon Groups",     summary.get("total_group_ids", 0),  COLOR["warning"],  "🔗"),
        kpi_card("Net Difference",
                 f"{totals.get('overall_difference', 0):,.2f}",
                 COLOR["danger"] if abs(totals.get("overall_difference", 0)) > 0 else COLOR["success"],
                 "⚖️"),
    ], className="g-2")

    tbl_ledger  = table_from_df(l_df, "dt-ledger")
    tbl_bank    = table_from_df(b_df, "dt-bank")

    unmatched_l, unmatched_b = get_unreconciled(l_df, b_df)
    unmatched_combined = pd.concat(
        [unmatched_l.assign(Source="Ledger"), unmatched_b.assign(Source="Bank")],
        ignore_index=True,
    )
    tbl_unrecon = table_from_df(unmatched_combined, "dt-unrecon")

    rule_df  = pd.DataFrame(rule_sum_data) if rule_sum_data else pd.DataFrame()
    tbl_rule = table_from_df(rule_df, "dt-rule-summary", page_size=15)

    if rule_df is not None and not rule_df.empty:
        fig_donut = px.pie(rule_df, names="Rule", values="Transactions_Reconciled",
                           hole=0.45, title="Transactions Reconciled by Rule",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig_donut.update_layout(margin=dict(l=20, r=20, t=50, b=20),
                                legend=dict(font=dict(size=11)), title_font=dict(size=13))
    else:
        fig_donut = go.Figure()
        fig_donut.update_layout(title="No data to chart")

    # ── Group-wise review cards (NEW) ─────────────────────────────────────
    review_ui = build_bank_review_cards(l_df, b_df)

    return kpi_row, tbl_ledger, tbl_bank, tbl_unrecon, tbl_rule, fig_donut, review_ui


# Bank Manual Assign — prepare tables

@app.callback(
    Output("dt-assign-ledger", "children"),
    Output("dt-assign-bank",   "children"),
    Input("store-results-ledger", "data"),
    Input("store-results-bank",   "data"),
    prevent_initial_call=True,
)
def populate_bank_assign_tables(l_data, b_data):
    """Render row-selectable unmatched tables for Manual Assign panel."""
    if not l_data or not b_data:
        empty = html.Div("Run reconciliation first.", style={"color": COLOR["muted"]})
        return empty, empty

    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)

    unmatched_l = l_df[l_df["Matched"] == False].copy()
    unmatched_b = b_df[b_df["Matched"] == False].copy()

    tbl_l = _df_to_assign_table(unmatched_l, "dt-assign-ledger-table")
    tbl_b = _df_to_assign_table(unmatched_b, "dt-assign-bank-table")
    return tbl_l, tbl_b


# Bank Manual Assign — apply decisions

@app.callback(
    Output("store-results-ledger", "data",     allow_duplicate=True),
    Output("store-results-bank",   "data",     allow_duplicate=True),
    Output("bank-review-status",   "children"),
    Input("btn-apply-bank-decisions", "n_clicks"),
    State({"type": "bank-review-decision", "group": ALL}, "value"),
    State({"type": "bank-review-decision", "group": ALL}, "id"),
    State("store-results-ledger", "data"),
    State("store-results-bank",   "data"),
    prevent_initial_call=True,
)
def apply_bank_review_decisions(n_clicks, values, ids, l_data, b_data):
    """
    Accept: keep Matched=True, update Rule/Comment to 'Matched (Reviewed)'.
    Reject: set Matched=False, clear GroupID, Rule, Comment.
    Stats (KPI row) auto-refresh because stores update triggers populate_results.
    """
    if not n_clicks or not l_data or not b_data:
        return no_update, no_update, no_update

    decisions = {id_dict["group"]: val for val, id_dict in zip(values, ids) if val}
    if not decisions:
        msg = dbc.Alert("⚠️ No decisions selected. Use the radio buttons on each group.",
                        color="warning", style={"fontSize": "0.82rem", "padding": "6px 12px"})
        return no_update, no_update, msg

    l_df     = store_to_df(l_data)
    b_df     = store_to_df(b_data)
    accepted = 0
    rejected = 0

    for gid_str, decision in decisions.items():
        try:
            gid_val = float(gid_str)
        except Exception:
            gid_val = gid_str

        l_mask = l_df["GroupID"].fillna(-1) == gid_val
        b_mask = b_df["GroupID"].fillna(-1) == gid_val

        if decision == "accepted":
            l_df.loc[l_mask, "Rule"]    = "Matched (Reviewed)"
            l_df.loc[l_mask, "Comment"] = "Matched (Reviewed)"
            b_df.loc[b_mask, "Rule"]    = "Matched (Reviewed)"
            b_df.loc[b_mask, "Comment"] = "Matched (Reviewed)"
            accepted += 1

        elif decision == "rejected":
            for df_, mask in [(l_df, l_mask), (b_df, b_mask)]:
                df_.loc[mask, "Matched"]    = False
                df_.loc[mask, "GroupID"]    = np.nan
                df_.loc[mask, "Rule"]       = np.nan
                df_.loc[mask, "Comment"]    = "Rejected"
                df_.loc[mask, "AmountDiff"] = np.nan
            rejected += 1

    msg = dbc.Alert(
        f"✅ {accepted} group(s) accepted as Matched (Reviewed), "
        f"{rejected} group(s) returned to Unmatched.",
        color="success", style={"fontSize": "0.82rem", "padding": "6px 12px"},
    )
    return df_to_store(l_df), df_to_store(b_df), msg


# Bank Manual Assign — apply decisions

@app.callback(
    Output("store-results-ledger", "data",     allow_duplicate=True),
    Output("store-results-bank",   "data",     allow_duplicate=True),
    Output("bank-assign-status",   "children"),
    Input("btn-bank-manual-assign", "n_clicks"),
    State("dt-assign-ledger-table", "selected_rows"),
    State("dt-assign-ledger-table", "data"),
    State("dt-assign-bank-table",   "selected_rows"),
    State("dt-assign-bank-table",   "data"),
    State("store-results-ledger",   "data"),
    State("store-results-bank",     "data"),
    prevent_initial_call=True,
)
def bank_manual_assign(n_clicks, l_sel, l_rows, b_sel, b_rows, l_data, b_data):
    """
    Assign selected unmatched Ledger + Bank rows into a new group.
    Uses OrigIdx column (added by _df_to_assign_table) to locate rows in the store DF.
    """
    if not n_clicks or not l_data or not b_data:
        return no_update, no_update, no_update

    l_sel = l_sel or []
    b_sel = b_sel or []
    if not l_sel and not b_sel:
        msg = dbc.Alert("⚠️ Select at least one Ledger or Bank row to assign.",
                        color="warning", style={"fontSize": "0.82rem", "padding": "6px 12px"})
        return no_update, no_update, msg

    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)

    # Next available GroupID
    all_gids = pd.concat([
        l_df["GroupID"].dropna(),
        b_df["GroupID"].dropna(),
    ])
    next_gid = int(all_gids.max()) + 1 if not all_gids.empty else 1

    l_orig_indices = [l_rows[i]["OrigIdx"] for i in l_sel] if l_rows else []
    b_orig_indices = [b_rows[i]["OrigIdx"] for i in b_sel] if b_rows else []

    # Compute AmountDiff for the group
    l_amounts = l_df.loc[l_orig_indices, "_Amount"].sum() if l_orig_indices else 0.0
    b_amounts = b_df.loc[b_orig_indices, "_Amount"].sum() if b_orig_indices else 0.0
    amt_diff  = round(abs(l_amounts - b_amounts), 4)

    for df_, indices in [(l_df, l_orig_indices), (b_df, b_orig_indices)]:
        if not indices:
            continue
        df_.loc[indices, "Matched"]    = True
        df_.loc[indices, "GroupID"]    = float(next_gid)
        df_.loc[indices, "Rule"]       = "Manual Assignment"
        df_.loc[indices, "Comment"]    = "Manual Assignment"
        df_.loc[indices, "AmountDiff"] = amt_diff

    msg = dbc.Alert(
        f"✅ Assigned {len(l_orig_indices)} Ledger row(s) and {len(b_orig_indices)} Bank row(s) "
        f"as Group {next_gid} (Rule: Manual Assignment).",
        color="success", style={"fontSize": "0.82rem", "padding": "6px 12px"},
    )
    return df_to_store(l_df), df_to_store(b_df), msg


# Bank Download results

def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out = df[[c for c in df.columns if not str(c).startswith("_")]]
        df_out.to_excel(writer, index=False, sheet_name="Reconciliation")
    return buf.getvalue()


@app.callback(Output("dl-ledger", "data"),
              Input("btn-dl-ledger", "n_clicks"),
              State("store-results-ledger", "data"), prevent_initial_call=True)
def download_ledger_excel(n, data):
    if not data: return no_update
    return dcc.send_bytes(_df_to_excel_bytes(store_to_df(data)), "reconciled_ledger.xlsx")


@app.callback(Output("dl-ledger-csv", "data"),
              Input("btn-dl-ledger-csv", "n_clicks"),
              State("store-results-ledger", "data"), prevent_initial_call=True)
def download_ledger_csv(n, data):
    if not data: return no_update
    df = store_to_df(data)
    return dcc.send_data_frame(df[[c for c in df.columns if not c.startswith("_")]].to_csv,
                               "reconciled_ledger.csv", index=False)


@app.callback(Output("dl-bank", "data"),
              Input("btn-dl-bank", "n_clicks"),
              State("store-results-bank", "data"), prevent_initial_call=True)
def download_bank_excel(n, data):
    if not data: return no_update
    return dcc.send_bytes(_df_to_excel_bytes(store_to_df(data)), "reconciled_bank.xlsx")


@app.callback(Output("dl-bank-csv", "data"),
              Input("btn-dl-bank-csv", "n_clicks"),
              State("store-results-bank", "data"), prevent_initial_call=True)
def download_bank_csv(n, data):
    if not data: return no_update
    df = store_to_df(data)
    return dcc.send_data_frame(df[[c for c in df.columns if not c.startswith("_")]].to_csv,
                               "reconciled_bank.csv", index=False)


@app.callback(Output("dl-unrecon", "data"),
              Input("btn-dl-unrecon", "n_clicks"),
              State("store-results-ledger", "data"),
              State("store-results-bank",   "data"), prevent_initial_call=True)
def download_unrecon(n, l_data, b_data):
    if not l_data or not b_data: return no_update
    l_df = store_to_df(l_data)
    b_df = store_to_df(b_data)
    ul, ub = get_unreconciled(l_df, b_df)
    combined = pd.concat([ul.assign(Source="Ledger"), ub.assign(Source="Bank")], ignore_index=True)
    return dcc.send_bytes(_df_to_excel_bytes(combined), "unreconciled_items.xlsx")


###################################################################################################################
# BANK AI AGENT CALLBACKS

@app.callback(
    Output("ai-model-hint", "children"),
    Input("ai-provider", "value"),
    prevent_initial_call=True,
)
def update_model_hint(provider):
    hints = AI_MODEL_SUGGESTIONS.get(provider, [])
    return ("Suggestions: " + " · ".join(hints)) if hints else ""


@app.callback(
    Output("ai-conn-status", "children"),
    Output("store-ai-config", "data"),
    Input("btn-test-ai", "n_clicks"),
    State("ai-provider",    "value"),
    State("ai-apikey",      "value"),
    State("ai-model",       "value"),
    State("ai-max-tokens",  "value"),
    State("ai-temperature", "value"),
    prevent_initial_call=True,
)
def save_and_test_ai(n, provider, api_key, model, max_tokens, temperature):
    if not n: return no_update, no_update
    config = {
        "provider":      provider or AI_DEFAULT_CONFIG["provider"],
        "api_key":       api_key  or "",
        "model":         model    or AI_DEFAULT_CONFIG["model"],
        "max_tokens":    int(max_tokens  or 1000),
        "temperature":   float(temperature or 0.3),
        "system_prompt": AI_SYSTEM_PROMPT,
    }
    agent   = AIAgent(config)
    ok, msg = agent.test_connection()
    status  = html.Span(msg, style={"color": COLOR["success"] if ok else COLOR["danger"]})
    return status, config if ok else no_update


@app.callback(
    Output("chat-history",       "children"),
    Output("store-chat-history", "data"),
    Output("chat-input",         "value"),
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
    triggered = ctx.triggered_id
    if triggered == "btn-clear-chat":
        return [], [], ""
    if not user_input or not user_input.strip():
        return no_update, no_update, ""
    history = history or []
    history.append({"role": "user", "content": user_input.strip()})
    if ai_config:
        try:
            reply = AIAgent(ai_config).chat(history, context=ai_context or "")
        except Exception as exc:
            reply = f"⚠️ Error: {str(exc)}"
    else:
        reply = "⚠️ AI Agent not configured. Set your API key in the Agent tab."
    history.append({"role": "assistant", "content": reply})
    bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubbles.append(html.Div([
            html.Div("You" if is_user else "🤖 Agent",
                     style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": COLOR["accent"] if is_user else COLOR["primary"],
                            "marginBottom": "2px"}),
            html.Div(msg["content"],
                     style={"background": "#dbeafe" if is_user else "#f0fdf4",
                            "borderRadius": "8px", "padding": "8px 12px",
                            "maxWidth": "85%",
                            "alignSelf": "flex-end" if is_user else "flex-start",
                            "fontSize": "0.84rem", "lineHeight": "1.55",
                            "whiteSpace": "pre-wrap"}),
        ], style={"display": "flex", "flexDirection": "column",
                  "alignItems": "flex-end" if is_user else "flex-start",
                  "marginBottom": "12px"}))
    return bubbles, history, ""


@app.callback(
    Output("dl-chat", "data"),
    Input("btn-export-chat", "n_clicks"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def export_chat(n, history):
    if not history: return no_update
    lines = [f"[{'YOU' if m['role']=='user' else 'AGENT'}]\n{m['content']}\n" for m in history]
    return dcc.send_string("\n".join(lines), "reconciliation_chat.txt")


###################################################################################################################
# IC AI AGENT CALLBACKS

@app.callback(
    Output("ic-ai-model-hint", "children"),
    Input("ic-ai-provider", "value"),
    prevent_initial_call=True,
)
def update_ic_model_hint(provider):
    hints = AI_MODEL_SUGGESTIONS.get(provider, [])
    return ("Suggestions: " + " · ".join(hints)) if hints else ""


@app.callback(
    Output("ic-ai-conn-status",  "children"),
    Output("store-ic-ai-config", "data"),
    Input("ic-btn-test-ai",      "n_clicks"),
    State("ic-ai-provider",      "value"),
    State("ic-ai-apikey",        "value"),
    State("ic-ai-model",         "value"),
    State("ic-ai-max-tokens",    "value"),
    State("ic-ai-temperature",   "value"),
    prevent_initial_call=True,
)
def save_and_test_ic_ai(n, provider, api_key, model, max_tokens, temperature):
    if not n: return no_update, no_update
    config = {
        "provider":      provider or AI_DEFAULT_CONFIG["provider"],
        "api_key":       api_key  or "",
        "model":         model    or AI_DEFAULT_CONFIG["model"],
        "max_tokens":    int(max_tokens  or 1000),
        "temperature":   float(temperature or 0.3),
        "system_prompt": AI_IC_SYSTEM_PROMPT,
    }
    agent   = AIAgent(config)
    ok, msg = agent.test_connection()
    status  = html.Span(msg, style={"color": COLOR["success"] if ok else COLOR["danger"]})
    return status, config if ok else no_update


@app.callback(
    Output("ic-chat-history",       "children"),
    Output("store-ic-chat-history", "data"),
    Output("ic-chat-input",         "value"),
    Input("ic-btn-send-chat",   "n_clicks"),
    Input("ic-btn-clear-chat",  "n_clicks"),
    Input("ic-chat-input",      "n_submit"),
    State("ic-chat-input",           "value"),
    State("store-ic-chat-history",   "data"),
    State("store-ic-ai-config",      "data"),
    State("store-ic-ai-context",     "data"),
    prevent_initial_call=True,
)
def handle_ic_chat(send_n, clear_n, submit_n, user_input, history, ai_config, ai_context):
    triggered = ctx.triggered_id
    if triggered == "ic-btn-clear-chat":
        return [], [], ""
    if not user_input or not user_input.strip():
        return no_update, no_update, ""
    history = history or []
    history.append({"role": "user", "content": user_input.strip()})
    if ai_config:
        try:
            reply = AIAgent(ai_config).chat(history, context=ai_context or "")
        except Exception as exc:
            reply = f"⚠️ Error: {str(exc)}"
    else:
        reply = "⚠️ IC AI Agent not configured. Set your API key in the IC Agent tab."
    history.append({"role": "assistant", "content": reply})
    bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubbles.append(html.Div([
            html.Div("You" if is_user else "🤖 IC Agent",
                     style={"fontSize": "0.72rem", "fontWeight": "700",
                            "color": COLOR["accent"] if is_user else COLOR["primary"],
                            "marginBottom": "2px"}),
            html.Div(msg["content"],
                     style={"background": "#dbeafe" if is_user else "#f0fdf4",
                            "borderRadius": "8px", "padding": "8px 12px",
                            "maxWidth": "85%",
                            "alignSelf": "flex-end" if is_user else "flex-start",
                            "fontSize": "0.84rem", "lineHeight": "1.55",
                            "whiteSpace": "pre-wrap"}),
        ], style={"display": "flex", "flexDirection": "column",
                  "alignItems": "flex-end" if is_user else "flex-start",
                  "marginBottom": "12px"}))
    return bubbles, history, ""


@app.callback(
    Output("ic-dl-chat", "data"),
    Input("ic-btn-export-chat",    "n_clicks"),
    State("store-ic-chat-history", "data"),
    prevent_initial_call=True,
)
def export_ic_chat(n, history):
    if not history: return no_update
    lines = [f"[{'YOU' if m['role']=='user' else 'IC AGENT'}]\n{m['content']}\n" for m in history]
    return dcc.send_string("\n".join(lines), "ic_reconciliation_chat.txt")


###################################################################################################################
# ACCEPT ALL CALLBACKS  (Bank + IC)

@app.callback(
    Output({"type": "bank-review-decision", "group": ALL}, "value"),
    Input("btn-bank-accept-all", "n_clicks"),
    State({"type": "bank-review-decision", "group": ALL}, "id"),
    prevent_initial_call=True,
)
def bank_accept_all(n_clicks, ids):
    """Set every Bank review group radio to 'accepted' in one click."""
    if not n_clicks or not ids:
        return no_update
    return ["accepted"] * len(ids)


@app.callback(
    Output({"type": "ic-review-decision", "group": ALL}, "value"),
    Input("btn-ic-accept-all", "n_clicks"),
    State({"type": "ic-review-decision", "group": ALL}, "id"),
    prevent_initial_call=True,
)
def ic_accept_all(n_clicks, ids):
    """Set every IC review group radio to 'accepted' in one click."""
    if not n_clicks or not ids:
        return no_update
    return ["accepted"] * len(ids)


###----------------------------------------------------------------------------------------------------------------###
# ENTRY POINT

if __name__ == "__main__":
    app.run(debug=APP_DEBUG, host=APP_HOST, port=APP_PORT)

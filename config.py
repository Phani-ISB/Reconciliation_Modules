"""

### Global Configuration ###

# Purpose: Single point of reference for all tunable parameters
           Changing vaue in config.py propagates initial params in all remaining files.
           Simply , all the default values/settings

           
"""
###----------------------------------------------------------------------------------------------------------------###
# Engine.py (Default params/ tolerances)

# Maximum difference in calendar days allowed between ledger and bank statement
DATE_TOLERANCE    = 3
# Maximum absolute rupee / currency difference allowed. 0.00 means exact match only.
AMOUNT_TOLERANCE  = 0.00
# Minimum similarity score (0-100) for narration text , uses Partial ratio function in rapidfuzz library.
FUZZY_THRESHOLD   = 70

###----------------------------------------------------------------------------------------------------------------###
# Engine.py (Rule Names / sequence of Rules)

RULES_CONFIG = [
    {
        "key"    : "duplicate_detection",
        "label"  : "Duplicate Detection",
        "enabled": True,
        "tooltip": "Flags and removes exact duplicate rows within ledger or bank data."
    },
    {
        "key"    : "narration_exact",
        "label"  : "Narration Exact",
        "enabled": True,
        "tooltip": "Match when narration text is exactly identical (case-insensitive)."
    },
    {
        "key"    : "narration_fuzzy",
        "label"  : "Narration Exact / Fuzzy",
        "enabled": True,
        "tooltip": "Match when narration is exact OR fuzzy score >= FUZZY_THRESHOLD."
    },
    {
        "key"    : "narration_date_exact",
        "label"  : "Narration + Date Exact",
        "enabled": True,
        "tooltip": "Narration exact match AND posting dates are identical."
    },
    {
        "key"    : "narration_date_range",
        "label"  : "Narration + Date Range",
        "enabled": True,
        "tooltip": "Narration exact match AND date difference <= DATE_TOLERANCE days."
    },
    {
        "key"    : "narration_fuzzy_date_range",
        "label"  : "Narration Fuzzy + Date Range",
        "enabled": True,
        "tooltip": "Narration fuzzy match AND date difference <= DATE_TOLERANCE days."
    },
    {
        "key"    : "date_exact",
        "label"  : "Date Exact",
        "enabled": True,
        "tooltip": "Match purely on identical posting date (irrespective of narration)."
    },
    {
        "key"    : "date_range",
        "label"  : "Date Range",
        "enabled": True,
        "tooltip": "Match when date difference <= DATE_TOLERANCE."
    },
    {
        "key"    : "many_to_one",
        "label"  : "Many Ledger → One Bank  (Group Match)",
        "enabled": True,
        "tooltip": "Multiple ledger txns whose amounts sum to one bank txn."
    },
    {
        "key"    : "one_to_many",
        "label"  : "One Ledger → Many Bank  (Group Match)",
        "enabled": True,
        "tooltip": "One ledger txn whose amount equals the sum of several bank txns."
    },
]


###----------------------------------------------------------------------------------------------------------------###
# AI_Agent.py (Default settings in LLM configuration in dashboard)

AI_PROVIDERS = ["OpenAI", "Anthropic (Claude)", "Groq", "Google Gemini"]

AI_DEFAULT_CONFIG = {
    "provider"   : "OpenAI",
    "model"      : "gpt-3.5-turbo",
    "api_key"    : "",            
    "max_tokens" : 1000,
    "temperature": 0.3,
}

# Model suggestions
AI_MODEL_SUGGESTIONS = {
    "OpenAI"            : ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
    "Anthropic (Claude)": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022",
                           "claude-opus-4-5"],
    "Groq"              : ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
    "Google Gemini"     : ["gemini-1.5-flash", "gemini-1.5-pro"],
}

# System prompt
AI_SYSTEM_PROMPT = """You are a Bank Reconciliation Assistant for EY.
You have access to the results of an automated bank reconciliation run.
Answer questions about matched transactions, unreconciled items, rule usage, 
amounts, dates, and anything else from the reconciliation data/results provided to you.
Be concise, factual, and professional. When referring to amounts, use the currency
as-is from the data. If you are unsure, say so — do not fabricate transaction data."""


###----------------------------------------------------------------------------------------------------------------###
# app.py (Dash board default settings)

APP_TITLE   = "Reconciliation Modules"   
APP_PORT    = 8050
APP_DEBUG   = True                        
APP_HOST    = "0.0.0.0"                  


# Color palette in dashboard
COLOR = {
    "primary"    : "#1e3a5f",   # Dark navy — header, tab borders
    "accent"     : "#2563eb",   # Blue — buttons, highlights
    "success"    : "#16a34a",   # Green — matched / reconciled
    "warning"    : "#d97706",   # Amber — partial / review
    "danger"     : "#dc2626",   # Red — unreconciled
    "background" : "#f1f5f9",   # Light slate — page background
    "card"       : "#ffffff",   # White — card backgrounds
    "text"       : "#1e293b",   # Dark text
    "muted"      : "#64748b",   # Muted text / labels
    "border"     : "#e2e8f0",   # Light border
}

###----------------------------------------------------------------------------------------------------------------###
# Export Settings

EXPORT_RECONCILED_FILENAME    = "reconciled_output.xlsx"
EXPORT_UNRECONCILED_FILENAME  = "unreconciled_items.xlsx"
EXPORT_SUMMARY_FILENAME       = "match_summary.xlsx"

# Columns always written to the output Excel (added by the engine)
ENGINE_OUTPUT_COLUMNS = ["Matched", "GroupID", "Comment", "Rule", "AmountDiff"]





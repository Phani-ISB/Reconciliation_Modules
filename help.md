# Reconciliation Modules — Help \& Credits

\---

## 📖 How to Use

### Step 1 — Data Ingestion

* Upload your **Ledger file** (CSV / XLS / XLSX) on the left panel.
* Upload your **Bank Statement file** on the right panel.
* Map the **mandatory columns** for each file:

  * 📅 **Date** — the transaction posting / value date column
  * 💰 **Amount** — the debit/credit amount column  
*(positive = debit/inflow, negative = credit/outflow — or vice versa, engine handles both)*
  * 📝 **Narration** — one or more description/remarks columns  
*(these are concatenated into a single text field for matching)*
* Click **Confirm Mapping \& Proceed to Rules**.

\---

### Step 2 — Rules \& Configuration

* Enable or disable individual reconciliation rules using the checkboxes.
* Use **Enable All** / **Disable All** to toggle all rules at once.
* Adjust the three KPI parameters:

|Parameter|Default|Meaning|
|-|-|-|
|Date Tolerance|3 days|Max allowable date gap for a match|
|Amount Tolerance|0.00|Max allowable amount difference for a match|
|Fuzzy Threshold|70|Min narration similarity score (0–100)|

* Click **Run Reconciliation** to execute.

\---

### Step 3 — Results

The Results tab has five sub-views:

|Tab|Contents|
|-|-|
|✅ Reconciled Ledger|All ledger rows with GroupID, Rule, Matched status|
|🏦 Reconciled Bank|All bank rows with GroupID, Rule, Matched status|
|❌ Unreconciled Items|Rows that could not be matched by any rule|
|📊 Rule Summary \& Chart|Count of matches per rule + donut chart|
|🔍 Manual Review|Group-matched items that need human sign-off|

Download options: **Excel** or **CSV** available on each sub-tab.

\---

### Step 4 — AI Agent

* Select your **LLM Provider** (OpenAI, Anthropic Claude, Groq, Google Gemini).
* Enter your **Model name** (see suggestions shown for each provider).
* Paste your **API Key** (stored only in memory — never saved to disk).
* Click **Save \& Test Connection** — a green ✅ confirms the key works.
* Type questions in the chat box and press **Send** or hit Enter.
* Export the full conversation as a TXT file with **Export Chat**.

\---

## 🔧 Reconciliation Rules (Priority Order)

1. **Duplicate Detection** — Exact duplicate rows are tagged and removed before matching
2. **Narration Exact** — Both narration strings are identical (case-insensitive)
3. **Narration Exact/Fuzzy** — Exact OR fuzzy text match (score ≥ Fuzzy Threshold)
4. **Narration + Date Exact** — Exact narration AND identical posting date
5. **Narration + Date Range** — Exact narration AND date gap ≤ Date Tolerance
6. **Narration Fuzzy + Date Range** — Fuzzy narration AND date gap ≤ Date Tolerance
7. **Date Exact** — Identical date (narration ignored)
8. **Date Range** — Date gap ≤ tolerance (widest / last resort 1-to-1 rule)
9. **Many-to-One** — Several ledger lines whose amounts sum to one bank transaction
10. **One-to-Many** — One ledger line matched to several bank lines that sum together

\---

## 🤖 Supported LLM Providers

|Provider|Install Command|Example Models|
|-|-|-|
|OpenAI|`pip install openai`|gpt-4o, gpt-3.5-turbo|
|Anthropic (Claude)|`pip install anthropic`|claude-3-5-sonnet, claude-opus-4-5|
|Groq|`pip install groq`|llama3-70b-8192, mixtral-8x7b|
|Google Gemini|`pip install google-generativeai`|gemini-1.5-pro, gemini-1.5-flash|

\---

## 📦 Installation

```bash
# Core requirements
pip install dash dash-bootstrap-components plotly pandas openpyxl xlrd rapidfuzz

# LLM providers (install only what you need)
pip install openai
pip install anthropic
pip install groq
pip install google-generativeai

# Run the app
python app.py
```

Then open your browser to: **http://localhost:8050**

\---

## 📁 File Structure

```
reconciliation/
├── app.py                  ← Plotly Dash dashboard (run this)
├── config.py               ← All global settings and defaults
├── data\_ingestion.py       ← File loading and column preprocessing
├── reconciliation\_engine.py← Rule-based matching engine
├── inferences.py           ← Analytics, KPIs, totals, AI context builder
├── ai\_agent.py             ← Multi-provider LLM abstraction layer
└── assets/
    └── help.md             ← This help file (edit to add credits)
```

\---

## 👥 Credits \& Acknowledgements

**Project:** Bank Reconciliation Automation Platform  
**Organisation:**  EY — Ernst \& Young
**Collaboration:** Indian School of Business (ISB)

\---

*Add team member credits, version history, and notes below this line:*

```
Team:
  - \[Name]  \[Role]  \[Contact]

Version History:
  v1.0  —  \[Date]  —  Initial release
```

\---

*Reconciliation Modules — v1.0*


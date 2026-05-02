# Reconciliation Modules — Help \& Credits

|  **Organisation:** EY — Ernst \& Young  |  **Collaboration:** Indian School of Business (ISB)

\---

## 📌 Overview

The **Reconciliation Modules** platform automates two core financial reconciliation workflows:

|Module|Purpose|Input|
|-|-|-|
|🏢 **Intercompany Reconciliation (ICR)**|Match transactions between group entities to eliminate intercompany balances|Single IC dataset file containing all entity pairs|
|🏦 **Bank Reconciliation**|Match ledger entries against bank statement lines to identify timing differences and errors|Separate Ledger file + Bank Statement file|

Each module follows the same four-step workflow:
**Data Ingestion → Rules \& Configuration → Results → AI Agent**

\---

## 🏢 Intercompany Reconciliation (ICR)

### What is ICR?

Intercompany reconciliation is the process of matching transactions recorded between two entities within the same corporate group. Entity A records a payable to Entity B; Entity B records a receivable from Entity A. These should offset each other to zero. Unmatched balances cause consolidation errors and delayed financial close.

\---

### Step 1 — Data Ingestion (ICR)

#### Uploading the IC File

* Upload a **single file** (CSV / XLS / XLSX) containing transactions from **all entities**.
* The file must have at least one row per transaction with columns for entity, partner entity, date, amount, and a description/narration.
* Typical source: exported from SAP / Oracle / ERP intercompany ledger reports.

#### Mandatory Column Mapping

|Field|Description|Example Column Names|
|-|-|-|
|**Entity Column \***|The legal entity that recorded the transaction|`Entity`, `Company Code`, `BookingEntity`|
|**Partner Entity Column \***|The counterpart entity in the same group|`PartnerEntity`, `CounterParty`, `IC\\\\\\\_Partner`|
|**Transaction Date Column \***|Posting or value date of the transaction|`TransactionDate`, `PostingDate`, `Date`|
|**Amount Column \***|Transaction amount (positive = debit, negative = credit)|`Amount`, `LC\\\\\\\_Amount`, `TransAmt`|
|**Narration Column(s) \***|One or more description/reference columns (concatenated for matching)|`Narration1`, `Narration2`, `Reference`, `Text`|

> \\\\\\\*\\\\\\\*Tip:\\\\\\\*\\\\\\\* Select multiple narration columns — they will be concatenated into a single text field used for fuzzy matching. This improves match rates when reference numbers are split across columns.

Click **✅ Confirm Mapping \& Proceed to Rules** once all columns are mapped. A green confirmation message shows the row count loaded.

\---

### Step 2 — Rules \& Configuration (ICR)

#### ICR Matching Parameters

|Parameter|Default|Description|
|-|-|-|
|**Date Tolerance (days)**|5|Maximum allowable gap in calendar days between the posting dates of two matching transactions. Set to 0 for exact date matching only.|
|**Amount Tolerance**|2.00|Maximum absolute currency difference allowed between offsetting amounts. Covers FX rounding, conversion differences, and minor posting mismatches.|
|**Fuzzy Threshold (0–100)**|70|Minimum similarity score for narration text matching using RapidFuzz `partial\\\\\\\_ratio`. 100 = exact match only; 70 = moderately similar text accepted.|

#### ICR Reconciliation Rules (Priority Order)

Rules are applied **sequentially** — a transaction matched by an earlier rule is never re-examined by a later one. Only unmatched transactions enter each subsequent rule.

|#|Rule|Narration Check|Date Check|Description|
|-|-|-|-|-|
|1|**Narration Exact + Date Exact + Amount Match**|Exact (case-insensitive)|Identical date|Strictest rule — all three criteria must match exactly. Highest confidence matches.|
|2|**Narration Fuzzy + Date Exact + Amount Match**|Fuzzy score ≥ threshold|Identical date|Allows for minor text differences (typos, abbreviations) while keeping exact date.|
|3|**Narration Exact + Date Range + Amount Match**|Exact (case-insensitive)|Within date tolerance|Handles posting timing differences (e.g. month-end cut-off) with exact narration.|
|4|**Narration Fuzzy + Date Range + Amount Match**|Fuzzy score ≥ threshold|Within date tolerance|Most flexible standard rule — accommodates both text variation and timing differences.|
|5|**Date Exact + Amount Match**|Ignored|Identical date|Narration is not checked; matches purely on date and offsetting amounts.|
|6|**Amount Match Only**|Ignored|Ignored|Widest 1-to-1 net — last resort before multi-leg check. Flag for review if volume is high.|
|7|**Within Company Reversal + Amount Match**|Ignored|Ignored|Detects same entity–partner reversals where both postings are on the same side.|
|8|**Multiple Group Match (Suggest Manual Review)**|—|—|Groups of 3+ transactions or near-zero net pairs that cannot be cleanly paired 1-to-1. Routed to the Manual Review tab for human sign-off.|

> \\\\\\\*\\\\\\\*Enable All / Disable All\\\\\\\*\\\\\\\* buttons let you toggle all rules at once. Individual rules can be toggled to control which matching logic is active.

Click **▶ Run ICR Reconciliation** to execute.

\---

### Step 3 — Results (ICR)

The Results tab has five sub-views:

#### KPI Summary Bar

|KPI|Description|
|-|-|
|**Total Rows**|Total IC transactions in the uploaded dataset|
|**Matched**|Transactions successfully paired by a rule|
|**Under Review**|Multi-leg groups routed to manual sign-off|
|**Unmatched**|Transactions not matched by any rule|
|**Match %**|Percentage of transactions reconciled (Matched / Total)|
|**Entity Pairs**|Number of unique entity–partner combinations detected|

#### Sub-tabs

|Tab|Contents|
|-|-|
|**✅ Matched**|All matched transactions with `Recon\\\\\\\_ID`, `Rule\\\\\\\_Applied`, `Amt\\\\\\\_Diff`|
|**❌ Unmatched**|Transactions that could not be matched by any enabled rule|
|**🔍 Manual Review**|Multi-leg groups whose net is near zero — require human sign-off before consolidation|
|**📋 Rule Summary**|Count of transactions and groups matched per rule + donut chart|
|**🏢 Entity Matrix**|Pivot table: Entity (rows) × PartnerEntity (columns), net matched amounts|

#### Output Columns Added by ICR Engine

|Column|Values|Description|
|-|-|-|
|`Recon\\\\\\\_Status`|`Matched` / `Unmatched` / `Review`|Final reconciliation status of each row|
|`Rule\\\\\\\_Applied`|Rule name string|Which rule produced the match|
|`Recon\\\\\\\_ID`|`ICR\\\\\\\_000001`, `ICR\\\\\\\_000002`, …|Shared ID linking the two (or more) rows of a matched group|
|`Amt\\\\\\\_Diff`|Numeric|Net sum of amounts within the `Recon\\\\\\\_ID` group (should be \~0 for valid matches)|

**Download options:** ⬇ Excel available on Matched, Unmatched, and Manual Review sub-tabs.

\---

### Step 4 — AI Agent (ICR)

The ICR AI Agent has full context of the reconciliation results and can answer questions such as:

* *"How many transactions between Entity A and Entity B remain unmatched?"*
* *"What is the net unreconciled balance for BHEL\_CORP?"*
* *"Which rule matched the most transactions?"*
* *"List all Review items and explain why they need manual sign-off."*

See the **AI Agent** section below for configuration details.

\---

## 🏦 Bank Reconciliation

### What is Bank Reconciliation?

Bank reconciliation matches entries in the company's **general ledger** against the **bank statement** to identify timing differences (outstanding cheques, deposits in transit), errors, and unauthorized transactions. It is a key internal control performed at every period close.

\---

### Step 1 — Data Ingestion (Bank)

#### Uploading Files

Upload **two separate files**:

* **Ledger File** — General ledger / cash book entries (CSV / XLS / XLSX)
* **Bank Statement File** — Bank-provided statement (CSV / XLS / XLSX)

#### Mandatory Column Mapping (both files)

|Field|Description|Example Column Names|
|-|-|-|
|**Date Column \***|Transaction posting or value date|`Date`, `PostingDate`, `ValueDate`, `Txn Date`|
|**Amount Column \***|Debit/credit amount (positive or negative — engine handles both conventions)|`Amount`, `Debit`, `Credit`, `Dr/Cr`|
|**Narration Column(s) \***|One or more description columns (concatenated)|`Narration`, `Description`, `Remarks`, `Reference`, `Payee`|

> \\\\\\\*\\\\\\\*Amount convention:\\\\\\\*\\\\\\\* The engine treats amounts as signed numbers. Debits and credits from the same transaction should offset each other (e.g. +10,000 in ledger matches −10,000 in bank). If your file uses separate Debit/Credit columns, preprocess them into a single signed column before uploading.

Once both files are uploaded and all columns are mapped, click **✅ Confirm Mapping \& Proceed to Rules**. The button enables only when all six mandatory fields are mapped.

\---

### Step 2 — Rules \& Configuration (Bank)

#### Bank Reconciliation Parameters

|Parameter|Default|Description|
|-|-|-|
|**Date Tolerance (days)**|3|Maximum allowable date gap for a match. Covers float periods, bank processing delays, and weekend timing.|
|**Amount Tolerance**|0.00|Maximum absolute difference between ledger and bank amounts. 0.00 = exact amounts only. Increase for rounding differences.|
|**Fuzzy Threshold (0–100)**|70|Minimum RapidFuzz `partial\\\\\\\_ratio` score for narration text matches.|

#### Bank Reconciliation Rules (Priority Order)

|#|Rule|Description|
|-|-|-|
|1|**Duplicate Detection**|Flags and removes exact duplicate rows within each dataset before matching begins. Duplicates are tagged and excluded from all subsequent rules.|
|2|**Narration Exact**|Both narration strings are exactly identical (case-insensitive, whitespace-normalised).|
|3|**Narration Exact / Fuzzy**|Exact match OR fuzzy similarity score ≥ Fuzzy Threshold. Useful when bank narrations abbreviate ledger descriptions.|
|4|**Narration + Date Exact**|Exact narration match AND posting dates are identical.|
|5|**Narration + Date Range**|Exact narration match AND date gap ≤ Date Tolerance. Handles cut-off timing differences.|
|6|**Narration Fuzzy + Date Range**|Fuzzy narration AND date gap ≤ Date Tolerance. Broadest text + date rule.|
|7|**Date Exact**|Identical posting date regardless of narration.|
|8|**Date Range**|Date gap ≤ tolerance, any narration. Widest last-resort 1-to-1 rule.|
|9|**Many Ledger → One Bank (Group Match)**|Multiple ledger transactions whose amounts **sum** to a single bank transaction amount.|
|10|**One Ledger → Many Bank (Group Match)**|A single ledger transaction that equals the **sum** of several bank transactions.|

Click **▶ Run Reconciliation** to execute. A progress bar shows engine activity.

\---

### Step 3 — Results (Bank)

#### KPI Summary Bar

|KPI|Description|
|-|-|
|**Ledger Rows**|Total rows in the uploaded ledger file|
|**Ledger Matched**|Ledger rows successfully matched to a bank line|
|**Ledger Unmatched**|Ledger rows with no bank counterpart|
|**Match %**|Ledger match rate|
|**Bank Rows**|Total rows in the bank statement|
|**Bank Matched**|Bank rows matched to a ledger entry|
|**Recon Groups**|Number of unique matched groups (each group = one reconciled event)|
|**Net Difference**|Ledger net − Bank net (should be zero for a fully reconciled period)|

#### Sub-tabs

|Tab|Contents|
|-|-|
|**✅ Reconciled Ledger**|Full ledger file with `Matched`, `GroupID`, `Rule`, `AmountDiff` columns added|
|**🏦 Reconciled Bank**|Full bank statement with the same engine output columns|
|**❌ Unreconciled Items**|Combined view of all unmatched ledger + bank rows|
|**📊 Rule Summary \& Chart**|Transactions matched per rule + donut chart breakdown|
|**🔍 Manual Review**|Transactions matched via Many-to-One or One-to-Many rules — require sign-off|

#### Output Columns Added by Bank Engine

|Column|Values|Description|
|-|-|-|
|`Matched`|`True` / `False`|Whether the row was matched|
|`GroupID`|`GRP\\\\\\\_001`, `GRP\\\\\\\_002`, …|Shared ID linking ledger and bank rows of a matched group|
|`Rule`|Rule name string|Which rule produced the match|
|`Comment`|Descriptive note|Additional match context|
|`AmountDiff`|Numeric|Absolute difference between matched ledger and bank amounts|

**Download options:** ⬇ Excel and ⬇ CSV available for Reconciled Ledger, Reconciled Bank, and Unreconciled Items.

\---

### Step 4 — AI Agent (Bank)

The Bank AI Agent has full context of reconciliation results and can answer questions such as:

* *"What is the net unreconciled difference between ledger and bank?"*
* *"How many transactions were matched by the fuzzy narration rule?"*
* *"List all unmatched bank entries above ₹1,00,000."*
* *"What percentage of the ledger was auto-reconciled?"*

\---

## 🤖 AI Agent — Shared Configuration

Both modules include an independent AI Agent with an identical configuration panel.

### Supported LLM Providers

|Provider|Install Command|Recommended Models|
|-|-|-|
|**OpenAI**|`pip install openai`|`gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`|
|**Anthropic (Claude)**|`pip install anthropic`|`claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-opus-4-5`|
|**Groq**|`pip install groq`|`llama3-70b-8192`, `mixtral-8x7b-32768`, `llama3-8b-8192`|
|**Google Gemini**|`pip install google-generativeai`|`gemini-1.5-pro`, `gemini-1.5-flash`|

### Configuration Steps

1. Select your **Provider** from the dropdown.
2. Type your **Model name** (suggestions are shown below the field for each provider).
3. Paste your **API Key** (stored only in browser memory — never written to disk or logs).
4. Click **💾 Save \& Test Connection** — a ✅ green message confirms the key works.
5. Type questions in the chat box and press **➤ Send** or hit **Enter**.
6. Click **⬇ Export Chat (TXT)** to save the full conversation.

> \\\\\\\*\\\\\\\*Security note:\\\\\\\*\\\\\\\* API keys are stored in `dcc.Store` which lives only in the browser session. They are never logged, persisted, or sent anywhere except directly to the LLM provider's API endpoint.

\---

## 📦 Installation \& Setup

### Prerequisites

* Python 3.9 or higher
* pip

### Install All Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# Core framework
pip install dash>=2.14.0 dash-bootstrap-components>=1.5.0

# Data \\\\\\\& charts
pip install pandas>=2.0.0 numpy>=1.24.0 plotly>=5.18.0

# File formats
pip install openpyxl>=3.1.0 xlrd>=2.0.1

# Fuzzy matching
pip install rapidfuzz>=3.0.0

# Deployment (if render is used) for Cloud deployment
pip install gunicorn

# LLM providers — install only what you need
pip install openai
pip install anthropic
pip install groq
pip install google-generativeai
```

### Run Locally

```bash
python app.py
```

Open your browser to: **http://localhost:8050**

### Deploy on Render / Railway / Heroku

The `server = app.server` line in `app.py` exposes the underlying Flask server. Use `gunicorn` as the process manager:

```bash
gunicorn app:server --bind 0.0.0.0:$PORT
```

Set the `PORT` environment variable on your hosting platform (typically 8050 or auto-assigned).

\---

## 📁 File Structure

```
reconciliation/
├── app.py                       ← Main Dash dashboard (run this file)
│
├── config.py                    ← Recon global settings \\\\\\\& defaults
├── data\\\\\\\_ingestion.py            ← ICR \\\& Bank file loading \\\\\\\& column preprocessing
├── engine.py                    ← Sequential Rule-based matching engine
├── inferences.py                ← Reconciliation Analytics, KPIs \\\\\\\& AI context builder│
├── ai\\\\\\\_agent.py                 ← Multi-provider LLM abstraction layer (shared)
│
├── requirements.txt             ← Python dependencies
└── assets/
    └── help.md                  ← This help file
```

\---

## 🔧 Tuning Guide

### Improving Match Rates

|Situation|Suggested Fix|
|-|-|
|Low match rate despite correct data|Lower **Fuzzy Threshold** from 70 → 60|
|Many near-misses on dates (month-end)|Increase **Date Tolerance** from 3 → 7 days|
|FX / rounding mismatches|Increase **Amount Tolerance** from 0 → 1.00 or 2.00|
|Too many false positives|Raise **Fuzzy Threshold** to 85–90 and reduce **Date Tolerance**|
|Large multi-payment settlements|Ensure **Many-to-One** and **One-to-Many** rules are enabled|

### Data Preparation Tips

* **Dates:** Ensure dates are consistently formatted before upload. Mixed formats (DD/MM/YYYY and MM/DD/YYYY in the same column) will cause parse errors.
* **Amounts:** Use a single signed amount column where possible. Avoid using separate Debit and Credit columns without preprocessing.
* **Narration:** The more narration columns you select, the richer the text field for fuzzy matching. Include reference numbers, payment IDs, and invoice numbers where available.
* **Encoding:** Save CSV files as UTF-8. Files with BOM or Windows-1252 encoding may cause character display issues.

\---

## ❓ Frequently Asked Questions

**Q: Can I upload files with different currency amounts?**
A: Yes. The engine works on absolute numeric values. If you have multi-currency IC data, ensure amounts are already converted to a single base currency before uploading, or set a wider Amount Tolerance to absorb FX rounding differences.

**Q: Why are some transactions in "Manual Review" instead of "Matched"?**
A: Rule 8 (ICR) and the Group Match rules (Bank) route transactions to Manual Review when a clean 1-to-1 offset cannot be found, but the net of a group of transactions is near zero or the group has 3+ legs. These require a human to confirm the business reason before marking as reconciled.

**Q: What does Amt\_Diff / AmountDiff mean?**
A: It is the net sum of all amounts within a matched group. For a valid 1-to-1 match, this should be ≤ your Amount Tolerance. For group matches, it shows the residual after netting all legs. Non-zero values highlight transactions that match on narration/date but have a slight amount difference.

**Q: The AI Agent says "No API key configured." What do I do?**
A: Go to the **AI Agent** sub-tab for the relevant module, enter your Provider, Model, and API Key, then click **💾 Save \& Test Connection**. The key must be saved successfully before sending chat messages.

**Q: Can I run both ICR and Bank Reconciliation in the same session?**
A: Yes. Each module maintains its own independent data stores. You can run ICR on the first tab and Bank Recon on the second tab simultaneously within the same browser session.

**Q: Why does the match rate drop when I disable certain rules?**
A: Each rule is a fallback for transactions that could not be matched by stricter rules above it. Disabling Rules 5–8 (especially Amount Match Only and Group Match) will leave transactions unmatched that have inconsistent narrations or dates. Disable rules only when you want to understand which rules are producing matches.

\---

## 📊 Data Flow Summary

```
                    ┌─────────────────────────────────┐
                    │         Upload IC / Ledger        │
                    │         \\\\\\\& Bank Files              │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Column Mapping \\\\\\\& Validation   │
                    │  (icr\\\\\\\_data.py / data\\\\\\\_ingestion.py)│
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │    Rule-Based Matching Engine     │
                    │   (icr\\\\\\\_engine.py / engine.py)    │
                    │                                  │
                    │  Rule 1 → Rule 2 → … → Rule N   │
                    │  (sequential, unmatched pool)     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Analytics \\\\\\\& KPIs               │
                    │  (icr\\\\\\\_inferences.py /            │
                    │   inferences.py)                 │
                    └──────────────┬──────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
    ┌──────────▼────────┐ ┌────────▼────────┐ ┌───────▼────────┐
    │  Results Dashboard │ │  AI Agent Chat  │ │  Excel Export  │
    │  (KPIs, Tables,   │ │  (ai\\\\\\\_agent.py)  │ │  (.xlsx files) │
    │   Charts, Matrix) │ │                 │ │                │
    └───────────────────┘ └─────────────────┘ └────────────────┘
```

\---

## 👥 Credits \& Acknowledgements

**Project:** Reconciliation Automation Platform
**Organisation:** EY — Ernst \& Young
**Academic Collaboration:** Indian School of Business (ISB)

\---

ISB Team:
  - \\\\PHANI KUMAR K R CH    \\\\\\\\\\\[12420063 / Phani\\\_Ramachandra\\\_ampba2025W@isb.edu]
  - \\\\V Hemanth Kumar       \\\\\\\\\\\[12420085 / Hemanth\\\_Kumar\\\_ampba2025W@isb.edu ]
  - \\\\Saketh Gutha          \\\\\\\\\\\[12420070 / Saketh\\\_Gutha\\\_ampba2025W@isb.edu ]
  - \\\\Bhaskar Yerramilli    \\\\\\\\\\\[12420007 / Bhaskar\\\_yerramilli\\\_ampba2025w@isb.edu ]
  - \\\\Vydeepthi Dhulipalla  \\\\\\\\\\\[12420007 / Bhaskar\\\_yerramilli\\\_ampba2025w@isb.edu ]

EY Team:
  - \\\\Satish Penta          \\\\\\\\\\\[ satish.penta@in.ey.com ]
  - \\\\Manikanta S Kasa      \\\\\\\\\\\[ manikanta.kasa@in.ey.com ]
  - \\\\Chandrika M P         \\\\\\\\\\\[ chandrika.p@in.ey.com ]
  - \\\\Darshan Varma         \\\\\\\\\\\[ darshan.varma@in.ey.com ]



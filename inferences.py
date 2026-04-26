"""

### Analytics & Inferences ###

# Purpose: inferences.py module computes all summary statistics, KPIs and analytical outputs for results tab.
           These can be inferenced easily be sending them as context to AI Agent.

           
"""

###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries

import numpy as np
import pandas as pd

###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries

def compute_totals(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> dict:
    if ledger_df is None or "_Amount" not in ledger_df.columns:
        return _empty_totals()
    if bank_df is None or "_Amount" not in bank_df.columns:
        return _empty_totals()

    lamt = ledger_df["_Amount"].fillna(0.0)
    bamt = bank_df["_Amount"].fillna(0.0)

    ledger_debit  = float(lamt[lamt > 0].sum())
    ledger_credit = float(lamt[lamt < 0].sum())   # negative value
    ledger_net    = float(lamt.sum())

    bank_credit   = float(bamt[bamt > 0].sum())
    bank_debit    = float(bamt[bamt < 0].sum())   # negative value
    bank_net      = float(bamt.sum())

    overall_diff  = round(ledger_net - bank_net, 4)

    return {
        "ledger_debit"       : round(ledger_debit,  2),
        "ledger_credit"      : round(ledger_credit, 2),
        "ledger_net"         : round(ledger_net,    2),
        "bank_credit"        : round(bank_credit,   2),
        "bank_debit"         : round(bank_debit,    2),
        "bank_net"           : round(bank_net,      2),
        "overall_difference" : overall_diff,
    }

def _empty_totals() -> dict:
    return {k: 0.0 for k in [
        "ledger_debit", "ledger_credit", "ledger_net",
        "bank_credit", "bank_debit", "bank_net", "overall_difference"
    ]}

###----------------------------------------------------------------------------------------------------------------###
# Reconciliation Summaries

def compute_reconciliation_summary(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    rule_summary_df: pd.DataFrame = None
) -> dict:
    
    if ledger_df is None or bank_df is None:
        return {}

    l_total     = len(ledger_df)
    l_matched   = int(ledger_df["Matched"].sum()) if "Matched" in ledger_df.columns else 0
    l_unmatched = l_total - l_matched
    l_pct       = round((l_matched / l_total * 100) if l_total > 0 else 0, 1)

    b_total     = len(bank_df)
    b_matched   = int(bank_df["Matched"].sum()) if "Matched" in bank_df.columns else 0
    b_unmatched = b_total - b_matched
    b_pct       = round((b_matched / b_total * 100) if b_total > 0 else 0, 1)

    # Count unique GroupIDs (each group = one reconciled event)
    all_gids = set()
    if "GroupID" in ledger_df.columns:
        all_gids.update(ledger_df["GroupID"].dropna().unique())
    if "GroupID" in bank_df.columns:
        all_gids.update(bank_df["GroupID"].dropna().unique())
    total_groups = len(all_gids)

    # Rules that actually produced matches
    rules_used = []
    if rule_summary_df is not None and not rule_summary_df.empty:
        rules_used = rule_summary_df["Rule"].tolist()

    return {
        "ledger_total"      : l_total,
        "ledger_matched"    : l_matched,
        "ledger_unmatched"  : l_unmatched,
        "ledger_match_pct"  : l_pct,
        "bank_total"        : b_total,
        "bank_matched"      : b_matched,
        "bank_unmatched"    : b_unmatched,
        "bank_match_pct"    : b_pct,
        "total_group_ids"   : total_groups,
        "rules_used"        : rules_used,
    }

###----------------------------------------------------------------------------------------------------------------###
# Duplicates in Ledger/Bank statements

def find_duplicates(df: pd.DataFrame, source_label: str = "Data") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    dup_mask = df.duplicated(keep=False)
    dups     = df[dup_mask].copy()

    if dups.empty:
        return pd.DataFrame()

    dups.sort_values(by=list(df.columns), inplace=True)
    dups.insert(0, "Source", source_label)
    return dups.reset_index(drop=True)


###----------------------------------------------------------------------------------------------------------------###
# Summaries wrt currency
def compute_currency_summary(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    currency_col: str = None
) -> pd.DataFrame:
    if currency_col is None:
        return pd.DataFrame()

    rows = []
    for source_label, df in [("Ledger", ledger_df), ("Bank", bank_df)]:
        if df is None or currency_col not in df.columns:
            continue
        grp = df.groupby(currency_col).agg(
            Total_Transactions = ("_Amount", "count"),
            Total_Amount       = ("_Amount", "sum"),
            Matched_Count      = ("Matched",  "sum"),
        ).reset_index()
        grp.rename(columns={currency_col: "Currency"}, inplace=True)
        grp.insert(0, "Source", source_label)
        grp["Unmatched_Count"] = grp["Total_Transactions"] - grp["Matched_Count"]
        rows.append(grp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


###----------------------------------------------------------------------------------------------------------------###
# Amount differences Summary

def compute_amount_diff_summary(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> dict:
    if ledger_df is None or "AmountDiff" not in ledger_df.columns:
        return {}

    diffs = ledger_df["AmountDiff"].dropna().astype(float)

    return {
        "max_diff"        : round(float(diffs.max()),  4) if not diffs.empty else 0.0,
        "avg_diff"        : round(float(diffs.mean()), 4) if not diffs.empty else 0.0,
        "total_diff_value": round(float(diffs.sum()),  4) if not diffs.empty else 0.0,
        "zero_diff_count" : int((diffs == 0.0).sum()),
    }

###----------------------------------------------------------------------------------------------------------------###
# AI Context Block

def build_ai_context(
    ledger_df:       pd.DataFrame,
    bank_df:         pd.DataFrame,
    rule_summary_df: pd.DataFrame,
    params:          dict
) -> str:
    
    totals  = compute_totals(ledger_df, bank_df)
    summary = compute_reconciliation_summary(ledger_df, bank_df, rule_summary_df)
    diffs   = compute_amount_diff_summary(ledger_df, bank_df)

    lines = [
        "=== BANK RECONCILIATION RESULTS CONTEXT ===",
        "",
        "PARAMETERS USED:",
        f"  Date Tolerance   : {params.get('date_tolerance', 3)} days",
        f"  Amount Tolerance : {params.get('amount_tolerance', 0.0)}",
        f"  Fuzzy Threshold  : {params.get('fuzzy_threshold', 70)}%",
        "",
        "LEDGER SUMMARY:",
        f"  Total rows       : {summary.get('ledger_total', 'N/A')}",
        f"  Matched          : {summary.get('ledger_matched', 'N/A')} ({summary.get('ledger_match_pct','?')}%)",
        f"  Unmatched        : {summary.get('ledger_unmatched', 'N/A')}",
        f"  Total Debits     : {totals.get('ledger_debit', 0):,.2f}",
        f"  Total Credits    : {totals.get('ledger_credit', 0):,.2f}",
        f"  Net              : {totals.get('ledger_net', 0):,.2f}",
        "",
        "BANK STATEMENT SUMMARY:",
        f"  Total rows       : {summary.get('bank_total', 'N/A')}",
        f"  Matched          : {summary.get('bank_matched', 'N/A')} ({summary.get('bank_match_pct','?')}%)",
        f"  Unmatched        : {summary.get('bank_unmatched', 'N/A')}",
        f"  Total Credits    : {totals.get('bank_credit', 0):,.2f}",
        f"  Total Debits     : {totals.get('bank_debit', 0):,.2f}",
        f"  Net              : {totals.get('bank_net', 0):,.2f}",
        "",
        f"OVERALL DIFFERENCE (Ledger Net - Bank Net): {totals.get('overall_difference', 0):,.4f}",
        f"TOTAL RECONCILED GROUPS: {summary.get('total_group_ids', 0)}",
        "",
    ]

    # Rule breakdown
    if rule_summary_df is not None and not rule_summary_df.empty:
        lines.append("RULE BREAKDOWN:")
        for _, row in rule_summary_df.iterrows():
            lines.append(f"  {row['Rule']}: {row['Transactions_Reconciled']} transactions")
        lines.append("")

    # Amount diff stats
    if diffs:
        lines += [
            "AMOUNT DIFFERENCE STATS (on matched pairs):",
            f"  Max diff   : {diffs.get('max_diff', 0):,.4f}",
            f"  Avg diff   : {diffs.get('avg_diff', 0):,.4f}",
            f"  Total diff : {diffs.get('total_diff_value', 0):,.4f}",
            f"  Zero diff  : {diffs.get('zero_diff_count', 0)} pairs",
            "",
        ]

    lines.append("=== END OF CONTEXT ===")
    return "\n".join(lines)


###----------------------------------------------------------------------------------------------------------------###
# IC AI Context Block

def build_ic_ai_context(
    matched_df:      pd.DataFrame,
    unmatched_df:    pd.DataFrame,
    review_df:       pd.DataFrame,
    rule_summary_df: pd.DataFrame,
    params:          dict
) -> str:

    matched_count   = len(matched_df)   if matched_df   is not None else 0
    unmatched_count = len(unmatched_df) if unmatched_df is not None else 0
    review_count    = len(review_df)    if review_df    is not None else 0
    total_count     = matched_count + unmatched_count + review_count
    match_pct       = round(100 * matched_count / total_count, 1) if total_count > 0 else 0.0

    # Amount stats from matched set
    matched_amt_sum  = 0.0
    matched_amt_max  = 0.0
    offset_sum       = 0.0
    if matched_df is not None and not matched_df.empty and "_Amount" in matched_df.columns:
        amounts         = matched_df["_Amount"].fillna(0.0)
        matched_amt_sum = round(float(amounts.sum()), 2)
        matched_amt_max = round(float(amounts.abs().max()), 2)
    if matched_df is not None and "Amt_Diff" in matched_df.columns:
        offset_sum = round(float(matched_df["Amt_Diff"].dropna().astype(float).sum()), 4)

    # Unique entity pairs from matched set
    entity_pairs = []
    if matched_df is not None and "_Entity" in matched_df.columns and "_PartnerEntity" in matched_df.columns:
        pairs = (
            matched_df[["_Entity", "_PartnerEntity"]]
            .dropna()
            .apply(lambda r: f"{r['_Entity']} ↔ {r['_PartnerEntity']}", axis=1)
            .unique()
            .tolist()
        )
        entity_pairs = pairs[:20]  # cap to avoid context overflow

    lines = [
        "=== INTER-COMPANY RECONCILIATION RESULTS CONTEXT ===",
        "",
        "PARAMETERS USED:",
        f"  Date Tolerance   : {params.get('date_tolerance', 3)} days",
        f"  Amount Tolerance : {params.get('amount_tolerance', 0.0)}",
        f"  Fuzzy Threshold  : {params.get('fuzzy_threshold', 70)}%",
        "",
        "OVERALL SUMMARY:",
        f"  Total Transactions : {total_count}",
        f"  Matched            : {matched_count} ({match_pct}%)",
        f"  Unmatched          : {unmatched_count}",
        f"  For Manual Review  : {review_count}",
        "",
        "MATCHED TRANSACTIONS — AMOUNT STATS:",
        f"  Net Amount Sum     : {matched_amt_sum:,.2f}",
        f"  Max Abs Amount     : {matched_amt_max:,.2f}",
        f"  Total Offset (Amt_Diff) : {offset_sum:,.4f}",
        "",
    ]

    # Entity pairs
    if entity_pairs:
        lines.append("ENTITY PAIRS IN MATCHED DATA:")
        for pair in entity_pairs:
            lines.append(f"  {pair}")
        lines.append("")

    # Rule breakdown
    if rule_summary_df is not None and not rule_summary_df.empty:
        lines.append("RULE BREAKDOWN:")
        for _, row in rule_summary_df.iterrows():
            rule_col  = row.get("Rule",  row.get("rule",  "Unknown"))
            count_col = row.get("Count", row.get("count", row.get("Transactions_Reconciled", "?")))
            lines.append(f"  {rule_col}: {count_col} transactions")
        lines.append("")

    # Sample unmatched narrations (top 5) for context
    if unmatched_df is not None and not unmatched_df.empty and "_Narration" in unmatched_df.columns:
        samples = unmatched_df["_Narration"].dropna().head(5).tolist()
        if samples:
            lines.append("SAMPLE UNMATCHED NARRATIONS (top 5):")
            for s in samples:
                lines.append(f"  • {s}")
            lines.append("")

    lines.append("=== END OF IC CONTEXT ===")
    return "\n".join(lines)







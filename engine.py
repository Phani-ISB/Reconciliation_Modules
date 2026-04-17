"""

### Reconciliation Engine ###

# Purpose : The core reconciliation logic. Take two processed dataframes (ledger & bank)
and applies configurable sequence of rules to match transactions.

# Matching hierarchy (in order of priority):
    0.  Duplicate Detection   — flags & removes exact internal duplicates
    1.  Narration Exact
    2.  Narration Exact / Fuzzy
    3.  Narration + Date Exact
    4.  Narration + Date Range
    5.  Narration Fuzzy + Date Range
    6.  Date Exact
    7.  Date Range
    8.  Many Ledger → One Bank  (Group / aggregate matching)
    9.  One Ledger → Many Bank  (Group / aggregate matching)

"""

###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries
import pandas as pd
import numpy as np
from itertools import combinations
from rapidfuzz import fuzz


###----------------------------------------------------------------------------------------------------------------###
# Helper Functions for rules

def amount_match(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance

def date_exact(d1, d2) -> bool:
    return pd.Timestamp(d1).date() == pd.Timestamp(d2).date()

def date_range(d1, d2, tolerance_days : int) -> bool :
    return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days) <= tolerance_days

def narration_exact(n1, n2) -> bool:
    return str(n1).strip() == str(n2).strip()

def narration_fuzzy(n1, n2, threshold : int) -> bool:
    if pd.isna(n1) or pd.isna(n2):
        return False
    return fuzz.partial_ratio(str(n1), str(n2)) >= threshold


###----------------------------------------------------------------------------------------------------------------###
# Reconciliation Rules

"""
    Create a dict of   rule_key → callable(lrow, brow) → bool
    Each callable takes one ledger row and one bank row and returns whether
    those two rows satisfy that particular rule (EXCLUDING amount check,
    which is applied separately as a first-pass filter).

    Parameters
    ----------
    params : dict  with keys  date_tolerance, amount_tolerance, fuzzy_threshold
    """

def _build_rule_funcs(params : dict) -> dict :
  dt = int(params.get("date_tolerance" , 3))
  ft = int(params.get("fuzzy_threshold" , 70))

  return {
        "narration_exact": lambda l, b: (
            _narration_exact(l["_Narration"], b["_Narration"])
        ),
        "narration_fuzzy": lambda l, b: (
            _narration_exact(l["_Narration"], b["_Narration"]) or
            _narration_fuzzy(l["_Narration"], b["_Narration"], ft)
        ),
        "narration_date_exact": lambda l, b: (
            _narration_exact(l["_Narration"], b["_Narration"]) and
            _date_exact(l["_Date"], b["_Date"])
        ),
        "narration_date_range": lambda l, b: (
            _narration_exact(l["_Narration"], b["_Narration"]) and
            _date_range(l["_Date"], b["_Date"], dt)
        ),
        "narration_fuzzy_date_range": lambda l, b: (
            (_narration_exact(l["_Narration"], b["_Narration"]) or
             _narration_fuzzy(l["_Narration"], b["_Narration"], ft)) and
            (_date_exact(l["_Date"], b["_Date"]) or
             _date_range(l["_Date"], b["_Date"], dt))
        ),
        "date_exact": lambda l, b: (
            _date_exact(l["_Date"], b["_Date"])
        ),
        "date_range": lambda l, b: (
            _date_range(l["_Date"], b["_Date"], dt)
        ),
    }

###----------------------------------------------------------------------------------------------------------------###
# 0.  Internal Duplicates
"""
    Find rows that are 100% identical across ALL original columns and tag them
    with a shared GroupID and Rule = "Duplicate Data".

    Duplicate rows are removed (keep first occurrence) so they do not
    interfere with the 1-to-1 matching rules.

    Parameters
    ----------
    df           : DataFrame  (must have Matched, GroupID, Comment, Rule columns)
    group_start  : int        first GroupID to use

    Returns
    -------
    (cleaned_df, next_group_id)
    """

def _tag_internal_duplicates(df: pd.DataFrame, group_start: int) -> tuple:

    df = df.copy()

    # Represent every row as a concatenation of all cell values for duplicate check
    row_signature = df.astype(str).agg("|".join, axis=1)

    # Finding all rows that have at least one duplicate somewhere
    duplicated_mask = row_signature.duplicated(keep=False)
    dup_sigs = row_signature[duplicated_mask]

    gid = group_start
    group_map = {}

    # Group identical rows together and assign the same GroupID
    for _sig, idx_series in dup_sigs.groupby(dup_sigs).groups.items():
        for i in idx_series:
            group_map[i] = gid
        gid += 1

    # Assign GroupID and Rule to duplicate rows
    df["GroupID"] = df.index.map(group_map)
    df.loc[df["GroupID"].notna(), "Rule"]    = "Duplicate Data"
    df.loc[df["GroupID"].notna(), "Comment"] = "Duplicate Data"
    df.loc[df["GroupID"].notna(), "Matched"] = False

    # Remove duplicate rows — keep only the first occurrence
    keep_mask = ~row_signature.duplicated(keep="first")
    df = df[keep_mask].copy()
    df.reset_index(drop=True, inplace=True)

    return df, gid

###----------------------------------------------------------------------------------------------------------------###
# 1.  One-to-One Rule based matching
"""
    Apply each enabled 1-to-1 rule in sequence.

    For each unmatched ledger row:
        1. Filter unmatched bank rows whose amount is within tolerance.
        2. Among those, apply the rule function to find the best candidate.
        3. If found, assign a shared GroupID, mark both as Matched.

    Parameters
    ----------
    ledger_df, bank_df : DataFrames (modified in place via .at assignments)
    rule_funcs         : dict  (built by _build_rule_funcs)
    enabled_rules      : list  of rule keys the user has ticked ON
    params             : dict  with amount_tolerance etc.
    start_gid          : int   starting GroupID counter

    Returns
    -------
    (match_log_list, next_gid)
        match_log_list: list of dicts, one entry per matched pair
    """

def _run_one_to_one_rules(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    rule_funcs: dict,
    enabled_rules: list,
    params: dict,
    start_gid: int
) -> tuple:

    at = float(params.get("amount_tolerance", 0.0))
    match_log = []
    gid = start_gid

    ONE_TO_ONE_RULE_KEYS = [
        "narration_exact",
        "narration_fuzzy",
        "narration_date_exact",
        "narration_date_range",
        "narration_fuzzy_date_range",
        "date_exact",
        "date_range",
    ]

    for rule_key in ONE_TO_ONE_RULE_KEYS:

        if rule_key not in enabled_rules:
            continue

        rule_fn = rule_funcs[rule_key]

        l_unmatched = ledger_df[~ledger_df["Matched"]]
        b_unmatched = bank_df[~bank_df["Matched"]]

        for li, lrow in l_unmatched.iterrows():
            ledger_amount = lrow["_Amount"]

            # First Filter with Amount Tolerance
            candidates = b_unmatched[
                abs(b_unmatched["_Amount"] - ledger_amount) <= at
            ]

            if candidates.empty:
                continue
            # Applying Rules against above candidates
            rule_hits = candidates[
                candidates.apply(lambda brow: rule_fn(lrow, brow), axis=1)
            ]

            if rule_hits.empty:
                continue

            # Take First match against rules
            bi = rule_hits.index[0]
            brow = bank_df.loc[bi]

            # Mark both rows as matched against ledger & bank
            ledger_df.at[li, "Matched"]    = True
            ledger_df.at[li, "GroupID"]    = gid
            ledger_df.at[li, "Comment"]    = rule_key
            ledger_df.at[li, "Rule"]       = rule_key
            ledger_df.at[li, "AmountDiff"] = round(abs(lrow["_Amount"] - brow["_Amount"]), 4)

            bank_df.at[bi, "Matched"]    = True
            bank_df.at[bi, "GroupID"]    = gid
            bank_df.at[bi, "Comment"]    = rule_key
            bank_df.at[bi, "Rule"]       = rule_key
            bank_df.at[bi, "AmountDiff"] = round(abs(lrow["_Amount"] - brow["_Amount"]), 4)

            # Append to Match summary
            match_log.append({
                "GroupID"      : gid,
                "Rule"         : rule_key,
                "MatchType"    : "1-to-1",
                "Ledger_Index" : li,
                "Bank_Index"   : bi,
                "LedgerAmount" : lrow["_Amount"],
                "BankAmount"   : brow["_Amount"],
                "AmountDiff"   : round(abs(lrow["_Amount"] - brow["_Amount"]), 4),
            })

            gid += 1

            # Unmatched transactions
            b_unmatched = bank_df[~bank_df["Matched"]]

    return match_log, gid

###----------------------------------------------------------------------------------------------------------------###
# 2.  Group Matching

def _subset_sum_match(df: pd.DataFrame, target: float, tol: float) -> list | None:
    """
    Try all combinations of 2–5 rows in df whose '_Amount' sums to target ± tol.

    Returns the list of matching row indices, or None if no combination found.
    This is an exhaustive search — works well for small candidate sets.
    """
    for r in range(2, min(6, len(df) + 1)):
        for combo in combinations(df.index, r):
            combo_sum = df.loc[list(combo), "_Amount"].sum()
            if abs(combo_sum - target) <= tol:
                return list(combo)
    return None


def _greedy_match(df: pd.DataFrame, target: float, tol: float) -> list | None:
    """
    Greedy approach: pick rows with the largest absolute amounts first and
    accumulate until we are within tol of target.

    Returns list of matching indices, or None if unsuccessful.
    """
    total    = 0.0
    selected = []

    # Sort descending by absolute amount — tackle biggest values first
    for idx, amt in df["_Amount"].abs().sort_values(ascending=False).items():
        val = df.at[idx, "_Amount"]

        # Only add this value if we're not already overshooting
        if abs(total + val) <= abs(target) + tol:
            total += val
            selected.append(idx)

            if abs(total - target) <= tol:
                return selected   # success

    return None

def _run_group_matching(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    direction: str,   # "Many-to-One" or "One-to-Many"
    params: dict,
    start_gid: int
) -> tuple:
    """
    Match groups of transactions where:
        Many-to-One: several ledger lines sum to one bank line
        One-to-Many: one ledger line equals several bank lines summed

    Uses a greedy approach (fast) for larger candidate sets.

    Returns
    -------
    (match_log_list, next_gid)
    """
    at = float(params.get("amount_tolerance", 0.0))
    match_log = []
    gid = start_gid

    # Decide which DataFrame is the "one" side and which is the "many" side
    if direction == "many_to_one":
        source_df, target_df = ledger_df, bank_df
        rule_label = "Many-to-One"
    else:
        source_df, target_df = bank_df, ledger_df
        rule_label = "One-to-Many"

    # Iterate over every unmatched row on the "one" (target) side
    for ti, trow in target_df[~target_df["Matched"]].iterrows():
        target_amount = trow["_Amount"]

        mask = (
            (~source_df["Matched"]) &
            (source_df["_Amount"] * target_amount > 0) &           # same sign
            (source_df["_Amount"].abs() <= abs(target_amount) + at) # smaller magnitude
        )
        candidates = source_df[mask]

        if candidates.empty or len(candidates) > 50:
            # Skip if no candidates, or too many
            continue

        # Try to find a combination that sums to target_amount
        group_indices = _greedy_match(candidates, target_amount, at)

        if group_indices is None and len(candidates) <= 10:
            group_indices = _subset_sum_match(candidates, target_amount, at)

        if group_indices is None:
            continue

        # Mark the "many" side rows
        source_df.loc[group_indices, "Matched"]    = True
        source_df.loc[group_indices, "GroupID"]    = gid
        source_df.loc[group_indices, "Comment"]    = rule_label
        source_df.loc[group_indices, "Rule"]       = rule_label
        source_df.loc[group_indices, "AmountDiff"] = 0.0   # sum matches, individual diff irrelevant

        # Mark the "one" side row
        target_df.at[ti, "Matched"]    = True
        target_df.at[ti, "GroupID"]    = gid
        target_df.at[ti, "Comment"]    = rule_label
        target_df.at[ti, "Rule"]       = rule_label
        target_df.at[ti, "AmountDiff"] = round(
            abs(source_df.loc[group_indices, "_Amount"].sum() - target_amount), 4
        )

        # Append to Match summary
        if direction == "many_to_one":
            l_indices = group_indices
            b_indices = [ti]
        else:
            l_indices = [ti]
            b_indices = group_indices

        match_log.append({
            "GroupID"      : gid,
            "Rule"         : rule_label,
            "MatchType"    : rule_label,
            "Ledger_Indices": l_indices,
            "Bank_Indices"  : b_indices,
            "TargetAmount"  : target_amount,
            "SourceSum"     : source_df.loc[group_indices, "_Amount"].sum(),
        })

        gid += 1

    return match_log, gid

###----------------------------------------------------------------------------------------------------------------###
# Function to Run the above reconciliation pipeline

def run_full_reconciliation(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    params: dict,
    enabled_rules: list
) -> tuple:

# Copy the ledger and bank dataframes
    ledger_df = ledger_df.copy()
    bank_df   = bank_df.copy()

    all_match_logs = []
    gid = 1   # GroupID starts at 1

    # Duplicate Detection
    if "duplicate_detection" in enabled_rules:
        ledger_df, gid = _tag_internal_duplicates(ledger_df, gid)
        bank_df,   gid = _tag_internal_duplicates(bank_df,   gid)

    # One-to-One Rule Matching
    rule_funcs = _build_rule_funcs(params)
    one_to_one_log, gid = _run_one_to_one_rules(
        ledger_df, bank_df, rule_funcs, enabled_rules, params, gid
    )
    all_match_logs.extend(one_to_one_log)

    # Group Matching
    if "many_to_one" in enabled_rules:
        group_log, gid = _run_group_matching(
            ledger_df, bank_df, "many_to_one", params, gid
        )
        all_match_logs.extend(group_log)

    if "one_to_many" in enabled_rules:
        group_log, gid = _run_group_matching(
            ledger_df, bank_df, "one_to_many", params, gid
        )
        all_match_logs.extend(group_log)

    # Appending Matchings to a dataframe
    if all_match_logs:
        match_log_df = pd.DataFrame(all_match_logs)
    else:
        match_log_df = pd.DataFrame(columns=[
            "GroupID", "Rule", "MatchType",
            "Ledger_Index", "Bank_Index", "AmountDiff"
        ])

    # Summary of applied reconciliation rules
    rule_counts = (
        pd.concat([
            ledger_df[ledger_df["Matched"] & ledger_df["Rule"].notna()]["Rule"],
            bank_df[bank_df["Matched"] & bank_df["Rule"].notna()]["Rule"],
        ])
        .value_counts()
        .reset_index()
    )
    rule_counts.columns = ["Rule", "Transactions_Reconciled"]

    return ledger_df, bank_df, match_log_df, rule_counts

###----------------------------------------------------------------------------------------------------------------###
# INFERENCES

# Unreconciled Data in Ledger & Bank statement (For inferencing)

def get_unreconciled(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> tuple:

    unmatched_ledger = ledger_df[~ledger_df["Matched"]].copy()
    unmatched_bank   = bank_df[~bank_df["Matched"]].copy()
    return unmatched_ledger, unmatched_bank

# Returns Manual review for all Many-to-one and One-to-Many Transactions
def get_manual_review_items(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:

    group_rules = {"Many-to-One", "One-to-Many"}

    ledger_review = ledger_df[ledger_df["Rule"].isin(group_rules)].copy()
    bank_review   = bank_df[bank_df["Rule"].isin(group_rules)].copy()

    ledger_review.insert(0, "Source", "Ledger")
    bank_review.insert(0, "Source", "Bank")

    return pd.concat([ledger_review, bank_review], ignore_index=True)
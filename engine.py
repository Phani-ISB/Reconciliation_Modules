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
            narration_exact(l["_Narration"], b["_Narration"])
        ),
        "narration_fuzzy": lambda l, b: (
            narration_exact(l["_Narration"], b["_Narration"]) or
            narration_fuzzy(l["_Narration"], b["_Narration"], ft)
        ),
        "narration_date_exact": lambda l, b: (
            narration_exact(l["_Narration"], b["_Narration"]) and
            date_exact(l["_Date"], b["_Date"])
        ),
        "narration_date_range": lambda l, b: (
            narration_exact(l["_Narration"], b["_Narration"]) and
            date_range(l["_Date"], b["_Date"], dt)
        ),
        "narration_fuzzy_date_range": lambda l, b: (
            (narration_exact(l["_Narration"], b["_Narration"]) or
             narration_fuzzy(l["_Narration"], b["_Narration"], ft)) and
            (date_exact(l["_Date"], b["_Date"]) or
             date_range(l["_Date"], b["_Date"], dt))
        ),
        "date_exact": lambda l, b: (
            date_exact(l["_Date"], b["_Date"])
        ),
        "date_range": lambda l, b: (
            date_range(l["_Date"], b["_Date"], dt)
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
    row_signature = df.apply(lambda row: "|".join(map(str, row)),axis=1)

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
    ledger_df["Matched"] = ledger_df["Matched"].fillna(False).astype(bool)
    bank_df["Matched"]   = bank_df["Matched"].fillna(False).astype(bool)
    
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

        l_unmatched = ledger_df.loc[ledger_df["Matched"] == False]
        b_unmatched = bank_df.loc[bank_df["Matched"] == False]

        for li, lrow in l_unmatched.iterrows():
            ledger_amount = pd.to_numeric(lrow["_Amount"] , errors ="coerce")
            if pd.isna(ledger_amount) :
                continue
            # First Filter with Amount Tolerance
            amt_series = pd.to_numeric(b_unmatched["_Amount"], errors ="coerce")
            mask = (amt_series - float(ledger_amount)).abs() <= float(at)
            mask = mask.fillna(False)
            candidates = b_unmatched.loc[mask]

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
            b_unmatched = bank_df.loc[bank_df["Matched"] == False]

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

###################################################################################################################
# INTER-COMPANY RECONCILIATION ENGINE

"""
Inter-Company (IC) Reconciliation matches transactions between counterparty entities.
Key differences from Bank Reconciliation:
    - Amount matching is OFFSET (a + b ≈ 0, not a ≈ b)
    - Entity/PartnerEntity fields define the counterparty relationship
    - Unmatched and Matched rows coexist in same DataFrame
    - Rule priority sequence differs (8 IC-specific rules)
"""

###----------------------------------------------------------------------------------------------------------------###
# IC Helper Functions

def _ic_amount_offset_match(amt1: float, amt2: float, tol: float) -> bool:
    """Check if two amounts offset each other (sum to ~0) within tolerance."""
    return abs(amt1 + amt2) <= tol


def _ic_date_match(d1, d2, days=None) -> bool:
    """Check if dates match: exact if days=0, within range if days>0, any if days=None."""
    if days is None:
        return True
    d1_ts = pd.Timestamp(d1)
    d2_ts = pd.Timestamp(d2)
    return abs((d1_ts - d2_ts).days) <= days


def _ic_narration_match(n1, n2, fuzzy=False, threshold=70) -> bool:
    """Check narration match: exact if fuzzy=False, fuzzy if fuzzy=True, None to skip."""
    if fuzzy is None:
        return True
    if pd.isna(n1) or pd.isna(n2):
        return False
    n1_str = str(n1).strip().lower()
    n2_str = str(n2).strip().lower()
    if not fuzzy:
        return n1_str == n2_str
    return fuzz.partial_ratio(n1_str, n2_str) >= threshold


###----------------------------------------------------------------------------------------------------------------###
# IC Reconciliation Rules (8 sequential rules)

IC_RULES = [
    {"id": 1, "name": "Narration Exact + Date Exact + Amount Offset", "dt": 0,    "fuzzy": False},
    {"id": 2, "name": "Narration Fuzzy + Date Exact + Amount Offset", "dt": 0,    "fuzzy": True},
    {"id": 3, "name": "Narration Exact + Date Range + Amount Offset", "dt": 5,    "fuzzy": False},
    {"id": 4, "name": "Narration Fuzzy + Date Range + Amount Offset", "dt": 5,    "fuzzy": True},
    {"id": 5, "name": "Date Exact + Amount Offset",                   "dt": 0,    "fuzzy": None},
    {"id": 6, "name": "Amount Offset Only",                           "dt": None, "fuzzy": None},
    {"id": 7, "name": "Within Company Reversal + Amount Offset",      "dt": None, "fuzzy": None},
    {"id": 8, "name": "Multiple Matchings (Manual Review)",           "dt": None, "fuzzy": None},
]


def run_ic_reconciliation(df: pd.DataFrame, params: dict, enabled_rules: list = None) -> tuple:
    """
    Run full Inter-Company reconciliation on a single DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: _Entity, _PartnerEntity, _Amount, _Date, _Narration,
        Recon_Status, Rule_Applied, Recon_ID
    params : dict
        Keys: amount_tolerance, date_tolerance, fuzzy_threshold
    enabled_rules : list
        Rule names to enable (all enabled by default)

    Returns
    -------
    (reconciled_df, matched_df, unmatched_df, review_df, rule_summary)
    """
    if df is None or df.empty:
        return df, None, None, None, None

    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    # Extract parameters
    amt_tol  = float(params.get("amount_tolerance", 2.0))
    date_tol = int(params.get("date_tolerance", 5))
    fuzzy_thr= int(params.get("fuzzy_threshold", 70))

    if enabled_rules is None:
        enabled_rules = [r["name"] for r in IC_RULES]

    recon_id_counter = 1

    # ─────────────────────────────────────────────────────────────
    # Rules 1-7: Sequential matching
    # ─────────────────────────────────────────────────────────────

    for rule in IC_RULES[:7]:  # Rules 1-7 (exclude Rule 8 which is special)
        if rule["name"] not in enabled_rules:
            continue

        unmatched = df[df["Recon_Status"] == "Unmatched"]

        for i, row in unmatched.iterrows():
            if df.loc[i, "Recon_Status"] != "Unmatched":
                continue

            entity   = row["_Entity"]
            partner  = row["_PartnerEntity"]
            amt      = row["_Amount"]
            date_val = row["_Date"]
            narr     = row["_Narration"]

            # Find counterparty candidates (entity A ↔ entity B swapped relationship)
            candidates = df[
                (df["Recon_Status"] == "Unmatched") &
                (df["_Entity"] == partner) &
                (df["_PartnerEntity"] == entity) &
                (df.index != i)
            ]

            # Also check within-company reversals (same entity pair, opposite direction)
            same_company = df[
                (df["Recon_Status"] == "Unmatched") &
                (df["_Entity"] == entity) &
                (df["_PartnerEntity"] == partner) &
                (df.index != i)
            ]

            candidates = pd.concat([candidates, same_company]).drop_duplicates()

            if candidates.empty:
                continue

            # Apply rule conditions
            for j, cand in candidates.iterrows():
                if df.loc[j, "Recon_Status"] != "Unmatched":
                    continue

                c_amt  = cand["_Amount"]
                c_date = cand["_Date"]
                c_narr = cand["_Narration"]

                # Amount offset check
                if not _ic_amount_offset_match(amt, c_amt, amt_tol):
                    continue

                # Skip date/narration for amount-only rule
                if rule["name"] != "Amount Offset Only":
                    if not _ic_date_match(date_val, c_date, rule["dt"]):
                        continue
                    if not _ic_narration_match(narr, c_narr, rule["fuzzy"], fuzzy_thr):
                        continue

                # Match found!
                recon_id = f"REC_{recon_id_counter}"
                df.loc[[i, j], "Recon_Status"] = "Matched"
                df.loc[[i, j], "Rule_Applied"] = rule["name"]
                df.loc[[i, j], "Recon_ID"]     = recon_id

                recon_id_counter += 1
                break  # Move to next unmatched row

    # ─────────────────────────────────────────────────────────────
    # Rule 8: Multiple matchings (suggest manual review)
    # ─────────────────────────────────────────────────────────────

    if "Multiple Matchings (Manual Review)" in enabled_rules:
        multi_match = df[df["Recon_Status"] == "Unmatched"].copy()

        if not multi_match.empty:
            multi_match["pair"] = multi_match.apply(
                lambda x: tuple(sorted([x["_Entity"], x["_PartnerEntity"]])), axis=1
            )

            for pair, group in multi_match.groupby("pair"):
                if len(group) >= 2:
                    total = group["_Amount"].sum()
                    # If sum ≈ 0 OR complex structure (3+ transactions) → send to review
                    if abs(total) <= amt_tol or len(group) >= 3:
                        df.loc[group.index, "Recon_Status"] = "Review"
                        df.loc[group.index, "Rule_Applied"] = "Multiple Matchings (Manual Review)"
                        df.loc[group.index, "Recon_ID"]     = f"REC_{recon_id_counter}"
                        recon_id_counter += 1

    # ─────────────────────────────────────────────────────────────
    # Calculate amount differences per reconciliation group
    # ─────────────────────────────────────────────────────────────

    df["Amt_Diff"] = df.groupby("Recon_ID")["_Amount"].transform("sum")

    # ─────────────────────────────────────────────────────────────
    # Extract results
    # ─────────────────────────────────────────────────────────────

    matched_df   = df[df["Recon_Status"] == "Matched"].copy()
    unmatched_df = df[df["Recon_Status"] == "Unmatched"].copy()
    review_df    = df[df["Recon_Status"] == "Review"].copy()

    # Rule summary
    if not matched_df.empty:
        rule_summary = matched_df["Rule_Applied"].value_counts().reset_index()
        rule_summary.columns = ["Rule_Applied", "Transaction_Count"]
    else:
        rule_summary = pd.DataFrame(columns=["Rule_Applied", "Transaction_Count"])

    return df, matched_df, unmatched_df, review_df, rule_summary


###----------------------------------------------------------------------------------------------------------------###
# IC Analysis Functions

def get_ic_entity_matrix(df: pd.DataFrame, status_filter="Matched") -> pd.DataFrame:
    """
    Create Entity → PartnerEntity matrix for reconciled transactions.
    Shows net amounts between entity pairs.
    """
    if df is None or df.empty:
        return None

    if status_filter:
        filtered = df[df["Recon_Status"] == status_filter].copy()
    else:
        filtered = df.copy()

    if filtered.empty:
        return None

    # Aggregate amounts by entity pair
    agg = filtered.groupby(["_Entity", "_PartnerEntity"])["_Amount"].sum().reset_index()

    # Pivot to create matrix
    matrix = agg.pivot(index="_Entity", columns="_PartnerEntity", values="_Amount").fillna(0)

    return matrix


###----------------------------------------------------------------------------------------------------------------###
# INFERENCES

# Bank Reconciliation Inferences

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


###----------------------------------------------------------------------------------------------------------------###
# INTER-COMPANY INFERENCES

def get_ic_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """Get all unmatched IC transactions."""
    if df is None:
        return None
    return df[df["Recon_Status"] == "Unmatched"].copy()


def get_ic_review_items(df: pd.DataFrame) -> pd.DataFrame:
    """Get all IC transactions flagged for manual review."""
    if df is None:
        return None
    return df[df["Recon_Status"] == "Review"].copy()

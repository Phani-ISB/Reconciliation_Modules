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
    8.  Many Ledger -> One Bank  (Group / aggregate matching)
    9.  One Ledger -> Many Bank  (Group / aggregate matching)


"""

###----------------------------------------------------------------------------------------------------------------###
import pandas as pd
import numpy as np
from itertools import combinations
from rapidfuzz import fuzz


###----------------------------------------------------------------------------------------------------------------###
# Helper Functions for reconciliation

def amount_match(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance

def date_exact(d1, d2) -> bool:
    return pd.Timestamp(d1).date() == pd.Timestamp(d2).date()

def date_range(d1, d2, tolerance_days: int) -> bool:
    return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days) <= tolerance_days

def narration_exact(n1, n2) -> bool:
    return str(n1).strip() == str(n2).strip()

def narration_fuzzy(n1, n2, threshold: int) -> bool:
    if pd.isna(n1) or pd.isna(n2):
        return False
    return fuzz.partial_ratio(str(n1), str(n2)) >= threshold


###----------------------------------------------------------------------------------------------------------------###
# Rule functions for reconciliations (Optimised for iterations across candidates)

def _build_rule_funcs(params: dict) -> dict:
    dt = int(params.get("date_tolerance", 3))
    ft = int(params.get("fuzzy_threshold", 70))
    return {
        "narration_exact": lambda l, b: narration_exact(l["_Narration"], b["_Narration"]),
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
        "date_exact": lambda l, b: date_exact(l["_Date"], b["_Date"]),
        "date_range": lambda l, b: date_range(l["_Date"], b["_Date"], dt),
    }


###----------------------------------------------------------------------------------------------------------------###
# Dictionary-based indices for faster candidate retrieval

def _build_amount_index(df: pd.DataFrame, tolerance: float) -> tuple:
    bucket_size = max(float(tolerance), 0.01)
    index = {}
    for idx, amt in pd.to_numeric(df["_Amount"], errors="coerce").items():
        if pd.isna(amt):
            continue
        bucket = int(float(amt) / bucket_size)
        index.setdefault(bucket, []).append(idx)
    return index, bucket_size


def _get_bucket_candidates(amt_index: dict, bucket_size: float, query_amt: float) -> list:
    bucket = int(query_amt / bucket_size)
    raw = []
    for b in (bucket - 1, bucket, bucket + 1):
        raw.extend(amt_index.get(b, []))
    return raw


def _build_entity_pair_index(df: pd.DataFrame) -> dict:
    index = {}
    for idx, row in df[["_Entity", "_PartnerEntity"]].iterrows():
        key = (row["_Entity"], row["_PartnerEntity"])
        index.setdefault(key, []).append(idx)
    return index


###----------------------------------------------------------------------------------------------------------------###
# 0. Internal Duplicates( Included dictionary based hashing)

def _tag_internal_duplicates(df: pd.DataFrame, group_start: int) -> tuple:
    df = df.copy()
    row_hash        = pd.util.hash_pandas_object(df, index=False)
    duplicated_mask = row_hash.duplicated(keep=False)
    dup_hashes      = row_hash[duplicated_mask]
    gid             = group_start
    group_map       = {}
    for _h, idx_series in dup_hashes.groupby(dup_hashes).groups.items():
        for i in idx_series:
            group_map[i] = gid
        gid += 1
    df["GroupID"] = df.index.map(group_map)
    df.loc[df["GroupID"].notna(), "Rule"]    = "Duplicate Data"
    df.loc[df["GroupID"].notna(), "Comment"] = "Duplicate Data"
    df.loc[df["GroupID"].notna(), "Matched"] = False
    keep_mask = ~row_hash.duplicated(keep="first")
    df        = df[keep_mask].copy()
    df.reset_index(drop=True, inplace=True)
    return df, gid


###----------------------------------------------------------------------------------------------------------------###
# Helpers for Bank reconciliation

def _amt_bucket_pairs(l_un: pd.DataFrame, b_un: pd.DataFrame, at: float) -> pd.DataFrame:
    """
    Build all amount-compatible (ledger, bank) pair candidates via +-1 bucket merge.
    Returns DataFrame with _l_idx, _b_idx, _Amount_L, _Amount_R, _Narration_L,
    _Narration_R, _Date_L, _Date_R.
    O(N*k) where k = average rows per bucket (typically 1-5 for financial data).
    """
    if l_un.empty or b_un.empty:
        return pd.DataFrame()

    bucket_size = max(float(at), 0.01)

    l_amt = pd.to_numeric(l_un["_Amount"], errors="coerce")
    l = pd.DataFrame({
        "_l_idx"      : l_un.index.astype(int),
        "_Amount_L"   : l_amt.values,
        "_Narration_L": l_un["_Narration"].astype(str).str.strip().values,
        "_Date_L"     : l_un["_Date"].values,
        "_bkt"        : np.floor(l_amt.fillna(np.nan).values / bucket_size).astype("int64"),
    }).dropna(subset=["_bkt"])

    b_amt = pd.to_numeric(b_un["_Amount"], errors="coerce")
    b = pd.DataFrame({
        "_b_idx"      : b_un.index.astype(int),
        "_Amount_R"   : b_amt.values,
        "_Narration_R": b_un["_Narration"].astype(str).str.strip().values,
        "_Date_R"     : b_un["_Date"].values,
        "_bkt"        : np.floor(b_amt.fillna(np.nan).values / bucket_size).astype("int64"),
    }).dropna(subset=["_bkt"])

    if l.empty or b.empty:
        return pd.DataFrame()

    frames = []
    for delta in (-1, 0, 1):
        b_shift = b.copy()
        b_shift["_bkt"] = b_shift["_bkt"] + delta
        frames.append(pd.merge(l, b_shift, on="_bkt"))

    pairs = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["_l_idx", "_b_idx"])
    pairs = pairs[(pairs["_Amount_L"] - pairs["_Amount_R"]).abs() <= float(at)]
    return pairs.drop(columns=["_bkt"]).reset_index(drop=True)


def _apply_bank_rule_filter(pairs: pd.DataFrame, rule_key: str, ft: int, dt: int) -> pd.DataFrame:
    """
    Apply rule-specific conditions on amount-compatible pairs DataFrame.
    All operations are vectorized (pandas Series / numpy). Fuzzy scoring runs
    via list comprehension only on the already-small amount-filtered set.
    """
    if pairs.empty:
        return pairs

    narr_L     = pairs["_Narration_L"].str.lower()
    narr_R     = pairs["_Narration_R"].str.lower()
    exact_narr = narr_L == narr_R

    date_L    = pd.to_datetime(pairs["_Date_L"], errors="coerce")
    date_R    = pd.to_datetime(pairs["_Date_R"], errors="coerce")
    date_diff = (date_L - date_R).dt.days.abs().fillna(9999).astype(int)
    exact_date = date_diff == 0
    range_date = date_diff <= dt

    if rule_key == "narration_exact":
        return pairs[exact_narr]
    if rule_key == "narration_date_exact":
        return pairs[exact_narr & exact_date]
    if rule_key == "narration_date_range":
        return pairs[exact_narr & range_date]
    if rule_key == "date_exact":
        return pairs[exact_date]
    if rule_key == "date_range":
        return pairs[range_date]

    if rule_key in ("narration_fuzzy", "narration_fuzzy_date_range"):
        scores = [fuzz.partial_ratio(n1, n2) for n1, n2 in zip(narr_L, narr_R)]
        fuzzy_match = pd.Series(
            exact_narr.values | (np.array(scores) >= ft), index=pairs.index
        )
        if rule_key == "narration_fuzzy":
            return pairs[fuzzy_match]
        return pairs[fuzzy_match & range_date]

    return pd.DataFrame()


def _commit_matched_pairs(
    pairs: pd.DataFrame, ledger_df: pd.DataFrame, bank_df: pd.DataFrame,
    rule_key: str, gid: int
) -> tuple:
    """
    Apply 1-to-1 constraint, then bulk-assign GroupID/Rule/Matched via df.loc.
    Returns (match_log_list, next_gid).
    """
    if pairs.empty:
        return [], gid

    pairs = pairs.sort_values(["_l_idx", "_b_idx"])
    pairs = pairs.drop_duplicates(subset=["_l_idx"])
    pairs = pairs.drop_duplicates(subset=["_b_idx"])

    l_indices = pairs["_l_idx"].tolist()
    b_indices = pairs["_b_idx"].tolist()

    l_matched = ledger_df.loc[l_indices, "Matched"].values
    b_matched = bank_df.loc[b_indices, "Matched"].values
    valid     = ~l_matched & ~b_matched

    if not valid.any():
        return [], gid

    valid_l    = [li for li, v in zip(l_indices, valid) if v]
    valid_b    = [bi for bi, v in zip(b_indices, valid) if v]
    valid_rows = pairs.iloc[[i for i, v in enumerate(valid) if v]]

    n         = len(valid_l)
    group_ids = list(range(gid, gid + n))
    amt_l     = valid_rows["_Amount_L"].values
    amt_r     = valid_rows["_Amount_R"].values
    diffs     = np.round(np.abs(amt_l - amt_r), 4).tolist()
    g_flt     = [float(g) for g in group_ids]

    # Ensure string columns are object dtype (they may start as float64 when
    # initialised with np.nan, which would reject string assignment in pandas 2.x)
    for df_ in (ledger_df, bank_df):
        for col in ("Rule", "Comment"):
            if col in df_.columns and df_[col].dtype != object:
                df_[col] = df_[col].astype(object)

    rule_list = [rule_key] * n        # list avoids pandas dtype-broadcast issues

    for df_, idx_list in ((ledger_df, valid_l), (bank_df, valid_b)):
        df_.loc[idx_list, "Matched"]    = True
        df_.loc[idx_list, "GroupID"]    = g_flt
        df_.loc[idx_list, "Rule"]       = rule_list
        df_.loc[idx_list, "Comment"]    = rule_list
        df_.loc[idx_list, "AmountDiff"] = diffs

    match_log = [
        {"GroupID": g, "Rule": rule_key, "MatchType": "1-to-1",
         "Ledger_Index": li, "Bank_Index": bi,
         "LedgerAmount": float(al), "BankAmount": float(ar), "AmountDiff": d}
        for g, li, bi, al, ar, d in zip(group_ids, valid_l, valid_b, amt_l, amt_r, diffs)
    ]
    return match_log, gid + n


###----------------------------------------------------------------------------------------------------------------###
# 1. One-to-One Rule based matching 

def _run_one_to_one_rules(
    ledger_df: pd.DataFrame,
    bank_df:   pd.DataFrame,
    rule_funcs: dict,         # kept for API compatibility - not used internally
    enabled_rules: list,
    params: dict,
    start_gid: int,
) -> tuple:
    """
    Vectorized 1-to-1 matching pipeline.
    rule_funcs parameter is accepted but unused (logic is now vectorized).
    Same signature and return type as the original.
    """
    at = float(params.get("amount_tolerance", 0.0))
    ft = int(params.get("fuzzy_threshold", 70))
    dt = int(params.get("date_tolerance", 3))

    match_log = []
    gid       = start_gid

    ledger_df["Matched"] = ledger_df["Matched"].fillna(False).astype(bool)
    bank_df["Matched"]   = bank_df["Matched"].fillna(False).astype(bool)

    ONE_TO_ONE_RULE_KEYS = [
        "narration_exact", "narration_fuzzy",
        "narration_date_exact", "narration_date_range",
        "narration_fuzzy_date_range", "date_exact", "date_range",
    ]

    for rule_key in ONE_TO_ONE_RULE_KEYS:
        if rule_key not in enabled_rules:
            continue

        l_un = ledger_df.loc[~ledger_df["Matched"]]
        b_un = bank_df.loc[~bank_df["Matched"]]

        if l_un.empty or b_un.empty:
            break

        # Step 1: amount-compatible pairs (vectorized merge)
        pairs = _amt_bucket_pairs(l_un, b_un, at)
        if pairs.empty:
            continue

        # Step 2: rule-specific filter (vectorized)
        filtered = _apply_bank_rule_filter(pairs, rule_key, ft, dt)

        # Step 3: commit matches (bulk df.loc assignment)
        new_log, gid = _commit_matched_pairs(filtered, ledger_df, bank_df, rule_key, gid)
        match_log.extend(new_log)

    return match_log, gid


###----------------------------------------------------------------------------------------------------------------###
# 2. Group Matching

def _subset_sum_match(df: pd.DataFrame, target: float, tol: float) -> list | None:
    """Try all combinations of 2-5 rows summing to target +/- tol."""
    for r in range(2, min(6, len(df) + 1)):
        for combo in combinations(df.index, r):
            if abs(df.loc[list(combo), "_Amount"].sum() - target) <= tol:
                return list(combo)
    return None


def _greedy_match(df: pd.DataFrame, target: float, tol: float) -> list | None:
    """Greedy: accumulate by descending abs amount until within tol of target."""
    total, selected = 0.0, []
    for idx, _ in df["_Amount"].abs().sort_values(ascending=False).items():
        val = df.at[idx, "_Amount"]
        if abs(total + val) <= abs(target) + tol:
            total += val
            selected.append(idx)
            if abs(total - target) <= tol:
                return selected
    return None


def _run_group_matching(
    ledger_df: pd.DataFrame, bank_df: pd.DataFrame,
    direction: str, params: dict, start_gid: int
) -> tuple:
    """Many-to-One / One-to-Many group matching. Unchanged from original."""
    at        = float(params.get("amount_tolerance", 0.0))
    match_log = []
    gid       = start_gid

    if direction == "many_to_one":
        source_df, target_df, rule_label = ledger_df, bank_df, "Many-to-One"
    else:
        source_df, target_df, rule_label = bank_df, ledger_df, "One-to-Many"

    for ti, trow in target_df[~target_df["Matched"]].iterrows():
        target_amount = trow["_Amount"]
        mask = (
            (~source_df["Matched"]) &
            (source_df["_Amount"] * target_amount > 0) &
            (source_df["_Amount"].abs() <= abs(target_amount) + at)
        )
        candidates = source_df[mask]
        if candidates.empty or len(candidates) > 50:
            continue

        group_indices = _greedy_match(candidates, target_amount, at)
        if group_indices is None and len(candidates) <= 10:
            group_indices = _subset_sum_match(candidates, target_amount, at)
        if group_indices is None:
            continue

        source_df.loc[group_indices, "Matched"]    = True
        source_df.loc[group_indices, "GroupID"]    = gid
        source_df.loc[group_indices, "Comment"]    = rule_label
        source_df.loc[group_indices, "Rule"]       = rule_label
        source_df.loc[group_indices, "AmountDiff"] = 0.0

        target_df.at[ti, "Matched"]    = True
        target_df.at[ti, "GroupID"]    = gid
        target_df.at[ti, "Comment"]    = rule_label
        target_df.at[ti, "Rule"]       = rule_label
        target_df.at[ti, "AmountDiff"] = round(
            abs(source_df.loc[group_indices, "_Amount"].sum() - target_amount), 4
        )

        l_indices = group_indices if direction == "many_to_one" else [ti]
        b_indices = [ti] if direction == "many_to_one" else group_indices
        match_log.append({
            "GroupID": gid, "Rule": rule_label, "MatchType": rule_label,
            "Ledger_Indices": l_indices, "Bank_Indices": b_indices,
            "TargetAmount": target_amount,
            "SourceSum": source_df.loc[group_indices, "_Amount"].sum(),
        })
        gid += 1

    return match_log, gid


###----------------------------------------------------------------------------------------------------------------###
# run_full_reconciliation

def run_full_reconciliation(
    ledger_df: pd.DataFrame, bank_df: pd.DataFrame,
    params: dict, enabled_rules: list
) -> tuple:

# Copy the ledger and bank dataframes
    ledger_df = ledger_df.copy()
    bank_df   = bank_df.copy()

    all_match_logs = []
    gid = 1

    if "duplicate_detection" in enabled_rules:
        ledger_df, gid = _tag_internal_duplicates(ledger_df, gid)
        bank_df,   gid = _tag_internal_duplicates(bank_df,   gid)

    rule_funcs = _build_rule_funcs(params)
    one_to_one_log, gid = _run_one_to_one_rules(
        ledger_df, bank_df, rule_funcs, enabled_rules, params, gid
    )
    all_match_logs.extend(one_to_one_log)

    if "many_to_one" in enabled_rules:
        group_log, gid = _run_group_matching(ledger_df, bank_df, "many_to_one", params, gid)
        all_match_logs.extend(group_log)

    if "one_to_many" in enabled_rules:
        group_log, gid = _run_group_matching(ledger_df, bank_df, "one_to_many", params, gid)
        all_match_logs.extend(group_log)

    match_log_df = pd.DataFrame(all_match_logs) if all_match_logs else pd.DataFrame(
        columns=["GroupID", "Rule", "MatchType", "Ledger_Index", "Bank_Index", "AmountDiff"]
    )

    rule_counts = (
        pd.concat([
            ledger_df[ledger_df["Matched"] & ledger_df["Rule"].notna()]["Rule"],
            bank_df[bank_df["Matched"] & bank_df["Rule"].notna()]["Rule"],
        ])
        .value_counts().reset_index()
    )
    rule_counts.columns = ["Rule", "Transactions_Reconciled"]

    return ledger_df, bank_df, match_log_df, rule_counts


###################################################################################################################
# INTER-COMPANY RECONCILIATION ENGINE

###----------------------------------------------------------------------------------------------------------------###
# IC Helper Functions (unchanged)

def _ic_amount_offset_match(amt1: float, amt2: float, tol: float) -> bool:
    return abs(amt1 + amt2) <= tol

def _ic_date_match(d1, d2, days=None) -> bool:
    if days is None:
        return True
    return abs((pd.Timestamp(d1) - pd.Timestamp(d2)).days) <= days

def _ic_narration_match(n1, n2, fuzzy=False, threshold=70) -> bool:
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
# IC Reconciliation Rules

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


###----------------------------------------------------------------------------------------------------------------###
# IC reconciliation helper functions

def _ic_offset_bucket_pairs(un_df: pd.DataFrame, amt_tol: float) -> pd.DataFrame:
    """
    Build all IC counterparty + same-company pairs within amount-offset tolerance.

    Uses entity-pair + +-1 amount-offset bucket merge.
    Left bucket = floor(X / bucket_size).
    Right bucket = floor(-X / bucket_size)  ->  same value for exact offset.
    Merge on (L.Entity, L.Partner, L.bkt) = (R.Partner, R.Entity, R.bkt_neg)
    for counterparty, and on (R.Entity, R.Partner) for same-company reversals.

    Returns DataFrame with _L_idx, _R_idx, _L_Amt, _R_Amt, _L_Narr, _R_Narr,
    _L_Date, _R_Date. Only pairs where _L_idx < _R_idx (symmetric dedup).
    """
    if len(un_df) < 2:
        return pd.DataFrame()

    bucket_size = max(float(amt_tol), 0.01)
    amt_num     = pd.to_numeric(un_df["_Amount"], errors="coerce")

    left = pd.DataFrame({
        "_L_idx"    : un_df.index.astype(int),
        "_L_Entity" : un_df["_Entity"].values,
        "_L_Partner": un_df["_PartnerEntity"].values,
        "_L_Amt"    : amt_num.values,
        "_L_Narr"   : un_df["_Narration"].astype(str).str.strip().values,
        "_L_Date"   : un_df["_Date"].values,
        "_L_bkt"    : np.floor(amt_num.values / bucket_size).astype("int64"),
    }).dropna(subset=["_L_bkt"])

    right = pd.DataFrame({
        "_R_idx"      : un_df.index.astype(int),
        "_R_Entity"   : un_df["_Entity"].values,
        "_R_Partner"  : un_df["_PartnerEntity"].values,
        "_R_Amt"      : amt_num.values,
        "_R_Narr"     : un_df["_Narration"].astype(str).str.strip().values,
        "_R_Date"     : un_df["_Date"].values,
        "_R_bkt_neg"  : np.floor(-amt_num.values / bucket_size).astype("int64"),
    }).dropna(subset=["_R_bkt_neg"])

    frames = []
    for delta in (-1, 0, 1):
        r_shifted = right.copy()
        r_shifted["_R_bkt_neg"] = r_shifted["_R_bkt_neg"] + delta

        # Counterparty: L.Entity==R.Partner AND L.Partner==R.Entity
        cp = pd.merge(
            left,  r_shifted,
            left_on=["_L_Entity",  "_L_Partner",  "_L_bkt"],
            right_on=["_R_Partner", "_R_Entity",   "_R_bkt_neg"],
        )
        frames.append(cp)

        # Same-company: L.Entity==R.Entity AND L.Partner==R.Partner
        sc = pd.merge(
            left,  r_shifted,
            left_on=["_L_Entity",  "_L_Partner",  "_L_bkt"],
            right_on=["_R_Entity",  "_R_Partner",  "_R_bkt_neg"],
        )
        frames.append(sc)

    all_pairs = pd.concat(frames, ignore_index=True)
    all_pairs["_L_idx"] = all_pairs["_L_idx"].astype(int)
    all_pairs["_R_idx"] = all_pairs["_R_idx"].astype(int)
    all_pairs = all_pairs[all_pairs["_L_idx"] != all_pairs["_R_idx"]]
    all_pairs = all_pairs[all_pairs["_L_idx"]  < all_pairs["_R_idx"]]   # symmetric dedup
    all_pairs = all_pairs.drop_duplicates(subset=["_L_idx", "_R_idx"])
    all_pairs = all_pairs[(all_pairs["_L_Amt"] + all_pairs["_R_Amt"]).abs() <= float(amt_tol)]

    keep = ["_L_idx", "_R_idx", "_L_Amt", "_R_Amt", "_L_Narr", "_R_Narr", "_L_Date", "_R_Date"]
    return all_pairs[keep].reset_index(drop=True)


def _apply_ic_rule_filter(pairs: pd.DataFrame, rule: dict, fuzzy_thr: int) -> pd.DataFrame:
    """
    Apply IC rule narration and date conditions on amount-offset pairs.
    Fully vectorized; fuzzy scoring only on the small amount-filtered set.
    """
    if pairs.empty:
        return pairs

    fuzzy_setting = rule.get("fuzzy")
    if fuzzy_setting is not None:
        narr_L = pairs["_L_Narr"].str.lower()
        narr_R = pairs["_R_Narr"].str.lower()
        exact  = narr_L == narr_R

        if fuzzy_setting is False:
            pairs = pairs[exact]
        else:
            scores = [fuzz.partial_ratio(n1, n2) for n1, n2 in zip(narr_L, narr_R)]
            fuzzy_match = pd.Series(
                exact.values | (np.array(scores) >= fuzzy_thr), index=pairs.index
            )
            pairs = pairs[fuzzy_match]

        if pairs.empty:
            return pairs

    dt_setting = rule.get("dt")
    if dt_setting is not None:
        date_L    = pd.to_datetime(pairs["_L_Date"], errors="coerce")
        date_R    = pd.to_datetime(pairs["_R_Date"], errors="coerce")
        date_diff = (date_L - date_R).dt.days.abs().fillna(9999)
        pairs     = pairs[date_diff <= dt_setting]

    return pairs


def _commit_ic_matches(
    pairs: pd.DataFrame, df: pd.DataFrame, rule_name: str, recon_id_counter: int
) -> int:
    """
    Apply 1-to-1 constraint, then bulk-assign Recon_Status/Rule_Applied/Recon_ID.
    Both rows of each pair receive the same Recon_ID.
    Returns next recon_id_counter.
    """
    if pairs.empty:
        return recon_id_counter
    
    pairs = pairs.copy()
    pairs["_L_idx"] = pairs["_L_idx"].astype(int)
    pairs["_R_idx"] = pairs["_R_idx"].astype(int)
    pairs     = pairs.sort_values(["_L_idx", "_R_idx"])
    pairs     = pairs.drop_duplicates(subset=["_L_idx"])
    pairs     = pairs.drop_duplicates(subset=["_R_idx"])
    valid_l   = pairs["_L_idx"].tolist()
    valid_r   = pairs["_R_idx"].tolist()
    n         = len(valid_l)

    if not n:
        return recon_id_counter

    recon_ids = [f"REC_{recon_id_counter + i}" for i in range(n)]

    # Interleaved index/value lists: [L0, R0, L1, R1, ...]
    all_idx = [idx for pair in zip(valid_l, valid_r) for idx in pair]
    all_ids = [rid for rid in recon_ids for _ in range(2)]

    for col in ("Recon_Status", "Rule_Applied", "Recon_ID"):
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)
    n2 = len(all_idx)
    df.loc[all_idx, "Recon_Status"] = ["Matched"]  * n2
    df.loc[all_idx, "Rule_Applied"] = [rule_name] * n2
    df.loc[all_idx, "Recon_ID"]     = all_ids

    return recon_id_counter + n


###----------------------------------------------------------------------------------------------------------------###
# run_ic_reconciliation

def run_ic_reconciliation(df: pd.DataFrame, params: dict, enabled_rules: list = None) -> tuple:
    """
    Run full Inter-Company reconciliation on a single DataFrame.

    Parameters
    ----------
    df            : Must contain _Entity, _PartnerEntity, _Amount, _Date, _Narration,
                    Recon_Status, Rule_Applied, Recon_ID
    params        : Keys: amount_tolerance, date_tolerance, fuzzy_threshold
    enabled_rules : Rule names to enable (all enabled by default)

    Returns
    -------
    (reconciled_df, matched_df, unmatched_df, review_df, rule_summary)
    """
    if df is None or df.empty:
        return df, None, None, None, None

    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    amt_tol   = float(params.get("amount_tolerance", 2.0))
    date_tol  = int(params.get("date_tolerance", 5))
    fuzzy_thr = int(params.get("fuzzy_threshold", 70))

    if enabled_rules is None:
        enabled_rules = [r["name"] for r in IC_RULES]

    recon_id_counter = 1

    # Rules 1-7: vectorized sequential matching
    for rule in IC_RULES[:7]:
        if rule["name"] not in enabled_rules:
            continue

        unmatched = df[df["Recon_Status"] == "Unmatched"]
        if len(unmatched) < 2:
            break

        # Step 1: entity-pair + amount-offset bucket pairs
        pairs = _ic_offset_bucket_pairs(unmatched, amt_tol)
        if pairs.empty:
            continue

        # Step 2: rule narration/date filter
        filtered = _apply_ic_rule_filter(pairs, rule, fuzzy_thr)

        # Step 3: bulk commit
        recon_id_counter = _commit_ic_matches(filtered, df, rule["name"], recon_id_counter)

    # Rule 8: Multiple matchings (manual review)
    if "Multiple Matchings (Manual Review)" in enabled_rules:
        multi_match = df[df["Recon_Status"] == "Unmatched"].copy()
        if not multi_match.empty:
            multi_match["pair"] = multi_match.apply(
                lambda x: tuple(sorted([x["_Entity"], x["_PartnerEntity"]])), axis=1
            )
            for pair, group in multi_match.groupby("pair"):
                if len(group) >= 2:
                    total = group["_Amount"].sum()
                    if abs(total) <= amt_tol or len(group) >= 3:
                        df.loc[group.index, "Recon_Status"] = "Review"
                        df.loc[group.index, "Rule_Applied"] = "Multiple Matchings (Manual Review)"
                        df.loc[group.index, "Recon_ID"]     = f"REC_{recon_id_counter}"
                        recon_id_counter += 1

    df["Amt_Diff"] = df.groupby("Recon_ID")["_Amount"].transform("sum")

    matched_df   = df[df["Recon_Status"] == "Matched"].copy()
    unmatched_df = df[df["Recon_Status"] == "Unmatched"].copy()
    review_df    = df[df["Recon_Status"] == "Review"].copy()

    if not matched_df.empty:
        rule_summary = matched_df["Rule_Applied"].value_counts().reset_index()
        rule_summary.columns = ["Rule_Applied", "Transaction_Count"]
    else:
        rule_summary = pd.DataFrame(columns=["Rule_Applied", "Transaction_Count"])

    return df, matched_df, unmatched_df, review_df, rule_summary


###----------------------------------------------------------------------------------------------------------------###
# IC Analysis Functions (Entity matrix and inference helpers)

def get_ic_entity_matrix(df: pd.DataFrame, status_filter="Matched") -> pd.DataFrame:
    """Create Entity -> PartnerEntity matrix for reconciled transactions."""
    if df is None or df.empty:
        return None
    filtered = df[df["Recon_Status"] == status_filter].copy() if status_filter else df.copy()
    if filtered.empty:
        return None
    agg    = filtered.groupby(["_Entity", "_PartnerEntity"])["_Amount"].sum().reset_index()
    matrix = agg.pivot(index="_Entity", columns="_PartnerEntity", values="_Amount").fillna(0)
    return matrix


###----------------------------------------------------------------------------------------------------------------###
# INFERENCES

def get_unreconciled(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> tuple:
    return ledger_df[~ledger_df["Matched"]].copy(), bank_df[~bank_df["Matched"]].copy()


def get_manual_review_items(ledger_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:
    group_rules   = {"Many-to-One", "One-to-Many"}
    ledger_review = ledger_df[ledger_df["Rule"].isin(group_rules)].copy()
    bank_review   = bank_df[bank_df["Rule"].isin(group_rules)].copy()
    ledger_review.insert(0, "Source", "Ledger")
    bank_review.insert(0, "Source", "Bank")
    return pd.concat([ledger_review, bank_review], ignore_index=True)


###----------------------------------------------------------------------------------------------------------------###
# INTER-COMPANY INFERENCES

def get_ic_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    return df[df["Recon_Status"] == "Unmatched"].copy()


def get_ic_review_items(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    return df[df["Recon_Status"] == "Review"].copy()

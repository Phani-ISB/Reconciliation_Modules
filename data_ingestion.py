
"""

### Data Ingestion & Preparation ###

# Purpose : Handles loading raw data and preparing them for reconciliation engine

# Key Respnsibilities :
    1. Load CSV / XLS / XLSX files into pandas DataFrames
    2. Standardise column names (strip whitespace, consistent casing)
    3. Apply user-defined column mappings  (date, amount, narration)
    4. Parse and validate date columns
    5. Build a unified 'Narration' field by concatenating selected columns
    6. Build a unified 'Amount' field (positive = credit, negative = debit)
    7. Return clean DataFrames ready for the reconciliation engine

"""

###----------------------------------------------------------------------------------------------------------------###
# Import all required libraries

import numpy as np
import pandas as pd
import io
import base64

###----------------------------------------------------------------------------------------------------------------###
# Detect file type and load the file

def load_file(contents: str, filename : str):
    if contents is None :
        return None, "No File provided"
    try :
        content_type, content_text = contents.split(",",1)
        decoded = base64.b64decode(content_text)
        file_buffer = io.BytesIO(decoded)
        file_name = filename.lower()

        if file_name.endswith(".csv") :
            df = pd.read_csv(file_buffer, encoding ="utf-8")
        elif file_name.endswith(".xlsx") :
            df = pd.read_excel(file_buffer, engine = "openpyxl")
        elif file_name.endswith(".xls") :
            df = pd.read_excel(file_buffer, engine = "xlrd")
        else :
            return None, f"Unsupported File Type : '{filename}' . Please upload csv, xls or xlsx file."

        # Basic Checks :
        if df.empty :
            return None, f"File '{filename}' is empty. No rows Found"
        if df.shape[1] < 2 :
            return None, f"File '{filename}' has fewer than 2 columns- Check the File/Data"

        #Standardise the Columns
        df.columns = df.columns.astype(str).str.strip()

        # Drop completely empty rows and columns
        df.dropna(how ="all", inplace = True)
        df.dropna(axis=1, how ="all", inplace = True)

        return df, None

    except Exception as exc :
        return None, f"Error reading '{filename}' : {str(exc)}"

###----------------------------------------------------------------------------------------------------------------###
# Build List of columns (Column Headers data from dataframe)

def get_column_options(df : pd.DataFrame) -> list :
    if df is None :
        return []
    return [{"label" : col, "value" :col} for col in df.columns]

###----------------------------------------------------------------------------------------------------------------###
# Validate mappings of columns

def validate_mapping(mapping : dict) -> tuple :
    date_col = mapping.get("date_col")
    amount_col = mapping.get("amount_col")
    narration_col = mapping.get("narration_cols", [])

    if not date_col :
        return False, "Date Column in mandatory - Please Select it"
    if not amount_col :
        return False, "Amount Column in mandatory - Please Select it"
    if not narration_col :
        return False, "Atlease one Narration Column in mandatory - Please Select it"

    return True, None

###----------------------------------------------------------------------------------------------------------------###
# Preprocessing and Clean the raw data using above validated and mapped columns

def preprocess(df : pd.DataFrame, mapping : dict, source_label: str = "Data") -> tuple :
    # Input guard
    if df is None or df.empty :
        return None, f"{source_label} : No data to process"
    ok, err = validate_mapping(mapping)
    if not ok :
        return None, f"{source_label} : {err} "

    date_col = mapping.get("date_col")
    amount_col = mapping.get("amount_col")
    narration_cols = mapping.get("narration_cols") or []
    # Ensure narration columns as list
    if not isinstance(narration_cols, list):
        narration_cols = [narration_cols]
    narration_cols = [str(c) for c in narration_cols if pd.notna(c)]    

    # Check missing columns any in DataFrame
    missing = [c for c in [date_col, amount_col] + narration_cols if c not in df.columns]
    if missing :
        return None, f"{source_label} : Missing Columns : {', '.join(map(str,missing))}"

    # Copy the DataFrame to carryout preprocessing steps
    df = df.copy()

    # 1. Parse the Date column
    try:
      df["_Date"] = pd.to_datetime(df[date_col],dayfirst = True, errors = "coerce")
    except Exception as exc :
      return None, f"{source_label} : Error parsing Date column '{date_col}': {str(exc)}"

    # 2. Parse the Amount column
    try:
      df["_Amount"] = pd.to_numeric(df[amount_col], errors ="coerce").fillna(0.0)

    except Exception as exc :
      return None, f"{source_label} : Error parsing Amount column '{amount_col}': {str(exc)}"

    # 3. Unified Narration Field
    try :
        df["_Narration"] = (
            df[narration_cols]
            .fillna("")
            .apply(lambda row: " ".join(
                str(v).strip()
                for v in row
                if pd.notna(v) and str(v).strip().lower() not in ("", "nan", "none")
            ), axis=1)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )     
    except Exception as exc :
      return None, f"{source_label} : Error building Narration Field : {str(exc)}"

    # 4. Initiate columns for reconciliation (Initialising for reconciliation)
    df["Matched"] = False         # Will be set to True when a match is Found
    df["GroupID"] = None          # Shared ID assigned to matched pair/group
    df["Comment"] = None          # Rule name that caused the match
    df["Rule"] = None             # Same as comment
    df["AmountDiff"] = None       # Filled after matching : |ledger_amount - bank_amount|

    # Return the clean dataframe
    return df, None

###----------------------------------------------------------------------------------------------------------------###
# INTER-COMPANY RECONCILIATION - Preprocessing

def validate_ic_mapping(mapping : dict) -> tuple :
    """Validate mapping for Inter-Company reconciliation (requires Entity & PartnerEntity fields)."""
    date_col = mapping.get("date_col")
    amount_col = mapping.get("amount_col")
    entity_col = mapping.get("entity_col")
    partner_col = mapping.get("partner_entity_col")
    narration_cols = mapping.get("narration_cols", [])

    if not date_col :
        return False, "Date Column is mandatory - Please Select it"
    if not amount_col :
        return False, "Amount Column is mandatory - Please Select it"
    if not entity_col :
        return False, "Entity Column is mandatory - Please Select it"
    if not partner_col :
        return False, "Partner Entity Column is mandatory - Please Select it"
    if not narration_cols :
        return False, "At least one Narration Column is mandatory - Please Select it"

    return True, None


def preprocess_ic(df : pd.DataFrame, mapping : dict, source_label: str = "Data") -> tuple :
    """Preprocess Inter-Company reconciliation data with Entity and PartnerEntity fields."""
    # Input guard
    if df is None or df.empty :
        return None, f"{source_label} : No data to process"

    ok, err = validate_ic_mapping(mapping)
    if not ok :
        return None, f"{source_label} : {err}"

    date_col = mapping.get("date_col")
    amount_col = mapping.get("amount_col")
    entity_col = mapping.get("entity_col")
    partner_col = mapping.get("partner_entity_col")
    narration_cols = mapping.get("narration_cols", [])

    # Check missing columns in DataFrame
    required_cols = [date_col, amount_col, entity_col, partner_col] + narration_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing :
        return None, f"{source_label} : Missing Columns : {', '.join(missing)}"

    # Copy the DataFrame
    df = df.copy()

    # 1. Parse the Date column
    try:
        df["_Date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        if df["_Date"].isna().all():
            return None, f"{source_label} : No valid dates found in '{date_col}'"
    except Exception as exc :
        return None, f"{source_label} : Error parsing Date column '{date_col}': {str(exc)}"

    # 2. Parse the Amount column
    try:
        df["_Amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    except Exception as exc :
        return None, f"{source_label} : Error parsing Amount column '{amount_col}': {str(exc)}"

    # 3. Entity and PartnerEntity columns
    try:
        df["_Entity"] = df[entity_col].astype(str).str.strip()
        df["_PartnerEntity"] = df[partner_col].astype(str).str.strip()
    except Exception as exc :
        return None, f"{source_label} : Error processing Entity columns: {str(exc)}"

    # 4. Unified Narration Field (lowercase for fuzzy matching)
    try :
        narration = [df[c].astype(str).str.strip() for c in narration_cols]
        df["_Narration"] = (
            pd.concat(narration, axis=1)
            .apply(lambda row : " ".join(v for v in row if v not in ("nan", "", "None")), axis=1)
            .str.lower()
            .str.strip()
        )
    except Exception as exc :
        return None, f"{source_label} : Error building Narration Field : {str(exc)}"

    # 5. Initialize IC reconciliation columns
    df["Recon_Status"] = "Unmatched"       # Unmatched, Matched, Review
    df["Rule_Applied"] = None              # Which rule matched this row
    df["Recon_ID"] = None                  # Shared ID for matched pairs/groups
    df["Amt_Diff"] = None                  # Sum of amounts in the reconciliation group

    # Return the clean dataframe
    return df, None

###----------------------------------------------------------------------------------------------------------------###
# Utility - serialise/ deserialise dataframe via JSON
# Convertion of dataframe to JSON serialisable dict

def df_to_store(df : pd.DataFrame) -> dict | None :
    if df is None or df.empty :
        return None
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime64[ns]","datetimetz"]).columns :
      df_copy[col] = df_copy[col].astype(str)

    return df_copy.to_dict("records")

# Reconstruction of datafram from JSON stored data

def store_to_df(records : list | None) -> pd.DataFrame | None :
    if not records :
        return None
    df = pd.DataFrame(records)
    if "_Date" in df.columns :
      df["_Date"] = pd.to_datetime(df["_Date"], errors = "coerce")
    return df


###----------------------------------------------------------------------------------------------------------------###
# Utility - Specialized for IC (Inter-Company) Reconciliation

def df_to_store_ic(df : pd.DataFrame) -> dict | None :
    """Convert IC dataframe to JSON-serializable format, preserving IC-specific fields."""
    if df is None or df.empty :
        return None
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime64[ns]","datetimetz"]).columns :
        df_copy[col] = df_copy[col].astype(str)
    return df_copy.to_dict("records")


def store_to_df_ic(records : list | None) -> pd.DataFrame | None :
    """Reconstruct IC dataframe from JSON stored data."""
    if not records :
        return None
    df = pd.DataFrame(records)
    if "_Date" in df.columns :
        df["_Date"] = pd.to_datetime(df["_Date"], errors = "coerce")
    return df

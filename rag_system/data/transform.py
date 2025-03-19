import pandas as pd
import re
import json
import numpy as np
from datetime import datetime
import spacy

def normalize_value(val):
    """
    Normalize a string value:
    - Remove extra spaces, commas, and dollar signs.
    - If the value contains parentheses (e.g., "-2913 ( 2913 )"), take the part before the parenthesis.
    - Remove percentage symbols and convert to float when possible.
    """
    if not isinstance(val, str):
        return val
    val = val.strip()
    if not val:
        return None
    # Remove commas and dollar signs
    val = val.replace(",", "")
    if val.startswith("$"):
        val = val.replace("$", "").strip()
    # Handle parentheses for negative numbers like "-2913 ( 2913 )"
    if "(" in val and ")" in val:
        # Extract the pattern: number followed by parentheses with the same number
        pattern = r"(-?\d+(?:\.\d+)?)\s*\(\s*\d+(?:\.\d+)?\s*\)"
        match = re.search(pattern, val)
        if match:
            val = match.group(1).strip()
        else:
            # Just take the part before the parenthesis if pattern doesn't match exactly
            val = val.split("(")[0].strip()
    # Handle percentages
    if "%" in val:
        val = val.replace("%", "").strip()
        try:
            return float(val) / 100  # Convert percentage to decimal
        except ValueError:
            return val
    try:
        return float(val)
    except ValueError:
        return val

def normalize_date_column(header):
    """
    Normalize date columns to a standard format (YYYY-MM-DD if possible).
    - Remove unnecessary words like "year ended", "unaudited", etc.
    - Extract year, month, day information
    - Handle ranges like "from 2008 to 2009" by using the end date
    """
    if not isinstance(header, str):
        return header
    
    header = header.lower().strip()
    
    # Skip empty or non-date headers
    if not header or header == "date":
        return header
    
    # Common patterns to remove
    patterns_to_remove = [
        r'\s*\(\s*unaudited\s*\)\s*',
        r'\s*year\s+ended\s+',
        r'\s*quarter\s+ended\s+',
        r'\s*period\s+ended\s+',
        r'\s*fiscal\s+year\s+',
        r'\s*fiscal\s+'
    ]
    
    for pattern in patterns_to_remove:
        header = re.sub(pattern, ' ', header)
    
    # Handle date ranges like "from 2008 to 2009" - extract the end date
    range_pattern = r'(?:from|between)\s+\d{4}\s+(?:to|and|through|[-])\s+(\d{4})'
    range_match = re.search(range_pattern, header)
    if range_match:
        # Use the end date from the range
        return range_match.group(1)
    
    # Extract date components
    date_pattern = r'(\d{4})[-/\s]+(\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december)[-/\s]*(\d{1,2})?'
    match = re.search(date_pattern, header)
    
    if match:
        year = match.group(1)
        month_str = match.group(2)
        day = match.group(3) or "1"  # Default to 1st if day not specified
        
        # Convert month name to number
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        if month_str.isalpha():
            month = month_map.get(month_str.lower(), '01')
        else:
            month = month_str.zfill(2)
        
        day = day.zfill(2) if day.isdigit() else "01"
        
        try:
            # Validate date
            datetime(int(year), int(month), int(day))
            return f"{year}-{month}-{day}"
        except (ValueError, TypeError):
            # If invalid date, return just the year
            return year
    
    # If just a year is present
    year_pattern = r'(\d{4})'
    year_match = re.search(year_pattern, header)
    if year_match:
        return year_match.group(1)
    
    # If no date pattern found, return cleaned header
    return header.strip()

def identify_metric_column(column_name, nlp=None):
    """
    Identify and standardize metric column names using spaCy for semantic understanding.
    This helps in merging similar metrics across different tables.
    """
    if not isinstance(column_name, str) or not column_name.strip():
        return "unnamed_metric"
    
    column_name = column_name.lower().strip()
    
    # Common financial metrics mapping (expand as needed)
    metric_map = {
        'net income': 'net_income',
        'income': 'net_income',
        'revenue': 'revenue',
        'sales': 'revenue',
        'net sales': 'revenue',
        'gross margin': 'gross_margin',
        'net cash': 'net_cash',
        'cash flow': 'cash_flow',
        'operating activities': 'operating_activities',
        'eps': 'earnings_per_share',
        'earnings per share': 'earnings_per_share',
        'basic earnings per share': 'basic_earnings_per_share',
        'diluted earnings per share': 'diluted_earnings_per_share',
        'cost of sales': 'cost_of_sales',
        'expenses': 'expenses',
        'non-cash expenses': 'non_cash_expenses',
        'change in receivables': 'change_in_receivables',
        'receivables': 'receivables',
        'deferred revenue': 'deferred_revenue',
        'change in deferred revenue': 'change_in_deferred_revenue',
        'assets': 'assets',
        'liabilities': 'liabilities',
        'change in other assets and liabilities': 'change_in_other_assets_liabilities',
        'net cash from operating activities': 'net_cash_from_operating_activities'
    }
    
    # Direct mapping if exact match found
    for key, value in metric_map.items():
        if key in column_name:
            return value
    
    # If no direct match and spaCy is available, use semantic analysis
    if nlp:
        doc = nlp(column_name)
        for key, value in metric_map.items():
            key_doc = nlp(key)
            # If similarity is high, consider it a match
            if doc.similarity(key_doc) > 0.8:
                return value
    
    # If no match found, clean up the column name for use as identifier
    clean_name = re.sub(r'[^a-z0-9]', '_', column_name)
    clean_name = re.sub(r'_+', '_', clean_name)
    return clean_name.strip('_')

def extract_organization_from_id(doc_id):
    """
    Extract organization name from document ID.
    
    Args:
        doc_id (str): Document ID in format like "Single_JKHY/2009/page_28.pdf-3"
    
    Returns:
        str: Organization name (e.g., "JKHY")
    """
    if not isinstance(doc_id, str):
        return None
    
    # Handle patterns like "Single_JKHY/2009/page_28.pdf-3"
    match = re.search(r'(?:Single|Double)_([A-Za-z0-9]+)/', doc_id)
    if match:
        return match.group(1)
    
    # Handle patterns without Single/Double prefix like "JKHY/2009/..."
    match = re.search(r'^([A-Za-z0-9]+)/', doc_id)
    if match:
        return match.group(1)
    
    # Handle other formats as needed
    return None

def standardize_organization_name(org_name):
    """
    Standardize organization names by removing prefixes like "Single_" or "Double_"
    
    Args:
        org_name (str): Organization name to standardize
    
    Returns:
        str: Standardized organization name
    """
    if not isinstance(org_name, str):
        return None
    
    # Remove prefixes like "Single_" or "Double_"
    org_name = re.sub(r'^(?:Single_|Double_)', '', org_name)
    
    return org_name

def process_and_transpose_table(table_data, doc_id=None, nlp=None):
    """
    Process a table from the JSON data with improved handling.
    
    The table_data is a list of rows. The first row is the header.
    - The first header cell (which is empty) is replaced with "date".
    - The remaining cells in the first row are date values.
    - Each subsequent row has a metric name in its first cell and the metric values for each date.
    
    This function creates a DataFrame, then transposes it so that each row corresponds
    to a date and each column to a metric.
    
    Args:
        table_data: List of rows of data
        doc_id: The document ID to extract organization from
        nlp: Optional spaCy model for semantic matching
    """
    if not table_data or len(table_data) < 2:
        return None
    
    header = table_data[0].copy()
    
    # Check if first header is empty or contains a date-like value
    if not header[0] or header[0].lower() == "date" or re.search(r'\d{4}', header[0]):
        header[0] = "date"
    
    # Normalize date headers
    for i in range(1, len(header)):
        header[i] = normalize_date_column(header[i])
    
    # Build DataFrame: rows are metrics; columns are: "date", date1, date2, etc.
    df = pd.DataFrame(table_data[1:], columns=header)
    
    # Standardize metric names in the first column
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: identify_metric_column(x, nlp))
    
    # Set the first column (metric names) as index
    df.set_index(header[0], inplace=True)
    
    # Transpose: now rows are dates, and columns are metrics
    df_transposed = df.transpose().reset_index()
    df_transposed.rename(columns={"index": "date"}, inplace=True)
    
    # Normalize each metric column (skip the "date" column)
    for col in df_transposed.columns:
        if col != "date":
            df_transposed[col] = df_transposed[col].apply(normalize_value)
    
    # Clean up the date column
    df_transposed["date"] = df_transposed["date"].apply(lambda x: normalize_date_column(x) if isinstance(x, str) else x)
    
    # Add organization column if we have a document ID
    if doc_id:
        org_name = standardize_organization_name(extract_organization_from_id(doc_id))
        if org_name:
            df_transposed["organization"] = org_name
    
    return df_transposed

def ensure_str_columns(df):
    """
    Convert any non-string column names (like tuples) into strings
    """
    if df is None:
        return None
    # Convert any non-string column names (like tuples) into strings
    df.columns = [col if isinstance(col, str) else '_'.join(map(str, col)) for col in df.columns]
    return df

def merge_tables(dfs):
    """
    Merge a list of DataFrames (each with a "date" column) into one unified DataFrame.
    
    This function iteratively merges DataFrames on the "date" column using an outer join.
    If overlapping metric columns occur, it combines them by taking non-null values.
    """
    if not dfs or len(dfs) == 0:
        return None
    
    # Check if any dataframe has the organization column
    has_org = any("organization" in df.columns for df in dfs if df is not None)
    
    # Start with the first DataFrame
    merged_df = ensure_str_columns(dfs[0].copy())
    
    # Reset index to avoid duplicate index issues
    merged_df = merged_df.reset_index(drop=True)
    
    # Store original organization values if present
    org_values = {}
    if "organization" in merged_df.columns:
        for idx, row in merged_df.iterrows():
            if not pd.isna(row["organization"]):
                if row["date"] not in org_values:
                    org_values[row["date"]] = row["organization"]
    
    # Merge with each subsequent DataFrame
    for df in dfs[1:]:
        df = ensure_str_columns(df.copy())
        
        # Reset index to avoid duplicate index issues
        df = df.reset_index(drop=True)
        
        # Store organization values from this dataframe
        if "organization" in df.columns:
            for idx, row in df.iterrows():
                if not pd.isna(row["organization"]):
                    if row["date"] not in org_values:
                        org_values[row["date"]] = row["organization"]
        
        # Temporarily remove organization column from both dataframes to avoid conflicts
        merged_org_col = None
        df_org_col = None
        
        if "organization" in merged_df.columns:
            merged_org_col = merged_df["organization"].copy()
            merged_df = merged_df.drop(columns=["organization"])
            
        if "organization" in df.columns:
            df_org_col = df["organization"].copy()
            df = df.drop(columns=["organization"])
        
        # Check for duplicate column names within each dataframe before merging
        if len(df.columns) != len(df.columns.unique()):
            # Generate unique names for duplicate columns
            new_cols = []
            seen = {}
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols
        
        # Create a list of columns to be renamed (conflicts with merged_df)
        rename_dict = {}
        for col in df.columns:
            if col in merged_df.columns and col != "date":
                rename_dict[col] = f"{col}_new"
        
        # Rename conflicting columns
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Perform the merge
        merged_df = pd.merge(merged_df, df, on="date", how="outer")
        
        # After merge, reset index again to ensure we have clean indices
        merged_df = merged_df.reset_index(drop=True)
        
        # Handle _new columns one by one without using boolean masks across DataFrames
        new_cols = [col for col in merged_df.columns if col.endswith("_new")]
        
        for new_col in new_cols:
            base_col = new_col[:-4]  # Remove "_new" suffix
            
            if base_col in merged_df.columns:
                # Combine the columns manually without complex boolean operations
                for idx in merged_df.index:
                    new_val = merged_df.at[idx, new_col]
                    base_val = merged_df.at[idx, base_col]
                    
                    if pd.isna(base_val) and not pd.isna(new_val):
                        merged_df.at[idx, base_col] = new_val
                
                # Drop the new column safely
                merged_df = merged_df.drop(columns=[new_col], errors='ignore')
            else:
                # If base column doesn't exist, rename the new column to base name
                merged_df = merged_df.rename(columns={new_col: base_col})
    
    # Final reset of index to ensure clean dataframe
    merged_df = merged_df.reset_index(drop=True)
    
    # Add the organization column back with the consolidated values
    if has_org:
        merged_df["organization"] = None
        for idx, row in merged_df.iterrows():
            if row["date"] in org_values:
                merged_df.at[idx, "organization"] = org_values[row["date"]]
    
    return merged_df

def deduplicate_df(df):
    """
    Remove duplicate rows based on the "date" column and clean up NaN values.
    """
    if df is None:
        return None
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["date"])
    
    # Sort by date if possible
    try:
        # Convert date to string format first to avoid JSON serialization issues later
        # Fixed: Use proper loc indexing to avoid SettingWithCopyWarning
        df.loc[:, 'date_temp'] = df['date'].apply(lambda x: str(x) if isinstance(x, (pd.Timestamp, datetime)) else x)
        df = df.sort_values('date_temp')
        df.loc[:, 'date'] = df['date_temp']  # Use string dates for consistency
        df.drop('date_temp', axis=1, inplace=True)
    except Exception as e:
        print(f"Warning: Could not sort by date: {e}")
    
    return df

def extract_context_from_json(json_data):
    """
    Extract context information from the JSON data to help with table interpretation.
    """
    contexts = []
    for entry in json_data:
        context = {
            "id": entry.get("id", ""),
            "organization": standardize_organization_name(extract_organization_from_id(entry.get("id", ""))),
            "pre_text": entry.get("pre_text", ""),
            "post_text": entry.get("post_text", ""),
            "table_data": entry.get("table_data", [])
        }
        contexts.append(context)
    return contexts

# Custom JSON encoder to handle datetime, Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        return super().default(obj)

def clean_and_align_dates(df):
    """
    Clean up the date column and align metrics with the correct dates.
    This is needed because some financial metrics were misaligned in the data.
    """
    if df is None or df.empty:
        return df
    
    # Create a deep copy to avoid warnings
    clean_df = df.copy()
    
    # Standardize date format
    clean_df['date'] = clean_df['date'].apply(lambda x: str(x).strip() if x else x)
    
    # Identify rows with just year information
    year_pattern = r'^(\d{4})$'
    year_only_rows = clean_df[clean_df['date'].str.match(year_pattern, na=False)]
    
    if not year_only_rows.empty:
        print(f"Found {len(year_only_rows)} rows with year-only dates")
        
        # For rows with just years, convert to standard format (year-12-31)
        for idx, row in year_only_rows.iterrows():
            year = row['date']
            # Update date to standard format
            clean_df.at[idx, 'date'] = f"{year}-12-31"
    
    # Handle date ranges or periods
    range_pattern = r'(\d{4})[-_\s]+(\d{4})'
    range_rows = clean_df[clean_df['date'].str.contains(range_pattern, na=False, regex=True)]
    
    if not range_rows.empty:
        print(f"Found {len(range_rows)} rows with date ranges")
        
        for idx, row in range_rows.iterrows():
            match = re.search(range_pattern, row['date'])
            if match:
                # Use the end year of the range
                end_year = match.group(2)
                clean_df.at[idx, 'date'] = f"{end_year}-12-31"
    
    # Try to convert all dates to ISO format
    date_formats = [
        '%Y-%m-%d',    # 2020-12-31
        '%Y/%m/%d',    # 2020/12/31
        '%m/%d/%Y',    # 12/31/2020
        '%d/%m/%Y',    # 31/12/2020
        '%m/%d/%y',    # 12/31/20
        '%d/%m/%y',    # 31/12/20
        '%b %d, %Y',   # Dec 31, 2020
        '%B %d, %Y',   # December 31, 2020
        '%d %b %Y',    # 31 Dec 2020
        '%d %B %Y',    # 31 December 2020
        '%Y'           # 2020 (default to December 31st)
    ]
    
    def standardize_date(date_str):
        if not date_str or not isinstance(date_str, str):
            return date_str
            
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # Standard ISO format
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If we couldn't parse the date with any format, return as is
        return date_str
    
    clean_df['date'] = clean_df['date'].apply(standardize_date)
    
    # Sort by date
    try:
        clean_df = clean_df.sort_values('date')
    except:
        print("Warning: Could not sort by date")
    
    return clean_df

def main():
    # Try to load spaCy model for semantic understanding
    try:
        import spacy
        nlp = spacy.load("en_core_web_md")
        print("Using spaCy for semantic matching of metrics.")
    except (ImportError, OSError):
        print("spaCy model not available. Using basic string matching for metrics.")
        nlp = None
    
    # Load the dataset from the JSON file
    try:
        with open("rag_system/data/knowledge.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        try:
            with open("knowledge.json", "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("Knowledge file not found. Please check the path.")
            return
    
    # Extract contexts for better understanding
    contexts = extract_context_from_json(data)
    
    processed_tables = []
    # Process each entry that contains "table_data"
    for entry in data:
        if "table_data" in entry and entry["table_data"]:
            table = entry["table_data"]
            doc_id = entry.get("id", "")
            df = process_and_transpose_table(table, doc_id, nlp)
            if df is not None and not df.empty:
                processed_tables.append(df)
    
    if not processed_tables:
        print("No tables found in the data.")
        return
    
    # Merge the individual tables on the "date" column
    merged_df = merge_tables(processed_tables)
    merged_df = deduplicate_df(merged_df)
    
    # Clean and align dates
    merged_df = clean_and_align_dates(merged_df)
    
    # Ensure organization column exists and is properly populated
    if "organization" not in merged_df.columns:
        print("Warning: Organization column not found, adding empty column")
        merged_df["organization"] = None
    else:
        # Fill any missing organization values with the previous or next valid value
        merged_df["organization"] = merged_df["organization"].fillna(method='ffill').fillna(method='bfill')
    
    # Clean up NaN values for better readability
    merged_df = merged_df.replace({np.nan: None})
    
    print("Unified Financial Report:")
    print(merged_df.head())
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    
    # Move organization column to the front for better readability
    if "organization" in merged_df.columns:
        cols = ["date", "organization"] + [col for col in merged_df.columns if col not in ["date", "organization"]]
        merged_df = merged_df[cols]
    
    # Ensure no duplicate column names
    if len(merged_df.columns) != len(merged_df.columns.unique()):
        # Get duplicate column names
        dupes = merged_df.columns[merged_df.columns.duplicated()].tolist()
        print(f"Warning: Found duplicate column names: {dupes}")
        
        # Create a dictionary to count occurrences of each column name
        col_counts = {}
        for col in merged_df.columns:
            if col in col_counts:
                col_counts[col] += 1
            else:
                col_counts[col] = 1
        
        # Create a new list of column names with duplicates renamed more distinctively
        new_columns = []
        seen_cols = {}
        
        for col in merged_df.columns:
            if col_counts[col] > 1:
                # For duplicate columns, we need to examine the data to distinguish them
                if col not in seen_cols:
                    seen_cols[col] = 0
                    new_columns.append(col)
                else:
                    seen_cols[col] += 1
                    # Get a sample of non-null values from this column to use in the name
                    sample_values = merged_df[col].dropna().head(1).astype(str).values
                    sample_str = sample_values[0] if len(sample_values) > 0 else f"variant_{seen_cols[col]}"
                    # Clean the sample to make it a valid column name
                    sample_str = re.sub(r'[^\w]', '_', str(sample_str))
                    new_columns.append(f"{col}_{sample_str}")
            else:
                new_columns.append(col)
        
        # Set the new column names
        merged_df.columns = new_columns
        
        # Verify we have no more duplicates
        if len(merged_df.columns) != len(merged_df.columns.unique()):
            print("Warning: Still have duplicate columns after renaming. Using numerical suffixes.")
            # Fall back to simple numerical suffixes
            new_columns = []
            seen_cols = {}
            
            for col in merged_df.columns:
                if col in seen_cols:
                    seen_cols[col] += 1
                    new_columns.append(f"{col}_{seen_cols[col]}")
                else:
                    seen_cols[col] = 0
                    new_columns.append(col)
            
            merged_df.columns = new_columns
    
    # Save the unified table to CSV and JSON formats
    output_dir = "rag_system/data/processed"
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = f"{output_dir}/merged_financial_report.csv"
    json_path = f"{output_dir}/merged_financial_report.json"
    
    merged_df.to_csv(csv_path, index=False)
    
    # Convert to JSON with date as index for better semantic querying
    json_records = merged_df.to_dict(orient='records')
    with open(json_path, "w") as f:
        json.dump(json_records, f, indent=2, cls=CustomJSONEncoder)
    
    print(f"Files saved: {csv_path} and {json_path}")
    
    # Generate data schema information for reference
    schema_info = {
        "total_records": len(merged_df),
        "columns": {},
        "date_range": {
            "min": str(merged_df['date'].min()),
            "max": str(merged_df['date'].max())
        }
    }
    
    for col in merged_df.columns:
        if col == 'date':
            continue
        
        non_null_count = merged_df[col].count()
        schema_info["columns"][col] = {
            "non_null_values": int(non_null_count),
            "fill_rate": round(float(non_null_count / len(merged_df)) * 100, 2)
        }
    
    schema_path = f"{output_dir}/schema_info.json"
    with open(schema_path, "w") as f:
        json.dump(schema_info, f, indent=2)
    
    print(f"Schema information saved to: {schema_path}")

if __name__ == "__main__":
    main()

import pandas as pd
try:
    df = pd.read_csv("data_histori.csv")
    print("Columns:", df.columns.tolist())
    # Check if 'DateTime' exists
    if 'DateTime' in df.columns:
        # Convert to datetime if not already, coercing errors
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        
        print(f"Min Date: {df['DateTime'].min()}")
        print(f"Max Date: {df['DateTime'].max()}")
        print(f"Total rows: {len(df)}")
        print(f"Null DateTimes: {df['DateTime'].isna().sum()}")
    else:
        print("Column 'DateTime' not found.")
        print(df.head())
except Exception as e:
    print(f"Error reading CSV: {e}")

# processing/clean_data.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data, insert_data
import pandas as pd


def clean_data():
    print("\n🚀 Fetching raw data...")
    data = fetch_data('raw_ecommerce_data')

    if not data:
        print("❌ No raw data found. Run ingestion first.")
        return

    df = pd.DataFrame(data)

    print("\n✅ Raw Data Sample:")
    print(df.head())

    # 🔥 Basic Cleaning — Fill Nulls Safely
    if 'Status' not in df.columns:
        df['Status'] = 'Pending'
    else:
        df['Status'] = df['Status'].fillna('Pending')

    if 'PaymentMethod' not in df.columns:
        df['PaymentMethod'] = 'UPI'
    else:
        df['PaymentMethod'] = df['PaymentMethod'].fillna('UPI')

    text_cols = ['CustomerName', 'ProductName', 'Category']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    num_cols = ['Quantity', 'Price']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df['OrderDate'] = df['OrderDate'].fillna(pd.Timestamp('2023-07-01'))

    # 🔥 Duplicate Removal
    initial_rows = len(df)

    df.drop_duplicates(subset=['OrderID', 'ProductName', 'CustomerName'], inplace=True)

    removed_rows = initial_rows - len(df)
    print(f"\n🗑️ Removed {removed_rows} duplicate rows based on OrderID, ProductName, and CustomerName.")

    # 🔥 Reset index (optional but clean)
    df.reset_index(drop=True, inplace=True)

    print("\n✅ Cleaned Data Sample:")
    print(df.head())

    print(f"\n📦 Total Rows after cleaning: {len(df)}")

    # ✅ Insert into processed_ecommerce_data collection
    cleaned_data = df.to_dict(orient='records')
    insert_data('processed_ecommerce_data', cleaned_data)

    print("\n🎉 Cleaned data inserted into 'processed_ecommerce_data' collection successfully!")


if __name__ == "__main__":
    clean_data()

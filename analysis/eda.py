# analysis/eda.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data
import pandas as pd


def run_eda():
    # ✅ Step 1: Fetch Processed Data
    data = fetch_data('processed_ecommerce_data')

    if not data:
        print("❌ No data found in 'processed_ecommerce_data'. Run data cleaning first.")
        return

    print(f"✅ {len(data)} records fetched for analysis.")

    # ✅ Step 2: Convert to DataFrame
    df = pd.DataFrame(data)

    # ✅ Step 3: Validate Required Columns
    required_cols = ['OrderDate', 'ProductName', 'Category', 'Quantity', 'Price', 'PaymentMethod', 'Status']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return

    # ✅ Step 4: Type Conversion
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%d-%m-%Y', errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Debug: Check for invalid parsing
    print("🔍 Invalid OrderDates:", df['OrderDate'].isna().sum())
    print("📅 Sample OrderDates:", df['OrderDate'].dropna().unique()[:5])

    df.dropna(subset=['OrderDate', 'Quantity', 'Price'], inplace=True)

    # ✅ Step 5: Basic Stats
    print("\n📊 Basic Stats:")
    print(df.describe(include='all'))

    total_orders = len(df)
    total_revenue = (df['Quantity'] * df['Price']).sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

    print(f"\n🛒 Total Orders: {total_orders}")
    print(f"💰 Total Revenue: ₹{total_revenue:.2f}")
    print(f"📦 Average Order Value: ₹{avg_order_value:.2f}")

    # ✅ Step 6: Category-wise Revenue
    print("\n📊 Category-wise Revenue:")
    cat_rev = df.groupby('Category').apply(lambda x: (x['Quantity'] * x['Price']).sum())
    print(cat_rev.sort_values(ascending=False))

    # ✅ Step 7: Payment Method Distribution
    print("\n💳 Payment Method Distribution:")
    print(df['PaymentMethod'].value_counts())

    # ✅ Step 8: Order Status Distribution
    print("\n🚚 Order Status Distribution:")
    print(df['Status'].value_counts())

    # ✅ Step 9: Monthly Revenue Trend
    df['Month'] = df['OrderDate'].dt.to_period('M')
    month_rev = df.groupby('Month').apply(lambda x: (x['Quantity'] * x['Price']).sum())

    print("\n📅 Monthly Revenue Trend:")
    print(month_rev.sort_index())

    # ✅ Step 10: (City analysis removed as per dataset)
    # No print or warning about City since it doesn't exist


if __name__ == "__main__":
    run_eda()

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

    # ✅ Step 3: Basic Overview
    print("\n📊 Basic Stats:")
    print(df.describe(include='all'))

    total_orders = len(df)
    total_revenue = (df['Quantity'] * df['Price']).sum()
    avg_order_value = total_revenue / total_orders

    print(f"\n🛒 Total Orders: {total_orders}")
    print(f"💰 Total Revenue: {total_revenue}")
    print(f"📦 Average Order Value: {avg_order_value}")

    # ✅ Category-wise Sales
    print("\n📊 Category-wise Revenue:")
    cat_rev = df.groupby('Category').apply(lambda x: (x['Quantity'] * x['Price']).sum())
    print(cat_rev)

    # ✅ Payment Method Distribution
    print("\n💳 Payment Method Distribution:")
    print(df['PaymentMethod'].value_counts())

    # ✅ Order Status Distribution
    print("\n🚚 Order Status Distribution:")
    print(df['Status'].value_counts())

    # ✅ Monthly Trend
    df['Month'] = pd.to_datetime(df['OrderDate']).dt.to_period('M')
    month_rev = df.groupby('Month').apply(lambda x: (x['Quantity'] * x['Price']).sum())

    print("\n📅 Monthly Revenue Trend:")
    print(month_rev)

    # ✅ Top 5 Cities by Revenue
    city_rev = df.groupby('City').apply(lambda x: (x['Quantity'] * x['Price']).sum()).sort_values(ascending=False)
    print("\n🏙️ Top 5 Cities by Revenue:")
    print(city_rev.head(5))


if __name__ == "__main__":
    run_eda()

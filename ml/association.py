# ml/association.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data
import pandas as pd


def association_analysis():
    # ✅ Fetch Data
    data = fetch_data()

    if not data:
        print("❌ No data found.")
        return

    df = pd.DataFrame(data)

    # ✅ Create Transactions: group by OrderID
    transactions = df.groupby('OrderID')['ProductName'].apply(list)

    # ✅ Create Co-Occurrence Matrix
    product_list = df['ProductName'].unique()
    co_matrix = pd.DataFrame(0, index=product_list, columns=product_list)

    for items in transactions:
        for item in items:
            for other_item in items:
                if item != other_item:
                    co_matrix.loc[item, other_item] += 1

    print("\n✅ Product Co-occurrence Matrix:")
    print(co_matrix)

    # ✅ Generate Recommendation List
    recommendations = {}

    for product in co_matrix.index:
        sorted_items = co_matrix.loc[product].sort_values(ascending=False)
        related_products = sorted_items[sorted_items > 0].index.tolist()
        recommendations[product] = related_products

    # ✅ Display Recommendations
    print("\n✅ Product-wise Frequently Bought Together Recommendations:")
    for product, related in recommendations.items():
        print(f"→ {product}: {', '.join(related) if related else 'No related products'}")

    # ✅ Optional: Save to CSV
    # co_matrix.to_csv('product_cooccurrence_matrix.csv')


if __name__ == "__main__":
    association_analysis()

# ml/customer_segmentation.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def customer_segmentation():
    # ✅ Fetch Processed Data
    data = fetch_data('processed_ecommerce_data')

    if not data:
        print("❌ No data found.")
        return

    df = pd.DataFrame(data)

    # ✅ Create TotalSales column
    df['TotalSales'] = df['Quantity'] * df['Price']

    # ✅ Base Customer Features
    customer_df = df.groupby('CustomerName').agg({
        'TotalSales': 'sum',
        'OrderID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()

    customer_df.rename(columns={
        'TotalSales': 'TotalSpend',
        'OrderID': 'TotalOrders',
        'Quantity': 'TotalQuantity'
    }, inplace=True)

    # ✅ Average Order Value
    customer_df['AvgOrderValue'] = customer_df['TotalSpend'] / customer_df['TotalOrders']

    # ✅ Product Preference (Pivot Table)
    product_pivot = pd.pivot_table(
        df, index='CustomerName',
        columns='ProductName', values='Quantity',
        aggfunc='sum', fill_value=0
    ).reset_index()

    # ✅ Merge All Features
    merged = pd.merge(customer_df, product_pivot, on='CustomerName')

    print("\n✅ Customer Dataset Preview:")
    print(merged.head())

    # ✅ Features for Clustering
    X = merged.drop(columns=['CustomerName'])

    # ✅ Elbow Method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # ✅ Apply KMeans (Set k based on elbow)
    k = 3  # You can change based on elbow curve
    kmeans = KMeans(n_clusters=k, random_state=42)
    merged['Cluster'] = kmeans.fit_predict(X)

    print("\n🏷️ Clustered Customers:")
    print(merged[['CustomerName', 'TotalSpend', 'TotalOrders', 'TotalQuantity', 'Cluster']])

    # ✅ 2D Scatter Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=merged,
        x='TotalSpend',
        y='TotalOrders',
        hue='Cluster',
        palette='Set1'
    )
    plt.title('Customer Segmentation: Spend vs Orders')
    plt.xlabel('Total Spend')
    plt.ylabel('Total Orders')
    plt.show()

    # ✅ 2D Scatter Plot Quantity vs Spend
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=merged,
        x='TotalSpend',
        y='TotalQuantity',
        hue='Cluster',
        palette='Set2'
    )
    plt.title('Customer Segmentation: Spend vs Quantity')
    plt.xlabel('Total Spend')
    plt.ylabel('Total Quantity')
    plt.show()

    # ✅ Bar Plot - Avg Spend per Cluster
    avg_spend = merged.groupby('Cluster')['TotalSpend'].mean()
    avg_spend.plot(kind='bar', color='skyblue')
    plt.title('Average Spend per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Avg Spend')
    plt.show()

    # ✅ Pie Chart - Customer Distribution
    counts = merged['Cluster'].value_counts()
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title('Customer Distribution by Cluster')
    plt.ylabel('')
    plt.show()


if __name__ == "__main__":
    customer_segmentation()

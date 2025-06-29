# ml/sales_prediction.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def sales_prediction():
    # ✅ Fetch Processed Data
    data = fetch_data('processed_ecommerce_data')

    if not data:
        print("❌ No data found in 'processed_ecommerce_data'. Run cleaning first.")
        return

    df = pd.DataFrame(data)

    # ✅ Feature Engineering
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Month'] = df['OrderDate'].dt.month

    # ✅ Calculate Average Price Per Product
    avg_price = df.groupby('ProductName')['Price'].mean().to_dict()

    # ✅ Map Product to Category
    category_map = df.set_index('ProductName')['Category'].to_dict()

    # ✅ Target Variable: Quantity
    y = df['Quantity']

    # ✅ Features for Prediction
    product_dummies = pd.get_dummies(df['ProductName'], prefix='Product')
    category_dummies = pd.get_dummies(df['Category'], prefix='Category')
    payment_dummies = pd.get_dummies(df['PaymentMethod'], prefix='Payment')

    X = pd.concat([
        df[['Month']],
        product_dummies,
        category_dummies,
        payment_dummies
    ], axis=1)

    # ✅ Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Model Training for Quantity Prediction
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ✅ Model Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Quantity Prediction Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # ✅ Product-wise and Category-wise Revenue Prediction
    print("\n🛍️ Product-wise Quantity & Revenue Prediction (for July):")

    products = df['ProductName'].unique()
    category_wise_revenue = {}

    for product in products:
        input_data = {
            'Month': [7]  # Example month (can change)
        }

        # ✅ Set product encoding
        for col in product_dummies.columns:
            input_data[col] = [1 if product in col else 0]

        # ✅ Set category encoding
        product_category = category_map.get(product)
        for col in category_dummies.columns:
            input_data[col] = [1 if product_category in col else 0]

        # ✅ Neutralize payment method
        for col in payment_dummies.columns:
            input_data[col] = [0]

        input_df = pd.DataFrame(input_data)

        # ✅ Predict Quantity
        quantity_pred = model.predict(input_df)[0]
        quantity_pred = max(int(round(quantity_pred)), 0)  # Convert to integer, avoid negatives

        # ✅ Calculate Revenue
        price = avg_price.get(product, 0)
        total_sales = quantity_pred * price

        print(f"→ {product}: Predicted Quantity: {quantity_pred}, Predicted Sales: ₹{total_sales:.2f}")

        # ✅ Accumulate Category-wise Revenue
        if product_category in category_wise_revenue:
            category_wise_revenue[product_category] += total_sales
        else:
            category_wise_revenue[product_category] = total_sales

    # ✅ Display Category-wise Revenue
    print("\n🏢 Category-wise Revenue Prediction (for July):")
    for category, revenue in category_wise_revenue.items():
        print(f"→ {category}: ₹{revenue:.2f}")


if __name__ == "__main__":
    sales_prediction()

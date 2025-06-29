# ml/sales_prediction_all.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_operations import fetch_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def sales_prediction_for_all():
    # ✅ Fetch Data
    data = fetch_data('processed_ecommerce_data')

    if not data:
        print("❌ No data found. Run cleaning first.")
        return

    df = pd.DataFrame(data)

    # ✅ Feature Engineering
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Month'] = df['OrderDate'].dt.month

    df['TotalSales'] = df['Quantity'] * df['Price']

    # ✅ Encode Product & Category
    product_dummies = pd.get_dummies(df['ProductName'], prefix='Product')
    category_dummies = pd.get_dummies(df['Category'], prefix='Category')

    X = pd.concat([df[['Quantity', 'Month']], product_dummies, category_dummies], axis=1)
    y = df['TotalSales']

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ✅ Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # ✅ Predict for every Product (Quantity=1, Month=7)
    print("\n🛍️ Product-wise Predicted Sales (for Quantity=1 in July):")
    products = df['ProductName'].unique()

    for product in products:
        input_data = {
            'Quantity': [1],
            'Month': [7]
        }
        for col in product_dummies.columns:
            input_data[col] = [1 if product in col else 0]
        for col in category_dummies.columns:
            input_data[col] = [0]  # Not setting category for product-only

        input_df = pd.DataFrame(input_data)
        pred = model.predict(input_df)[0]
        print(f"→ {product}: ₹{pred:.2f}")

    # ✅ Predict for every Category (Quantity=1, Month=7)
    print("\n🏬 Category-wise Predicted Sales (for Quantity=1 in July):")
    categories = df['Category'].unique()

    for category in categories:
        input_data = {
            'Quantity': [1],
            'Month': [7]
        }
        for col in product_dummies.columns:
            input_data[col] = [0]  # No specific product
        for col in category_dummies.columns:
            input_data[col] = [1 if category in col else 0]

        input_df = pd.DataFrame(input_data)
        pred = model.predict(input_df)[0]
        print(f"→ {category}: ₹{pred:.2f}")


if __name__ == "__main__":
    sales_prediction_for_all()

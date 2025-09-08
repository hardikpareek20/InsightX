import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from storage.db_operations import fetch_data

def sales_prediction():
    # Get target month input
    target_input = input("📅 Enter target month & year for prediction (MM-YYYY): ").strip()
    try:
        target_date = datetime.strptime(target_input, "%m-%Y")
    except ValueError:
        print("❌ Invalid format! Please use MM-YYYY (e.g., 01-2026).")
        return

    # Fetch data from MongoDB
    data = fetch_data()
    if not data:
        print("❌ No data found in MongoDB.")
        return

    df = pd.DataFrame(data)
    
    # Convert and process dates
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], format="%d-%m-%Y", errors='coerce')
    df = df.dropna(subset=['OrderDate'])
    
    # Create time-based features
    df['Month'] = df['OrderDate'].dt.month
    df['Year'] = df['OrderDate'].dt.year
    df['DaysInMonth'] = df['OrderDate'].dt.daysinmonth
    
    # Use more historical data (last 12 months)
    start_date = target_date - relativedelta(months=12)
    df = df[df['OrderDate'] >= start_date]
    
    if df.empty:
        print("❌ Not enough historical data available.")
        return

    # Feature engineering
    df['PriceQuantity'] = df['Price'] * df['Quantity']
    
    # Prepare features and target
    features = pd.get_dummies(df[['ProductName', 'Category', 'PaymentMethod', 'Month', 'Year']], 
                            columns=['ProductName', 'Category', 'PaymentMethod'])
    target = df['Quantity']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Use Random Forest which handles categorical data better
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    print("\n✅ Model Evaluation:")
    print(f"→ MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"→ R² Score: {r2_score(y_test, y_pred):.2f}")
    
    # Prepare prediction input
    prediction_template = pd.DataFrame(0, index=[0], columns=features.columns)
    prediction_template['Month'] = target_date.month
    prediction_template['Year'] = target_date.year
    
    # Get unique products and categories
    products = df['ProductName'].unique()
    categories = df['Category'].unique()
    
    category_qty = {cat: 0 for cat in categories}
    category_rev = {cat: 0.0 for cat in categories}
    avg_prices = df.groupby('ProductName')['Price'].mean().to_dict()
    category_map = df.drop_duplicates('ProductName').set_index('ProductName')['Category'].to_dict()
    
    # Make predictions for each product
    for product in products:
        temp = prediction_template.copy()
        
        # Set product and category features
        product_col = f"ProductName_{product}"
        if product_col in temp.columns:
            temp[product_col] = 1
            
        category = category_map.get(product, '')
        category_col = f"Category_{category}"
        if category_col in temp.columns:
            temp[category_col] = 1
            
        # Predict quantity
        quantity_pred = max(0, int(round(model.predict(temp)[0])))
        price = avg_prices.get(product, 0)
        revenue = quantity_pred * price
        
        # Aggregate by category
        category_qty[category] += quantity_pred
        category_rev[category] += revenue
    
    # Print results
    print(f"\n📊 Category-wise Prediction for {target_input}:")
    for category in categories:
        print(f"→ {category}: Predicted Quantity: {category_qty[category]}, "
              f"Predicted Revenue: ₹{category_rev[category]:.2f}")

if __name__ == "__main__":
    sales_prediction()
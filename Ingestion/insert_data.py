# ingestion/insert_data.py

import os
import sys
import json
import pandas as pd

# ✅ Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Now import works perfectly
from storage.db_config import get_db_connection


def insert_data():
    db = get_db_connection()
    collection = db['raw_ecommerce_data']  # ✅ Collection name

    # ✅ File paths
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))

    json_file = os.path.join(data_folder, 'ecommerce_data.json')
    csv_file = os.path.join(data_folder, 'ecommerce_data.csv')

    # ✅ Insert JSON data
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            if isinstance(json_data, list):
                if json_data:
                    collection.insert_many(json_data)
                    print("✅ JSON data inserted successfully!")
                else:
                    print("⚠️ JSON file is empty.")
            else:
                collection.insert_one(json_data)
                print("✅ Single JSON record inserted!")
    else:
        print("❌ JSON file not found.")

    # ✅ Insert CSV data
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        csv_data = df.to_dict(orient='records')
        if csv_data:
            collection.insert_many(csv_data)
            print("✅ CSV data inserted successfully!")
        else:
            print("⚠️ CSV file is empty.")
    else:
        print("❌ CSV file not found.")


if __name__ == "__main__":
    insert_data()

# storage/db_operations.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_config import get_db_connection

# Get the MongoDB collection
db = get_db_connection()
collection_name = 'processed_ecommerce_data'
collection = db[collection_name]


def insert_data(data):
    """Insert one or many documents into the processed_ecommerce_data collection."""
    if isinstance(data, list):
        result = collection.insert_many(data)
        print(f"✅ {len(result.inserted_ids)} documents inserted into '{collection_name}'")
    else:
        result = collection.insert_one(data)
        print(f"✅ 1 document inserted into '{collection_name}'")


def fetch_data(query={}):
    """Fetch documents from processed_ecommerce_data based on optional query."""
    results = collection.find(query)
    return list(results)


def delete_data(query={}):
    """Delete documents from processed_ecommerce_data based on optional query."""
    result = collection.delete_many(query)
    print(f"🗑️ {result.deleted_count} documents deleted from '{collection_name}'")


def update_data(query, new_values):
    """Update documents in processed_ecommerce_data using query and update dict."""
    result = collection.update_many(query, {'$set': new_values})
    print(f"🔧 {result.modified_count} documents updated in '{collection_name}'")


# ✅ ✅ ✅ Test Block Starts Here
if __name__ == "__main__":
    sample_data = {
        "OrderID": 9999,
        "CustomerName": "Test User",
        "ProductName": "Test Product",
        "Quantity": 2,
        "Price": 1000,
        "Status": "Completed"
    }

    print("\n🚀 Inserting Sample Data...")
    insert_data(sample_data)

    print("\n🔍 Fetching Data...")
    data = fetch_data({"OrderID": 9999})
    print(data)

    print("\n🔧 Updating Data...")
    update_data({"OrderID": 9999}, {"Status": "Cancelled"})

    print("\n🗑️ Deleting Data...")
    delete_data({"OrderID": 9999})

    print("\n✅ CRUD Operations Test Completed!\n")

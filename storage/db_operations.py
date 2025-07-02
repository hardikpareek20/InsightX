# storage/db_operations.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_config import get_db_connection

db = get_db_connection()

def insert_data(collection_name, data):
    collection = db[collection_name]
    if isinstance(data, list):
        result = collection.insert_many(data)
        print(f"✅ {len(result.inserted_ids)} documents inserted into '{collection_name}'")
    else:
        result = collection.insert_one(data)
        print(f"✅ 1 document inserted into '{collection_name}'")


def fetch_data(collection_name, query={}):
    collection = db[collection_name]
    results = collection.find(query)
    return list(results)


def delete_data(collection_name, query={}):
    collection = db[collection_name]
    result = collection.delete_many(query)
    print(f"🗑️ {result.deleted_count} documents deleted from '{collection_name}'")


def update_data(collection_name, query, new_values):
    collection = db[collection_name]
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
    insert_data('raw_ecommerce_data', sample_data)

    print("\n🔍 Fetching Data...")
    data = fetch_data('raw_ecommerce_data', {"OrderID": 9999})
    print(data)

    print("\n🔧 Updating Data...")
    update_data('raw_ecommerce_data', {"OrderID": 9999}, {"Status": "Cancelled"})

    print("\n🗑️ Deleting Data...")
    delete_data('raw_ecommerce_data', {"OrderID": 9999})

    print("\n✅ CRUD Operations Test Completed!\n")











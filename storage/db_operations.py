# storage/db_operations.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.db_config import get_db_connection

# ✅ Get MongoDB Database
db = get_db_connection()


# 🔹 Insert Data
def insert_data(collection_name, data):
    """Insert one or many documents into the given collection."""
    collection = db[collection_name]

    if isinstance(data, list):
        if not data:
            print(f"⚠️ No data provided to insert into '{collection_name}'")
            return None
        result = collection.insert_many(data)
        print(f"✅ {len(result.inserted_ids)} documents inserted into '{collection_name}'")
        return result
    else:
        result = collection.insert_one(data)
        print(f"✅ 1 document inserted into '{collection_name}'")
        return result


# 🔹 Fetch Data
def fetch_data(collection_name, query=None):
    """Fetch documents from the given collection based on an optional query."""
    collection = db[collection_name]
    if query is None:
        query = {}
    print(f"📥 Fetching from collection: {collection_name}, Query: {query}")  # Debug
    results = collection.find(query)
    return list(results)


# 🔹 Delete Data
def delete_data(collection_name, query=None):
    """Delete documents from the given collection based on an optional query."""
    collection = db[collection_name]
    if query is None:
        query = {}
    result = collection.delete_many(query)
    print(f"🗑️ {result.deleted_count} documents deleted from '{collection_name}'")
    return result


# 🔹 Update Data
def update_data(collection_name, query, new_values):
    """Update documents in the given collection using query and update dict."""
    collection = db[collection_name]
    result = collection.update_many(query, {'$set': new_values})
    print(f"🔧 {result.modified_count} documents updated in '{collection_name}'")
    return result


# ✅ Test Block
if __name__ == "__main__":
    test_collection = "processed_ecommerce_data"

    sample_data = {
        "OrderID": 9999,
        "CustomerName": "Test User",
        "ProductName": "Test Product",
        "Quantity": 2,
        "Price": 1000,
        "Status": "Completed"
    }

    print("\n🚀 Inserting Sample Data...")
    insert_data(test_collection, sample_data)

    print("\n🔍 Fetching Data...")
    data = fetch_data(test_collection, {"OrderID": 9999})
    print(data)

    print("\n🔧 Updating Data...")
    update_data(test_collection, {"OrderID": 9999}, {"Status": "Cancelled"})

    print("\n🗑️ Deleting Data...")
    delete_data(test_collection, {"OrderID": 9999})

    print("\n✅ CRUD Operations Test Completed!\n")

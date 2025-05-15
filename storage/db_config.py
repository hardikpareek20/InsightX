from pymongo import MongoClient

def get_db_connection():
    client = MongoClient('mongodb://localhost:27017/')  # or use your MongoDB URI
    db = client['InsightX']  # Your database name
    return db

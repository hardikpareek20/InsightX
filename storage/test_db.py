from db_config import get_db_connection

def test_connection():
    db = get_db_connection()
    print("Collections in DB:", db.list_collection_names())

if __name__ == "__main__":
    test_connection()
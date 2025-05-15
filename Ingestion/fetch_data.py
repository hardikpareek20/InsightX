import json
import os

def load_json(file_path):
    """Load data from JSON file and return it."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from JSON.")
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

if __name__ == "__main__":
    # Resolve absolute path to sample_data.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, '..', 'data', 'sample_data.json')

    # Load data
    json_data = load_json(json_file)
    
    # Optional: print first 2 entries for preview
    if json_data:
        print("Preview of first 2 records:")
        for item in json_data[:2]:
            print(item)

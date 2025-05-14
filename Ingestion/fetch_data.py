import requests
import json

# Function to fetch data from a URL
def fetch_data(url):
    try:
        # Make a GET request to the API
        response = requests.get(url)
        
        # If the request is successful
        if response.status_code == 200:
            data = response.json()  # Parse JSON data from the response
            
            # Save the data to a local file (optional, can skip if not needed)
            with open('data/sample_data.json', 'w') as f:
                json.dump(data, f, indent=4)
            
            print("Data fetched and saved successfully.")
        else:
            print(f"Failed to retrieve data: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    fetch_data('https://jsonplaceholder.typicode.com/todos')

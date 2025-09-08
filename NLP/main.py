import ollama
import pandas as pd
import re
import requests
from pymongo import MongoClient
import traceback

def connect_to_mongodb():
    """Connect to MongoDB with error handling"""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        # Verify connection
        client.server_info()
        db = client['InsightX']
        collection = db['processed_ecommerce_data']
        
        # Convert MongoDB documents to DataFrame
        data = pd.DataFrame(list(collection.find()))
        
        # Drop MongoDB's default '_id' column if present
        if '_id' in data.columns:
            data.drop('_id', axis=1, inplace=True)
            
        # Validate required columns
        required_columns = ['OrderID', 'CustomerName', 'ProductName', 'Category', 
                           'Quantity', 'Price', 'OrderDate', 'PaymentMethod']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # Create placeholder data for testing if columns are missing
            for col in missing_columns:
                if col == 'OrderID':
                    data[col] = range(1, len(data) + 1)
                elif col == 'CustomerName':
                    data[col] = f'Customer_{range(1, len(data) + 1)}'
                elif col == 'ProductName':
                    data[col] = f'Product_{range(1, len(data) + 1)}'
                elif col == 'Category':
                    data[col] = 'Unknown'
                elif col == 'Quantity':
                    data[col] = 1
                elif col == 'Price':
                    data[col] = 10.0
                elif col == 'OrderDate':
                    data[col] = pd.Timestamp.now()
                elif col == 'PaymentMethod':
                    data[col] = 'Credit Card'
            
        # Parse OrderDate
        data['OrderDate'] = pd.to_datetime(data['OrderDate'], errors='coerce')
        data.dropna(subset=['OrderDate'], inplace=True)
        
        # Preprocessing
        data['TotalSales'] = data['Quantity'] * data['Price']
        data['Category_clean'] = data['Category'].str.strip().str.lower()
        data['ProductName_clean'] = data['ProductName'].str.strip().str.lower()
        
        return data
        
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        print(traceback.format_exc())
        # Create sample data for testing if MongoDB is not available
        print("🔄 Creating sample data for testing...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if MongoDB is not available"""
    sample_data = {
        'OrderID': [1, 2, 3, 4, 5],
        'CustomerName': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'ProductName': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
        'Quantity': [1, 2, 1, 1, 2],
        'Price': [1000.0, 25.0, 75.0, 300.0, 50.0],
        'OrderDate': pd.date_range('2023-01-01', periods=5),
        'PaymentMethod': ['Credit Card', 'PayPal', 'Credit Card', 'Bank Transfer', 'PayPal']
    }
    
    data = pd.DataFrame(sample_data)
    data['TotalSales'] = data['Quantity'] * data['Price']
    data['Category_clean'] = data['Category'].str.strip().str.lower()
    data['ProductName_clean'] = data['ProductName'].str.strip().str.lower()
    
    return data

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama is running and API is responding!")
            return True
        else:
            print(f"⚠️ Ollama responded unexpectedly: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ollama is NOT running. Error: {e}")
        return False

def get_schema_description():
    return f"""
The dataset 'data' has the following columns:
{list(data.columns)}

Column details:
- OrderID (integer): Unique identifier for each order
- CustomerName (string): Name of the customer
- ProductName (string): Name of the product
- Category (string): Product category
- Quantity (integer): Number of items purchased
- Price (float): Price per item
- OrderDate (datetime): Date of the order
- PaymentMethod (string): Payment method used

Derived columns:
- TotalSales = Quantity * Price
- Category_clean = lowercase version of Category
- ProductName_clean = lowercase version of ProductName

Sample data (first 3 rows):
{data.head(3).to_string()}

⚠️ Important Rules:
- Text matching → use Category_clean/ProductName_clean (lowercase).
- For 'max' type queries use → .idxmax() not .argmax().
- ❌ Never use broken syntax like .reset_index(). CustomerName.
- ❌ Never use .name(0) — it's invalid.
- ✅ Return ONLY executable pandas code — no explanation, no markdown.
- ✅ Always reference the dataframe as 'data'

Example queries:
1. "Which customer made the most purchases?" → data.groupby('CustomerName')['Quantity'].sum().idxmax()
2. "What is the total sales by category?" → data.groupby('Category_clean')['TotalSales'].sum()
3. "Show me all orders from the electronics category" → data[data['Category_clean'] == 'electronics']
"""

def ask_llm(user_query):
    prompt = get_schema_description() + f"\nNow write pandas code for this query:\n'{user_query}'\n"
    try:
        response = ollama.chat(
            model='mistral',
            messages=[{"role": "user", "content": prompt}],
            options={'temperature': 0.1}  # Lower temperature for more deterministic code generation
        )
        
        # Handle different response formats from Ollama
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content.strip()
        elif isinstance(response, dict) and 'message' in response and 'content' in response['message']:
            return response['message']['content'].strip()
        else:
            print(f"⚠️ Unexpected response format: {response}")
            return str(response)
            
    except Exception as e:
        return f"❌ Ollama API error: {e}"

def extract_code(text):
    if not text.strip():
        return ""
    
    # Try to extract code from markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks, look for lines that look like code
    lines = text.strip().split('\n')
    code_lines = []
    
    for line in lines:
        # Skip explanation lines
        if re.match(r'^(here (is|are)|this code|output:|result=|assuming|if your dataset|to get|you can|this will|the pandas code|note:|explanation:|# explanation|import |from |def |class |print\()', line.lower()):
            continue
        
        # Look for code patterns
        if re.search(r'(data\[|data\.|\.groupby|\.sort_values|\.head|\.sum|\.max|\.idxmax|\.mean|\.count)', line):
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # If all else fails, return the text as is
    return text.strip()

def is_code_syntax_valid(code):
    # Check for common issues
    if re.search(r'\.\s*[a-zA-Z_]\w*$', code): 
        return False
    if re.search(r'\.\s*$', code): 
        return False
    if 'argmax()' in code and 'idxmax()' not in code:
        return False
    return True

def auto_correct_code(code):
    original_code = code
    
    # Common fixes
    fixes = [
        (r'\.argmax\(\)', '.idxmax()'),
        (r'\.name\(0\)', ''),
        (r'\.reset_index\(\)\.\s*(\w+)', r".reset_index()['\1']"),
        (r'data\.data\b', 'data'),  # Fix data.data references
        (r'print\(([^)]+)\)', r'\1'),  # Remove print statements for cleaner output
    ]
    
    for pattern, replacement in fixes:
        code = re.sub(pattern, replacement, code)
    
    if code != original_code:
        print(f"✅ Auto-corrected code:\n{code}")
    
    return code

def execute_code_safely(code):
    """Safely execute pandas code with limited scope"""
    try:
        # Create a safe environment for execution
        local_vars = {'data': data, 'pd': pd}
        
        # Check if it's an expression or statement
        if '=' in code or '\n' in code:
            # It's a statement (multiple lines or assignment)
            exec(code, local_vars)
            # Try to return the result if it exists
            if 'result' in local_vars:
                return local_vars['result']
            else:
                # Try to find the last expression
                lines = code.strip().split('\n')
                last_line = lines[-1].strip()
                if not last_line.startswith('#') and not last_line.startswith('print'):
                    return eval(last_line, local_vars)
                return "Code executed successfully but no explicit result to display."
        else:
            # It's a simple expression
            return eval(code, local_vars)
            
    except Exception as e:
        print(f"❌ Error executing code: {e}")
        raise e

def execute_code_with_retries(code, user_query, retries=2):
    attempt = 0
    while attempt <= retries:
        code = auto_correct_code(code)
        if not code.strip() or code.lower() == 'error':
            return "❌ No valid code generated. Retry."
        
        if not is_code_syntax_valid(code):
            print("❌ Detected likely syntax issue in generated code. Asking LLM again...")
            raw_output = ask_llm(f"The previous code had syntax errors. Fix it. Query: {user_query}")
            print("\n🪵 Raw LLM Output (Retry):\n", raw_output)
            code = extract_code(raw_output)
            print(f"\n🧠 Extracted Code (Retry):\n{code}")
            attempt += 1
            continue
        
        try:
            print(f"\n🔧 Running Code:\n{code}")
            result = execute_code_safely(code)
            
            if result is None:
                return "❌ Code executed but returned no result."
            
            if isinstance(result, (pd.Series, pd.DataFrame, list)) and len(result) == 0:
                return "❌ No matching records found."
            
            # Format the result for better display
            if isinstance(result, pd.DataFrame):
                return result.head(10)  # Limit large DataFrames
            elif isinstance(result, pd.Series):
                return result.head(20)  # Limit large Series
            else:
                return result
                
        except Exception as e:
            print(f"❌ Error executing code: {e}")
            print(traceback.format_exc())
            attempt += 1
            if attempt <= retries:
                raw_output = ask_llm(f"The previous code had execution error '{e}'. Fix it. Query: {user_query}")
                print("\n🪵 Raw LLM Output (Execution Retry):\n", raw_output)
                code = extract_code(raw_output)
                print(f"\n🧠 Extracted Code (Execution Retry):\n{code}")
            else:
                return f"❌ Failed after multiple attempts. Error: {e}"
    
    return "❌ Failed after multiple attempts."

def main():
    global data
    print("\n🤖 Ollama LLM NLP Query Assistant (MongoDB mode) is ready!")
    
    # Load data
    data = connect_to_mongodb()
    print(f"📊 Loaded {len(data)} records")
    print(f"📋 Columns: {list(data.columns)}")
    
    if not check_ollama_status():
        print("❌ Please start 'ollama serve' in another terminal before running this.")
        # Continue anyway for testing
        print("🔄 Continuing with sample mode...")
    
    while True:
        user_query = input("\n🗣️ Enter your query (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']:
            print("👋 Exiting. Goodbye!")
            break
        
        if not user_query.strip():
            print("⚠️ Please enter a valid query.")
            continue
        
        print("\n👉 Asking Ollama to generate the code...")
        raw_output = ask_llm(user_query)
        print("\n🪵 Raw LLM Output:\n", raw_output)
        
        code = extract_code(raw_output)
        print(f"\n🧠 Extracted Code:\n{code}")
        
        if not code.strip():
            print("❌ No code could be extracted from the response.")
            continue
            
        output = execute_code_with_retries(code, user_query)
        print(f"\n📄 Result:\n{output}")

if __name__ == "__main__":
    main()
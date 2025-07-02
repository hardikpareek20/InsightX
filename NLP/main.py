import ollama
import pandas as pd
import re
import requests
import os


# ✅ Load dataset from InsightX/data/raw/
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'raw', 'ecommerce_data.json')

data = pd.read_json(data_path)

# ✅ Preprocess
data['TotalSales'] = data['Quantity'] * data['Price']
data['Category_clean'] = data['Category'].str.strip().str.lower()
data['ProductName_clean'] = data['ProductName'].str.strip().str.lower()

# ✅ Ollama Health Check
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("✅ Ollama is running and API is responding!")
            return True
        else:
            print(f"⚠️ Ollama responded unexpectedly: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ollama is NOT running. Error: {e}")
        return False


# ✅ Schema Prompt
def get_schema_description():
    return f"""
The dataset 'data' has the following columns:
- OrderID (integer)
- CustomerName (string)
- ProductName (string)
- Category (string)
- Quantity (integer)
- Price (float)
- OrderDate (date string in YYYY-MM-DD)
- PaymentMethod (string)

Derived columns:
- TotalSales = Quantity * Price
- Category_clean = lowercase version of Category
- ProductName_clean = lowercase version of ProductName

⚠️ Important Rules:
- Text matching → use Category_clean/ProductName_clean (lowercase).
- For 'max' type queries use → .idxmax() not .argmax().
- ❌ Never use broken syntax like .reset_index(). CustomerName.
- ❌ Never use .name(0) — it's invalid.
- ✅ Return ONLY executable pandas code — no explanation, no markdown.

Example:
data.groupby('CustomerName')['Quantity'].sum().idxmax()
"""


# ✅ Ask Ollama
def ask_llm(user_query):
    prompt = get_schema_description() + f"\nNow write pandas code for this query:\n'{user_query}'\n"
    try:
        response = ollama.chat(
            model='mistral',
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"❌ Ollama API error: {e}"


# ✅ Extract Code
def extract_code(text):
    if not text.strip():
        return ""

    code_block = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    if code_block:
        return code_block[0].strip()

    lines = text.strip().split('\n')
    code_lines = [
        line for line in lines
        if not line.lower().startswith((
            'here is', 'this code', 'output:', 'result=', 'assuming',
            'if your dataset', 'to get', 'you can', 'this will', 'the pandas code'
        )) and line.strip() != ''
    ]

    final_code = ' '.join(code_lines).strip()
    final_code = re.sub(' +', ' ', final_code)

    return final_code


# ✅ Code Syntax Validator
def is_code_syntax_valid(code):
    if re.search(r'\.\s*[a-zA-Z_]\w*$', code):
        return False
    if re.search(r'\.\s*$', code):
        return False
    if not any(op in code for op in ['=', '(', ')', '[', ']']):
        return False
    return True


# ✅ Code Auto-Corrector
def auto_correct_code(code):
    original_code = code
    if '.argmax()' in code:
        print("⚠️ Detected .argmax(), auto-correcting to .idxmax()")
        code = code.replace('.argmax()', '.idxmax()')

    if '.name(0)' in code:
        print("⚠️ Detected .name(0), removing as invalid")
        code = code.replace('.name(0)', '')

    if '.reset_index(). CustomerName' in code:
        print("⚠️ Detected broken chaining .reset_index(). CustomerName, auto-fixing")
        code = code.replace('.reset_index(). CustomerName', ".reset_index()['CustomerName']")

    if code != original_code:
        print(f"✅ Auto-corrected code:\n{code}")

    return code


# ✅ Code Executor with Retries
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
            result = eval(code)

            if isinstance(result, (pd.Series, pd.DataFrame, list)) and len(result) == 0:
                return "❌ No matching records found."

            return result

        except Exception as e:
            print(f"❌ Error executing code: {e}")
            attempt += 1
            raw_output = ask_llm(f"The previous code had execution error '{e}'. Fix it. Query: {user_query}")
            print("\n🪵 Raw LLM Output (Execution Retry):\n", raw_output)
            code = extract_code(raw_output)
            print(f"\n🧠 Extracted Code (Execution Retry):\n{code}")

    return "❌ Failed after multiple attempts."


# ✅ Main Loop
def main():
    print("\n🤖 Ollama LLM NLP Query Assistant is ready!")

    if not check_ollama_status():
        print("❌ Please start 'ollama serve' in another terminal before running this.")
        exit()

    while True:
        user_query = input("\n🗣️ Enter your query (or 'exit'): ")

        if user_query.lower() in ['exit', 'quit']:
            print("👋 Exiting. Goodbye!")
            break

        print("\n👉 Asking Ollama to generate the code...")

        raw_output = ask_llm(user_query)
        print("\n🪵 Raw LLM Output:\n", raw_output)

        code = extract_code(raw_output)
        print(f"\n🧠 Extracted Code:\n{code}")

        output = execute_code_with_retries(code, user_query)
        print(f"\n📄 Result:\n{output}")


if __name__ == "__main__":
    main()

import ollama
import pandas as pd
import re
import requests
import os
import time
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class OptimizedDataProcessor:
    def __init__(self):
        self.data = None
        self.data_info = {}
        self.query_cache = {}
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with optimizations"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', 'data', 'raw', 'ecommerce_data.json')
            
            # Try JSON first, fallback to CSV if needed
            try:
                self.data = pd.read_json(data_path)
            except (FileNotFoundError, ValueError):
                # Fallback to CSV if JSON doesn't exist
                csv_path = os.path.join(current_dir, 'ecommerce_data.csv')
                self.data = pd.read_csv(csv_path)
                logger.info("Loaded data from CSV file")
            
            # Optimize data types for memory efficiency
            self.data = self._optimize_dtypes(self.data)
            
            # Enhanced preprocessing
            self.data['TotalSales'] = self.data['Quantity'] * self.data['Price']
            self.data['Category_clean'] = self.data['Category'].str.strip().str.lower()
            self.data['ProductName_clean'] = self.data['ProductName'].str.strip().str.lower()
            self.data['CustomerName_clean'] = self.data['CustomerName'].str.strip().str.lower()
            
            # Convert date column with error handling
            if 'OrderDate' in self.data.columns:
                self.data['OrderDate'] = pd.to_datetime(self.data['OrderDate'], errors='coerce', dayfirst=True)
                self.data['Year'] = self.data['OrderDate'].dt.year
                self.data['Month'] = self.data['OrderDate'].dt.month
                self.data['Quarter'] = self.data['OrderDate'].dt.quarter
            
            # Pre-calculate commonly used aggregations
            self._precompute_aggregations()
            
            # Store data information for schema description
            self._analyze_data_structure()
            
            logger.info(f"Successfully loaded and preprocessed {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better memory usage and performance"""
        # Convert object columns to categories if they have limited unique values
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['OrderDate'] and df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
        
        return df
    
    def _precompute_aggregations(self):
        """Pre-compute commonly used aggregations for faster query processing"""
        try:
            self.aggregations = {
                'total_sales_by_customer': self.data.groupby('CustomerName')['TotalSales'].sum(),
                'total_quantity_by_customer': self.data.groupby('CustomerName')['Quantity'].sum(),
                'sales_by_category': self.data.groupby('Category_clean')['TotalSales'].sum(),
                'sales_by_product': self.data.groupby('ProductName_clean')['TotalSales'].sum(),
                'order_count_by_customer': self.data.groupby('CustomerName').size(),
                'avg_order_value': self.data.groupby('OrderID')['TotalSales'].sum().mean(),
            }
            
            if 'OrderDate' in self.data.columns:
                self.aggregations.update({
                    'monthly_sales': self.data.groupby(['Year', 'Month'])['TotalSales'].sum(),
                    'quarterly_sales': self.data.groupby(['Year', 'Quarter'])['TotalSales'].sum(),
                })
            
            logger.info("Pre-computed aggregations successfully")
        except Exception as e:
            logger.warning(f"Could not pre-compute all aggregations: {e}")
            self.aggregations = {}
    
    def _analyze_data_structure(self):
        """Analyze data structure for better schema descriptions"""
        self.data_info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns},
            'sample_values': {col: list(self.data[col].dropna().unique()[:5]) for col in self.data.columns},
            'null_counts': self.data.isnull().sum().to_dict(),
        }


class EnhancedOllamaClient:
    def __init__(self, model='mistral', timeout=60, max_retries=3):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_cache = {}
        
    @lru_cache(maxsize=100)
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running with caching"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                logger.info(f"✅ Ollama is running. Available models: {available_models}")
                
                # Check if our preferred model is available
                if not any(self.model in model for model in available_models):
                    logger.warning(f"⚠ Model '{self.model}' not found. Using first available model.")
                    if available_models:
                        self.model = available_models[0].split(':')[0]
                
                return True
            else:
                logger.warning(f"⚠ Ollama responded unexpectedly: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Ollama is NOT running. Error: {e}")
            return False
    
    def ask_llm(self, prompt: str, use_cache: bool = True) -> str:
        """Enhanced LLM interaction with caching and retry logic"""
        # Check cache first
        if use_cache and prompt in self.request_cache:
            logger.info("Using cached response")
            return self.request_cache[prompt]
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Sending request to Ollama (attempt {attempt + 1})")
                start_time = time.time()
                
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        'temperature': 0.1,  # Lower temperature for more consistent code generation
                        'top_p': 0.9,
                        'num_predict': 512,  # Limit response length for faster processing
                    }
                )
                
                response_time = time.time() - start_time
                logger.info(f"Response received in {response_time:.2f} seconds")
                
                result = response['message']['content'].strip()
                
                # Cache successful responses
                if use_cache:
                    self.request_cache[prompt] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return f"❌ Ollama API error after {self.max_retries} attempts: {e}"
                time.sleep(2 ** attempt)  # Exponential backoff


class SmartQueryProcessor:
    def __init__(self, data_processor: OptimizedDataProcessor, ollama_client: EnhancedOllamaClient):
        self.data_processor = data_processor
        self.ollama_client = ollama_client
        self.execution_cache = {}
        
    def get_enhanced_schema_description(self) -> str:
        """Generate comprehensive schema description"""
        data_info = self.data_processor.data_info
        
        schema_desc = f"""
The dataset 'data' contains {data_info['shape'][0]} records with {data_info['shape'][1]} columns:

MAIN COLUMNS:
- OrderID (integer): Unique order identifier
- CustomerName (string): Customer identifier
- ProductName (string): Product name
- Category (string): Product category
- Quantity (integer): Number of items ordered  
- Price (float): Unit price per item
- OrderDate (datetime): Order date in YYYY-MM-DD format
- PaymentMethod (string): Payment method used

DERIVED COLUMNS:
- TotalSales = Quantity * Price (calculated revenue)
- Category_clean = lowercase Category for text matching
- ProductName_clean = lowercase ProductName for text matching  
- CustomerName_clean = lowercase CustomerName for text matching
- Year, Month, Quarter = extracted from OrderDate

SAMPLE DATA INSIGHTS:
- Categories: {', '.join(map(str, data_info['sample_values'].get('Category', [])))[:100]}
- Products: {', '.join(map(str, data_info['sample_values'].get('ProductName', [])))[:100]}
- Payment Methods: {', '.join(map(str, data_info['sample_values'].get('PaymentMethod', [])))[:50]}

PRE-COMPUTED AGGREGATIONS AVAILABLE:
You can directly use these for faster queries:
- processor.aggregations['total_sales_by_customer']
- processor.aggregations['sales_by_category'] 
- processor.aggregations['sales_by_product']
- processor.aggregations['order_count_by_customer']

OPTIMIZATION RULES:
✅ Use .idxmax() instead of .argmax()
✅ Use _clean columns for text matching (e.g., Category_clean, ProductName_clean)
✅ For aggregations, prefer groupby operations
✅ Use .loc[] for filtering instead of boolean indexing when possible
✅ Consider using pre-computed aggregations when available

❌ NEVER use .name(0), .argmax(), or broken chaining like .reset_index(). CustomerName
❌ Avoid complex string operations on large datasets
❌ Don't use deprecated pandas syntax

RETURN REQUIREMENTS:
- Return ONLY executable pandas code
- No markdown, no explanations
- Code should be production-ready and optimized
- Handle edge cases (empty results, missing values)

PERFORMANCE TIPS:
- Use vectorized operations
- Leverage pre-computed aggregations when possible
- Filter data early to reduce processing load
"""
        return schema_desc
    
    def extract_and_validate_code(self, text: str) -> Tuple[str, bool]:
        """Enhanced code extraction and validation"""
        if not text.strip():
            return "", False
        
        # Remove common non-code patterns
        text = re.sub(r'```python\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Extract code blocks
        code_patterns = [
            r'```(?:python)?\s*(.*?)```',  # Code blocks
            r'`([^`]+)`',  # Inline code
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = matches[0].strip()
                # If multi-line code, try to extract the final expression
                if '\n' in code:
                    lines = [line.strip() for line in code.split('\n') if line.strip()]
                    # Look for the last line that looks like a return expression
                    for line in reversed(lines):
                        if not line.startswith(('import', '#', 'print')) and '=' not in line:
                            code = line
                            break
                    else:
                        # If no good final line, try to convert multi-line to single expression
                        if len(lines) == 2 and 'idxmax()' in lines[0]:
                            code = lines[0]  # Use the idxmax line
                
                if self._is_valid_pandas_code(code):
                    return code, True
        
        # Fallback: extract from lines
        lines = text.strip().split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            if (line and 
                not line.lower().startswith(('here', 'this', 'output:', 'result:', 'to get', 'you can', 'the pandas', 'import', '#')) and
                ('data' in line or 'processor.aggregations' in line or any(op in line for op in ['=', '(', ')', '[', ']']))):
                # Skip assignment lines, prefer expressions
                if '=' in line and not line.startswith('data[') and not line.startswith('processor'):
                    continue
                code_lines.append(line)
        
        if code_lines:
            # Prefer single expressions over assignments
            for line in code_lines:
                if '=' not in line and ('idxmax()' in line or 'sum()' in line or 'max()' in line):
                    return line, self._is_valid_pandas_code(line)
            
            # Fallback to first valid line
            for line in code_lines:
                if self._is_valid_pandas_code(line):
                    return line, True
        
        return text.strip(), self._is_valid_pandas_code(text.strip())
    
    def _is_valid_pandas_code(self, code: str) -> bool:
        """Enhanced code validation"""
        if not code.strip():
            return False
            
        # Check for obvious syntax issues
        invalid_patterns = [
            r'\.\s*[a-zA-Z_]\w*\(',  # Hanging method calls - FIXED
            r'\.name\(0\)',  # Invalid .name(0)
            r'\.argmax\(\)',  # Should be idxmax
            r'\.reset_index\(\)\.\s*[A-Z]',  # Broken chaining
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, code):
                return False
        
        # Must contain valid operations - accept either data or processor references
        has_data_ref = bool(re.search(r'\bdata\b', code))
        has_processor_ref = bool(re.search(r'\bprocessor\b', code))
        has_operations = bool(re.search(r'[\.\[\(]', code))
        
        return (has_data_ref or has_processor_ref) and has_operations
    
    def auto_correct_code(self, code: str) -> str:
        """Enhanced auto-correction with more patterns"""
        original_code = code
        corrections = [
            (r'\.argmax\(\)', '.idxmax()', "Corrected .argmax() to .idxmax()"),
            (r'\.name\(0\)', '', "Removed invalid .name(0)"),
            (r'\.reset_index\(\)\.\s*([A-Z]\w+)', r".reset_index()['\1']", "Fixed broken chaining"),
            (r'data\.groupby\((.*?)\)\.sum\(\)\.max\(\)', r'data.groupby(\1).sum().idxmax()', "Fixed max to idxmax on grouped data"),
            (r'\bCategory\b(?!_clean)', 'Category_clean', "Use cleaned category column"),
            (r'\bProductName\b(?!_clean)', 'ProductName_clean', "Use cleaned product column"),
            # Fix processor access issues
            (r'processor\[processor\[', 'data[data[', "Fixed processor DataFrame access"),
            (r'processor\.aggregations\[', 'processor.aggregations[', "Keep aggregation access"),
        ]
        
        for pattern, replacement, message in corrections:
            if re.search(pattern, code):
                code = re.sub(pattern, replacement, code)
                logger.info(f"⚠ {message}")
        
        if code != original_code:
            logger.info(f"✅ Auto-corrected code:\n{code}")
        
        return code
    
    def execute_code_safely(self, code: str, context: Dict[str, Any]) -> Any:
        """Safely execute code with proper context and error handling"""
        try:
            # Create execution context
            exec_context = {
                'data': context['data'],
                'processor': context['processor'],
                'pd': pd,
                'np': np,
                're': re,
                **context.get('additional_context', {})
            }
            
            # Execute code
            result = eval(code, exec_context)
            
            # Handle different result types
            if isinstance(result, (pd.Series, pd.DataFrame)):
                if len(result) == 0:
                    return "❌ No matching records found."
                elif len(result) > 100:  # Limit large results
                    logger.info(f"Large result set ({len(result)} items), showing top 20")
                    return result.head(20)
                    
            return result
            
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            raise e
    
    def process_query_with_retries(self, user_query: str, max_retries: int = 3) -> Any:
        """Process query with intelligent retries and caching"""
        # Check execution cache
        if user_query in self.execution_cache:
            logger.info("Using cached execution result")
            return self.execution_cache[user_query]
        
        context = {
            'data': self.data_processor.data,
            'processor': self.data_processor,
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing query attempt {attempt + 1}: {user_query[:50]}...")
                
                # Generate prompt
                schema_desc = self.get_enhanced_schema_description()
                prompt = f"{schema_desc}\n\nQuery: '{user_query}'\n\nGenerate optimized pandas code:"
                
                # Get LLM response
                raw_output = self.ollama_client.ask_llm(prompt)
                logger.info(f"Raw LLM Output:\n{raw_output}")
                
                # Extract and validate code
                code, is_valid = self.extract_and_validate_code(raw_output)
                logger.info(f"Extracted Code:\n{code}")
                
                if not is_valid:
                    if attempt < max_retries - 1:
                        logger.warning("Invalid code detected, retrying...")
                        continue
                    else:
                        return "❌ Could not generate valid code after multiple attempts."
                
                # Auto-correct common issues
                code = self.auto_correct_code(code)
                
                # Execute code
                result = self.execute_code_safely(code, context)
                
                # Cache successful results
                self.execution_cache[user_query] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Generate error-specific retry prompt
                    error_prompt = f"""
                    The previous code had error: {str(e)}
                    Original query: {user_query}
                    Failed code: {code if 'code' in locals() else 'N/A'}
                    
                    Generate corrected pandas code that handles this error:
                    """
                    continue
                else:
                    return f"❌ Failed after {max_retries} attempts. Last error: {str(e)}"
        
        return "❌ Unexpected error in query processing."


def main():
    """Enhanced main function with better error handling and user experience"""
    print("🚀 Starting Enhanced Ollama Query Assistant...")
    
    try:
        # Initialize components
        print("📊 Loading and preprocessing data...")
        data_processor = OptimizedDataProcessor()
        
        print("🤖 Initializing Ollama client...")
        ollama_client = EnhancedOllamaClient()
        
        if not ollama_client.check_ollama_status():
            print("❌ Please start 'ollama serve' in another terminal before running this.")
            return
        
        print("⚙️ Setting up query processor...")
        query_processor = SmartQueryProcessor(data_processor, ollama_client)
        
        print("\n✅ Enhanced Ollama Query Assistant is ready!")
        print(f"📈 Loaded {len(data_processor.data)} records with {len(data_processor.data.columns)} columns")
        print("💡 Try queries like:")
        print("  - 'Which customer spent the most?'")
        print("  - 'Show top 5 products by sales'")
        print("  - 'Monthly sales trend for Electronics category'")
        print("  - 'Average order value by payment method'")
        
        query_count = 0
        start_time = time.time()
        
        while True:
            try:
                user_query = input(f"\n🗣 Query #{query_count + 1} (or 'exit', 'stats', 'clear'): ")
                
                if user_query.lower() in ['exit', 'quit']:
                    elapsed = time.time() - start_time
                    print(f"👋 Session complete! Processed {query_count} queries in {elapsed:.1f} seconds.")
                    break
                
                if user_query.lower() == 'stats':
                    print(f"📊 Session Stats:")
                    print(f"  - Queries processed: {query_count}")
                    print(f"  - Cache hits: {len(query_processor.execution_cache)}")
                    print(f"  - Session time: {(time.time() - start_time):.1f}s")
                    continue
                
                if user_query.lower() == 'clear':
                    query_processor.execution_cache.clear()
                    print("🧹 Cleared query cache")
                    continue
                
                if not user_query.strip():
                    continue
                
                # Process query
                query_start = time.time()
                result = query_processor.process_query_with_retries(user_query)
                query_time = time.time() - query_start
                
                print(f"\n📄 Result (processed in {query_time:.2f}s):")
                print(result)
                
                query_count += 1
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"❌ An unexpected error occurred: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}")
        print(f"💥 Fatal error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
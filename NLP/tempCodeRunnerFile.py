current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'raw', 'ecommerce_data.json')

data = pd.read_json(data_path)

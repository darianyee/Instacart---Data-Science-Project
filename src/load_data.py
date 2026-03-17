from pathlib import Path
import pandas as pd

# 1. Get the path of THIS file (src/load_data.py)
# 2. .resolve to get the full path
# 3. .parent to go to src folder and .parent again to go to Project 1 - InstaCart folder
# 4. Now we point to the 'dataset' folder starting from the Project root
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "dataset"

def get_data():
    orders = pd.read_csv(DATA_PATH / "orders.csv")
    order_products_prior = pd.read_csv(DATA_PATH / "order_products__prior.csv")
    order_products_train = pd.read_csv(DATA_PATH / "order_products__train.csv")
    products = pd.read_csv(DATA_PATH / "products.csv")
    aisles = pd.read_csv(DATA_PATH / "aisles.csv")
    departments = pd.read_csv(DATA_PATH / "departments.csv")
    return orders, order_products_prior, order_products_train, products, aisles, departments
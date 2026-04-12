import os
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
PRODUCTS_FILE = Path(__file__).resolve().parent.parent / "products.json"

SUGGESTIONS = [
    "lightweight laptop for students",
    "budget phone with good camera",
    "winter jacket for travel",
    "books for self improvement",
]

CATEGORY_COLORS = {
    "Electronics": "#0ea5e9",
    "Clothing": "#fb7185",
    "Footwear": "#f59e0b",
    "Books": "#22c55e",
    "Home Appliances": "#14b8a6",
}

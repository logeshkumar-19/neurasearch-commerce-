import json
from pathlib import Path

from endee import Endee, Precision
from sentence_transformers import SentenceTransformer


BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
PRODUCTS_FILE = Path(__file__).with_name("products.json")


def connect_client() -> Endee:
    """Create and configure the Endee client."""
    client = Endee()
    client.set_base_url(BASE_URL)
    return client


def ensure_index(client: Endee):
    """Create the products index if it does not already exist, then return it."""
    try:
        existing_indexes = client.list_indexes()
        if isinstance(existing_indexes, dict):
            index_items = existing_indexes.get("indexes", [])
        elif isinstance(existing_indexes, list):
            index_items = existing_indexes
        else:
            index_items = []

        existing_names = {
            item.get("name") if isinstance(item, dict) else str(item)
            for item in index_items
        }
    except Exception:
        existing_names = set()

    if INDEX_NAME not in existing_names:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8,
        )

    return client.get_index(name=INDEX_NAME)


def load_products() -> list[dict]:
    """Read the product catalog from disk and validate the expected item count."""
    with PRODUCTS_FILE.open("r", encoding="utf-8") as file:
        products = json.load(file)

    if len(products) != 50:
        raise ValueError(f"Expected 50 products in {PRODUCTS_FILE.name}, found {len(products)}.")

    return products


def build_embedding_text(product: dict) -> str:
    """Combine searchable product fields into a single embedding text."""
    return (
        f"Name: {product['name']}. "
        f"Category: {product['category']}. "
        f"Description: {product['description']}. "
        f"Price: Rs. {product['price']}."
    )


def main() -> None:
    """Load products, create embeddings, and upsert them into Endee."""
    try:
        print("Connecting to Endee...")
        client = connect_client()
        index = ensure_index(client)

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Reading product data from {PRODUCTS_FILE.name}...")
        products = load_products()

        for product in products:
            try:
                embedding = model.encode(build_embedding_text(product), normalize_embeddings=True)

                index.upsert(
                    [
                        {
                            "id": product["id"],
                            "vector": embedding.tolist(),
                            "meta": {
                                "name": product["name"],
                                "description": product["description"],
                                "category": product["category"],
                                "price": product["price"],
                            },
                            "filter": {"category": product["category"]},
                        }
                    ]
                )

                print(f"Successfully upserted product {product['id']}: {product['name']}")
            except Exception as product_error:
                print(
                    f"Failed to process product {product.get('id', 'unknown')}: "
                    f"{product_error}"
                )

        print("Product data loading completed.")

    except FileNotFoundError as error:
        print(f"File error: {error}")
    except json.JSONDecodeError as error:
        print(f"JSON parsing error: {error}")
    except Exception as error:
        print(f"Unexpected error while loading data: {error}")


if __name__ == "__main__":
    main()



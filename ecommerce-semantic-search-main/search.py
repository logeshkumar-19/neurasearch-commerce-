from endee import Endee
from sentence_transformers import SentenceTransformer


BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def connect_client() -> Endee:
    """Create an Endee client configured for the local products index."""
    client = Endee()
    client.set_base_url(BASE_URL)
    return client


def similarity_to_percent(similarity_value) -> float:
    """Convert similarity values from Endee into percentages."""
    try:
        similarity = float(similarity_value)
        similarity = max(0.0, min(1.0, similarity))
        return round(similarity * 100, 2)
    except Exception:
        return 0.0


def search_products(query, top_k=5):
    """Run semantic search directly against Endee and return formatted results."""
    try:
        if not query or not query.strip():
            return []

        model = SentenceTransformer(EMBEDDING_MODEL)
        query_embedding = model.encode(query.strip(), normalize_embeddings=True).tolist()

        client = connect_client()
        index = client.get_index(INDEX_NAME)
        results = index.query(vector=query_embedding, top_k=top_k, ef=128)

        formatted_results = []
        for item in results:
            meta = item.get("meta", {})
            formatted_results.append(
                {
                    "id": item.get("id", ""),
                    "name": meta.get("name", "Unknown Product"),
                    "category": meta.get("category", "Unknown"),
                    "price": meta.get("price", 0),
                    "description": meta.get("description", ""),
                    "similarity_score": similarity_to_percent(item.get("similarity", 0)),
                }
            )

        return formatted_results

    except Exception as error:
        print(f"An error occurred during search: {error}")
        return []


if __name__ == "__main__":
    sample_results = search_products("lightweight laptop for students")
    for result in sample_results:
        print(result)

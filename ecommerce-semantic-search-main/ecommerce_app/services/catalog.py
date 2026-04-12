import json
from typing import Any

import numpy as np
import streamlit as st
from endee import Endee
from sentence_transformers import SentenceTransformer

from ecommerce_app.config import BASE_URL, EMBEDDING_MODEL, INDEX_NAME, PRODUCTS_FILE

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@st.cache_data(show_spinner=False)
def load_products() -> list[dict[str, Any]]:
    with PRODUCTS_FILE.open("r", encoding="utf-8") as file:
        products = json.load(file)

    required = {"id", "name", "category", "price", "description"}
    cleaned = []
    for i, product in enumerate(products, start=1):
        missing = required.difference(product)
        if missing:
            raise ValueError(f"Product #{i} is missing: {', '.join(sorted(missing))}")
        cleaned.append(
            {
                "id": str(product["id"]),
                "name": str(product["name"]),
                "category": str(product["category"]),
                "price": float(product["price"]),
                "description": str(product["description"]),
            }
        )
    return cleaned


def embed_text(product: dict[str, Any]) -> str:
    return (
        f"Product name: {product['name']}. Category: {product['category']}. "
        f"Price: {int(product['price'])} rupees. Description: {product['description']}"
    )


@st.cache_resource(show_spinner=False)
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def catalog_embeddings(products: list[dict[str, Any]]) -> list[list[float]]:
    texts = [embed_text(product) for product in products]
    return embedding_model().encode(texts, normalize_embeddings=True).tolist()


def encode_query(text: str) -> list[float]:
    return embedding_model().encode(text.strip(), normalize_embeddings=True).tolist()


def pct(score: float) -> float:
    return round(max(0.0, min(1.0, float(score))) * 100, 1)


def connect_client() -> Endee:
    client = Endee()
    client.set_base_url(BASE_URL)
    return client


def extract_matches(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("results", "matches", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


@st.cache_resource(show_spinner=False)
def vector_backend(products: list[dict[str, Any]], embeddings: list[list[float]]) -> dict[str, Any]:
    matrix = np.array(embeddings, dtype="float32")
    try:
        index = connect_client().get_index(name=INDEX_NAME)
        index.query(vector=embeddings[0], top_k=1, ef=64)
        return {"kind": "Endee", "index": index}
    except Exception:
        pass

    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return {"kind": "FAISS", "index": index, "matrix": matrix}

    return {"kind": "NumPy", "matrix": matrix}


def local_search(
    query_vector: list[float],
    products: list[dict[str, Any]],
    backend: dict[str, Any],
    top_k: int,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    query_np = np.array(query_vector, dtype="float32").reshape(1, -1)
    if backend["kind"] == "FAISS":
        scores, ids = backend["index"].search(query_np, min(top_k + 1, len(products)))
        ranked = zip(ids[0].tolist(), scores[0].tolist())
    else:
        scores = (backend["matrix"] @ query_np.T).reshape(-1)
        order = np.argsort(scores)[::-1][: min(top_k + 1, len(products))]
        ranked = ((int(i), float(scores[i])) for i in order)

    results = []
    for i, score in ranked:
        if i < 0 or i >= len(products):
            continue
        product = dict(products[i])
        if exclude_id and product["id"] == exclude_id:
            continue
        product["similarity_percent"] = pct(score)
        product["similarity_score"] = float(score)
        results.append(product)
        if len(results) >= top_k:
            break
    return results


def semantic_search(query: str, top_k: int = 6) -> tuple[list[dict[str, Any]], str | None]:
    if not query or not query.strip():
        return [], "Please enter a search query to find products."
    try:
        products = load_products()
        embeddings = catalog_embeddings(products)
        backend = vector_backend(products, embeddings)
        query_vector = encode_query(query)
        if backend["kind"] == "Endee":
            raw = backend["index"].query(vector=query_vector, top_k=top_k, ef=128)
            results = []
            for item in extract_matches(raw):
                meta = item.get("meta", {})
                results.append(
                    {
                        "id": str(item.get("id", meta.get("id", ""))),
                        "name": meta.get("name", "Unknown Product"),
                        "category": meta.get("category", "Unknown"),
                        "price": float(meta.get("price", 0)),
                        "description": meta.get("description", ""),
                        "similarity_percent": pct(item.get("similarity", 0)),
                        "similarity_score": float(item.get("similarity", 0)),
                    }
                )
            return results, None
        return local_search(query_vector, products, backend, top_k), None
    except Exception as error:
        return [], f"Search is temporarily unavailable: {error}"


def get_recommendations(product_id: str, top_k: int = 3) -> tuple[list[dict[str, Any]], str | None]:
    if not product_id:
        return [], "Select a product to generate recommendations."
    try:
        products = load_products()
        embeddings = catalog_embeddings(products)
        backend = vector_backend(products, embeddings)
        if backend["kind"] == "Endee":
            vector_record = backend["index"].get_vector(product_id)
            vector = vector_record.get("vector") if isinstance(vector_record, dict) else None
            if vector:
                raw = backend["index"].query(vector=vector, top_k=top_k + 1, ef=128)
                recommendations = []
                for item in extract_matches(raw):
                    if str(item.get("id")) == product_id:
                        continue
                    meta = item.get("meta", {})
                    recommendations.append(
                        {
                            "id": str(item.get("id", meta.get("id", ""))),
                            "name": meta.get("name", "Unknown Product"),
                            "category": meta.get("category", "Unknown"),
                            "price": float(meta.get("price", 0)),
                            "description": meta.get("description", ""),
                            "similarity_percent": pct(item.get("similarity", 0)),
                            "similarity_score": float(item.get("similarity", 0)),
                        }
                    )
                    if len(recommendations) >= top_k:
                        break
                return recommendations, None

        index = next((i for i, product in enumerate(products) if product["id"] == product_id), None)
        if index is None:
            return [], "The selected product vector could not be found."
        return local_search(embeddings[index], products, backend, top_k, exclude_id=product_id), None
    except Exception as error:
        return [], f"Recommendations are unavailable right now: {error}"

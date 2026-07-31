import html
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import numpy as np
import streamlit as st
from endee import Endee
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
PRODUCTS_FILE = Path(__file__).with_name("products.json")
SUGGESTIONS = [
    "lightweight laptop for students",
    "budget phone with good camera",
    "winter jacket for travel",
    "books for self improvement",
]
COLORS = {
    "Electronics": "#0ea5e9",
    "Clothing": "#fb7185",
    "Footwear": "#f59e0b",
    "Books": "#22c55e",
    "Home Appliances": "#14b8a6",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap');
        :root{
            --primary:#0f766e;
            --primary-mid:#14b8a6;
            --primary-end:#06b6d4;
            --accent:#f59e0b;
            --accent-2:#fb7185;
            --success:#16a34a;
            --bg-1:#fff7ed;
            --bg-2:#fef3c7;
            --text-1:#1f2937;
            --text-2:#4b5563;
            --text-3:#9ca3af;
            --line:rgba(15,118,110,.18);
            --glass:rgba(255,255,255,.75);
            --glass-strong:rgba(255,255,255,.9);
            --shadow-lg:0 28px 70px rgba(15,118,110,.12);
            --shadow-md:0 16px 40px rgba(17,24,39,.08);
            --match-bg:rgba(15,118,110,.12);
            --match-text:#0f766e;
            --match-border:rgba(15,118,110,.25);
        }
        .stApp{
            background:
                radial-gradient(circle at top center, rgba(14,116,144,.12) 0%, rgba(14,116,144,0) 32%),
                linear-gradient(180deg,var(--bg-1) 0%,var(--bg-2) 100%);
            color:var(--text-1);
            font-family:"Manrope","Segoe UI",sans-serif;
            position:relative;
            overflow:hidden;
        }
        .stApp:before,
        .stApp:after{
            content:"";
            position:absolute;
            width:540px;
            height:540px;
            border-radius:50%;
            opacity:.25;
            pointer-events:none;
            z-index:0;
        }
        .stApp:before{
            top:-140px;
            right:-140px;
            background:radial-gradient(circle, rgba(15,118,110,.6) 0%, rgba(15,118,110,0) 70%);
        }
        .stApp:after{
            bottom:-220px;
            left:-180px;
            background:radial-gradient(circle, rgba(245,158,11,.55) 0%, rgba(245,158,11,0) 70%);
        }
        .block-container, .stApp > header, .stApp > div{
            position:relative;
            z-index:1;
        }
        [data-testid="stSidebar"],[data-testid="stSidebarNav"],[data-testid="collapsedControl"]{display:none!important}
        .block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1180px}
        .hero{
            text-align:center;
            padding:2.2rem 0 1.1rem;
            max-width:900px;
            margin:0 auto;
        }
        .eyebrow{
            display:inline-flex;
            align-items:center;
            gap:.4rem;
            padding:.45rem .95rem;
            border-radius:999px;
            background:rgba(15,118,110,.12);
            border:1px solid rgba(15,118,110,.2);
            color:var(--primary);
            font-size:.78rem;
            font-weight:700;
            letter-spacing:.06em;
            text-transform:uppercase
        }
        .hero-title{
            font-family:"Space Grotesk","Manrope",sans-serif;
            font-size:clamp(2.7rem,4vw,4.35rem);
            line-height:1.02;
            font-weight:800;
            letter-spacing:-.045em;
            margin:1.05rem 0 .7rem
        }
        .hero-copy{
            max-width:700px;
            margin:0 auto;
            color:var(--text-2);
            font-size:1.02rem;
            line-height:1.8
        }
        .search-shell{
            max-width:860px;
            margin:1.6rem auto 0;
            padding:1.1rem;
            background:var(--glass);
            backdrop-filter:blur(18px);
            border:1px solid rgba(255,255,255,.55);
            border-radius:28px;
            box-shadow:var(--shadow-lg)
        }
        .chip-caption{
            color:var(--text-3);
            font-size:.84rem;
            font-weight:600;
            margin:.8rem 0 .55rem;
            text-align:center
        }
        .panel{
            background:var(--glass);
            backdrop-filter:blur(18px);
            border:1px solid rgba(255,255,255,.55);
            border-radius:28px;
            box-shadow:var(--shadow-lg);
            padding:1.5rem;
            margin:1.35rem 0
        }
        .title{
            font-family:"Space Grotesk","Manrope",sans-serif;
            font-size:1.35rem;
            font-weight:760;
            letter-spacing:-.025em;
            margin-bottom:.28rem
        }
        .copy{
            color:var(--text-2);
            margin-bottom:1.05rem;
            line-height:1.65
        }
        .answer{
            background:var(--glass-strong);
            backdrop-filter:blur(14px);
            border:1px solid var(--line);
            border-radius:20px;
            padding:1.05rem 1.1rem;
            line-height:1.75;
            color:var(--text-1);
            box-shadow:0 12px 32px rgba(15,118,110,.08)
        }
        .card{
            background:rgba(255,255,255,.72);
            backdrop-filter:blur(18px);
            -webkit-backdrop-filter:blur(18px);
            border:1px solid rgba(255,255,255,.7);
            border-radius:22px;
            padding:1.25rem;
            box-shadow:var(--shadow-md);
            min-height:320px;
            transition:transform .24s ease,box-shadow .24s ease,border-color .24s ease
        }
        .card:hover{
            transform:translateY(-6px);
            box-shadow:0 24px 50px rgba(15,118,110,.16);
            border-color:rgba(15,118,110,.25)
        }
        .name{
            font-family:"Space Grotesk","Manrope",sans-serif;
            font-size:1.12rem;
            font-weight:760;
            line-height:1.45;
            letter-spacing:-.02em;
            margin-bottom:.95rem
        }
        .meta{
            display:flex;
            flex-wrap:wrap;
            gap:.55rem;
            margin-bottom:.95rem
        }
        .pill{
            display:inline-flex;
            align-items:center;
            border-radius:999px;
            padding:.4rem .82rem;
            font-size:.78rem;
            font-weight:700;
            border:1px solid transparent
        }
        .price{
            font-size:1.18rem;
            font-weight:780;
            color:var(--success);
            margin-bottom:.88rem
        }
        .body{
            color:var(--text-2);
            line-height:1.75;
            font-size:.95rem
        }
        .match{
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin:1.05rem 0 .45rem;
            color:var(--text-2);
            font-size:.82rem;
            font-weight:650
        }
        .bar{
            width:100%;
            height:10px;
            border-radius:999px;
            overflow:hidden;
            background:rgba(15,118,110,.12)
        }
        .fill{
            height:100%;
            border-radius:999px;
            background:linear-gradient(90deg,#0f766e 0%,#06b6d4 100%)
        }
        .section-heading{
            font-family:"Space Grotesk","Manrope",sans-serif;
            font-size:1.75rem;
            font-weight:780;
            letter-spacing:-.035em;
            margin:.25rem 0 1rem
        }
        .stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div{
            border-radius:18px!important;
            border:1px solid rgba(15,118,110,.2)!important;
            min-height:3.5rem;
            background:rgba(255,255,255,.94)!important;
            color:var(--text-1)!important;
            box-shadow:0 8px 24px rgba(17,24,39,.05)!important;
            transition:border-color .22s ease, box-shadow .22s ease!important
        }
        .stTextArea textarea{
            border-radius:22px!important;
            min-height:7rem!important
        }
        .stTextInput input:focus,.stTextArea textarea:focus{
            border-color:rgba(15,118,110,.45)!important;
            box-shadow:0 0 0 4px rgba(15,118,110,.16)!important
        }
        .stButton>button{
            border-radius:999px!important;
            min-height:3rem;
            padding:0 1.2rem!important;
            font-weight:700!important;
            border:1px solid rgba(15,118,110,.18)!important;
            color:var(--text-1)!important;
            background:rgba(255,255,255,.85)!important;
            transition:transform .22s ease,box-shadow .22s ease,filter .22s ease!important;
            box-shadow:0 10px 24px rgba(17,24,39,.05)
        }
        .stButton>button:hover{
            transform:translateY(-1px);
            box-shadow:0 14px 28px rgba(17,24,39,.08)!important
        }
        .stButton>button[kind="primary"]{
            background:linear-gradient(135deg,#0f766e 0%,#14b8a6 55%,#06b6d4 100%)!important;
            color:#fff!important;
            border-color:transparent!important
        }
        .stButton>button[kind="primary"]:hover{
            filter:brightness(.96)
        }
        .stAlert{
            border-radius:18px!important;
            border:1px solid var(--line)!important
        }
        .match-pill{
            background:var(--match-bg);
            color:var(--match-text);
            border-color:var(--match-border);
        }
        @media (max-width: 900px){
            .block-container{padding-left:1rem;padding-right:1rem}
            .panel{padding:1.1rem}
            .search-shell{padding:.9rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    """Run semantic search for the top matching products."""
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
    """Return similar products but exclude the selected one."""
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


@st.cache_resource(show_spinner=False)
def generator_bundle() -> dict[str, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    try:
        return {"mode": "pipeline", "generator": pipeline("text2text-generation", model=LLM_MODEL)}
    except Exception:
        return {
            "mode": "manual",
            "tokenizer": AutoTokenizer.from_pretrained(LLM_MODEL),
            "model": AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL),
        }


def generate_text(prompt: str, max_new_tokens: int = 180) -> str:
    try:
        bundle = generator_bundle()
        if bundle["mode"] == "pipeline":
            response = bundle["generator"](prompt, max_new_tokens=max_new_tokens, do_sample=False)
            return response[0]["generated_text"].strip() if response else ""
        inputs = bundle["tokenizer"](prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = bundle["model"].generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return bundle["tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def build_context(products: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {product['name']} | Category: {product['category']} | Price: Rs. {int(product['price'])} | Description: {product['description']}"
        for product in products
    )


def rag_answer(question: str, top_k: int = 3) -> dict[str, Any]:
    """Retrieve product context and generate a grounded answer."""
    if not question or not question.strip():
        return {"answer": "", "sources": [], "error": "Ask a product question to use RAG."}
    sources, error = semantic_search(question, top_k=top_k)
    if error:
        return {"answer": "", "sources": [], "error": error}
    if not sources:
        return {
            "answer": "I could not find relevant products in the catalog for that question.",
            "sources": [],
            "error": None,
        }
    prompt = (
        "You are a concise ecommerce AI assistant. Use only the product context below. "
        "Answer clearly, mention the best-fit products, and explain your reasoning in shopper-friendly language.\n\n"
        f"Question: {question.strip()}\n\nContext:\n{build_context(sources)}\n\nAnswer:"
    )
    answer = generate_text(prompt)
    if not answer:
        best = sources[0]
        answer = (
            f"The strongest match is {best['name']} because it aligns well with the question, belongs to "
            f"{best['category']}, and is priced at Rs. {int(best['price'])}. {best['description']}"
        )
    return {"answer": answer, "sources": sources, "error": None}


def compare_targets(user_input: str, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered = user_input.lower()
    explicit = [product for product in products if product["name"].lower() in lowered]
    if len(explicit) >= 2:
        return explicit[:2]
    if " vs " in lowered or "compare" in lowered:
        guesses, _ = semantic_search(user_input, top_k=2)
        return guesses[:2]
    return []


def detect_intent(user_input: str, selected_product_id: str | None, products: list[dict[str, Any]]) -> str:
    lowered = user_input.lower()
    if compare_targets(user_input, products):
        return "compare"
    if selected_product_id and any(word in lowered for word in ("why", "explain", "recommended")):
        return "explain"
    if any(word in lowered for word in ("best", "recommend", "suggest", "which should i buy")):
        return "recommend"
    return "guide"


def shopping_agent(user_input: str, selected_product_id: str | None = None) -> dict[str, Any]:
    """Agentic assistant that recommends, compares, and explains products."""
    if not user_input or not user_input.strip():
        return {"intent": "empty", "answer": "", "products": [], "error": "Tell the shopping agent what you need."}

    products = load_products()
    product_lookup = {product["id"]: product for product in products}
    intent = detect_intent(user_input, selected_product_id, products)

    if intent == "compare":
        compared = compare_targets(user_input, products)
        if len(compared) < 2:
            return {"intent": intent, "answer": "I could not identify two strong products to compare.", "products": compared, "error": None}
        prompt = (
            "You are an AI shopping agent. Compare the two products below for a buyer. "
            "Highlight ideal use case, pricing, and who should choose which one.\n\n"
            f"User request: {user_input.strip()}\n\nProducts:\n{build_context(compared)}\n\nComparison:"
        )
        answer = generate_text(prompt)
        if not answer:
            left, right = compared
            answer = (
                f"{left['name']} is stronger if you want {left['category'].lower()} value around Rs. {int(left['price'])}, "
                f"while {right['name']} is better if its strengths match your workflow more closely."
            )
        return {"intent": intent, "answer": answer, "products": compared, "error": None}

    if intent == "explain" and selected_product_id and selected_product_id in product_lookup:
        selected = product_lookup[selected_product_id]
        related, _ = get_recommendations(selected_product_id, top_k=2)
        context_products = [selected, *related]
        prompt = (
            "You are an AI shopping agent. Explain why the selected product is recommended. "
            "Mention its strengths, ideal buyer, and nearby alternatives.\n\n"
            f"User request: {user_input.strip()}\n\nContext:\n{build_context(context_products)}\n\nExplanation:"
        )
        answer = generate_text(prompt)
        if not answer:
            answer = (
                f"{selected['name']} is recommended because it fits the current intent well, sits at Rs. {int(selected['price'])}, "
                f"and stands out in {selected['category']}. {selected['description']}"
            )
        return {"intent": intent, "answer": answer, "products": context_products, "error": None}

    results, error = semantic_search(user_input, top_k=3)
    if error:
        return {"intent": intent, "answer": "", "products": [], "error": error}
    if not results:
        return {"intent": intent, "answer": "I could not find suitable products for that request in the current catalog.", "products": [], "error": None}
    prompt = (
        "You are an AI shopping agent. Understand the user intent, recommend the best product, "
        "mention one or two alternatives, and explain why the top pick is right.\n\n"
        f"User request: {user_input.strip()}\n\nCatalog context:\n{build_context(results)}\n\nRecommendation:"
    )
    answer = generate_text(prompt)
    if not answer:
        top = results[0]
        alt = results[1]["name"] if len(results) > 1 else "another catalog item"
        answer = (
            f"My top recommendation is {top['name']} because it best matches the request semantically, "
            f"offers a strong fit in {top['category']}, and is priced at Rs. {int(top['price'])}. "
            f"If you want an alternative, consider {alt}."
        )
    return {"intent": intent, "answer": answer, "products": results, "error": None}


def init_state() -> None:
    defaults = {
        "search_query": "",
        "selected_product_name": "",
        "rag_question": "",
        "agent_prompt": "",
        "search_results": [],
        "rag_result": None,
        "recommendations": [],
        "agent_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_search_query(query: str) -> None:
    st.session_state["search_query"] = query


def price_text(price: float) -> str:
    return f"Rs. {int(price):,}"


def category_badge(category: str) -> str:
    color = COLORS.get(category, "#4b5563")
    return f"<span class='pill' style='background:{color};color:white'>{category}</span>"


def product_card(product: dict[str, Any], show_match: bool = True) -> None:
    description = product["description"]
    if len(description) > 170:
        description = f"{description[:167].rstrip()}..."
    safe_name = html.escape(product["name"])
    safe_description = html.escape(description)
    safe_category = html.escape(product["category"])
    score = float(product.get("similarity_percent", 0.0))
    match_badge = (
        f"<span class='pill match-pill'>Match {score:.1f}%</span>"
        if show_match else ""
    )
    progress = (
        "<div class='match'><span>Semantic match</span>"
        f"<span>{score:.1f}%</span></div><div class='bar'><div class='fill' style='width:{score}%;'></div></div>"
        if show_match else ""
    )
    card_html = (
        "<div class='card'>"
        f"<div class='name'>{safe_name}</div>"
        "<div class='meta'>"
        f"{category_badge(safe_category)}"
        f"{match_badge}"
        "</div>"
        f"<div class='price'>{price_text(product['price'])}</div>"
        f"<div class='body'>{safe_description}</div>"
        f"{progress}"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def product_grid(products: list[dict[str, Any]], show_match: bool = True) -> None:
    cols = st.columns(3)
    for i, product in enumerate(products):
        with cols[i % 3]:
            product_card(product, show_match=show_match)


def top_picks(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = {"1", "2", "11", "23", "31", "47"}
    picks = [product for product in products if product["id"] in preferred]
    return picks[:6] if picks else products[:6]


def render_header(products: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI Product Discovery</div>
            <div class="hero-title">Smart Product Search</div>
            <div class="hero-copy">
                Discover products with a cleaner, smarter shopping interface powered by semantic search,
                grounded answers, and thoughtful AI recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search() -> None:
    st.markdown("<div class='search-shell'>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2.4, 1])
    with center:
        st.text_input("Search", key="search_query", label_visibility="collapsed", placeholder="Search for products by intent, budget, or use case")
    st.markdown("<div class='chip-caption'>Try a few popular searches</div>", unsafe_allow_html=True)
    chip_cols = st.columns(len(SUGGESTIONS))
    for col, suggestion in zip(chip_cols, SUGGESTIONS):
        with col:
            st.button(suggestion, use_container_width=True, on_click=set_search_query, args=(suggestion,))
    _, action, _ = st.columns([1.2, 1, 1.2])
    with action:
        if st.button("Search Products", type="primary", use_container_width=True):
            with st.spinner("Searching the product graph..."):
                results, error = semantic_search(st.session_state["search_query"], top_k=6)
            st.session_state["search_results"] = results
            if error:
                st.warning(error)
            elif not results:
                st.info("No results found. Try a broader query, category, or use case.")
    if st.session_state["search_results"]:
        st.markdown("<div class='section-heading'>Search Results</div>", unsafe_allow_html=True)
        product_grid(st.session_state["search_results"], show_match=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_top_picks(products: list[dict[str, Any]]) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Top Picks for You</div>", unsafe_allow_html=True)
    st.markdown("<div class='copy'>A curated starter shelf that makes the experience feel polished before the first search.</div>", unsafe_allow_html=True)
    product_grid(top_picks(products), show_match=False)
    st.markdown("</div>", unsafe_allow_html=True)


def render_rag() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Ask AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='copy'>Retrieve the most relevant products first, then answer using that catalog context only.</div>", unsafe_allow_html=True)
    st.text_input("RAG", key="rag_question", label_visibility="collapsed", placeholder="Example: What is the best laptop under 50000 for a student?")
    if st.button("Generate Answer", type="primary", use_container_width=False):
        with st.spinner("Retrieving products and generating answer..."):
            st.session_state["rag_result"] = rag_answer(st.session_state["rag_question"])
    result = st.session_state["rag_result"]
    if result:
        if result.get("error"):
            st.warning(result["error"])
        elif result.get("answer"):
            st.markdown(f"<div class='answer'>{result['answer']}</div>", unsafe_allow_html=True)
        if result.get("sources"):
            st.markdown("<div class='section-heading'>Retrieved Context</div>", unsafe_allow_html=True)
            product_grid(result["sources"], show_match=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_recommendations(products: list[dict[str, Any]]) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Similar Products</div>", unsafe_allow_html=True)
    st.markdown("<div class='copy'>Select a product and explore nearby alternatives without repeating the original item.</div>", unsafe_allow_html=True)
    names = sorted(product["name"] for product in products)
    if not st.session_state["selected_product_name"] and names:
        st.session_state["selected_product_name"] = names[0]
    st.selectbox("Pick", options=names, key="selected_product_name", label_visibility="collapsed")
    if st.button("Show Recommendations", type="primary", use_container_width=False):
        selected = next((product for product in products if product["name"] == st.session_state["selected_product_name"]), None)
        with st.spinner("Finding similar products..."):
            recommendations, error = get_recommendations(selected["id"] if selected else "", top_k=3)
        st.session_state["recommendations"] = recommendations
        if error:
            st.warning(error)
        elif not recommendations:
            st.info("No recommendations were found for the selected item.")
    if st.session_state["recommendations"]:
        product_grid(st.session_state["recommendations"], show_match=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_agent(products: list[dict[str, Any]]) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>AI Shopping Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='copy'>Ask for the best product, compare options, or understand why something fits your intent.</div>", unsafe_allow_html=True)
    st.text_area(
        "Agent",
        key="agent_prompt",
        label_visibility="collapsed",
        placeholder="Recommend the best winter jacket for travel | Compare AstraBook Student Laptop 14 vs VisionPad 11 Tablet | Why is this product recommended?",
        height=110,
    )
    if st.button("Run AI Agent", type="primary", use_container_width=False):
        selected = next((product for product in products if product["name"] == st.session_state["selected_product_name"]), None)
        with st.spinner("The AI agent is reasoning over your catalog..."):
            st.session_state["agent_result"] = shopping_agent(st.session_state["agent_prompt"], selected["id"] if selected else None)
    result = st.session_state["agent_result"]
    if result:
        if result.get("error"):
            st.warning(result["error"])
        elif result.get("answer"):
            st.markdown(f"<div class='answer'>{result['answer']}</div>", unsafe_allow_html=True)
        if result.get("products"):
            st.markdown("<div class='section-heading'>Agent-Supported Products</div>", unsafe_allow_html=True)
            product_grid(result["products"], show_match=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()
    init_state()
    try:
        products = load_products()
    except Exception as error:
        st.error(f"Unable to load the product catalog: {error}")
        return
    render_header(products)
    render_search()
    render_top_picks(products)
    left, right = st.columns([1.2, 1])
    with left:
        render_rag()
    with right:
        render_recommendations(products)
    render_agent(products)



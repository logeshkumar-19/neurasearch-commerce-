import json
import os
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from endee import Endee
from sentence_transformers import SentenceTransformer
from ecommerce_app.ui.app import main as smart_main


BASE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "products"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"
PRODUCTS_FILE = Path(__file__).with_name("products.json")
CATEGORY_COLORS = {
    "Electronics": "#2563eb",
    "Clothing": "#db2777",
    "Footwear": "#ea580c",
    "Books": "#059669",
    "Home Appliances": "#7c3aed",
}
CATEGORY_ICONS = {
    "Electronics": "💻",
    "Clothing": "🧥",
    "Footwear": "👟",
    "Books": "📚",
    "Home Appliances": "🏠",
}



def inject_custom_css() -> None:
    """Apply custom styling for a polished Streamlit experience."""
    st.markdown(
        """
        <style>
            :root {
                --blue-1: #1d4ed8;
                --blue-2: #2563eb;
                --blue-3: #eff6ff;
                --green-1: #16a34a;
                --text-1: #0f172a;
                --text-2: #475569;
                --border-1: #dbeafe;
                --surface-1: #ffffff;
                --surface-2: #f8fafc;
            }
            .stApp {
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                color: var(--text-1);
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }
            [data-testid="stSidebar"] {
                background: #f8fbff;
                border-right: 1px solid #e2e8f0;
            }
            [data-testid="stSidebar"] > div:first-child {
                background: #f8fbff;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }
            .top-shell {
                background: linear-gradient(135deg, #ffffff 0%, #f8fbff 68%, #eef4ff 100%);
                border: 1px solid #dbeafe;
                border-radius: 24px;
                padding: 1.4rem 1.5rem 1.2rem;
                box-shadow: 0 20px 45px rgba(37, 99, 235, 0.08);
                margin-bottom: 1.25rem;
            }
            .empty-panel {
                background: #ffffff;
                border: 1px dashed #93c5fd;
                border-radius: 14px;
                padding: 1.2rem;
                color: var(--text-2);
                text-align: center;
                box-shadow: 0 8px 20px rgba(37, 99, 235, 0.04);
            }
            .page-title {
                color: var(--text-1);
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
            }
            .page-subtitle {
                color: var(--text-2);
                font-size: 1rem;
                font-weight: 500;
                margin-bottom: 1rem;
            }
            .feature-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin-top: 0.85rem;
            }
            .feature-tile {
                background: #ffffff;
                border: 1px solid #dbeafe;
                border-radius: 16px;
                padding: 0.9rem 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            }
            .feature-kicker {
                color: #2563eb;
                font-size: 0.82rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .feature-copy {
                color: var(--text-1);
                font-size: 0.95rem;
                font-weight: 600;
                margin-top: 0.35rem;
                line-height: 1.4;
            }
            .section-intro {
                background: var(--surface-2);
                border: 1px solid var(--border-1);
                border-radius: 16px;
                padding: 1rem 1.15rem;
                color: var(--text-2);
                margin-bottom: 1rem;
            }
            .product-card {
                background: var(--surface-1);
                border: 1px solid var(--border-1);
                border-radius: 12px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
                min-height: 280px;
                margin-bottom: 1rem;
            }
            .product-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 18px 40px rgba(37, 99, 235, 0.14);
                border-color: #bfdbfe;
            }
            .mini-card {
                min-height: 220px;
            }
            .product-name {
                color: var(--text-1);
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.45;
                margin-bottom: 0.7rem;
            }
            .meta-row {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
                align-items: center;
                margin-bottom: 0.75rem;
            }
            .category-badge, .score-badge {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
            }
            .score-badge {
                background: #eff6ff;
                color: var(--blue-1);
            }
            .price-tag {
                font-size: 1.08rem;
                font-weight: 700;
                color: var(--green-1);
                margin-bottom: 0.7rem;
            }
            .description-text {
                color: var(--text-2);
                font-size: 0.93rem;
                line-height: 1.55;
            }
            .match-label {
                color: var(--text-2);
                font-size: 0.82rem;
                font-weight: 600;
                margin: 0.1rem 0 0.35rem;
            }
            .progress-shell {
                width: 100%;
                height: 9px;
                background: #e8eef8;
                border-radius: 999px;
                overflow: hidden;
                margin-bottom: 0.75rem;
            }
            .progress-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%);
            }
            .source-label {
                font-size: 0.85rem;
                color: #334155;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .sidebar-box {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }
            .sidebar-title {
                font-size: 1.1rem;
                font-weight: 800;
                color: var(--text-1);
                margin-bottom: 0.25rem;
            }
            .sidebar-subtitle {
                color: var(--text-2);
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
            }
            .category-stack {
                display: grid;
                gap: 0.65rem;
                margin-top: 0.8rem;
            }
            .category-chip {
                display: flex;
                align-items: center;
                gap: 0.6rem;
                padding: 0.75rem 0.8rem;
                border-radius: 14px;
                border: 1px solid #e2e8f0;
                background: #ffffff;
            }
            .category-icon {
                width: 34px;
                height: 34px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.95rem;
            }
            .category-copy {
                color: var(--text-2);
                font-size: 0.9rem;
                line-height: 1.35;
            }
            .category-name {
                color: var(--text-1);
                font-weight: 700;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.75rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: #f8fbff;
                border: 1px solid var(--border-1);
                border-radius: 12px;
                padding: 0.7rem 1rem;
                color: var(--text-1);
                font-weight: 700;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
                color: white !important;
                border-color: transparent !important;
            }
            .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
                border-radius: 12px !important;
                border: 1px solid var(--border-1) !important;
                background: white !important;
                min-height: 3rem;
            }
            .stTextInput label, .stSelectbox label {
                color: var(--text-1) !important;
                font-weight: 600 !important;
            }
            .stButton > button {
                border-radius: 999px !important;
                border: 1px solid #bfdbfe !important;
                font-weight: 700 !important;
                box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08);
                min-height: 2.8rem;
                color: #0f172a !important;
                background: #ffffff !important;
            }
            .search-button button,
            .find-button button,
            .ask-button button {
                background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
                color: white !important;
                border-color: transparent !important;
            }
            .example-blue button {
                background: #eff6ff !important;
                color: #1d4ed8 !important;
            }
            .example-green button {
                background: #ecfdf3 !important;
                color: #15803d !important;
            }
            .example-orange button {
                background: #fff7ed !important;
                color: #c2410c !important;
            }
            .example-purple button {
                background: #f5f3ff !important;
                color: #6d28d9 !important;
            }
            @media (max-width: 980px) {
                .feature-strip {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_products() -> list[dict]:
    """Load all products from the local JSON catalog."""
    try:
        with PRODUCTS_FILE.open("r", encoding="utf-8") as file:
            products = json.load(file)
        return products
    except FileNotFoundError:
        st.error("products.json was not found. Please keep it in the project folder.")
    except json.JSONDecodeError as error:
        st.error(f"products.json could not be parsed: {error}")
    except Exception as error:
        st.error(f"Unexpected error while loading products: {error}")
    return []


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence transformer used for semantic search."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_generator():
    """Load and cache the FLAN-T5 generator, with a fallback for newer transformers builds."""
    try:
        return {
            "mode": "pipeline",
            "generator": pipeline("text2text-generation", model=LLM_MODEL),
        }
    except Exception:
        # Some newer transformers builds omit the classic text2text pipeline task name.
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        return {
            "mode": "manual",
            "tokenizer": tokenizer,
            "model": model,
        }


def generate_text(prompt: str) -> str:
    """Generate text from FLAN-T5 using either the requested pipeline or a direct-model fallback."""
    generator_bundle = get_generator()

    if generator_bundle["mode"] == "pipeline":
        response = generator_bundle["generator"](
            prompt,
            max_new_tokens=160,
            do_sample=False,
        )
        return response[0]["generated_text"].strip() if response else ""

    tokenizer = generator_bundle["tokenizer"]
    model = generator_bundle["model"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def connect_client() -> Endee:
    """Create an Endee client configured for the local vector database."""
    client = Endee()
    client.set_base_url(BASE_URL)
    return client


def get_index():
    """Fetch the products index and surface a helpful error if it is missing."""
    try:
        client = connect_client()
        return client.get_index(name=INDEX_NAME)
    except Exception as error:
        raise RuntimeError(
            "Could not connect to the Endee 'products' index. "
            "Start Endee and run 'python load_data.py' first."
        ) from error


def build_embedding_text(product: dict) -> str:
    """Create a rich text representation so product meaning is encoded well."""
    return (
        f"Name: {product['name']}. "
        f"Category: {product['category']}. "
        f"Description: {product['description']}. "
        f"Price: Rs. {product['price']}."
    )


def encode_text(text: str) -> list[float]:
    """Encode text into a normalized 384-dimensional embedding."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def extract_matches(raw_results) -> list[dict]:
    """Normalize Endee query responses into a plain list of match dictionaries."""
    if isinstance(raw_results, dict):
        for key in ("results", "matches", "data"):
            value = raw_results.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(raw_results, list):
        return raw_results
    return []


def similarity_to_percent(similarity_value) -> float:
    """Convert a similarity score into a user-friendly percentage."""
    try:
        similarity = float(similarity_value)
        similarity = max(0.0, min(1.0, similarity))
        return round(similarity * 100, 2)
    except Exception:
        return 0.0


def semantic_search(query, top_k=5):
    """Run semantic similarity search in Endee and return the top matching products."""
    try:
        if not query or not query.strip():
            return []

        index = get_index()
        query_embedding = encode_text(query.strip())
        raw_results = index.query(vector=query_embedding, top_k=top_k, ef=128)
        matches = extract_matches(raw_results)

        formatted_results = []
        for item in matches:
            meta = item.get("meta", {})
            formatted_results.append(
                {
                    "id": item.get("id", ""),
                    "name": meta.get("name", "Unknown Product"),
                    "description": meta.get("description", ""),
                    "category": meta.get("category", "Unknown"),
                    "price": meta.get("price", 0),
                    "similarity_percent": similarity_to_percent(item.get("similarity", 0)),
                }
            )

        return formatted_results
    except Exception as error:
        st.error(f"Semantic search failed: {error}")
        return []


def rag_answer(question):
    """Retrieve relevant products, build context, and generate an answer with FLAN-T5."""
    try:
        if not question or not question.strip():
            return {"answer": "", "sources": []}

        index = get_index()
        question_embedding = encode_text(question.strip())
        raw_results = index.query(vector=question_embedding, top_k=3, ef=128)
        matches = extract_matches(raw_results)

        sources = []
        context_lines = []
        for item in matches:
            meta = item.get("meta", {})
            source = {
                "id": item.get("id", ""),
                "name": meta.get("name", "Unknown Product"),
                "description": meta.get("description", ""),
                "category": meta.get("category", "Unknown"),
                "price": meta.get("price", 0),
                "similarity_percent": similarity_to_percent(item.get("similarity", 0)),
            }
            sources.append(source)
            context_lines.append(
                f"Product: {source['name']} | Category: {source['category']} | "
                f"Price: Rs. {source['price']} | Description: {source['description']}"
            )

        if not sources:
            return {
                "answer": "I could not find any relevant products in the catalog for that question.",
                "sources": [],
            }

        prompt = (
            "You are an AI shopping assistant. Answer the user's question using only the provided "
            "product context. Mention specific products when useful, compare options briefly, and "
            "be honest if the catalog does not fully answer the question.\n\n"
            f"Question: {question.strip()}\n\n"
            "Context:\n"
            + "\n".join(context_lines)
            + "\n\nAnswer:"
        )

        answer_text = generate_text(prompt)

        if not answer_text:
            answer_text = "I found relevant products, but I could not generate a detailed answer right now."

        return {"answer": answer_text, "sources": sources}
    except Exception as error:
        st.error(f"RAG answer generation failed: {error}")
        return {
            "answer": "An error occurred while generating the AI answer.",
            "sources": [],
        }


def get_recommendations(product_id, top_k=4):
    """Look up a product vector in Endee and return the most similar alternatives."""
    try:
        if not product_id:
            return []

        index = get_index()
        vector_record = index.get_vector(product_id)
        product_vector = vector_record.get("vector") if isinstance(vector_record, dict) else None

        if not product_vector:
            st.warning("The selected product embedding was not found in Endee.")
            return []

        raw_results = index.query(
            vector=product_vector,
            top_k=top_k + 1,
            ef=128,
        )
        matches = extract_matches(raw_results)

        recommendations = []
        for item in matches:
            if item.get("id") == product_id:
                continue

            meta = item.get("meta", {})
            recommendations.append(
                {
                    "id": item.get("id", ""),
                    "name": meta.get("name", "Unknown Product"),
                    "description": meta.get("description", ""),
                    "category": meta.get("category", "Unknown"),
                    "price": meta.get("price", 0),
                    "similarity_percent": similarity_to_percent(item.get("similarity", 0)),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations
    except Exception as error:
        st.error(f"Recommendation lookup failed: {error}")
        return []


def category_badge(category: str) -> str:
    """Return an HTML badge for the given product category."""
    color = CATEGORY_COLORS.get(category, "#475569")
    return (
        f"<span class='category-badge' style='background:{color}; color:white;'>{category}</span>"
    )


def render_product_card(product: dict, show_match: bool = True, compact: bool = False) -> None:
    """Render one product result as a styled HTML card."""
    description = product.get("description", "")
    if len(description) > (150 if compact else 190):
        description = description[: (147 if compact else 187)].rstrip() + "..."

    match_markup = ""
    progress_markup = ""
    if show_match:
        match_markup = (
            f"<span class='score-badge'>Match {product.get('similarity_percent', 0)}%</span>"
        )
        progress_markup = (
            "<div class='match-label'>Semantic match</div>"
            "<div class='progress-shell'>"
            f"<div class='progress-fill' style='width: {product.get('similarity_percent', 0)}%;'></div>"
            "</div>"
        )

    card_class = "product-card mini-card" if compact else "product-card"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="product-name">{product.get('name', 'Unknown Product')}</div>
            <div class="meta-row">
                {category_badge(product.get('category', 'Unknown'))}
                {match_markup}
            </div>
            <div class="price-tag">&#8377;{product.get('price', 0):,}</div>
            {progress_markup}
            <div class="description-text">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Seed session state values used by example buttons and form interactions."""
    defaults = {
        "semantic_query": "",
        "rag_question": "",
        "selected_product_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_semantic_query(query: str) -> None:
    """Populate the semantic search text input from an example button."""
    st.session_state["semantic_query"] = query


def set_rag_question(question: str) -> None:
    """Populate the RAG question input from an example button."""
    st.session_state["rag_question"] = question


def render_sidebar() -> None:
    """Render a minimal modern sidebar for quick browsing."""
    with st.sidebar:
        category_markup = []
        for category, color in CATEGORY_COLORS.items():
            icon = CATEGORY_ICONS.get(category, "&#128717;")
            category_markup.append(
                f"<div class='category-chip'>"
                f"<div class='category-icon' style='background:{color};'>{icon}</div>"
                f"<div class='category-copy'>"
                f"<div class='category-name'>{category}</div>"
                "Browse this collection"
                f"</div>"
                f"</div>"
            )

        st.markdown(
            """
            <div class="sidebar-box">
                <div class="sidebar-title">Explore Catalog</div>
                <div class="sidebar-subtitle">Clean browsing for product discovery, answers, and recommendations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-box">
                <div class="sidebar-title">Collections</div>
                <div class="category-stack">
                    {''.join(category_markup)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_semantic_tab() -> None:
    """Build the semantic search workflow and results UI."""
    st.markdown(
        """
        <div class="section-intro">
            <strong>Semantic Search</strong><br>
            Describe the product you want in natural language and the app will find the closest matches from the vector index.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input(
        "Describe what you want",
        key="semantic_query",
        placeholder="Example: lightweight laptop for students",
    )

    example_columns = st.columns(4)
    example_queries = [
        "lightweight laptop for students",
        "warm clothes for winter",
        "books for self improvement",
        "budget phone with good camera",
    ]
    example_classes = ["example-blue", "example-green", "example-orange", "example-purple"]
    for column, example, css_class in zip(example_columns, example_queries, example_classes):
        with column:
            st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
            st.button(example, use_container_width=True, on_click=set_semantic_query, args=(example,))
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='search-button'>", unsafe_allow_html=True)
    if st.button("Search", type="primary", use_container_width=True):
        results = semantic_search(st.session_state["semantic_query"], top_k=5)
        if results:
            st.success(f"Found {len(results)} relevant products.")
            columns = st.columns(2)
            for index, product in enumerate(results):
                with columns[index % 2]:
                    render_product_card(product, show_match=True)
        else:
            st.markdown(
                """
                <div class="empty-panel">
                    No search results were found for that query. Try describing the product in a broader or more natural way.
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_rag_tab() -> None:
    """Build the RAG question-answering workflow and show supporting sources."""
    st.markdown(
        """
        <div class="section-intro">
            <strong>Ask AI (RAG)</strong><br>
            Ask a product question in plain English. The app retrieves the most relevant products first, then generates an answer from that context.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input(
        "Ask anything about products",
        key="rag_question",
        placeholder="Example: best laptop under 50000?",
    )

    question_columns = st.columns(3)
    sample_questions = [
        "best laptop under 50000?",
        "what to buy for winter trekking?",
        "suggest home workout equipment",
    ]
    for column, question in zip(question_columns, sample_questions):
        with column:
            st.button(question, use_container_width=True, on_click=set_rag_question, args=(question,))

    st.markdown("<div class='ask-button'>", unsafe_allow_html=True)
    if st.button("Ask AI", type="primary", use_container_width=True):
        response = rag_answer(st.session_state["rag_question"])
        if response["answer"]:
            st.info(response["answer"])

        if response["sources"]:
            st.success("Source products used for the answer")
            source_columns = st.columns(3)
            for index, product in enumerate(response["sources"]):
                with source_columns[index % 3]:
                    st.markdown("<div class='source-label'>Retrieved for the answer</div>", unsafe_allow_html=True)
                    render_product_card(product, show_match=True, compact=True)
        else:
            st.markdown(
                """
                <div class="empty-panel">
                    No supporting products were retrieved for that question. Try asking about budget, category, or use case.
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_recommendation_tab(products: list[dict]) -> None:
    """Build the similar-products workflow using a selected catalog item."""
    st.markdown(
        """
        <div class="section-intro">
            <strong>Similar Products</strong><br>
            Choose a product from the catalog and discover nearby items based on stored vector similarity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    product_lookup = {product["name"]: product["id"] for product in products}
    product_names = sorted(product_lookup.keys())

    if not st.session_state["selected_product_name"] and product_names:
        st.session_state["selected_product_name"] = product_names[0]

    st.selectbox(
        "Choose a product",
        options=product_names,
        key="selected_product_name",
    )

    st.markdown("<div class='find-button'>", unsafe_allow_html=True)
    if st.button("Find Similar", type="primary", use_container_width=True):
        selected_id = product_lookup.get(st.session_state["selected_product_name"], "")
        recommendations = get_recommendations(selected_id, top_k=4)
        if recommendations:
            columns = st.columns(2)
            for index, product in enumerate(recommendations):
                with columns[index % 2]:
                    render_product_card(product, show_match=True)
        else:
            st.markdown(
                """
                <div class="empty-panel">
                    No similar products were found for the selected item right now.
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Assemble the Streamlit page with semantic search, RAG, and recommendations."""
    try:
        inject_custom_css()
        initialize_state()
        products = load_products()
        render_sidebar()

        st.markdown(
            """
            <div class="top-shell">
                <div class="page-title">&#128722; AI Product Search Engine</div>
                <div class="page-subtitle">Discover products with natural-language search, AI answers, and similarity-driven exploration.</div>
                <div class="feature-strip">
                    <div class="feature-tile">
                        <div class="feature-kicker">Search</div>
                        <div class="feature-copy">Find relevant products by intent instead of exact keywords.</div>
                    </div>
                    <div class="feature-tile">
                        <div class="feature-kicker">Ask</div>
                        <div class="feature-copy">Get concise AI guidance grounded in the catalog you already loaded.</div>
                    </div>
                    <div class="feature-tile">
                        <div class="feature-kicker">Discover</div>
                        <div class="feature-copy">Explore similar items and compare options without leaving the page.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        search_tab, rag_tab, recommendations_tab = st.tabs(
            ["\U0001F50D Semantic Search", "\U0001F916 Ask AI (RAG)", "\U0001F4A1 Similar Products"]
        )

        with search_tab:
            render_semantic_tab()

        with rag_tab:
            render_rag_tab()

        with recommendations_tab:
            render_recommendation_tab(products)

    except Exception as error:
        st.error(f"The application failed to start correctly: {error}")


if __name__ == "__main__":
    smart_main()


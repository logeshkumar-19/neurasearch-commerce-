import html
from typing import Any

import streamlit as st

from ecommerce_app.config import CATEGORY_COLORS, SUGGESTIONS
from ecommerce_app.services.assistant import rag_answer, shopping_agent
from ecommerce_app.services.catalog import get_recommendations, semantic_search


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
    color = CATEGORY_COLORS.get(category, "#4b5563")
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
        if show_match
        else ""
    )
    progress = (
        "<div class='match'><span>Semantic match</span>"
        f"<span>{score:.1f}%</span></div><div class='bar'><div class='fill' style='width:{score}%;'></div></div>"
        if show_match
        else ""
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


def render_header() -> None:
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
        st.text_input(
            "Search",
            key="search_query",
            label_visibility="collapsed",
            placeholder="Search for products by intent, budget, or use case",
        )
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
    st.markdown(
        "<div class='copy'>A curated starter shelf that makes the experience feel polished before the first search.</div>",
        unsafe_allow_html=True,
    )
    product_grid(top_picks(products), show_match=False)
    st.markdown("</div>", unsafe_allow_html=True)


def render_rag() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Ask AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='copy'>Retrieve the most relevant products first, then answer using that catalog context only.</div>",
        unsafe_allow_html=True,
    )
    st.text_input(
        "RAG",
        key="rag_question",
        label_visibility="collapsed",
        placeholder="Example: What is the best laptop under 50000 for a student?",
    )
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
    st.markdown(
        "<div class='copy'>Select a product and explore nearby alternatives without repeating the original item.</div>",
        unsafe_allow_html=True,
    )
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
    st.markdown(
        "<div class='copy'>Ask for the best product, compare options, or understand why something fits your intent.</div>",
        unsafe_allow_html=True,
    )
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
            st.session_state["agent_result"] = shopping_agent(
                st.session_state["agent_prompt"],
                selected["id"] if selected else None,
            )
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



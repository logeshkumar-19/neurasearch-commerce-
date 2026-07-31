from typing import Any

import streamlit as st

from ecommerce_app.config import LLM_MODEL
from ecommerce_app.services.catalog import get_recommendations, load_products, semantic_search


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
    if not question or not question.strip():
        return {"answer": "", "sources": [], "error": "Ask a product question to use RAG."}
    sources, error = semantic_search(question, top_k=top_k)
    if error:
        return {"answer": "", "sources": [], "error": error}
    if not sources:
        return {"answer": "I could not find relevant products in the catalog for that question.", "sources": [], "error": None}

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



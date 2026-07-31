import streamlit as st

from ecommerce_app.services.catalog import load_products
from ecommerce_app.ui.components import (
    init_state,
    render_header,
    render_recommendations,
    render_search,
    render_top_picks,
)
from ecommerce_app.ui.styles import inject_css


def main() -> None:
    st.set_page_config(
        page_title="NeuraSearch Commerce",
        page_icon="🛍️",
        layout="wide",
    )
    inject_css()
    init_state()
    try:
        products = load_products()
    except Exception as error:
        st.error(f"Unable to load the product catalog: {error}")
        return

    render_header()
    render_search()
    render_top_picks(products)

    render_recommendations(products)





import streamlit as st

from ecommerce_app.services.catalog import load_products
from ecommerce_app.ui.components import init_state, render_agent
from ecommerce_app.ui.styles import inject_css


def main() -> None:
    st.set_page_config(
        page_title="AI Agent | NeuraSearch Commerce",
        page_icon="🤖",
        layout="wide",
    )
    inject_css()
    init_state()
    try:
        products = load_products()
    except Exception as error:
        st.error(f"Unable to load the product catalog: {error}")
        return

    st.markdown(
        """
        <div class="panel">
            <div class="title">AI Shopping Agent</div>
            <div class="copy">Compare products, ask for recommendations, or request a quick explanation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_agent(products)


if __name__ == "__main__":
    main()





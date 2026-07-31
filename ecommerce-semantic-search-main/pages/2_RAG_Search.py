import streamlit as st

from ecommerce_app.ui.components import init_state, render_rag
from ecommerce_app.ui.styles import inject_css


def main() -> None:
    st.set_page_config(
        page_title="RAG Search | NeuraSearch Commerce",
        page_icon="🧠",
        layout="wide",
    )
    inject_css()
    init_state()

    st.markdown(
        """
        <div class="panel">
            <div class="title">RAG Search</div>
            <div class="copy">Ask product questions and get grounded answers built from your catalog.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_rag()


if __name__ == "__main__":
    main()



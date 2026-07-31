# NeuraSearch Commerce

NeuraSearch Commerce is a semantic product search and AI shopping assistant built with Streamlit. It combines vector search, retrieval-augmented answers, and a guided AI agent to help shoppers discover the right products quickly.

**Author:** logesh  
**Contact:** logeshp297@gmail.com  
**Repository:** https://github.com/logeshkumar-19/endee.git

## What You Get
- Semantic product search by intent, budget, and use case
- RAG-powered Q&A that cites retrieved catalog items
- AI shopping agent for recommendations and comparisons
- Clean multi-page Streamlit experience

## Tech Stack
- Streamlit UI
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- Endee Vector DB (optional; local fallback supported)
- FLAN-T5 for grounded generation

## Quick Start
1. Create/activate the virtual environment at `..\.venv`
2. Install dependencies: `..\.venv\Scripts\python -m pip install -r requirements.txt`
3. Start Endee: `docker run -d -p 8080:8080 --name endee-server endeeio/endee-server:latest`
4. Load data: `..\.venv\Scripts\python load_data.py`
5. Run the app: `..\.venv\Scripts\streamlit run app.py`

## Project Structure
```text
ecommerce-semantic-search-main/
├── app.py
├── products.json
├── ecommerce_app/
│   ├── services/
│   └── ui/
└── pages/
    ├── 2_RAG_Search.py
    └── 3_AI_Agent.py
```

## Notes
- If model downloads fail on Windows, clear any proxy variables like `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`.





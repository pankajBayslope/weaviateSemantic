import os
import streamlit as st
import weaviate
from dotenv import load_dotenv

# =============================
# Load ENV
# =============================
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# =============================
# Weaviate Client
# =============================
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(
        api_key=WEAVIATE_API_KEY
    ),
    additional_headers={
        # 🔥 MOST IMPORTANT FIX
        "X-Weaviate-Cluster-Url": WEAVIATE_URL
    }
)


# =============================
# ✅ EXACT CLASS NAME (FIXED)
# =============================
PATENT_CLASS = "Patcat"

# =============================
# Hybrid Search Function (SAFE)
# =============================
def hybrid_search(query, alpha=0.75, limit=5):
    response = (
        client.query
        .get(
            PATENT_CLASS,
            [
                "patentNumber",
                "title",
                "abstract",
                "industryDomain",
                "technologyArea",
                "subTechnologyArea",
                "keywords",
            ]
        )
        .with_hybrid(
            query=query,
            alpha=alpha,
            properties=["title", "abstract", "keywords"]
        )
        .with_limit(limit)
        .with_additional(["score", "explainScore"])
        .do()
    )

    # ---------- ERROR HANDLING ----------
    if "errors" in response:
        return [], response["errors"]

    if "data" not in response:
        return [], "No data returned from Weaviate"

    return response["data"]["Get"].get(PATENT_CLASS, []), None


# =============================
# Streamlit UI
# =============================
st.set_page_config(
    page_title="Patent Hybrid Semantic Search",
    layout="wide"
)

st.title("🔍 Patent Hybrid Semantic Search")
st.caption("Weaviate | BM25 + Semantic Vector | Class: Patcat")

st.success("✅ Connected to Weaviate class: Patcat")

query = st.text_input(
    "Search patent idea / technology / problem",
    placeholder="e.g. AI based vehicle personalization system"
)

col1, col2 = st.columns(2)

with col1:
    alpha = st.slider(
        "Semantic vs Keyword (alpha)",
        0.0, 1.0, 0.75
    )

with col2:
    limit = st.slider("Results", 1, 20, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query")
    else:
        with st.spinner("Searching patents..."):
            results, error = hybrid_search(query, alpha, limit)

        if error:
            st.error(error)
        elif not results:
            st.info("No patents found")
        else:
            for i, p in enumerate(results, 1):
                st.markdown(f"## {i}. {p.get('title','')}")
                st.write(f"**Patent No:** {p.get('patentNumber','')}")
                st.write(f"**Industry:** {p.get('industryDomain','')}")
                st.write(
                    f"**Technology:** {p.get('technologyArea','')} → "
                    f"{p.get('subTechnologyArea','')}"
                )
                st.write(f"**Keywords:** {p.get('keywords','')}")
                st.markdown("**Abstract:**")
                st.write(p.get("abstract",""))
                st.caption(f"Score: {p['_additional']['score']}")
                st.divider()

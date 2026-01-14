import os
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import HybridFusion
from weaviate.auth import AuthApiKey
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# =============================
# ENV
# =============================
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("❌ Weaviate ENV variables not set")

# =============================
# CONSTANTS
# =============================
PATENT_CLASS = "Patcat"
ALPHA = 0.5
TOP_K = 30
WEAVIATE_FETCH = 80

# =============================
# LIFESPAN (modern replacement for startup/shutdown)
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Weaviate
    global client
    client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=120, insert=300)
    )
)


    if not client.is_ready():
        raise RuntimeError("Weaviate is not ready. Check URL/API key/cluster.")

    print("✅ Connected to Weaviate Cloud (v4 client)")

    yield  # Application runs here

    # Shutdown: close Weaviate client
    client.close()
    print("Weaviate client closed.")

# =============================
# FASTAPI APP with lifespan
# =============================
app = FastAPI(title="Patent Hybrid Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# EMBEDDER (load once at module level)
# =============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =============================
# HYBRID SEARCH
# =============================
def hybrid_search(query: str):
    if not query.strip():
        return []

    collection = client.collections.get(PATENT_CLASS)

    response = collection.query.hybrid(
        query=query,
        alpha=0.5,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        limit=WEAVIATE_FETCH,
        return_properties=[
            "patentNumber",
            "title",
            "abstract",
            "industryDomain",
            "technologyArea",
            "subTechnologyArea",
            "keywords",
        ]
    )

    docs_raw = response.objects
    if not docs_raw:
        return []

    docs = [obj.properties for obj in docs_raw]

    # Semantic scoring
    q_emb = embedder.encode(query, convert_to_numpy=True)

    def emb(text):
        return embedder.encode(text or "", convert_to_numpy=True)

    semantic_scores = []
    for d in docs:
        score = (
            0.25 * cosine_similarity([q_emb], [emb(d.get("industryDomain", ""))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("technologyArea", ""))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("subTechnologyArea", ""))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("keywords", ""))])[0][0]
        )
        semantic_scores.append(score)

    semantic_scores = np.array(semantic_scores)
    semantic_norm = np.zeros_like(semantic_scores) if semantic_scores.max() == semantic_scores.min() else \
        (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-9)

    # BM25
    corpus = [
        f"{d.get('title','')} {d.get('abstract','')} {d.get('keywords','')} "
        f"{d.get('technologyArea','')} {d.get('subTechnologyArea','')}"
        for d in docs
    ]
    tokenized = [re.findall(r"\w+", c.lower()) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))

    bm25_norm = np.zeros_like(bm25_scores) if bm25_scores.max() == bm25_scores.min() else \
        (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

    # Final score
    final_score = ALPHA * semantic_norm + (1 - ALPHA) * bm25_norm

    ranked = sorted(zip(docs, final_score), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:TOP_K]]

# =============================
# ENDPOINT
# =============================
class SearchQuery(BaseModel):
    query: str

@app.post("/search")
def search_patents(search: SearchQuery):
    results = hybrid_search(search.query)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
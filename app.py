import os
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import weaviate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Weaviate ENV variables not set")

# Constants
PATENT_CLASS = "Patcat"
ALPHA = 0.5
TOP_K = 30
WEAVIATE_FETCH = 80

# Weaviate Client
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers={"X-Weaviate-Cluster-Url": WEAVIATE_URL}
)

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Hybrid Search Function
def hybrid_search(query: str):
    # Fetch candidates from Weaviate
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
            alpha=0.5,
            properties=[
                "keywords",
                "title",
                "abstract",
                "technologyArea",
                "subTechnologyArea",
                "industryDomain"
            ]
        )
        .with_limit(WEAVIATE_FETCH)
        .do()
    )
    if "errors" in response:
        raise ValueError(response["errors"])
    docs = response["data"]["Get"].get(PATENT_CLASS, [])
    if not docs:
        return []

    # Semantic scoring
    q_emb = embedder.encode(query, convert_to_numpy=True)
    def emb(text):
        return embedder.encode(text or "", convert_to_numpy=True)
    semantic_scores = []
    for d in docs:
        score = (
            0.25 * cosine_similarity([q_emb], [emb(d.get("industryDomain"))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("technologyArea"))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("subTechnologyArea"))])[0][0] +
            0.25 * cosine_similarity([q_emb], [emb(d.get("keywords"))])[0][0]
        )
        semantic_scores.append(score)
    semantic_scores = np.array(semantic_scores)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (
        semantic_scores.max() - semantic_scores.min() + 1e-9
    )

    # BM25
    corpus = [
        f"{d.get('title','')} {d.get('abstract','')} {d.get('keywords','')} "
        f"{d.get('technologyArea','')} {d.get('subTechnologyArea','')}"
        for d in docs
    ]
    tokenized = [re.findall(r"\w+", c.lower()) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))
    bm25_norm = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-9
    )

    # Final score
    final_score = ALPHA * semantic_norm + (1 - ALPHA) * bm25_norm
    ranked = sorted(zip(docs, final_score), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:TOP_K]]

# FastAPI App
app = FastAPI(title="Patent Hybrid Search API")

# CORS for React frontend (allow localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class SearchQuery(BaseModel):
    query: str

# Search Endpoint
@app.post("/search")
def search_patents(search: SearchQuery):
    if not search.query.strip():
        raise HTTPException(status_code=400, detail="Please enter a search query")
    results = hybrid_search(search.query)
    if not results:
        return {"results": []}
    return {"results": results}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
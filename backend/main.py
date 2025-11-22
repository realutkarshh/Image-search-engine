# main.py (Image Search API)

import os
import math
import re
from collections import defaultdict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

# ------------------ Config ------------------ #

load_dotenv()

IMG_DB_URI = os.getenv("IMG_DB_URI") or os.getenv("IMG_MONGO_URI") or os.getenv("MONGO_URI")
IMG_DB_NAME = os.getenv("IMG_DB_NAME", "image_search_engine")

if not IMG_DB_URI:
    raise RuntimeError("IMG_DB_URI is not set in .env")

client = MongoClient(IMG_DB_URI)
db = client[IMG_DB_NAME]

IMG_DOCS = db["image_documents"]
IMG_INDEX = db["image_terms"]

# ------------------ FastAPI app ------------------ #

app = FastAPI(
    title="Image Search Engine API",
    description="TF-IDF based image search API",
    version="1.0.0",
)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Tokenizer ------------------ #

STOPWORDS = {
    "the","is","in","at","of","a","an","and","or","to","for","on","with","by",
    "this","that","it","as","are","was","were","be","from","which","into",
    "about","can","will","has","have","had","you","your","we","they","their",
    "our","not","image","jpg","jpeg","png","gif","webp"
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str):
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return tokens


# ------------------ Image Search Logic ------------------ #

def search_images(query: str, limit: int = 25):
    terms = tokenize(query)
    if not terms:
        return []

    # Fetch index entries for all query terms
    index_entries = list(IMG_INDEX.find({"term": {"$in": terms}}))
    if not index_entries:
        return []

    scores = defaultdict(float)

    # TF-IDF scoring
    for entry in index_entries:
        term = entry["term"]
        idf = entry.get("idf", 0.0)

        for posting in entry.get("docs", []):
            doc_id = posting["doc_id"]
            tf = posting["tf"]
            score = (1 + math.log(tf)) * idf
            scores[doc_id] += score

    if not scores:
        return []

    # Sort documents by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = sorted_docs[:limit]

    doc_ids = [d for d, _ in top_docs]
    normalized_ids = [ObjectId(d) if not isinstance(d, ObjectId) else d for d in doc_ids]

    # Fetch metadata
    cursor = IMG_DOCS.find(
        {"_id": {"$in": normalized_ids}},
        {
            "file_url": 1,
            "alt_text": 1,
            "caption_text": 1,
            "page_url": 1,
            "domain_name": 1,
            "format": 1,
            "snippet": 1,
        }
    )

    docs_by_id = {doc["_id"]: doc for doc in cursor}

    results = []
    for doc_id, score in top_docs:
        meta = docs_by_id.get(doc_id)
        if not meta:
            continue

        results.append({
            "id": str(doc_id),
            "file_url": meta.get("file_url", ""),
            "alt": meta.get("alt_text", ""),
            "caption": meta.get("caption_text", ""),
            "page_url": meta.get("page_url", ""),
            "domain": meta.get("domain_name", ""),
            "format": meta.get("format", ""),
            "snippet": meta.get("snippet", ""),
            "score": score,
        })

    return results


# ------------------ API Endpoints ------------------ #

@app.get("/search/images")
def image_search(q: str = Query(...), limit: int = 25):
    results = search_images(q, limit)
    return {
        "query": q,
        "count": len(results),
        "results": results
    }


@app.get("/")
def root():
    return {"message": "Image Search API. Use /search/images?q=your+query"}

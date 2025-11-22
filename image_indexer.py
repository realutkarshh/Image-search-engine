# image_indexer.py
"""
Image Indexer (Option D).
Indexes fields: alt text, caption text, page URL tokens, filename tokens, domain, format.
Writes:
 - image_documents collection: per-image metadata + token length
 - image_terms collection: inverted index entries with idf and postings (doc_id, tf)
"""

import os
import re
import math
from collections import defaultdict, Counter
from urllib.parse import urlparse, unquote
from os.path import basename

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

IMG_MONGO_URI = os.getenv("IMG_DB_URI") or os.getenv("IMG_MONGO_URI") or os.getenv("MONGO_URI")
IMG_DB_NAME = os.getenv("IMG_DB_NAME", "image_search_engine")

if not IMG_MONGO_URI:
    raise RuntimeError("IMG_DB_URI / IMG_MONGO_URI (or MONGO_URI) is not set in .env")

client = MongoClient(IMG_MONGO_URI)
db = client[IMG_DB_NAME]

IMAGE_COLL = db["image_files"]       # source collection (from crawler)
IMAGE_DOCS_COLL = db["image_documents"]
IMAGE_INDEX_COLL = db["image_terms"]

# ---------------- tokenization / stopwords ----------------

STOPWORDS = {
    "the", "is", "in", "at", "of", "a", "an", "and", "or", "to", "for",
    "on", "with", "by", "this", "that", "it", "as", "are", "was", "were",
    "be", "from", "which", "into", "about", "can", "will", "has", "have",
    "had", "you", "your", "we", "they", "their", "our", "not", "image", "jpg",
    "jpeg", "png", "webp", "gif"
}

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str):
    text = (text or "").lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return tokens


def tokens_from_url(u: str):
    """
    Extract tokens from URL path and filename.
    """
    if not u:
        return []
    try:
        parsed = urlparse(u)
        path = unquote(parsed.path or "")
    except Exception:
        path = u
    # split path parts and filename
    parts = re.split(r"[\/\-\._]+", path)
    parts = [p for p in parts if p]
    # also include hostname tokens
    try:
        host = urlparse(u).hostname or ""
    except Exception:
        host = ""
    host_tokens = re.split(r"[\.\-]+", host)
    all_tokens = parts + host_tokens
    joined = " ".join(all_tokens)
    return tokenize(joined)


# ---------------- indexing ----------------

def build_image_index():
    print("Fetching image documents from MongoDB...")

    projection = {
        "_id": 1,
        "file_url": 1,
        "alt_text": 1,
        "caption_text": 1,
        "page_url": 1,
        "domain_name": 1,
        "format": 1,
    }

    try:
        cursor = IMAGE_COLL.find({}, projection)
    except PyMongoError as e:
        print("Failed to query image_files collection:", e)
        return

    images = []
    for doc in cursor:
        try:
            images.append(doc)
        except Exception as e:
            print("Skipping problematic document while reading cursor:", e)
            continue

    if not images:
        print("No images found in 'image_files' collection. Run image crawler first.")
        return

    print(f"Found {len(images)} images. Building index...")

    inverted_index = defaultdict(lambda: defaultdict(int))  # term -> {doc_id: tf}
    doc_lengths = {}
    doc_metadata = {}

    for img in images:
        try:
            doc_id = img["_id"]
            file_url = img.get("file_url") or img.get("image_url") or ""
            alt = img.get("alt_text") or img.get("alt") or ""
            caption = img.get("caption_text") or img.get("caption") or ""
            page_url = img.get("page_url") or img.get("parent_url") or ""
            domain = img.get("domain_name") or img.get("site_name") or ""
            fmt = img.get("format") or img.get("image_type") or ""

            # Build a combined text that we will tokenize
            # Components: alt, caption, filename tokens, page tokens, domain, format
            filename = ""
            try:
                filename = basename(urlparse(file_url).path)
            except Exception:
                filename = ""

            parts = []
            if alt:
                parts.append(alt)
            if caption:
                parts.append(caption)
            if filename:
                # split filename into words
                parts.append(re.sub(r"[-_]+", " ", filename))
            if page_url:
                parts.append(page_url)
            if domain:
                parts.append(domain)
            if fmt:
                parts.append(fmt)

            combined = " ".join(parts)
            tokens = tokenize(combined)

            # also include tokens extracted from page_url and filename separately
            tokens += tokens_from_url(page_url)
            tokens += tokens_from_url(file_url)

            # dedupe tokens list? No — we want term frequency, so keep multiplicity.
            if not tokens:
                # skip indexing images with no textual signal
                continue

            doc_lengths[doc_id] = len(tokens)

            # snippet: prefer caption > alt > filename > page_url (short)
            snippet = caption or alt or (filename if filename else (page_url[:300] if page_url else ""))

            doc_metadata[doc_id] = {
                "file_url": file_url,
                "alt_text": alt,
                "caption_text": caption,
                "page_url": page_url,
                "domain_name": domain,
                "format": fmt,
                "snippet": snippet
            }

            tf_counter = Counter(tokens)
            for term, tf in tf_counter.items():
                inverted_index[term][doc_id] += tf

        except KeyError as e:
            print("Skipping doc due to missing key:", e)
            continue
        except Exception as e:
            print("Unexpected error while processing an image, skipping:", e)
            continue

    num_docs = len(doc_lengths)
    print(f"Indexed {num_docs} images with tokens.")

    if num_docs == 0:
        print("No images contained tokens. Aborting index build.")
        return

    # Build index documents
    index_docs = []
    for term, postings in inverted_index.items():
        df = len(postings)
        idf = math.log(num_docs / (1 + df))
        term_entry = {
            "term": term,
            "idf": float(idf),
            "docs": [
                {"doc_id": doc_id, "tf": int(tf)}
                for doc_id, tf in postings.items()
            ],
        }
        index_docs.append(term_entry)

    print(f"Built index for {len(index_docs)} unique terms.")

    # Persist results: drop old collections and insert fresh
    print("Dropping old 'image_documents' and 'image_terms' collections (if they exist)...")
    IMAGE_DOCS_COLL.drop()
    IMAGE_INDEX_COLL.drop()

    print("Inserting image metadata documents...")
    docs_bulk = []
    for doc_id, meta in doc_metadata.items():
        docs_bulk.append({
            "_id": doc_id,
            "file_url": meta["file_url"],
            "alt_text": meta["alt_text"],
            "caption_text": meta["caption_text"],
            "page_url": meta["page_url"],
            "domain_name": meta["domain_name"],
            "format": meta["format"],
            "length": doc_lengths[doc_id],
            "snippet": meta["snippet"]
        })

    if docs_bulk:
        IMAGE_DOCS_COLL.insert_many(docs_bulk)
    print(f"Inserted {len(docs_bulk)} documents into 'image_documents' collection.")

    print("Inserting index terms (this may take a moment)...")
    batch_size = 1000
    for i in range(0, len(index_docs), batch_size):
        batch = index_docs[i:i + batch_size]
        IMAGE_INDEX_COLL.insert_many(batch)
        print(f"Inserted {i + len(batch)} / {len(index_docs)} index terms...")

    print("Image index build complete ✅")


if __name__ == "__main__":
    build_image_index()

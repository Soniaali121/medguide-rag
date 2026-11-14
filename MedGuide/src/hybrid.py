"""Hybrid retrieval: dense (FAISS) + BM25 -> reciprocal rank fusion -> optional cross-encoder rerank.

Usage:
  python src/hybrid.py "your question" --mode dense|bm25|hybrid|rerank [--pool 100]

Setup: pip install rank-bm25
The reranker model (default BAAI/bge-reranker-base, roughly 1 GB) downloads on first use.
Set RERANK_MODEL to a smaller cross-encoder if it is too slow on your CPU.
"""
import argparse, json, os, re
from functools import lru_cache
from pathlib import Path
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss.index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
        "what", "how", "should", "be", "with", "who", "when", "which", "do", "does"}


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


@lru_cache(maxsize=1)
def load():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    index = faiss.read_index(str(INDEX))
    embedder = SentenceTransformer(EMBED_MODEL)
    bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])
    return chunks, index, embedder, bm25


@lru_cache(maxsize=1)
def load_reranker():
    return CrossEncoder(RERANK_MODEL)


def dense_rank(q: str, n: int):
    _, index, embedder, _ = load()
    v = embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)
    _, ids = index.search(np.asarray(v, dtype="float32"), n)
    return [int(i) for i in ids[0] if i >= 0]


def bm25_rank(q: str, n: int):
    bm25 = load()[3]
    scores = bm25.get_scores(tokenize(q))
    return [int(i) for i in np.argsort(-scores)[:n]]


def rrf(rank_lists, k: int = 60):
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank)."""
    score = {}
    for lst in rank_lists:
        for r, i in enumerate(lst, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def rerank(q: str, candidates, top_n=None):
    chunks = load()[0]
    ce = load_reranker()
    scores = ce.predict([(q, chunks[i]["embed_text"]) for i in candidates])
    order = [candidates[int(j)] for j in np.argsort(-scores)]
    return order[:top_n] if top_n else order


def search(q: str, mode: str = "hybrid", pool: int = 100):
    """Return chunk indices, best first. `pool` = candidates fetched per retriever
    (and, for mode 'rerank', how many fused candidates the reranker sees)."""
    if mode == "dense":
        return dense_rank(q, pool)
    if mode == "bm25":
        return bm25_rank(q, pool)
    fused = rrf([dense_rank(q, pool), bm25_rank(q, pool)])
    if mode == "hybrid":
        return fused
    if mode == "rerank":
        return rerank(q, fused[:pool])
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid", "rerank"])
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    chunks = load()[0]
    for r, i in enumerate(search(a.question, a.mode, a.pool)[: a.k], 1):
        c = chunks[i]
        print(f"[{r}] {c['section']}\n    {c['text'][:200]}...\n")

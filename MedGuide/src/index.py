"""Embed chunks with a free Hugging Face model and build a FAISS index."""
import json
from pathlib import Path
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss.index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small: runs fine on CPU


def main():
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode([c["embed_text"] for c in chunks], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim (vectors are normalised)
    index.add(np.asarray(vecs, dtype="float32"))
    faiss.write_index(index, str(INDEX))
    print(f"indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()

"""Baseline RAG: retrieve top-k chunks, then (optionally) answer with an open LLM.

Usage:
  python src/ask.py "What is the first-line treatment for ...?" --no-llm
  HF_TOKEN=hf_xxx python src/ask.py "your question"

HF_TOKEN is a free Hugging Face access token. The free Inference API has rate
limits, and model availability changes - if the default model fails, set
HF_MODEL to another instruct model that is available to you.
"""
import argparse, json, os
from pathlib import Path
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX = ROOT / "data" / "processed" / "faiss.index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "Cite passages like [1], [2]. If the context does not contain the answer, "
    "say exactly: 'I don't know based on the provided guidelines.' "
    "This is decision support, not medical advice."
)


def retrieve(question: str, k: int = 5):
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    index = faiss.read_index(str(INDEX))
    model = SentenceTransformer(EMBED_MODEL)
    q = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]


def answer(question: str, hits):
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=LLM, token=os.environ.get("HF_TOKEN"))
    ctx = "\n\n".join(f"[{i+1}] ({h['section']})\n{h['text']}" for i, h in enumerate(hits))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]
    out = client.chat_completion(messages=msgs, max_tokens=400, temperature=0.1)
    return out.choices[0].message.content


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    hits = retrieve(a.question, a.k)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.3f} | {h['section']}\n    {h['text'][:200]}...\n")
    if not a.no_llm:
        print("ANSWER:\n" + answer(a.question, hits))

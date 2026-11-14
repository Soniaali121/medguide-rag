"""Section-aware chunking.

Each chunk stays inside one section and gets its heading path prepended,
so the embedding knows the context. Long sections are split on sentence
boundaries with a small overlap.

Output: data/processed/chunks.jsonl
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "processed" / "sections.jsonl"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"

MAX_WORDS = 200
OVERLAP_WORDS = 30


def split_sentences(text: str):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def split_section(text: str, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
    sents, chunks, cur, cur_len = split_sentences(text), [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cur_len + w > max_words:
            chunks.append(" ".join(cur))
            tail, tl = [], 0  # carry last sentences as overlap
            for prev in reversed(cur):
                tl += len(prev.split())
                tail.insert(0, prev)
                if tl >= overlap:
                    break
            cur, cur_len = tail, sum(len(x.split()) for x in tail)
        cur.append(s)
        cur_len += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN.open(encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            sec = json.loads(line)
            path = " > ".join(sec["heading_path"]) or sec["title"]
            for piece in split_section(sec["text"]):
                rec = {
                    "chunk_id": f"{sec['doc_id']}::{n}",
                    "doc_id": sec["doc_id"],
                    "title": sec["title"],
                    "url": sec["url"],
                    "section": path,
                    "text": piece,
                    "embed_text": f"{path}\n{piece}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} chunks -> {OUT}")


if __name__ == "__main__":
    main()
